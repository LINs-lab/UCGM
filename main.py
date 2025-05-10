import warnings

warnings.filterwarnings("ignore")

import torch
import torch.backends.cuda
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import argparse
from time import time
from copy import deepcopy
from accelerate import Accelerator

# local imports
from networks import NETWORKS
from optimers import OPTIMERS
from autoencs import AUTOENCS
from methods import UCGMTS, UniODE
from utilities import ImgLatentDataset
from utilities import create_logger, load_config
from utilities import set_seed, update_ema, remove_module_prefix, remove_module_all


def do_train(train_config, accelerator):
    """
    Training a generative model.
    """
    # Setup accelerator:
    device = accelerator.device
    rank = accelerator.process_index
    seed = train_config["train"]["global_seed"] * accelerator.num_processes + rank
    set_seed(seed)

    experiment_dir = f"{train_config['output_dir']}/{train_config['exp_name']}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(
            train_config["output_dir"], exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, "train")
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    downsample_ratio = train_config["vae"]["downsample_ratio"]
    assert (
        train_config["data"]["image_size"] % downsample_ratio == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config["data"]["image_size"] // downsample_ratio
    model = NETWORKS[train_config["model"]["type"]](
        input_size=latent_size,
        num_classes=train_config["data"]["num_classes"],
        in_channels=(train_config["model"]["in_chans"]),
    ).to(device)
    ema = deepcopy(model).requires_grad_(False).to(device)

    if accelerator.is_main_process:
        demoimages_dir = f"{experiment_dir}/demoimages"  # Stores saved model demoimages
        os.makedirs(demoimages_dir, exist_ok=True)

        if train_config["data"]["image_size"] == 256:
            demo_y = torch.tensor([975, 3, 207, 387, 388, 88, 979, 279], device=device)
        elif train_config["data"]["image_size"] == 512:
            demo_y = torch.tensor([975, 207], device=device)
        else:
            demo_y = torch.tensor(list(range(0, 16)), device=device)
        demo_z = torch.randn(
            len(demo_y), model.in_channels, latent_size, latent_size, device=device
        )

        vae = AUTOENCS[train_config["vae"]["type"]](train_config["vae"]["type"])
        logger.info("Loaded VAE model")

    trainer = UCGMTS(
        transport_type=train_config["transport"]["type"],
        lab_drop_ratio=train_config["transport"]["lab_drop_ratio"],
        consistc_ratio=train_config["transport"]["consistc_ratio"],
        enhanced_ratio=train_config["transport"]["enhanced_ratio"],
        scaled_cbl_eps=train_config["transport"]["scaled_cbl_eps"],
        ema_decay_rate=train_config["transport"]["ema_decay_rate"],
        enhanced_range=train_config["transport"]["enhanced_range"],
        time_dist_ctrl=train_config["transport"]["time_dist_ctrl"],
        wt_cosine_loss=train_config["transport"]["wt_cosine_loss"],
        weight_funcion=train_config["transport"]["weight_funcion"],
    )
    if accelerator.is_main_process:
        logger.info(
            f"{train_config['model']['type']} Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
        )
        logger.info(
            f"Optimizer: {train_config['optimizer']['type']}, lr={train_config['optimizer']['lr']}, beta1={train_config['optimizer']['beta1']}, beta2={train_config['optimizer']['beta2']}"
        )
        logger.info(f'Use cosine loss: {train_config["transport"]["wt_cosine_loss"]}')
        logger.info(f'Use weight func: {train_config["transport"]["weight_funcion"]}')

    opt = OPTIMERS[train_config["optimizer"]["type"]](
        model.parameters(),
        lr=train_config["optimizer"]["lr"],
        weight_decay=train_config["optimizer"]["weight_decay"],
        betas=(train_config["optimizer"]["beta1"], train_config["optimizer"]["beta2"]),
    )

    dataset = ImgLatentDataset(
        data_dir=train_config["data"]["data_path"],
        latent_norm=(train_config["data"]["latent_norm"]),
        latent_multiplier=(train_config["data"]["latent_multiplier"]),
    )
    batch_size_per_gpu = (
        train_config["train"]["global_batch_size"] // accelerator.num_processes
    )

    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    if accelerator.is_main_process:
        mean, stad, latent_multiplier = (
            dataset._latent_mean.cuda(),
            dataset._latent_std.cuda(),
            dataset.latent_multiplier,
        )
        logger.info(
            f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}"
        )
        logger.info(
            f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size"
        )

    if "ckpt" in train_config["train"]:
        checkpoint_path = f"{checkpoint_dir}/{train_config['train']['ckpt']}"
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["model"])
        if train_config["train"]["no_reopt"] is not True:
            opt.load_state_dict(checkpoint["opt"])
        if train_config["train"]["no_reuni"] is not True:
            if ((trainer.cor > 0.0) or (trainer.enr > 0.0)) and trainer.emd > 0.0:
                trainer.mod = deepcopy(model).requires_grad_(False).to(device)
            trainer.load_state_dict(checkpoint["unigen"])
        ema.load_state_dict(checkpoint["ema"])
        train_steps = int(checkpoint_path.split("/")[-1].split(".")[0])
        del checkpoint
        if accelerator.is_main_process:
            logger.info(f"Loaded checkpoint at: {checkpoint_path}.")
    else:
        train_steps = 0
        update_ema(ema, model, decay=0)
        if accelerator.is_main_process:
            logger.info("Starting training from scratch.")

    # Prepare models for training:
    model.train()
    ema.eval()
    if train_config["train"]["no_buffer"] is True:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    model, opt, loader, trainer = accelerator.prepare(model, opt, loader, trainer)

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    while True:
        for x, y in loader:
            if accelerator.mixed_precision == "no":
                x = x.to(device, dtype=torch.float32)
                y = y.to(device)
            else:
                x = x.to(device)
                y = y.to(device)

            loss = trainer.training_step(model, x, y)

            opt.zero_grad()
            accelerator.backward(loss)
            if "max_grad_norm" in train_config["optimizer"]:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), train_config["optimizer"]["max_grad_norm"]
                    )
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            opt.step()
            update_ema(ema, model, train_config["train"]["ema_decay"])

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % train_config["train"]["log_every"] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if (
                train_steps % train_config["train"]["ckpt_every"] == 0
                and train_steps > 0
            ):
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": remove_module_prefix(model.state_dict()),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "unigen": remove_module_all(trainer.state_dict()),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    demox = UniODE(
                        transport_typ=train_config["transport"]["type"],
                    ).sample(demo_z, ema, **dict(y=demo_y))
                    demox = demox[1::2].reshape(-1, *demox.shape[2:])
                    demox = (demox * stad) / latent_multiplier + mean
                    demox = vae.decode_to_images(demox).cpu()
                    demoimages_path = f"{demoimages_dir}/{train_steps:07d}.png"
                    save_image(demox, os.path.join(demoimages_path), nrow=len(demo_y))
                    logger.info(f"Saved demoimages to {demoimages_path}")
                    del checkpoint, demox

                dist.barrier()

            if train_steps >= train_config["train"]["max_steps"]:
                break
        if train_steps >= train_config["train"]["max_steps"]:
            break
    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator


if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    par_path = args.config.split("/")
    train_config["exp_name"] = os.path.join(par_path[-2], par_path[-1].split(".")[0])
    do_train(train_config, accelerator)
