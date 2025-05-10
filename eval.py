import warnings

warnings.filterwarnings("ignore")

import os, argparse, torch, json
from tqdm import tqdm
import torch.distributed as dist
from accelerate import Accelerator
from torchvision.utils import save_image

# local imports
from networks import NETWORKS
from autoencs import AUTOENCS
from methods import UCGMTS, UniODE
from metrics import torch_quality_evaluate
from utilities import load_config, create_logger, parse_list


# sample function
@torch.no_grad
def do_sample(
    train_config,
    accelerator,
    model=None,
    demo_sample_mode=False,
    save_result=False,
):
    """
    Run sampling.
    """
    # get model
    latent_size = (
        train_config["data"]["image_size"] // train_config["vae"]["downsample_ratio"]
    )
    model = NETWORKS[train_config["model"]["type"]](
        input_size=latent_size,
        num_classes=train_config["data"]["num_classes"],
        in_channels=(train_config["model"]["in_chans"]),
    )
    vae = AUTOENCS[train_config["vae"]["type"]](train_config["vae"]["type"])

    checkpoint_dir = (
        f"{train_config['output_dir']}/{train_config['exp_name']}/checkpoints"
    )
    ckpt_path = f"{checkpoint_dir}/{train_config['sample']['ckpt']}"
    folder_name = f"evaluations/ckpt-{ckpt_path.split('/')[-1].split('.')[0]}".lower()
    sample_folder_dir = os.path.join(
        train_config["output_dir"], train_config["exp_name"], folder_name
    )

    if accelerator.is_main_process:
        logger = create_logger(sample_folder_dir, "eval")
        print(train_config["sample"])

        def report(loging, config):
            for name in config["sample"].keys():
                loging.info(f"{name}: {config['sample'][name]}")

    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)

    # Setup accelerator:
    device = accelerator.device
    seed = (
        train_config["train"]["global_seed"] * accelerator.num_processes
        + accelerator.process_index
    )
    torch.manual_seed(seed)
    rank = accelerator.local_process_index

    # Load model:
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    mode = train_config["sample"]["type"]

    if mode == "UNI":

        def sample_fn(z, model_fn, **model_kwargs):
            return UCGMTS(
                transport_type=train_config["transport"]["type"],
            ).uni_sample(
                inital_noise_z=z,
                sampling_model=model_fn,
                sampling_steps=train_config["sample"]["sampling_steps"],
                stochast_ratio=train_config["sample"]["stochast_ratio"],
                extrapol_ratio=train_config["sample"]["extrapol_ratio"],
                sampling_order=train_config["sample"]["sampling_order"],
                time_dist_ctrl=train_config["sample"]["time_dist_ctrl"],
                rfba_gap_steps=train_config["sample"]["rfba_gap_steps"],
                **model_kwargs,
            )

    elif mode == "ODE":

        def sample_fn(z, model_fn, **model_kwargs):
            return UniODE(
                transport_type=train_config["transport"]["type"],
            ).sample(
                z,
                model_fn,
                num_steps=train_config["sample"]["sampling_steps"],
                sampling_order=train_config["sample"]["sampling_order"],
                **model_kwargs,
            )

    else:
        raise NotImplementedError(f"Sampling mode {mode} is not supported.")

    # accelerator.wait_for_everyone()
    using_cfg = train_config["sample"]["cfg_scale"] > 0.0
    n = train_config["sample"]["per_batch_size"]
    global_batch_size = n * accelerator.num_processes
    total_samples = train_config["sample"]["fid_sample_num"]
    assert (
        total_samples % accelerator.num_processes == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert (
        samples_needed_this_gpu % n == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)

    # Generate labels:
    labels = torch.arange(train_config["data"]["num_classes"])
    labels = labels.repeat(total_samples // train_config["data"]["num_classes"])
    labels = labels[
        samples_needed_this_gpu * rank : samples_needed_this_gpu * (rank + 1)
    ]

    pbar = range(iterations)
    if not demo_sample_mode:
        pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    if train_config["data"]["latent_norm"]:
        mean_stad = torch.load(
            os.path.join(
                "./buffers/vaes/stat",
                f"{train_config['vae']['type']}_{train_config['data']['image_size']}.pt",
            )
        )
        latent_mean, latent_std = mean_stad["mean"].to(device), mean_stad["std"].to(
            device
        )
    else:
        latent_mean, latent_std = torch.tensor(0), torch.tensor(1)
    latent_multiplier = train_config["data"]["latent_multiplier"]

    # move to device
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)

    all_samples = []

    if demo_sample_mode:
        if accelerator.is_main_process:
            for label in tqdm(
                [975, 3, 207, 387, 388, 88, 979, 279], desc="Generating Demo Samples"
            ):
                z = torch.randn(
                    1, model.in_channels, latent_size, latent_size, device=device
                )
                y = torch.tensor([label], device=device)
                if using_cfg:
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([1000] * 1, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(
                        y=y,
                        cfg_scale=train_config["sample"]["cfg_scale"],
                        cfg_interval=train_config["sample"]["cfg_interval"],
                    )
                    model_fn = model.forward_with_cfg
                else:
                    model_kwargs = dict(y=y)
                    model_fn = model.forward
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                samples = (samples * latent_std) / latent_multiplier + latent_mean
                samples = vae.decode_to_images(samples).cpu()
                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                all_samples.append(samples)
    else:
        for i in pbar:
            # Sample inputs:
            z = torch.randn(
                n, model.in_channels, latent_size, latent_size, device=device
            )
            y = labels[(i * n) : (i * n + n)].to(device)

            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(
                    y=y,
                    cfg_scale=train_config["sample"]["cfg_scale"],
                    cfg_interval=train_config["sample"]["cfg_interval"],
                )
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = (samples * latent_std) / latent_multiplier + latent_mean
            samples = vae.decode_to_images(samples)
            samples = accelerator.gather(samples).cpu()

            if accelerator.is_main_process:
                all_samples.append(samples)
            total += global_batch_size
            accelerator.wait_for_everyone()

    if (not demo_sample_mode) and accelerator.is_main_process:
        # calculate FID
        all_samples = torch.cat(all_samples)[: train_config["sample"]["fid_sample_num"]]
        fid_reference_file = os.path.join(
            train_config["data"]["fid_reference_file"],
        )

        # metrics_dict = quality_evaluate(all_samples, fid_reference_file)
        metrics_dict = torch_quality_evaluate(all_samples, fid_reference_file)

        if save_result is True:
            result_dict = {"config": train_config["sample"], "result": metrics_dict}
            result_path = os.path.join(sample_folder_dir, "results.json")
            try:
                with open(result_path, "r") as f:
                    experiments = json.load(f)
            except FileNotFoundError:
                experiments = []
            experiments.append(result_dict)
            with open(result_path, "w") as f:
                json.dump(experiments, f, indent=2)

        report(logger, train_config)
        logger.info(metrics_dict)

        image_name = f"num={train_config['sample']['fid_sample_num']}_is={metrics_dict['inception_score_mean']:.3f}_fid={metrics_dict['frechet_inception_distance']:.3f}"

        save_image(
            all_samples[:64], os.path.join(sample_folder_dir, image_name + ".png")
        )
    elif accelerator.is_main_process:
        all_samples = torch.cat(all_samples)
        report(logger, train_config)
        logger.info(f"Pixel mean: {all_samples.mean().item()}")
        save_image(
            all_samples, os.path.join(sample_folder_dir, "demo_images.png"), nrow=4
        )

    return all_samples, sample_folder_dir


if __name__ == "__main__":

    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    # Parameters
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--type", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--cfg_interval", type=parse_list, default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--stochast_ratio", type=float, default=None)
    parser.add_argument("--extrapol_ratio", type=float, default=None)
    parser.add_argument("--sampling_order", type=int, default=None)
    parser.add_argument("--time_dist_ctrl", type=parse_list, default=None)
    parser.add_argument("--rfba_gap_steps", type=parse_list, default=None)
    parser.add_argument("--per_batch_size", type=int, default=None)
    parser.add_argument("--fid_sample_num", type=int, default=None)
    # Mode
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--save_result", action="store_true", default=False)
    args = parser.parse_args()
    accelerator = Accelerator()
    train_config = load_config(args.config)

    # Reset parameters
    for param_name in train_config["sample"].keys():
        param_value = getattr(args, param_name)
        if param_value is not None:
            train_config["sample"][param_name] = param_value

    # get ckpt_dir
    par_path = args.config.split("/")
    train_config["exp_name"] = os.path.join(par_path[-2], par_path[-1].split(".")[0])

    # naive sample
    all_samples, sample_folder_dir = do_sample(
        train_config,
        accelerator,
        demo_sample_mode=args.demo,
        save_result=args.save_result,
    )

    dist.destroy_process_group()
