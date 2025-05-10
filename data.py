import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime

from utilities import ImgLatentDataset
from autoencs import AUTOENCS


def main(args):
    """
    Run a vae on full dataset and save the features.
    """
    assert (
        torch.cuda.is_available()
    ), "Extract features currently requires at least one GPU."

    # Setup DDP:
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup feature folders:
    output_dir = os.path.join(
        args.output_path,
        os.path.splitext(os.path.basename(args.config))[0],
        f"{args.data_split}_{args.image_size}",
    )
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Create model:
    vae = AUTOENCS[args.config](args.config, args.image_size)

    # Setup data:
    datasets = [
        ImageFolder(args.data_path, transform=vae.img_transform(p_hflip=0.0)),
        ImageFolder(args.data_path, transform=vae.img_transform(p_hflip=1.0)),
    ]
    samplers = [
        DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed
        )
        for dataset in datasets
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        for dataset, sampler in zip(datasets, samplers)
    ]
    total_data_in_loop = len(loaders[0].dataset)
    if rank == 0:
        print(f"Total data in one loop: {total_data_in_loop}")

    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(
                f"{datetime.now()} processing {run_images} of {total_data_in_loop} images"
            )
        save_filename = os.path.join(
            output_dir, f"latents_rank{rank:02d}_shard{saved_files:03d}.safetensors"
        )

        if not os.path.exists(save_filename):
            for loader_idx, data in enumerate(batch_data):
                x = data[0]
                y = data[1]  # (N,)

                z = vae.encode_images(x).detach().cpu()  # (N, C, H, W)

                if batch_idx == 0 and rank == 0:
                    print("latent shape", z.shape, "dtype", z.dtype)

                if loader_idx == 0:
                    latents.append(z)
                    labels.append(y)
                else:
                    latents_flip.append(z)
        else:
            latents.append(0)

        if len(latents) == 10000 // args.batch_size:
            if not os.path.exists(save_filename):
                latents = torch.cat(latents, dim=0)
                latents_flip = torch.cat(latents_flip, dim=0)
                labels = torch.cat(labels, dim=0)
                save_dict = {
                    "latents": latents,
                    "latents_flip": latents_flip,
                    "labels": labels,
                }
                for key in save_dict:
                    if rank == 0:
                        print(key, save_dict[key].shape)
                save_file(
                    save_dict,
                    save_filename,
                    metadata={
                        "total_size": f"{latents.shape[0]}",
                        "dtype": f"{latents.dtype}",
                        "device": f"{latents.device}",
                    },
                )
                if rank == 0:
                    print(f"Saved {save_filename}")

            latents = []
            latents_flip = []
            labels = []
            saved_files += 1

    # save remainder latents that are fewer than 10000 images
    if len(latents) > 0 and isinstance(latents[0], torch.Tensor):
        latents = torch.cat(latents, dim=0)
        latents_flip = torch.cat(latents_flip, dim=0)
        labels = torch.cat(labels, dim=0)
        save_dict = {"latents": latents, "latents_flip": latents_flip, "labels": labels}
        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        save_filename = os.path.join(
            output_dir, f"latents_rank{rank:02d}_shard{saved_files:03d}.safetensors"
        )
        save_file(
            save_dict,
            save_filename,
            metadata={
                "total_size": f"{latents.shape[0]}",
                "dtype": f"{latents.dtype}",
                "device": f"{latents.device}",
            },
        )
        if rank == 0:
            print(f"Saved {save_filename}")

    # Calculate latents stats
    dist.barrier()
    if rank == 0:
        dataset = ImgLatentDataset(output_dir, latent_norm=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/path/to/your/data/train")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="./buffers/data")
    parser.add_argument("--config", type=str, default="sdvae_f8c4")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
