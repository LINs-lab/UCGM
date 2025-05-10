import torch
from PIL import Image
import os
import threading
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch_fidelity


def process_0to1_arrays(data, output_dir, num_threads=8):
    """
    Process numpy arrays with values in 0-1 range and save as images using multiple threads.

    Args:
        data: Numpy array of images (either single or multiple)
              Expected shape: (num_images, channels, height, width)
        output_dir: Directory to save output images
        num_threads: Number of threads to use for parallel processing
    """

    def process_chunk(chunk, start_idx, output_dir):
        """Process a chunk of data and save as images"""
        for i, array in enumerate(chunk):
            idx = start_idx + i  # Global index
            try:
                # Convert from (C, H, W) to (H, W, C)
                array = np.transpose(array, (1, 2, 0))

                # Convert 0-1 float array to 0-255 uint8
                if array.max() <= 1.0:
                    array = (array * 255).astype(np.uint8)

                # Handle both grayscale and color images
                if array.ndim == 2:  # Grayscale
                    img = Image.fromarray(array, mode="L")
                elif array.ndim == 3:  # Color
                    img = Image.fromarray(array)

                img.save(
                    os.path.join(output_dir, f"image_{idx:05d}.png")
                )  # 使用5位数字填充
            except Exception as e:
                print(f"Error processing image {idx}: {str(e)}")

    # Split data into chunks for parallel processing
    chunks = np.array_split(data, num_threads)

    # Calculate correct start indices for each chunk
    chunk_sizes = [len(chunk) for chunk in chunks]
    start_indices = [0] + np.cumsum(chunk_sizes[:-1]).tolist()

    # Create and start threads
    threads = []
    for chunk, start_idx in zip(chunks, start_indices):
        thread = threading.Thread(
            target=process_chunk, args=(chunk, start_idx, output_dir)
        )
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()


class FIDDataset(Dataset):
    def __init__(self, data):

        self.images = data
        self.transform = transforms.Compose(
            [
                # transforms.ToTensor(),  # Converts to [0, 1] range
                transforms.Lambda(lambda x: x * 255),  # Rescale to [0, 255]
                transforms.Lambda(lambda x: x.to(torch.uint8)),
                # transforms.Lambda(
                #     lambda x: x.permute(1, 2, 0) if x.shape[0] == 3 else x
                # ),  # (H,W,C)
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img


def torch_quality_evaluate(data, fid_reference_file, output_dir=None):
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=FIDDataset(data),
        input2=None,
        fid_statistics_file=fid_reference_file,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )
    return metrics_dict

    # data = data.numpy()
    # process_0to1_arrays(data, output_dir, 50)
    #

    # metrics_dict = torch_fidelity.calculate_metrics(
    #     input1=output_dir,
    #     input2=None,
    #     fid_statistics_file=fid_reference_file,
    #     cuda=True,
    #     isc=True,
    #     fid=True,
    #     kid=False,
    #     prc=False,
    #     verbose=True,
    # )
    # return metrics_dict
