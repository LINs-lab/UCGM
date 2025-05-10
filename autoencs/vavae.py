"""
VA_VAE by Maple (Jingfeng Yao) from HUST-VL
"""

import torch
from torchvision import transforms
from autoencs.autoencoder import AutoencoderKL
from autoencs.utils import center_crop_arr


class VA_VAE:
    """Vision-Aligned Variational Autoencoder (VAE) Implementation.

    This class implements a custom VAE architecture aligned with vision foundation models,
    using an AutoencoderKL backbone with specific channel multiplier configurations.
    """

    def __init__(self, name, img_size=256):
        """Initialize the Vision-Aligned VAE.

        Args:
            name (str): Model variant identifier. Currently supports:
                        - 'vavae_f16d32': 32-dim latent space variant
            img_size (int, optional): Default image processing size. Defaults to 256.
        """
        self.img_size = img_size
        self.load(name)

    def load(self, name):
        """Initialize and load pretrained VAE weights.

        Args:
            name (str): Model variant identifier matching available configurations.

        Returns:
            VA_VAE: The initialized model instance for method chaining.

        Raises:
            KeyError: If provided name doesn't match available configurations.

        Note:
            - Uses AutoencoderKL with embed_dim=32 and ch_mult=(1, 1, 2, 2, 4)
            - Automatically moves model to CUDA and sets to evaluation mode
            - Currently only supports 'vavae_f16d32' variant
        """
        name2path = {"vavae_f16d32": "./buffers/vaes/vavae_f16d32.ckpt"}

        if name not in name2path:
            raise KeyError(
                f"Unknown model name: {name}. Available options: {list(name2path.keys())}"
            )

        self.model = AutoencoderKL(embed_dim=32, ch_mult=(1, 1, 2, 2, 4))
        self.model.load_state_dict(torch.load(name2path[name]))
        self.model.cuda().eval()
        return self

    def img_transform(self, p_hflip=0, img_size=None):
        """Create image preprocessing transformation pipeline.

        Args:
            p_hflip (float, optional): Probability of horizontal flip [0, 1]. Defaults to 0.
            img_size (int, optional): Target image size. Uses instance default if None.

        Returns:
            transforms.Compose: Composition of transforms including:
                1. Center cropping to target size
                2. Optional horizontal flipping
                3. Conversion to tensor
                4. Normalization to [-1, 1] range (in-place)
        """
        img_size = img_size if img_size is not None else self.img_size
        img_transforms = [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
        return transforms.Compose(img_transforms)

    def encode_images(self, images):
        """Encode input images to latent space with stochastic sampling.

        Args:
            images (torch.Tensor): Input image tensor in [-1, 1] range with shape (B, C, H, W).

        Returns:
            torch.Tensor: Sampled latent representation from the posterior distribution.

        Note:
            - Uses stochastic sampling from the posterior (sample() instead of mean)
            - Automatically moves input to CUDA if available
            - Runs in no-gradient mode for inference
        """
        with torch.no_grad():
            posterior = self.model.encode(images.cuda())
            return posterior.sample()

    def decode_to_images(self, z):
        """Decode latent representations back to image space.

        Args:
            z (torch.Tensor): Latent tensor with shape matching the model's latent dimensions.

        Returns:
            torch.Tensor: Decoded images in [0, 1] range with shape (B, C, H, W).

        Note:
            - Output is clipped and scaled to [0, 1] range
            - Automatically moves input to CUDA if available
            - Runs in no-gradient mode for inference
            - Applies standard (z + 1)/2 scaling convention
        """
        with torch.no_grad():
            images = torch.clip((self.model.decode(z.cuda()) + 1) / 2, 0, 1)
        return images
