import torch
from torchvision import transforms
from diffusers.models import AutoencoderKL
from autoencs.utils import center_crop_arr


class SD_VAE:
    """Stable Diffusion Variational Autoencoder (VAE) implementation.

    This class provides an interface to Stability AI's pretrained VAE models used in Stable Diffusion.
    It supports both the EMA-regularized and MSE-optimized variants of the VAE.
    """

    def __init__(self, name, img_size=256):
        """Initialize the Stable Diffusion VAE wrapper.

        Args:
            name (str): Model variant identifier. Supported options:
                        - 'sdvae_ema_f8c4': EMA-regularized version
                        - 'sdvae_f8c4': MSE-optimized version
            img_size (int, optional): Default image size for processing. Defaults to 256.
        """
        self.img_size = img_size
        self.load(name)

    def load(self, name):
        """Load pretrained Stable Diffusion VAE model.

        Args:
            name (str): Model variant identifier matching available configurations.

        Returns:
            SD_VAE: The initialized model instance for method chaining.

        Raises:
            KeyError: If provided name doesn't match available configurations.

        Note:
            - Models can be loaded from either local paths or HuggingFace model hub
            - Automatically moves model to CUDA and sets to evaluation mode
        """
        name2type = {
            "sdvae_ema_f8c4": "/storage/lintaoLab/lintao/superdit/pretrains/stabilityai/sd-vae-ft-ema",
            "sdvae_f8c4": "stabilityai/sd-vae-ft-mse",
        }

        if name not in name2type:
            raise KeyError(
                f"Unknown model name: {name}. Available options: {list(name2type.keys())}"
            )

        self.model = AutoencoderKL.from_pretrained(name2type[name])
        self.model.cuda().eval()
        return self

    def img_transform(self, p_hflip=0, img_size=None):
        """Create image preprocessing pipeline compatible with Stable Diffusion.

        Args:
            p_hflip (float, optional): Probability of horizontal flip [0, 1]. Defaults to 0.
            img_size (int, optional): Target image size. Uses instance default if None.

        Returns:
            transforms.Compose: A composition of transforms including:
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
        """Encode input images to latent space representation.

        Args:
            images (torch.Tensor): Input image tensor in [-1, 1] range with shape (B, C, H, W).

        Returns:
            torch.Tensor: Mean of the latent distribution with shape (B, 4, H//8, W//8).

        Note:
            - Uses deterministic encoding (returns mean of the posterior distribution)
            - Automatically moves input to CUDA if available
            - Runs in no-gradient mode for inference
        """
        with torch.no_grad():
            posterior = self.model.encode(images.cuda()).latent_dist.mean
            return posterior

    def decode_to_images(self, z):
        """Decode latent representations back to image space.

        Args:
            z (torch.Tensor): Latent tensor with shape (B, 4, H//8, W//8).

        Returns:
            torch.Tensor: Decoded images in [0, 1] range with shape (B, C, H, W).

        Note:
            - Output is clipped and scaled to [0, 1] range
            - Uses stochastic sampling from latent distribution
            - Automatically moves input to CUDA if available
            - Runs in no-gradient mode for inference
        """
        with torch.no_grad():
            images = torch.clip((self.model.decode(z.cuda()).sample + 1) / 2, 0, 1)
        return images
