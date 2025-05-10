import torch
from torchvision import transforms
from diffusers.models import AutoencoderKL
from autoencs.utils import center_crop_arr


class EDM2_SD_VAE:
    """Stable Diffusion VAE Implementation for EDM2 framework.

    This class provides an interface to Stability AI's Stable Diffusion VAE models,
    supporting both EMA and MSE variants. It handles image preprocessing, encoding to latent space,
    and decoding back to image space with proper normalization and safety clipping.
    """

    def __init__(self, name, img_size=256):
        """Initialize the Stable Diffusion VAE wrapper.

        Args:
            name (str): Model variant identifier, either:
                       - 'edm2_sdvae_ema_f8c4' (EMA-regularized version)
                       - 'edm2_sdvae_f8c4' (MSE-optimized version)
            img_size (int, optional): Default image processing size. Defaults to 256.
        """
        self.img_size = img_size
        self.load(name)

    def load(self, name):
        """Load pretrained Stable Diffusion VAE model from HuggingFace or local path.

        Args:
            name (str): Model variant identifier matching available configurations.

        Returns:
            EDM2_SD_VAE: The initialized model instance for method chaining.

        Raises:
            ValueError: If provided name doesn't match available configurations.

        Note:
            - Automatically moves model to CUDA and sets to evaluation mode.
            - Supports both local paths and HuggingFace model hub identifiers.
        """
        name2type = {
            "edm2_sdvae_ema_f8c4": "/storage/lintaoLab/lintao/superdit/pretrains/stabilityai/sd-vae-ft-ema",
            "edm2_sdvae_f8c4": "stabilityai/sd-vae-ft-mse",
        }

        if name not in name2type:
            raise ValueError(
                f"Unknown model name: {name}. Available options: {list(name2type.keys())}"
            )

        self.model = AutoencoderKL.from_pretrained(name2type[name])
        self.model.cuda().eval()
        return self

    def img_transform(self, p_hflip=0, img_size=None):
        """Create image preprocessing pipeline compatible with Stable Diffusion.

        Args:
            p_hflip (float, optional): Probability [0,1] of random horizontal flip.
                                      Defaults to 0 (no flipping).
            img_size (int, optional): Target processing size. Uses instance default if None.

        Returns:
            transforms.Compose: Transform pipeline consisting of:
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
        """Encode images to latent space using Stable Diffusion's VAE.

        Args:
            images (torch.Tensor): Input image tensor of shape (B, C, H, W) in [-1,1] range.

        Returns:
            torch.Tensor: Mean of the latent distribution with shape (B, 4, H//8, W//8).

        Note:
            - Automatically moves input to CUDA if available.
            - Runs in inference mode (no gradients calculated).
            - Returns the mean of the posterior distribution (deterministic encoding).
        """
        with torch.no_grad():
            posterior = self.model.encode(images.cuda()).latent_dist.mean
            return posterior

    def decode_to_images(self, z):
        """Decode latent representations back to image space.

        Args:
            z (torch.Tensor): Latent tensor of shape (B, 4, H//8, W//8).

        Returns:
            torch.Tensor: Decoded images in [0, 1] range (clipped) with shape (B, C, H, W).

        Note:
            - Automatically moves input to CUDA if available.
            - Runs in inference mode (no gradients calculated).
            - Output is guaranteed to be in valid [0, 1] range.
            - Uses sample() from latent distribution for stochastic decoding.
        """
        with torch.no_grad():
            # For default SD VAE, we torch.clip((self.model.decode(z.cuda()).sample + 1) / 2, 0, 1)
            # Here we follow the EDM code to implement:
            images = torch.clip(self.model.decode(z.cuda()).sample, 0, 1)
        return images
