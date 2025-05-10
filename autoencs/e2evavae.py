import torch
from torchvision import transforms
from autoencs.e2evae import vae_models
from autoencs.utils import center_crop_arr


class E2E_VA_VAE:
    """End-to-End Vision-Aligned Variational Autoencoder (VAE) implementation.

    This class provides an interface to load and use pre-trained VAEs that are aligned with
    vision foundation models. It supports image preprocessing, encoding to latent space,
    and decoding back to image space.
    """

    def __init__(self, name, img_size=256):
        """Initialize the E2E_VA_VAE model with specified configuration.

        Args:
            name (str): Name identifier for the model variant to load.
            img_size (int, optional): Default image size for processing. Defaults to 256.
        """
        self.img_size = img_size
        self.load(name)

    def load(self, name):
        """Load and initialize the specified VAE model from disk.

        Args:
            name (str): Name identifier for the model variant to load.

        Returns:
            E2E_VA_VAE: The initialized model instance for method chaining.

        Note:
            - Currently supports 'e2evavae_f16d32' model variant.
            - Automatically moves model to CUDA and sets to evaluation mode.
            - Model weights are loaded from predefined checkpoint paths.
        """
        name2type = {"e2evavae_f16d32": "f16d32"}
        name2path = {"e2evavae_f16d32": "./buffers/vaes/e2evavae_f16d32.ckpt"}

        self.model = vae_models[name2type[name]]()
        self.model.load_state_dict(torch.load(name2path[name]))
        self.model.cuda().eval()
        return self

    def img_transform(self, p_hflip=0, img_size=None):
        """Create a composition of image preprocessing transformations.

        Args:
            p_hflip (float, optional): Probability [0,1] of applying random horizontal flip.
                                      Defaults to 0 (no flip).
            img_size (int, optional): Target output size for images. Uses instance default if None.

        Returns:
            transforms.Compose: A torchvision transform pipeline consisting of:
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
        """Encode input images to latent space using the VAE encoder.

        Args:
            images (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Mean of the latent distribution (posterior) with shape (B, D),
                          where D is the latent dimension.

        Note:
            - Automatically moves input to CUDA if available.
            - Runs in inference mode (no gradients calculated).
            - Only returns the mean of the posterior distribution.
        """
        with torch.no_grad():
            posterior = self.model.encode(images.cuda()).mean
            return posterior

    def decode_to_images(self, z):
        """Decode latent representations back to image space.

        Args:
            z (torch.Tensor): Latent representation tensor of shape (B, D).

        Returns:
            torch.Tensor: Decoded images in [0, 1] range (clipped and scaled) with shape (B, C, H, W).

        Note:
            - Automatically moves input to CUDA if available.
            - Runs in inference mode (no gradients calculated).
            - Output is guaranteed to be in valid [0, 1] range.
        """
        with torch.no_grad():
            images = torch.clip((self.model.decode(z.cuda()) + 1) / 2, 0, 1)
        return images
