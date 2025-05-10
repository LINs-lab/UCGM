import torch
from torchvision import transforms
from efficientvit.ae_model_zoo import DCAE_HF
from autoencs.utils import center_crop_arr


class DC_AE:
    """Implementation of a Vision Foundation Model Aligned Variational Autoencoder (VAE).

    This class provides functionality for loading a pre-trained VAE model, preprocessing images,
    encoding images to latent representations, and decoding latent representations back to images.
    """

    def __init__(self, name, img_size=256):
        """Initialize the DC_AE model with specified configuration.

        Args:
            name (str): Name identifier for the model variant to load.
            img_size (int, optional): Default image size for processing. Defaults to 256.
        """
        self.img_size = img_size
        self.load(name)

    def load(self, name):
        """Load and initialize the specified VAE model.

        Args:
            name (str): Name identifier for the model variant to load.

        Returns:
            DC_AE: The initialized model instance for method chaining.

        Note:
            - Automatically moves the model to CUDA and sets it to evaluation mode.
            - Currently supports 'dcae_f32c32' model variant.
        """
        name2type = {"dcae_f32c32": "dc-ae-f32c32-in-1.0"}
        name2path = {"dcae_f32c32": "./buffers/vaes/dcae_f32c32.ckpt"}

        self.model = DCAE_HF(name2type[name])
        self.model.load_state_dict(torch.load(name2path[name]))
        self.model.cuda().eval()
        return self

    def img_transform(self, p_hflip=0, img_size=None):
        """Create image preprocessing transformation pipeline.

        Args:
            p_hflip (float, optional): Probability of applying horizontal flip. Defaults to 0.
            img_size (int, optional): Target image size. Uses instance default if None.

        Returns:
            transforms.Compose: Composition of image transformations including:
                - Center cropping to target size
                - Optional horizontal flipping
                - Conversion to tensor
                - Normalization to [-1, 1] range
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
        """Encode input images to latent space representations.

        Args:
            images (torch.Tensor): Input image tensor (batch of images).

        Returns:
            torch.Tensor: Encoded latent representations from the VAE posterior.

        Note:
            - Automatically moves input to CUDA if available.
            - Runs in inference mode (no gradients calculated).
        """
        with torch.no_grad():
            posterior = self.model.encode(images.cuda())
            return posterior

    def decode_to_images(self, z):
        """Decode latent representations back to image space.

        Args:
            z (torch.Tensor): Latent representation tensor.

        Returns:
            torch.Tensor: Decoded images in [0, 1] range (clipped and scaled).

        Note:
            - Automatically moves input to CUDA if available.
            - Runs in inference mode (no gradients calculated).
            - Output is guaranteed to be in valid [0, 1] range.
        """
        with torch.no_grad():
            images = torch.clip((self.model.decode(z.cuda()) + 1) / 2, 0, 1)
        return images
