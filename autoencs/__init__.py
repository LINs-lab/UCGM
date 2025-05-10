from autoencs.vavae import VA_VAE
from autoencs.dcae import DC_AE
from autoencs.sdvae import SD_VAE
from autoencs.identity import ID_AE
from autoencs.e2evavae import E2E_VA_VAE
from autoencs.edm2_sdvae import EDM2_SD_VAE

AUTOENCS = {
    "vavae_f16d32": VA_VAE,
    "dcae_f32c32": DC_AE,
    "sdvae_f8c4": SD_VAE,
    "sdvae_ema_f8c4": SD_VAE,
    "idae_f1c3": ID_AE,
    "e2evavae_f16d32": E2E_VA_VAE,
    "edm2_sdvae_f8c4": EDM2_SD_VAE,
    "edm2_sdvae_ema_f8c4": EDM2_SD_VAE,
}
