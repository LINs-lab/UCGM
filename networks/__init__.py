from networks.unetplus import SongUNet
from networks.sit import SiT_models
from networks.ddt import DDT_models
from networks.lightningdit import LightningDiT_models
from networks.tit import TiT_models
from networks.edm2 import EDM2_models

NETWORKS = {
    "UNet+": SongUNet,
    "SiT-XL/1": SiT_models["SiT-XL/1"],
    "SiT-XL/2": SiT_models["SiT-XL/2"],
    "SiT-L/1": SiT_models["SiT-L/1"],
    "SiT-L/2": SiT_models["SiT-L/2"],
    "SiT-B/1": SiT_models["SiT-B/1"],
    "SiT-B/2": SiT_models["SiT-B/2"],
    "SiT-S/1": SiT_models["SiT-S/1"],
    "SiT-S/2": SiT_models["SiT-S/2"],
    "LightningDiT-S/1": LightningDiT_models["LightningDiT-S/1"],
    "LightningDiT-B/1": LightningDiT_models["LightningDiT-B/1"],
    "LightningDiT-B/2": LightningDiT_models["LightningDiT-B/2"],
    "LightningDiT-XL/1": LightningDiT_models["LightningDiT-XL/1"],
    "LightningDiT-XL/2": LightningDiT_models["LightningDiT-XL/2"],
    "DDT-XL/1": DDT_models["DDT-XL/1"],
    "DDT-XL/2": DDT_models["DDT-XL/2"],
    "TiT-B/1": TiT_models["TiT-B/1"],
    "TiT-B/2": TiT_models["TiT-B/2"],
    "TiT-XL/1": TiT_models["TiT-XL/1"],
    "TiT-XL/2": TiT_models["TiT-XL/2"],
    "TiT-XL/4": TiT_models["TiT-XL/4"],
    "EDM2-XS": EDM2_models["EDM2-XS"],
    "EDM2-S": EDM2_models["EDM2-S"],
    "EDM2-M": EDM2_models["EDM2-M"],
    "EDM2-L": EDM2_models["EDM2-L"],
    "EDM2-XL": EDM2_models["EDM2-XL"],
    "EDM2-XXL": EDM2_models["EDM2-XXL"],
}
