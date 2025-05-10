from optimers.lion import Lion
from optimers.adamw import AdamW
from optimers.cadamw import CAdamW
from optimers.clion import CLion
from optimers.dadamw import DAdamW
from torch.optim import RAdam, Adam

OPTIMERS = {
    "Adam": Adam,
    "RAdam": RAdam,
    "AdamW": AdamW,
    "Lion": Lion,
    "CAdamW": CAdamW,
    "CLion": CLion,
    "DAdamW": DAdamW,
}
