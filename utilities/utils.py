import os
import torch
import random
import numpy as np
from collections import OrderedDict
import torch.backends.cudnn as cudnn


def set_seed(seed):
    # For random number generation in CPU and GPU
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure that CUDA algorithms are deterministic
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Control over external libraries (if applicable)
    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def remove_module_prefix(state_dict):
    """Remove all leading 'module.' prefixes from the start of each key."""
    new_state_dict = {}
    for k, v in state_dict.items():
        while k.startswith("module."):
            k = k[len("module.") :]
        new_state_dict[k] = v
    return new_state_dict


def remove_module_all(state_dict):
    """Remove all leading all 'module.' prefixes from each key."""
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict
