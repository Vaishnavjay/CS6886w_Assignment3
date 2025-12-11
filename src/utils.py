import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location)


def model_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model: torch.nn.Module, bytes_per_param: int = 4) -> int:
    return model_num_params(model) * bytes_per_param