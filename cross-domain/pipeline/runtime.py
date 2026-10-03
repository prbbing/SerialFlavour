"""Explicit precision, device, and independently seeded random streams."""

import os
import random
import numpy as np
import torch


def configure(settings):
    torch.set_num_threads(int(settings.get("threads", 4)))
    device = torch.device(settings.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if settings.get("deterministic", True):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value
