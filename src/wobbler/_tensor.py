"""Tensors and devices: every function of the package takes NumPy arrays, numbers or torch tensors, works in float64 and returns tensors on one device. The default device is the GPU when torch sees one (ROCm builds report it as cuda), else the CPU."""
import numpy as np
import torch

DTYPE = torch.float64


def default_device():
    """'cuda' when torch sees a GPU, else 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_device(*values, device=None):
    """The device of a call: `device` if given, else the device of the first tensor among `values`, else the default."""
    if device is not None:
        return torch.device(device)
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return default_device()


def as_tensor(value, device):
    """`value` (a number, a list, a NumPy array or a tensor) as a float64 tensor on `device`."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=DTYPE)
    array = np.asarray(value, dtype=np.float64)
    if not array.flags.writeable:
        array = array.copy()  # torch warns about, and does not support, a tensor over a read-only array (a broadcast view, for one)
    return torch.as_tensor(array, device=device)
