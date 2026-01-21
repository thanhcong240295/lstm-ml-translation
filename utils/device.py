import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False


def get_array_module(device="cpu"):
    if device == "gpu":
        if not CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is not installed. Install it with: pip install cupy-cuda12x (or cupy-cuda11x for CUDA 11.x)"
            )
        return cp
    return np


def is_gpu_available():
    return CUPY_AVAILABLE and cp is not np


def get_device_name(xp):
    if xp is np:
        return "CPU"
    return "GPU"


def to_numpy(array):
    if hasattr(array, "get"):
        return array.get()
    return array


def to_device(array, xp):
    if xp is np:
        return to_numpy(array)
    return xp.asarray(array)
