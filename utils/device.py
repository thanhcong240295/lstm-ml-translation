"""
Device management utilities for GPU/CPU operations.

This module provides utilities for detecting and managing computational devices
(GPU with CuPy or CPU with NumPy).
"""

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False


def get_array_module(device="cpu"):
    """
    Get the appropriate array module (NumPy or CuPy) based on device.

    Args:
        device (str): Either 'cpu' or 'gpu'

    Returns:
        module: numpy module for CPU, cupy module for GPU (if available)

    Raises:
        ImportError: If GPU is requested but CuPy is not available
    """
    if device == "gpu":
        if not CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is not installed. Install it with: pip install cupy-cuda12x (or cupy-cuda11x for CUDA 11.x)"
            )
        return cp
    return np


def is_gpu_available():
    """
    Check if GPU (CuPy) is available.

    Returns:
        bool: True if CuPy is installed and GPU is available, False otherwise
    """
    return CUPY_AVAILABLE and cp is not np


def get_device_name(xp):
    """
    Get the device name from array module.

    Args:
        xp: Array module (numpy or cupy)

    Returns:
        str: 'GPU' if using CuPy, 'CPU' if using NumPy
    """
    if xp is np:
        return "CPU"
    return "GPU"


def to_numpy(array):
    """
    Convert array to NumPy (from either NumPy or CuPy).

    Args:
        array: NumPy or CuPy array

    Returns:
        numpy.ndarray: NumPy array
    """
    if hasattr(array, "get"):
        return array.get()
    return array


def to_device(array, xp):
    """
    Convert array to the specified device.

    Args:
        array: NumPy or CuPy array
        xp: Target array module (numpy or cupy)

    Returns:
        array: Array on the target device
    """
    if xp is np:
        return to_numpy(array)
    return xp.asarray(array)
