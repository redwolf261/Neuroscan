"""
BraTS preprocessing utilities.

Handles normalization, resizing, and label conversion for BraTS data.
"""

import numpy as np
from scipy.ndimage import zoom


def normalize_intensity(volume, method="minmax"):
    """
    Normalize volume intensity.

    Args:
        volume: 3D array
        method: 'minmax' (0-1), 'zscore', or 'robust'

    Returns:
        Normalized volume
    """
    volume = volume.astype(np.float32)

    if method == "minmax":
        v_min = np.min(volume)
        v_max = np.max(volume)
        if v_max > v_min:
            return (volume - v_min) / (v_max - v_min)
        else:
            return np.zeros_like(volume)

    elif method == "zscore":
        v_mean = np.mean(volume)
        v_std = np.std(volume)
        if v_std > 0:
            return (volume - v_mean) / v_std
        else:
            return volume - v_mean

    elif method == "robust":
        # Percentile-based normalization
        p5 = np.percentile(volume, 5)
        p95 = np.percentile(volume, 95)
        if p95 > p5:
            return np.clip((volume - p5) / (p95 - p5), 0, 1)
        else:
            return np.zeros_like(volume)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def convert_segmentation_to_binary(segmentation):
    """
    Convert BraTS 4-class segmentation to binary (tumor vs background).

    BraTS labels:
        0: background
        1: necrotic core
        2: edema
        3: enhancing tumor

    Returns:
        0: background
        1: tumor (any label > 0)
    """
    return (segmentation > 0).astype(np.float32)


def resize_volume(volume, target_shape=(64, 64, 64), order=1):
    """
    Resize volume to target shape using scipy zoom.

    Args:
        volume: 3D array
        target_shape: Target (D, H, W)
        order: Interpolation order (1=linear, 0=nearest for masks)

    Returns:
        Resized volume
    """
    current_shape = volume.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    resized = zoom(volume, zoom_factors, order=order)
    return resized


def get_volume_statistics(volume):
    """
    Compute statistics for a volume.

    Returns dict with mean, std, min, max, median.
    """
    return {
        "mean": float(np.mean(volume)),
        "std": float(np.std(volume)),
        "min": float(np.min(volume)),
        "max": float(np.max(volume)),
        "median": float(np.median(volume)),
    }
