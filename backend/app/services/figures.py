"""Small image helpers shared by the API (colours match the dashboard legend)."""
from __future__ import annotations

import numpy as np

WATER = (37, 99, 235)
NOT_WATER = (241, 245, 249)
NODATA = (148, 163, 184)


def label_rgb(label: np.ndarray) -> np.ndarray:
    img = np.empty(label.shape + (3,), np.uint8)
    img[:] = NOT_WATER
    img[label == 1] = WATER
    img[label == -1] = NODATA
    return img


def error_rgb(mask: np.ndarray, label: np.ndarray) -> np.ndarray:
    img = np.empty(label.shape + (3,), np.uint8)
    img[:] = NOT_WATER
    img[mask & (label == 1)] = WATER             # true positive
    img[mask & (label == 0)] = (239, 68, 68)     # false positive
    img[~mask & (label == 1)] = (245, 158, 11)   # false negative
    img[label == -1] = NODATA
    return img


def flood_rgba(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Map overlay: transparent where dry, blue with opacity rising with probability where wet."""
    rgba = np.zeros(prob.shape + (4,), np.uint8)
    p = np.nan_to_num(prob, nan=0.0)
    wet = p >= threshold
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 29, 78, 216
    rgba[..., 3] = np.where(wet, 110 + 120 * p, np.where(p >= 0.25, 50 * p, 0)).astype(np.uint8)
    return rgba
