"""Sen1Floods11 hand-labelled dataset: preprocessing helpers and PyTorch Dataset.

Label convention (from the Sen1Floods11 README, ``LabelHand`` layer):
    -1 -> no data / not valid  (ignored in loss and metrics)
     0 -> not water            (negative class)
     1 -> water                (positive class; flood water *and* permanent water)

Processed layout (written by ``training/prepare_data.py``)::

    <processed>/chips/<chip_id>.npz   s1 (2,H,W) f16 dB | s2 (13,H,W) f16 reflectance | rgb (3,H,W) u8
                                      label (H,W) i8 | otsu (H,W) i8 | jrc (H,W) i8
    <processed>/index.json            chip list with split, event, CRS, transform, bounds
    <processed>/stats.json            per-band mean/std computed on the TRAIN split only
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from training.utils import processed_dir

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
MODALITIES = {
    "s1": {"channels": 2, "bands": S1_BANDS, "description": "Sentinel-1 SAR (VV, VH) in dB"},
    "s2": {"channels": 13, "bands": S2_BANDS, "description": "Sentinel-2 L1C, 13 bands (TOA reflectance)"},
    "s1s2": {"channels": 15, "bands": S1_BANDS + S2_BANDS, "description": "Sentinel-1 + Sentinel-2 stacked"},
    "rgb": {"channels": 3, "bands": ["R", "G", "B"], "description": "True-colour RGB (S2 B4,B3,B2), 8-bit stretched"},
}
IGNORE = -1
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------- preprocessing
def rgb_stretch(rgb: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    """Per-image percentile stretch of a (3,H,W) array to uint8.

    One joint percentile over all three channels preserves colour balance. The same function
    is applied to S2 true-colour composites at training time and to uploaded RGB images at
    inference time, so both see the same input distribution.
    """
    x = rgb.astype(np.float32)
    # no-data = non-finite or all bands exactly 0 (S2 fill value). Do not test the sign: S1 dB is negative.
    valid = np.isfinite(x).all(0) & (x != 0).any(0)
    if valid.sum() < 16:
        return np.zeros(x.shape, dtype=np.uint8)
    v = x[:, valid]
    lo, hi = np.percentile(v, lo_pct), np.percentile(v, hi_pct)
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((x - lo) / (hi - lo), 0, 1) * 255.0
    out[:, ~valid] = 0
    return out.astype(np.uint8)


def s2_to_rgb8(s2_reflectance: np.ndarray) -> np.ndarray:
    return rgb_stretch(s2_reflectance[[3, 2, 1]])


@lru_cache(maxsize=4)
def load_stats(root: str | None = None) -> dict:
    return json.loads((Path(root) if root else processed_dir()).joinpath("stats.json").read_text())


def normalize(modality: str, s1: np.ndarray | None = None, s2: np.ndarray | None = None,
              rgb: np.ndarray | None = None, stats: dict | None = None) -> np.ndarray:
    """Return a float32 (C,H,W) model input. Non-finite values become 0 after standardisation.

    s1: (2,H,W) dB, s2: (13,H,W) TOA reflectance (0-1, i.e. raw/10000), rgb: (3,H,W) uint8.
    """
    if modality == "rgb":
        x = rgb.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return x.astype(np.float32)
    stats = stats or load_stats()
    parts = []
    if modality in ("s1", "s1s2"):
        parts.append(_std(np.clip(s1.astype(np.float32), -50, 10), stats["s1_mean"], stats["s1_std"]))
    if modality in ("s2", "s1s2"):
        parts.append(_std(s2.astype(np.float32), stats["s2_mean"], stats["s2_std"]))
    x = np.concatenate(parts, 0)
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32)


def _std(x: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    m = np.asarray(mean, dtype=np.float32)[:, None, None]
    s = np.asarray(std, dtype=np.float32)[:, None, None]
    return (x - m) / s


def chip_input(chip: dict, modality: str, stats: dict | None = None) -> np.ndarray:
    return normalize(modality, s1=chip.get("s1"), s2=chip.get("s2"), rgb=chip.get("rgb"), stats=stats)


# ----------------------------------------------------------------------------- index helpers
def load_index(root: Path | None = None) -> list[dict]:
    return json.loads((root or processed_dir()).joinpath("index.json").read_text())["chips"]


def split_ids(split: str, root: Path | None = None) -> list[str]:
    return [c["id"] for c in load_index(root) if c["split"] == split]


def load_chip(chip_id: str, root: Path | None = None) -> dict:
    with np.load((root or processed_dir()) / "chips" / f"{chip_id}.npz") as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- torch dataset
class FloodDataset(Dataset):
    """Returns (x float32 [C,H,W], y int64 [H,W] with -1 = ignore, chip_id)."""

    KEYS = {"s1": ("s1",), "s2": ("s2",), "s1s2": ("s1", "s2"), "rgb": ("rgb",)}

    def __init__(self, split: str, modality: str, crop_size: int | None = None, crops_per_chip: int = 1,
                 augment: bool = False, root: Path | None = None, seed: int = 0):
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}; choose from {list(MODALITIES)}")
        self.root = Path(root) if root else processed_dir()
        self.ids = split_ids(split, self.root)
        if not self.ids:
            raise RuntimeError(f"no chips for split {split!r} in {self.root}; run training/prepare_data.py")
        self.modality, self.crop, self.k, self.augment = modality, crop_size, crops_per_chip, augment
        self.stats = load_stats(str(self.root)) if modality != "rgb" else None
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids) * self.k

    def __getitem__(self, idx: int):
        i = idx // self.k
        # read only the arrays this modality needs (npz members are loaded lazily)
        with np.load(self.root / "chips" / f"{self.ids[i]}.npz") as z:
            raw = {k: z[k] for k in self.KEYS[self.modality]}
            y = z["label"]
        if self.crop and self.crop < y.shape[-1]:
            r, c = self._crop_origin(y)
            sl = (slice(r, r + self.crop), slice(c, c + self.crop))
            raw = {k: v[(slice(None),) + sl] for k, v in raw.items()}
            y = y[sl]
        # crop first, then normalise: ~4x cheaper than normalising the full chip
        x = chip_input(raw, self.modality, self.stats)
        y = y.astype(np.int64)
        if self.augment:
            x, y = self._augment(x, y)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)), self.ids[i]

    def _crop_origin(self, y: np.ndarray) -> tuple[int, int]:
        h, w = y.shape
        # retry a few times to avoid crops that are almost entirely no-data
        for _ in range(10):
            r = int(self.rng.integers(0, h - self.crop + 1))
            c = int(self.rng.integers(0, w - self.crop + 1))
            if (y[r:r + self.crop, c:c + self.crop] != IGNORE).mean() > 0.1:
                break
        return r, c

    def _augment(self, x, y):
        if self.rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        if self.rng.random() < 0.5:
            x, y = x[:, ::-1, :], y[::-1, :]
        k = int(self.rng.integers(0, 4))
        if k:
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(0, 1))
        if self.modality == "rgb":
            # brightness / contrast jitter in normalised space for robustness to other RGB sources
            x = x * self.rng.uniform(0.8, 1.2) + self.rng.uniform(-0.2, 0.2)
        return x, y


def worker_init(worker_id: int) -> None:
    """DataLoader worker_init_fn: give every worker's dataset copy an independent RNG stream."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)
