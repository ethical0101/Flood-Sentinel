"""v2 training-side features: index channels, radiometric augmentation invariants, GPU guard, registry."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from training.dataset import MODALITIES, normalize, radiometric_augment, water_indices
from training.model import build_model
from training.utils import processed_dir

STATS = {"s1_mean": [-10.0, -17.0], "s1_std": [4.0, 4.7], "s2_mean": [0.15] * 13, "s2_std": [0.08] * 13}
CFG = {"p": 1.0, "s2_gain": [0.75, 1.25], "s2_band_jitter": 0.05, "s1_offset_db": 2.0, "s1_pol_jitter_db": 0.5}


def test_water_indices_values_and_scale_invariance():
    s2 = np.full((13, 4, 4), 0.1, np.float32)
    s2[2] = 0.3   # green
    s2[7] = 0.1   # NIR
    s2[11] = 0.2  # SWIR1
    idx = water_indices(s2)
    assert idx.shape == (2, 4, 4)
    assert idx[0, 0, 0] == pytest.approx((0.3 - 0.1) / 0.4)
    assert idx[1, 0, 0] == pytest.approx((0.3 - 0.2) / 0.5)
    assert np.allclose(water_indices(s2 * 1.7), idx)  # invariant to a common brightness gain


def test_index_modalities_shapes_and_nodata():
    s1 = np.full((2, 8, 8), -12.0, np.float32)
    s2 = np.full((13, 8, 8), 0.2, np.float32)
    s2[:, 0, 0] = 0  # S2 no-data -> 0/0 in the indices
    for mod in ("s1s2i", "s2i"):
        x = normalize(mod, s1=s1, s2=s2, stats=STATS)
        assert x.shape[0] == MODALITIES[mod]["channels"] and np.isfinite(x).all()
        assert x[-1, 0, 0] == 0 and x[-2, 0, 0] == 0


@pytest.mark.parametrize("modality", ["s1s2i", "s2i"])
def test_index_models_build(modality):
    m = build_model("unet", "resnet34", modality, encoder_weights=None).eval()
    with torch.no_grad():
        assert m(torch.randn(1, MODALITIES[modality]["channels"], 64, 64)).shape == (1, 1, 64, 64)


def test_radiometric_augmentation_preserves_physics():
    rng = np.random.default_rng(0)
    s2 = rng.uniform(0.01, 0.4, (13, 16, 16)).astype(np.float32)
    s2[:, 0, 0] = 0  # no-data stays no-data under a multiplicative gain
    s1 = rng.uniform(-25, -5, (2, 16, 16)).astype(np.float32)
    s1[:, 1, 1] = np.nan
    out = radiometric_augment({"s1": s1, "s2": s2}, rng, CFG)
    assert (out["s2"][:, 0, 0] == 0).all() and np.isnan(out["s1"][:, 1, 1]).all()
    ratio = out["s2"][:, 5, 5] / s2[:, 5, 5]
    assert ratio.min() >= 0.75 * 0.95 - 1e-6 and ratio.max() <= 1.25 * 1.05 + 1e-6
    off = out["s1"][:, 5, 5] - s1[:, 5, 5]
    assert np.abs(off).max() <= 2.5 + 1e-5
    # a single per-chip gain: spatial structure within a band is preserved exactly
    band = out["s2"][3] / np.where(s2[3] == 0, 1, s2[3])
    assert np.allclose(band[s2[3] > 0], band[s2[3] > 0].mean(), rtol=1e-5)
    assert radiometric_augment({"s2": s2}, rng, {**CFG, "p": 0.0})["s2"] is s2  # p=0 -> untouched


@pytest.mark.skipif(not (processed_dir() / "index.json").exists(), reason="processed data absent")
def test_dataset_applies_radiometric_only_when_enabled():
    from training.dataset import FloodDataset

    off = FloodDataset("train", "s1s2i", 64, 1, augment=True, radiometric={**CFG, "enabled": False})
    on = FloodDataset("train", "s1s2i", 64, 1, augment=True, radiometric={**CFG, "enabled": True})
    val = FloodDataset("valid", "s1s2i", radiometric={**CFG, "enabled": True})
    assert off.radiometric is None and on.radiometric is not None and val.radiometric is None  # never on eval data
    x, y, _ = on[0]
    assert x.shape == (17, 64, 64) and torch.isfinite(x).all()


def test_gpu_guard_returns_list_and_never_lists_self():
    import os

    from training.gpu_guard import foreign_gpu_processes

    procs = foreign_gpu_processes()
    assert isinstance(procs, list) and all(p["pid"] != os.getpid() for p in procs)


def test_registry_fields():
    from training.registry import build

    rows = build()
    if not rows:
        pytest.skip("no runs")
    need = {"experiment_id", "dataset", "split", "architecture", "input_channels", "loss", "optimizer", "seed",
            "training_time_s", "best_val_iou@0.5", "test_iou@0.5", "peak_train_gpu_mem_mib", "checkpoint"}
    assert need <= set(rows[0])
    assert all(r["state"] != "running" or r["experiment_id"] for r in rows)
