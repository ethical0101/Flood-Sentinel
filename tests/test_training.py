"""Unit tests for the deep-learning pipeline (dataset, model, loss, metrics, checkpoints, inference)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from training.dataset import IGNORE, MODALITIES, normalize, rgb_stretch
from training.infer import Predictor, from_array
from training.metrics import chip_iou, confusion, masked_bce_dice, summarize
from training.model import build_model, load_for_inference, save_checkpoint
from training.utils import load_config, processed_dir, runs_dir

STATS = {"s1_mean": [-10.0, -17.0], "s1_std": [4.0, 4.7], "s2_mean": [0.15] * 13, "s2_std": [0.08] * 13}
HAS_DATA = (processed_dir() / "index.json").exists()


# ----------------------------------------------------------------------------- metrics
def test_confusion_ignores_nodata_and_counts_correctly():
    target = np.array([[1, 1, 0, 0], [-1, -1, 1, 0]])
    pred = np.array([[1, 0, 1, 0], [1, 1, 1, 0]])
    cm = confusion(pred, target)
    # TN=2 (0,3),(1,3); FP=1 (0,2); FN=1 (0,1); TP=2 (0,0),(1,2); the two -1 pixels are excluded
    assert cm.tolist() == [[2, 1], [1, 2]]
    m = summarize(cm)
    assert m["valid_pixels"] == 6
    assert m["iou_water"] == pytest.approx(2 / 4)
    assert m["iou_not_water"] == pytest.approx(2 / 4)
    assert m["precision_water"] == pytest.approx(2 / 3)
    assert m["recall_water"] == pytest.approx(2 / 3)
    assert m["dice_water"] == pytest.approx(m["f1_water"])
    assert m["pixel_accuracy"] == pytest.approx(4 / 6)


def test_iou_undefined_when_no_water_anywhere():
    cm = confusion(np.zeros((4, 4), int), np.zeros((4, 4), int))
    assert np.isnan(chip_iou(cm))
    assert np.isnan(summarize(cm)["precision_water"])


# ----------------------------------------------------------------------------- loss
def test_loss_ignores_masked_pixels_and_rewards_correct_predictions():
    target = torch.tensor([[[1, 0], [IGNORE, IGNORE]]])
    good = torch.tensor([[[[8.0, -8.0], [-8.0, 8.0]]]])     # correct on valid px, "wrong" on ignored px
    bad = torch.tensor([[[[-8.0, 8.0], [-8.0, 8.0]]]])
    lg, lb = masked_bce_dice(good, target), masked_bce_dice(bad, target)
    assert lg < 0.05 < lb
    flipped = torch.tensor([[[[8.0, -8.0], [8.0, -8.0]]]])  # differs only on ignored pixels
    assert masked_bce_dice(flipped, target) == pytest.approx(lg.item(), abs=1e-6)
    assert torch.isfinite(masked_bce_dice(torch.zeros(1, 1, 2, 2), torch.full((1, 2, 2), IGNORE)))


# ----------------------------------------------------------------------------- preprocessing
def test_normalize_shapes_and_nan_handling():
    s1 = np.full((2, 8, 8), -12.0, np.float32)
    s1[0, 0, 0] = np.nan
    s2 = np.full((13, 8, 8), 0.2, np.float32)
    for mod, c in (("s1", 2), ("s2", 13), ("s1s2", 15)):
        x = normalize(mod, s1=s1, s2=s2, stats=STATS)
        assert x.shape == (c, 8, 8) and x.dtype == np.float32 and np.isfinite(x).all()
    rgb = rgb_stretch(np.random.default_rng(0).random((3, 8, 8)))
    assert normalize("rgb", rgb=rgb).shape == (3, 8, 8)


def test_rgb_stretch_is_scale_invariant():
    a = np.random.default_rng(1).random((3, 32, 32)).astype(np.float32)
    assert np.array_equal(rgb_stretch(a), rgb_stretch(a * 1000))


def test_from_array_band_count_detection():
    assert from_array(np.zeros((2, 16, 16)) - 10).kind == "s1"
    assert from_array(np.ones((13, 16, 16)) * 1500).kind == "s2"
    assert from_array(np.zeros((15, 16, 16))).kind == "s1s2"
    assert from_array(np.zeros((3, 16, 16), np.uint8)).kind == "rgb"
    lin = from_array(np.full((2, 16, 16), 0.05, np.float32))  # linear power -> dB
    assert lin.s1.max() == pytest.approx(10 * np.log10(0.05), rel=1e-4)
    with pytest.raises(ValueError):
        from_array(np.zeros((5, 16, 16)))


# ----------------------------------------------------------------------------- model + checkpoints
@pytest.mark.parametrize("arch,encoder", [("unet", "resnet34"), ("deeplabv3plus", "tu-mobilenetv3_large_100")])
@pytest.mark.parametrize("modality", list(MODALITIES))
def test_model_output_dimensions(arch, encoder, modality):
    m = build_model(arch, encoder, modality, encoder_weights=None).eval()
    with torch.no_grad():
        y = m(torch.randn(1, MODALITIES[modality]["channels"], 64, 96))
    assert y.shape == (1, 1, 64, 96)


def test_checkpoint_save_and_reload(tmp_path):
    cfg = load_config(overrides=["data.modality=s1", "model.encoder_weights=null"])
    m = build_model("unet", "resnet34", "s1", None)
    opt = torch.optim.AdamW(m.parameters())
    save_checkpoint(tmp_path / "best.pt", m, opt, None, None, 3, 0.5, cfg, {"n_params": 1})
    m2, ck = load_for_inference(tmp_path / "best.pt")
    assert ck["epoch"] == 3 and ck["config"]["data"]["modality"] == "s1"
    x = torch.randn(1, 2, 64, 64)
    with torch.no_grad():
        assert torch.allclose(m.eval()(x), m2(x), atol=1e-6)


def test_tiled_prediction_handles_arbitrary_sizes(tmp_path):
    cfg = load_config(overrides=["data.modality=s1", "model.encoder_weights=null"])
    run = tmp_path / "run"
    m = build_model("unet", "resnet34", "s1", None)
    save_checkpoint(run / "checkpoints" / "best.pt", m, None, None, None, 1, 0.0, cfg, {})
    (run / "stats.json").write_text(__import__("json").dumps(STATS))
    p = Predictor(run, device="cpu")
    for h, w in ((70, 130), (600, 520)):
        prob = p.predict(np.random.default_rng(0).standard_normal((2, h, w)).astype(np.float32), tile=256, overlap=32)
        assert prob.shape == (h, w) and prob.min() >= 0 and prob.max() <= 1


# ----------------------------------------------------------------------------- real dataset (skipped if absent)
@pytest.mark.skipif(not HAS_DATA, reason="processed Sen1Floods11 not available")
def test_processed_dataset_labels_and_splits():
    from training.dataset import FloodDataset, load_index

    idx = load_index()
    splits = {c["split"] for c in idx}
    assert splits == {"train", "valid", "test", "bolivia"}
    ids = {s: {c["id"] for c in idx if c["split"] == s} for s in splits}
    assert not ids["train"] & ids["test"] and not ids["train"] & ids["valid"] and not ids["train"] & ids["bolivia"]
    assert not any(c["event"] == "Bolivia" for c in idx if c["split"] == "train")
    ds = FloodDataset("valid", "s1s2")
    x, y, cid = ds[0]
    assert x.shape == (15, 512, 512) and y.shape == (512, 512)
    assert set(torch.unique(y).tolist()) <= {-1, 0, 1}
    tr = FloodDataset("train", "s1", crop_size=128, augment=True)
    x, y, _ = tr[0]
    assert x.shape == (2, 128, 128) and y.shape == (128, 128)


@pytest.mark.skipif(not (runs_dir() / "unet_resnet34_s1s2" / "checkpoints" / "best.pt").exists(),
                    reason="main model not trained")
def test_trained_model_beats_constant_predictor_on_a_test_chip():
    from training.dataset import chip_input, load_chip

    p = Predictor("unet_resnet34_s1s2")
    chip = load_chip("Spain_7370579")
    prob = p.predict(chip_input(chip, "s1s2", p.stats))
    m = summarize(confusion((prob > 0.5).astype(int), chip["label"]))
    assert m["iou_water"] > 0.5
