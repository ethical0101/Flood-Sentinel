"""Binary segmentation metrics with an ignore label.

Positive class = water (label 1). Negative class = not-water (label 0). Pixels labelled -1
(no data) are excluded from every count. Metrics are computed from a confusion matrix
accumulated over *all valid pixels of all chips* ("pooled", the Sen1Floods11 "total IoU"),
and additionally the IoU is averaged per chip ("mean chip IoU", chips with no water in both
label and prediction are skipped because IoU is undefined there).

    confusion = [[TN, FP],
                 [FN, TP]]      rows = ground truth (0, 1), columns = prediction (0, 1)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def confusion(pred: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor, ignore: int = -1) -> np.ndarray:
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    valid = target != ignore
    p = pred[valid].astype(np.int64)
    t = target[valid].astype(np.int64)
    return np.bincount(t * 2 + p, minlength=4).reshape(2, 2)


def _div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def summarize(cm: np.ndarray, chip_ious: list[float] | None = None) -> dict:
    tn, fp, fn, tp = (float(v) for v in cm.ravel())
    iou_w = _div(tp, tp + fp + fn)
    iou_b = _div(tn, tn + fn + fp)
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    out = {
        "iou_water": iou_w,
        "iou_not_water": iou_b,
        "miou": float(np.nanmean([iou_w, iou_b])),
        "dice_water": _div(2 * tp, 2 * tp + fp + fn),  # identical to F1 for binary masks
        "f1_water": _div(2 * precision * recall, precision + recall) if precision + recall > 0 else float("nan"),
        "precision_water": precision,
        "recall_water": recall,
        "pixel_accuracy": _div(tp + tn, tp + tn + fp + fn),
        "omission_error": _div(fn, tp + fn),     # missed water, common in flood-mapping literature
        "commission_error": _div(fp, tp + fp),  # false water alarms
        "valid_pixels": int(cm.sum()),
        "water_fraction_gt": _div(tp + fn, cm.sum()),
        "confusion_matrix": cm.astype(int).tolist(),
    }
    if chip_ious is not None:
        vals = [v for v in chip_ious if not np.isnan(v)]
        out["mean_chip_iou_water"] = float(np.mean(vals)) if vals else float("nan")
        out["n_chips"] = len(chip_ious)
    return out


def chip_iou(cm: np.ndarray) -> float:
    tn, fp, fn, tp = cm.ravel()
    return float("nan") if tp + fp + fn == 0 else float(tp / (tp + fp + fn))


# ----------------------------------------------------------------------------- losses
def masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, ignore: int = -1, bce_weight: float = 1.0,
                    dice_weight: float = 1.0, pos_weight: float | None = None, eps: float = 1.0) -> torch.Tensor:
    """BCE-with-logits + soft Dice on the water class, both restricted to valid pixels.

    logits: (B,1,H,W) or (B,H,W); target: (B,H,W) in {-1,0,1}.
    """
    if logits.dim() == 4:
        logits = logits[:, 0]
    logits = logits.float()
    valid = (target != ignore).float()
    t = target.clamp(min=0).float()
    pw = torch.tensor(pos_weight, device=logits.device) if pos_weight else None
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none", pos_weight=pw)
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    inter = (p * t).sum()
    dice = 1 - (2 * inter + eps) / (p.sum() + (t * valid).sum() + eps)
    return bce_weight * bce + dice_weight * dice
