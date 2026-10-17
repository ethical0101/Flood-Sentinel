"""Compact segmentation models (segmentation_models_pytorch) and checkpoint I/O.

Pretrained weights: when ``encoder_weights="imagenet"`` the encoder is initialised from the
ImageNet weights published by torchvision/timm (downloaded by smp on first use). For inputs
with != 3 channels smp adapts the first convolution (the pretrained RGB filters are re-used
cyclically and rescaled). Everything after initialisation is trained locally on Sen1Floods11;
checkpoints record this provenance in ``meta["pretrained"]``.
"""
from __future__ import annotations

from pathlib import Path

import torch
import segmentation_models_pytorch as smp

from training.dataset import MODALITIES

ARCHS = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}


def build_model(arch: str, encoder: str, modality: str, encoder_weights: str | None = "imagenet") -> torch.nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {list(ARCHS)}")
    return ARCHS[arch](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=MODALITIES[modality]["channels"],
        classes=1,  # single logit: P(water)
    )


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best: float,
                    cfg: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best": best,
        "config": cfg,
        "meta": meta,
    }, tmp)
    tmp.replace(path)  # atomic: an interrupted save never corrupts the previous checkpoint


def load_for_inference(path: Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    m = cfg["model"]
    model = build_model(m["arch"], m["encoder"], cfg["data"]["modality"], encoder_weights=None)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck
