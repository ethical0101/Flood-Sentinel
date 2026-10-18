"""Experiment registry built from the run folders (single source of truth: files on disk).

    .venv\\Scripts\\python -m training.registry      # writes artifacts/experiments.{json,csv}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from training.dataset import MODALITIES
from training.utils import REPO, artifacts_dir, runs_dir, write_json

V1_RUNS = {"unet_resnet34_s1s2", "unet_resnet34_s2", "unet_resnet34_s1", "unet_resnet34_rgb", "deeplabv3plus_mnv3l_s1s2"}


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        import psutil

        return bool(pid) and psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower()
    except Exception:  # noqa: BLE001
        return False


def describe_changes(cfg: dict) -> str:
    """Human-readable difference from the v1 default recipe."""
    ch = []
    if cfg["data"]["modality"] not in ("s1s2",):
        ch.append(f"input={cfg['data']['modality']}")
    if cfg["model"]["arch"] != "unet" or cfg["model"]["encoder"] != "resnet34":
        ch.append(f"model={cfg['model']['arch']}/{cfg['model']['encoder']}")
    if (cfg["data"].get("radiometric") or {}).get("enabled"):
        ch.append("radiometric augmentation")
    if "wf_train" in cfg["data"].get("train_split", "train"):
        ch.append("+ WorldFloods training data")
    if cfg.get("seed", 42) != 42:
        ch.append(f"seed={cfg['seed']}")
    return ", ".join(ch) or "v1 default recipe"


def build() -> list[dict]:
    rows = []
    for d in sorted(runs_dir().iterdir()) if runs_dir().exists() else []:
        if d.name.startswith("smoke_") or not (d / "config.yaml").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        st = _read(d / "status.json") or {}
        if st.get("state") == "running" and not _alive(st.get("pid", 0)):
            st["state"] = "stalled"
        hist = _read(d / "history.json") or []
        ev = _read(d / "evaluation.json") or {}
        env = _read(d / "environment.json") or {}
        t = cfg["train"]
        sp = ev.get("splits", {})
        spv = ev.get("splits_at_val_threshold", {})
        best = max(hist, key=lambda h: h["val_iou_water"]) if hist else {}

        def g(src, split, k="iou_water"):
            return (src.get(split) or {}).get(k)

        rows.append({
            "experiment_id": d.name,
            "generation": "v1" if d.name in V1_RUNS else "v2",
            "change_vs_v1": describe_changes(cfg),
            "state": st.get("state"),
            "dataset": "Sen1Floods11 v1.1 HandLabeled" + (" + WorldFloods v2 subset (97 scenes)"
                                                          if "wf_train" in cfg["data"].get("train_split", "train") else ""),
            "split": "official train/valid/test + Bolivia held out",
            "architecture": f"{cfg['model']['arch']} / {cfg['model']['encoder']}",
            "pretrained": cfg["model"].get("encoder_weights"),
            "input_modality": cfg["data"]["modality"],
            "input_channels": MODALITIES[cfg["data"]["modality"]]["channels"],
            "train_resolution": f"{cfg['data']['crop_size']}x{cfg['data']['crop_size']} crops (10 m)",
            "eval_resolution": "512x512 chips (10 m)",
            "loss": f"BCE x{t['bce_weight']} + soft Dice x{t['dice_weight']} (masked)",
            "optimizer": f"AdamW lr={t['lr']} wd={t['weight_decay']}, warmup {t['warmup_epochs']} + cosine",
            "batch_size": t["batch_size"], "grad_accum": t["grad_accum"], "amp": t["amp"],
            "epochs_max": t["epochs"], "epochs_run": len(hist), "early_stopping_patience": t["early_stopping_patience"],
            "radiometric_aug": bool((cfg["data"].get("radiometric") or {}).get("enabled")),
            "seed": cfg["seed"],
            "training_time_s": st.get("total_time_s") or round(sum(h["epoch_time_s"] for h in hist)),
            "best_epoch": best.get("epoch"),
            "best_val_iou@0.5": best.get("val_iou_water"),
            "val_selected_threshold": ev.get("val_selected_threshold"),
            "val_iou@val_thr": g(spv, "valid"),
            "test_iou@0.5": g(sp, "test"), "test_iou@val_thr": g(spv, "test"),
            "test_f1@val_thr": g(spv, "test", "f1_water"),
            "bolivia_iou@0.5": g(sp, "bolivia"), "bolivia_iou@val_thr": g(spv, "bolivia"),
            "bolivia_recall@val_thr": g(spv, "bolivia", "recall_water"),
            "peak_train_gpu_mem_mib": max((h["peak_gpu_mem_mib"] for h in hist), default=None),
            "peak_infer_gpu_mem_mib": ev.get("inference_peak_gpu_mem_mib"),
            "latency_ms_model_only": (ev.get("latency") or {}).get("model_only_ms_mean"),
            "gpu": env.get("gpu"), "torch": env.get("torch"),
            "checkpoint": str((d / "checkpoints" / "best.pt").relative_to(REPO)).replace("\\", "/")
            if (d / "checkpoints" / "best.pt").exists() else None,
            "evaluated": bool(ev),
        })
    return rows


def main() -> None:
    rows = build()
    write_json(artifacts_dir() / "experiments.json", rows)
    with open(artifacts_dir() / "experiments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} experiments -> artifacts/experiments.json, .csv")


if __name__ == "__main__":
    main()
