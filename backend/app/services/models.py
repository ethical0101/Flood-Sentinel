"""Discovers trained runs and keeps loaded models in memory (loaded once, reused per request)."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from backend.app.config import settings

log = logging.getLogger("flood.models")


def _read(p: Path) -> dict | list | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


class ModelRegistry:
    def __init__(self, runs_root: Path | None = None, device: str | None = None):
        self.runs_root = runs_root or settings.artifacts / "runs"
        self.device = device or settings.device
        self._loaded: dict[str, object] = {}
        self._lock = threading.Lock()

    # -------------------------------------------------------------- discovery
    def available(self) -> list[dict]:
        out = []
        if not self.runs_root.exists():
            return out
        for d in sorted(self.runs_root.iterdir()):
            ck = d / "checkpoints" / "best.pt"
            if d.name.startswith("smoke_") or not ck.exists():
                continue
            status = _read(d / "status.json") or {}
            ev = _read(d / "evaluation.json") or {}
            cfg = ev.get("config") or {}
            if not cfg:
                import yaml

                try:
                    cfg = yaml.safe_load((d / "config.yaml").read_text())
                except OSError:
                    continue
            out.append({
                "name": d.name,
                "modality": cfg["data"]["modality"],
                "arch": cfg["model"]["arch"],
                "encoder": cfg["model"]["encoder"],
                "training_state": status.get("state", "unknown"),
                "best_val_iou": status.get("best"),
                "best_epoch": status.get("best_epoch"),
                "evaluated": bool(ev),
                "test_iou": (ev.get("splits") or {}).get("test", {}).get("iou_water"),
                "bolivia_iou": (ev.get("splits") or {}).get("bolivia", {}).get("iou_water"),
                "n_params": (ev.get("meta") or {}).get("n_params"),
                "checkpoint_mb": round(ck.stat().st_size / 2**20, 1),
                "loaded": d.name in self._loaded,
            })
        return out

    def choose(self, modalities: list[str], requested: str | None = None) -> str:
        # only finished runs are served; a checkpoint of a run still in progress is not a final model
        models = {m["name"]: m for m in self.available() if m["training_state"] == "completed"}
        if requested:
            if requested not in models:
                raise KeyError(f"unknown or unfinished model {requested!r}")
            if models[requested]["modality"] not in modalities:
                raise ValueError(f"model {requested!r} needs {models[requested]['modality']!r} input; "
                                 f"this image provides {modalities}")
            return requested
        ok = [m for m in models.values() if m["modality"] in modalities]
        if not ok:
            raise LookupError(f"no trained model accepts {modalities} input")
        # prefer the configured default, else the best validation IoU among compatible models
        for m in ok:
            if m["name"] == settings.default_model:
                return m["name"]
        return max(ok, key=lambda m: m["best_val_iou"] or 0)["name"]

    # -------------------------------------------------------------- loading
    def get(self, name: str):
        from training.infer import Predictor

        with self._lock:
            if name not in self._loaded:
                if name not in {m["name"] for m in self.available()}:
                    raise KeyError(f"unknown model {name!r}")
                log.info("loading model %s", name)
                self._loaded[name] = Predictor(self.runs_root / name, device=self.device)
            return self._loaded[name]

    def loaded(self) -> list[str]:
        return list(self._loaded)


registry = ModelRegistry()
