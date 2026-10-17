"""Shared helpers: paths, config loading, seeding, environment capture."""
from __future__ import annotations

import copy
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def raw_dir() -> Path:
    return Path(os.environ.get("FLOOD_DATA_DIR", REPO / "data" / "raw")) / "sen1floods11"


def processed_dir() -> Path:
    return Path(os.environ.get("FLOOD_PROCESSED_DIR", REPO / "data" / "processed"))


def artifacts_dir() -> Path:
    return Path(os.environ.get("FLOOD_ARTIFACTS_DIR", REPO / "artifacts"))


def runs_dir() -> Path:
    return artifacts_dir() / "runs"


def _coerce(v: str) -> Any:
    low = v.lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    with open(path or REPO / "training" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _coerce(val)
    if not cfg.get("run_name"):
        m = cfg["model"]
        cfg["run_name"] = f"{m['arch']}_{m['encoder']}_{cfg['data']['modality']}"
    return cfg


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def environment_info() -> dict:
    import torch

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    try:
        import segmentation_models_pytorch as smp

        info["segmentation_models_pytorch"] = smp.__version__
    except ImportError:
        pass
    try:
        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def deep_copy(d: dict) -> dict:
    return copy.deepcopy(d)
