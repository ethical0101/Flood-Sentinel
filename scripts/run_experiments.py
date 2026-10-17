"""Run the experiment suite sequentially (never in parallel: one 4 GB GPU).

Completed runs are skipped, interrupted runs are resumed, so this is safe to rerun:
    .venv\Scripts\python scripts\run_experiments.py            # train all, then evaluate
    .venv\Scripts\python scripts\run_experiments.py --only unet_resnet34_s1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training.utils import runs_dir  # noqa: E402

EXPERIMENTS = {
    # main model: SAR + optical
    "unet_resnet34_s1s2": [],
    # input-modality ablation (same architecture)
    "unet_resnet34_s2": ["data.modality=s2"],
    "unet_resnet34_s1": ["data.modality=s1"],
    "unet_resnet34_rgb": ["data.modality=rgb"],
    # architecture comparison (same input): compact DeepLabV3+ / MobileNetV3-Large
    "deeplabv3plus_mnv3l_s1s2": ["model.arch=deeplabv3plus", "model.encoder=tu-mobilenetv3_large_100"],
}


def state(run: str) -> str:
    p = runs_dir() / run / "status.json"
    return json.loads(p.read_text()).get("state", "unknown") if p.exists() else "new"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    for run, overrides in EXPERIMENTS.items():
        if a.only and run not in a.only:
            continue
        st = state(run)
        if st == "completed":
            print(f"== {run}: already completed, skipping")
            continue
        cmd = [py, "-m", "training.train", "--run", run]
        cmd += ["--resume"] if (runs_dir() / run / "checkpoints" / "last.pt").exists() else \
            [x for o in overrides for x in ("--set", o)]
        print(f"== {run}: {st} -> {' '.join(cmd[1:])}", flush=True)
        rc = subprocess.call(cmd, cwd=REPO)
        if rc != 0:
            print(f"== {run} exited with {rc}; stopping suite")
            return rc
    if not a.skip_eval:
        return subprocess.call([py, "-m", "training.evaluate", "--all"], cwd=REPO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
