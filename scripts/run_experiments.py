r"""Run the experiment suite sequentially (never in parallel: one 4 GB GPU).

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


# v2 controlled experiments: one change at a time vs the v1 recipe (unet_resnet34_s1s2, seed 42).
# Selection rule fixed in advance: highest validation IoU at its validation-selected threshold.
# Bolivia and WorldFloods are reported for every run but never used to choose between runs.
RADIO = ["data.radiometric.enabled=true"]
EXPERIMENTS_V2 = {
    "v2_r0b_unet_s1s2_seed7": ["seed=7"],                                  # seed-noise reference
    "v2_r1_unet_s1s2_radiometric": RADIO,                                  # + photometric augmentation
    "v2_r2_unet_s1s2i_indices": ["data.modality=s1s2i"],                   # + NDWI/MNDWI channels
    "v2_r3_unet_s1s2i_radiometric": ["data.modality=s1s2i", *RADIO],       # both
    "v2_r4_unet_s2i_radiometric": ["data.modality=s2i", *RADIO],           # S2-only version (WorldFloods)
}


def wait_for_idle_gpu(quiet_s: int) -> None:
    import time

    from training.gpu_guard import foreign_gpu_processes

    quiet_since = None
    announced = False
    while True:
        busy = foreign_gpu_processes()
        now = time.time()
        if busy:
            quiet_since = None
            if not announced:
                print(f"   waiting: GPU in use by pid {busy[0]['pid']} ({busy[0]['cmd'][-60:]})", flush=True)
                announced = True
        elif quiet_since is None:
            quiet_since = now
        elif now - quiet_since >= quiet_s:
            return
        time.sleep(15)


def state(run: str) -> str:
    p = runs_dir() / run / "status.json"
    return json.loads(p.read_text()).get("state", "unknown") if p.exists() else "new"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--suite", choices=["v1", "v2"], default="v1")
    ap.add_argument("--polite", action="store_true",
                    help="only use the GPU when no other project's GPU job has run for --quiet-s seconds; "
                         "runs pause at epoch boundaries when one appears and resume afterwards")
    ap.add_argument("--quiet-s", type=int, default=300)
    a = ap.parse_args()
    py = sys.executable
    suite = EXPERIMENTS if a.suite == "v1" else EXPERIMENTS_V2
    for run, overrides in suite.items():
        if a.only and run not in a.only:
            continue
        st = state(run)
        if st == "completed":
            print(f"== {run}: already completed, skipping")
            continue
        while True:
            if a.polite:
                wait_for_idle_gpu(a.quiet_s)
            cmd = [py, "-m", "training.train", "--run", run]
            cmd += ["--resume"] if (runs_dir() / run / "checkpoints" / "last.pt").exists() else \
                [x for o in overrides for x in ("--set", o)]
            if a.polite:
                cmd.append("--polite")
            print(f"== {run}: {state(run)} -> {' '.join(cmd[1:])}", flush=True)
            rc = subprocess.call(cmd, cwd=REPO)
            if rc != 75:
                break
            print(f"== {run}: paused for another GPU job; will resume when the GPU is idle", flush=True)
        if rc != 0:
            print(f"== {run} exited with {rc}; stopping suite")
            return rc
    if not a.skip_eval:
        return subprocess.call([py, "-m", "training.evaluate", "--all"], cwd=REPO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
