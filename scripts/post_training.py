"""After the v2 suite finishes: wait for an idle GPU, then evaluate everything (polite, sequential).

    .venv\\Scripts\\python scripts\\post_training.py [--no-wait-suite]
Runs: training.evaluate --all (Sen1Floods11 valid/test/Bolivia + baselines + val-selected thresholds),
scripts/evaluate_worldfloods.py (external test), training.registry (experiments.csv/json).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def suite_running() -> bool:
    import psutil

    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except Exception:  # noqa: BLE001
            continue
        if "run_experiments.py" in cmd and "--suite" in cmd and "v2" in cmd:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait-suite", action="store_true")
    ap.add_argument("--quiet-s", type=int, default=120)
    a = ap.parse_args()
    if not a.no_wait_suite:
        while suite_running():
            time.sleep(30)
    from scripts.run_experiments import wait_for_idle_gpu  # noqa: E402

    py = sys.executable
    steps = [[py, "-m", "training.evaluate", "--all"], [py, "scripts/evaluate_worldfloods.py"], [py, "-m", "training.registry"]]
    for cmd in steps:
        wait_for_idle_gpu(a.quiet_s)
        print("==", " ".join(cmd[1:]), time.strftime("%H:%M:%S"), flush=True)
        rc = subprocess.call(cmd, cwd=REPO)
        print("== exit", rc, flush=True)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
