r"""End-to-end multi-dataset pipeline (GPU-polite): prepare WorldFloods tiles, train the 'md' suite, evaluate.

    .venv\Scripts\python scripts\run_multidataset.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    py = sys.executable
    steps = [
        *([] if "--skip-prepare" in sys.argv else [[py, "-m", "training.prepare_worldfloods"]]),
        [py, "scripts/run_experiments.py", "--suite", "md", "--polite", "--skip-eval", "--quiet-s", "120"],
        [py, "-m", "training.evaluate", "--all", "--reuse", "--skip-baselines"],
        [py, "scripts/evaluate_worldfloods.py"],
        [py, "scripts/bootstrap_compare.py"],
        [py, "-m", "training.registry"],
    ]
    for cmd in steps:
        print("==", " ".join(cmd[1:]), flush=True)
        rc = subprocess.call(cmd, cwd=REPO)
        print("== exit", rc, flush=True)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
