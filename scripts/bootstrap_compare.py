"""Paired bootstrap over chips: is a v2 run's IoU difference vs the v1 baseline larger than chance?

Resamples chips with replacement (2,000 draws, seed 0) and recomputes pooled water IoU for each model at
its validation-selected threshold. Reports the observed difference and the 95% percentile interval, on the
test split and on Bolivia. Seed-to-seed spread (v1 vs R0b) is reported the same way as a noise reference.

    .venv\\Scripts\\python scripts\\bootstrap_compare.py
Writes artifacts/v2_bootstrap.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training.dataset import chip_input, load_chip, load_index  # noqa: E402
from training.infer import Predictor  # noqa: E402
from training.utils import artifacts_dir, runs_dir, write_json  # noqa: E402

BASE = "unet_resnet34_s1s2"
OTHERS = ["v2_r0b_unet_s1s2_seed7", "v2_r1_unet_s1s2_radiometric", "v2_r2_unet_s1s2i_indices", "v2_r3_unet_s1s2i_radiometric",
          "v2_r4_unet_s2i_radiometric"]


def per_chip_counts(run: str, ids: list[str]) -> np.ndarray:
    thr = json.loads((runs_dir() / run / "evaluation.json").read_text()).get("val_selected_threshold", 0.5)
    p = Predictor(run)
    out = []
    for cid in ids:
        ch = load_chip(cid)
        prob = p.predict(chip_input(ch, p.modality, p.stats))
        y = ch["label"]
        v = y != -1
        pr, t = (prob > thr) & v, (y == 1) & v
        out.append([(pr & t).sum(), (pr & ~t).sum(), (~pr & t).sum()])
    return np.array(out, np.float64)


def iou(c: np.ndarray) -> float:
    tp, fp, fn = c.sum(0)
    return tp / (tp + fp + fn)


def main() -> None:
    idx = load_index()
    rng = np.random.default_rng(0)
    out = {"method": "paired chip bootstrap, 2000 draws, pooled water IoU at each run's validation-selected threshold",
           "baseline": BASE, "splits": {}}
    for split in ("test", "bolivia"):
        ids = [c["id"] for c in idx if c["split"] == split]
        counts = {r: per_chip_counts(r, ids) for r in [BASE, *OTHERS]}
        draws = rng.integers(0, len(ids), size=(2000, len(ids)))
        res = {}
        for r in OTHERS:
            obs = iou(counts[r]) - iou(counts[BASE])
            boot = np.array([iou(counts[r][d]) - iou(counts[BASE][d]) for d in draws])
            res[r] = {"delta_iou": round(float(obs), 4), "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                                                                   round(float(np.percentile(boot, 97.5)), 4)],
                      "p_delta_le_0": round(float((boot <= 0).mean()), 4)}
            print(split, r, res[r], flush=True)
        out["splits"][split] = {"n_chips": len(ids), "comparisons": res}
    write_json(artifacts_dir() / "v2_bootstrap.json", out)


if __name__ == "__main__":
    main()
