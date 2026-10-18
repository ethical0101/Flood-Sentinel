"""Independent re-verification of reported results (upgrade audit, Phase 0).

Deliberately does NOT reuse training/evaluate.py or the processed .npz cache:
  * reads the RAW Sen1Floods11 GeoTIFFs with rasterio,
  * normalises with the run's saved stats.json,
  * runs the saved best.pt checkpoint,
  * computes IoU with an independent implementation (set arithmetic on boolean masks),
  * recomputes NDWI from raw S2 digital numbers.
Also checks split integrity: duplicate chips (content hashes), Bolivia isolation, and
geographic proximity between held-out chips and training chips.

    .venv\\Scripts\\python scripts\\audit_reproduce.py
Writes artifacts/audit/reproduction.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training.model import load_for_inference  # noqa: E402
from training.utils import artifacts_dir, raw_dir, write_json  # noqa: E402

HL = "v1.1/data/flood_events/HandLabeled"


def split_ids() -> dict[str, list[str]]:
    out = {}
    for s, fn in {"train": "flood_train_data.csv", "valid": "flood_valid_data.csv", "test": "flood_test_data.csv",
                  "bolivia": "flood_bolivia_data.csv"}.items():
        with open(raw_dir() / "v1.1/splits/flood_handlabeled" / fn, newline="") as f:
            out[s] = [r[0].replace("_S1Hand.tif", "") for r in csv.reader(f) if r]
    return out


def read(layer: str, cid: str):
    with rasterio.open(raw_dir() / HL / layer / f"{cid}_{layer}.tif") as d:
        return d.read().astype(np.float32), d.bounds


def iou_sets(pred: np.ndarray, label: np.ndarray) -> tuple[int, int]:
    valid = label != -1
    p = pred & valid
    t = (label == 1) & valid
    return int((p & t).sum()), int((p | t).sum())


def main() -> None:
    splits = split_ids()
    report: dict = {"splits": {k: len(v) for k, v in splits.items()}}

    # ---------------------------------------------------------------- integrity
    allids = [c for v in splits.values() for c in v]
    report["chip_id_duplicates_across_splits"] = len(allids) - len(set(allids))
    report["bolivia_in_train_or_valid"] = sum(c.startswith("Bolivia") for s in ("train", "valid") for c in splits[s])
    hashes = defaultdict(list)
    bounds = {}
    for s, ids in splits.items():
        for c in ids:
            s2, b = read("S2Hand", c)
            hashes[hashlib.sha1(s2.tobytes()).hexdigest()].append((s, c))
            bounds[c] = b
    report["identical_s2_content_groups"] = [v for v in hashes.values() if len(v) > 1]

    def center(b):
        return ((b.bottom + b.top) / 2, (b.left + b.right) / 2)

    def km(a, b):
        (la1, lo1), (la2, lo2) = a, b
        p1, p2 = math.radians(la1), math.radians(la2)
        h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
        return 2 * 6371 * math.asin(math.sqrt(h))

    def overlaps(a, b):
        return not (a.right <= b.left or b.right <= a.left or a.top <= b.bottom or b.top <= a.bottom)

    prox = {}
    for s in ("valid", "test", "bolivia"):
        d = [min(km(center(bounds[c]), center(bounds[t])) for t in splits["train"]) for c in splits[s]]
        ov = sum(any(overlaps(bounds[c], bounds[t]) for t in splits["train"]) for c in splits[s])
        prox[s] = {"median_km_to_nearest_train_chip": round(float(np.median(d)), 2),
                   "min_km": round(float(np.min(d)), 2),
                   "share_within_10km": round(float(np.mean(np.array(d) < 10)), 3),
                   "chips_whose_footprint_overlaps_a_train_chip": int(ov)}
    report["geographic_proximity_to_train"] = prox

    # ---------------------------------------------------------------- reproduce metrics
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    repro = {}
    for run in ("unet_resnet34_s1s2", "unet_resnet34_s2", "unet_resnet34_s1", "unet_resnet34_rgb",
                "deeplabv3plus_mnv3l_s1s2"):
        rd = artifacts_dir() / "runs" / run
        if not (rd / "checkpoints" / "best.pt").exists():
            repro[run] = "checkpoint missing"
            continue
        model, ck = load_for_inference(rd / "checkpoints" / "best.pt", dev)
        mod = ck["config"]["data"]["modality"]
        st = json.loads((rd / "stats.json").read_text())
        res = {}
        for s in ("test", "bolivia"):
            inter = union = 0
            for c in splits[s]:
                s1 = read("S1Hand", c)[0]
                s2 = read("S2Hand", c)[0] / 10000.0
                label = read("LabelHand", c)[0][0]
                if mod == "rgb":
                    from training.dataset import IMAGENET_MEAN, IMAGENET_STD, rgb_stretch
                    x = rgb_stretch(s2[[3, 2, 1]]).astype(np.float32) / 255.0
                    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
                else:
                    parts = []
                    if mod in ("s1", "s1s2"):
                        parts.append((np.clip(s1, -50, 10) - np.array(st["s1_mean"])[:, None, None]) / np.array(st["s1_std"])[:, None, None])
                    if mod in ("s2", "s1s2"):
                        parts.append((s2 - np.array(st["s2_mean"])[:, None, None]) / np.array(st["s2_std"])[:, None, None])
                    x = np.concatenate(parts)
                    x[~np.isfinite(x)] = 0
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    p = torch.sigmoid(model(torch.from_numpy(x[None].astype(np.float32)).to(dev)).float())[0, 0].cpu().numpy()
                i, u = iou_sets(p > 0.5, label)
                inter += i
                union += u
            res[s] = round(inter / union, 4)
        reported = json.loads((rd / "evaluation.json").read_text())["splits"]
        res["reported"] = {s: round(reported[s]["iou_water"], 4) for s in ("test", "bolivia")}
        repro[run] = res
        print(run, res, flush=True)
        del model
        torch.cuda.empty_cache()

    # NDWI from raw digital numbers (B3 = index 2, B8 = index 7)
    nd = {}
    for s in ("test", "bolivia"):
        inter = union = 0
        for c in splits[s]:
            s2 = read("S2Hand", c)[0]
            label = read("LabelHand", c)[0][0]
            with np.errstate(divide="ignore", invalid="ignore"):
                ndwi = (s2[2] - s2[7]) / (s2[2] + s2[7])
            i, u = iou_sets(ndwi > 0, label)
            inter += i
            union += u
        nd[s] = round(inter / union, 4)
    repro["ndwi_gt0"] = nd
    print("ndwi", nd)
    report["reproduction_iou_water"] = repro
    write_json(artifacts_dir() / "audit" / "reproduction.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "reproduction_iou_water"}, indent=1))


if __name__ == "__main__":
    main()
