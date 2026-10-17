"""Convert verified Sen1Floods11 GeoTIFFs into fast-loading per-chip .npz files.

* Official split CSVs are used unchanged: train / valid / test, plus the Bolivia event,
  which the dataset authors hold out entirely as an unseen-region test set.
* Normalisation statistics are computed on the TRAIN split only (no leakage).
* Georeferencing (CRS, affine transform, bounds) is preserved in index.json so that
  predictions can later be aligned with road networks.

    python -m training.prepare_data
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import rasterio

from training.dataset import s2_to_rgb8
from training.utils import processed_dir, raw_dir, write_json

HL = "v1.1/data/flood_events/HandLabeled"
SPLITS = {"train": "flood_train_data.csv", "valid": "flood_valid_data.csv",
          "test": "flood_test_data.csv", "bolivia": "flood_bolivia_data.csv"}


def read(layer: str, cid: str, root: Path):
    with rasterio.open(root / HL / layer / f"{cid}_{layer}.tif") as ds:
        return ds.read(), ds.crs.to_string(), ds.transform, ds.bounds


def main() -> int:
    raw, out = raw_dir(), processed_dir()
    (out / "chips").mkdir(parents=True, exist_ok=True)
    chips = []
    s1_sum = np.zeros(2); s1_sq = np.zeros(2); s1_n = np.zeros(2)
    s2_sum = np.zeros(13); s2_sq = np.zeros(13); s2_n = np.zeros(13)
    pos = neg = 0
    for split, fn in SPLITS.items():
        with open(raw / "v1.1/splits/flood_handlabeled" / fn, newline="") as f:
            ids = [r[0].replace("_S1Hand.tif", "") for r in csv.reader(f) if r]
        for cid in ids:
            s1, crs, tr, b = read("S1Hand", cid, raw)
            s2 = read("S2Hand", cid, raw)[0].astype(np.float32) / 10000.0
            label = read("LabelHand", cid, raw)[0][0].astype(np.int8)
            otsu = read("S1OtsuLabelHand", cid, raw)[0][0].astype(np.int8)
            jrc = read("JRCWaterHand", cid, raw)[0][0].astype(np.int8)
            s1 = s1.astype(np.float32)
            np.savez(out / "chips" / f"{cid}.npz", s1=s1.astype(np.float16), s2=s2.astype(np.float16),
                     rgb=s2_to_rgb8(s2), label=label, otsu=otsu, jrc=jrc)
            valid = label != -1
            chips.append({
                "id": cid, "event": cid.split("_")[0], "split": split, "crs": crs,
                "transform": list(tr)[:6], "bounds": [b.left, b.bottom, b.right, b.top],
                "height": int(label.shape[0]), "width": int(label.shape[1]),
                "valid_fraction": float(valid.mean()),
                "water_fraction": float((label == 1).sum() / max(valid.sum(), 1)),
            })
            if split == "train":
                for k in range(2):
                    v = s1[k][np.isfinite(s1[k])]
                    v = np.clip(v, -50, 10)
                    s1_sum[k] += v.sum(); s1_sq[k] += (v.astype(np.float64) ** 2).sum(); s1_n[k] += v.size
                for k in range(13):
                    v = s2[k][np.isfinite(s2[k])].astype(np.float64)
                    s2_sum[k] += v.sum(); s2_sq[k] += (v ** 2).sum(); s2_n[k] += v.size
                pos += int((label == 1).sum()); neg += int((label == 0).sum())
            print(f"{split:8s} {cid}", flush=True)

    def ms(s, q, n):
        m = s / n
        return m.tolist(), np.sqrt(np.maximum(q / n - m ** 2, 1e-12)).tolist()

    s1m, s1s = ms(s1_sum, s1_sq, s1_n)
    s2m, s2s = ms(s2_sum, s2_sq, s2_n)
    write_json(out / "stats.json", {"computed_on": "train", "s1_mean": s1m, "s1_std": s1s,
                                    "s2_mean": s2m, "s2_std": s2s,
                                    "train_water_pixels": pos, "train_notwater_pixels": neg,
                                    "train_pos_fraction": pos / max(pos + neg, 1)})
    write_json(out / "index.json", {"dataset": "Sen1Floods11 v1.1 HandLabeled", "chips": chips})
    print(f"wrote {len(chips)} chips to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
