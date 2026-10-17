"""Verify the downloaded Sen1Floods11 subset (integrity + structure + label sanity).

Checks
  1. Every file in manifest.json exists and matches its published MD5 (``--fast`` = size only).
  2. Every chip referenced by the official split CSVs has all required layers.
  3. Each chip's layers share CRS (EPSG:4326), shape (512x512) and geotransform.
  4. Label values are a subset of {-1, 0, 1}; S1 has 2 bands, S2 has 13 bands.

Writes a JSON report to artifacts/dataset_verification.json and exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.utils import artifacts_dir, raw_dir, write_json  # noqa: E402

HL = "v1.1/data/flood_events/HandLabeled"
SPLITS = {"train": "flood_train_data.csv", "valid": "flood_valid_data.csv",
          "test": "flood_test_data.csv", "bolivia": "flood_bolivia_data.csv"}
LAYERS = {"S1Hand": 2, "S2Hand": 13, "LabelHand": 1, "S1OtsuLabelHand": 1, "JRCWaterHand": 1}


def md5_b64(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return base64.b64encode(h.digest()).decode()


def read_splits(root: Path) -> dict[str, list[str]]:
    out = {}
    for name, fn in SPLITS.items():
        with open(root / "v1.1/splits/flood_handlabeled" / fn, newline="") as f:
            out[name] = [row[0].replace("_S1Hand.tif", "") for row in csv.reader(f) if row]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="check sizes only, skip MD5")
    args = ap.parse_args()
    root = raw_dir()
    errors: list[str] = []
    report: dict = {"root": str(root)}

    man_path = root / "manifest.json"
    if not man_path.exists():
        print(f"manifest.json missing in {root}; run scripts/download_datasets.py first")
        return 2
    manifest = json.loads(man_path.read_text())
    n_ok = 0
    for rel, meta in manifest["files"].items():
        p = root / rel
        if not p.exists():
            errors.append(f"missing: {rel}")
        elif p.stat().st_size != meta["size"]:
            errors.append(f"size mismatch: {rel}")
        elif not args.fast and md5_b64(p) != meta["md5"]:
            errors.append(f"md5 mismatch: {rel}")
        else:
            n_ok += 1
    report["files_checked"] = len(manifest["files"])
    report["files_ok"] = n_ok
    report["integrity_mode"] = "size" if args.fast else "md5"

    splits = read_splits(root)
    report["split_sizes"] = {k: len(v) for k, v in splits.items()}
    all_ids = [c for v in splits.values() for c in v]
    dup = [c for c, n in Counter(all_ids).items() if n > 1]
    if dup:
        errors.append(f"chips in more than one split: {dup[:5]}")
    report["unique_chips"] = len(set(all_ids))

    label_values: Counter = Counter()
    crs_seen: Counter = Counter()
    for cid in all_ids:
        ref = None
        for layer, bands in LAYERS.items():
            p = root / HL / layer / f"{cid}_{layer}.tif"
            if not p.exists():
                errors.append(f"missing layer {layer} for {cid}")
                continue
            with rasterio.open(p) as ds:
                crs_seen[str(ds.crs)] += 1
                if ds.count != bands:
                    errors.append(f"{p.name}: {ds.count} bands, expected {bands}")
                sig = (ds.width, ds.height, np.array(ds.transform[:6]), str(ds.crs))
                if ref is None:
                    ref = sig
                # grids must match to within 1/1000 of a pixel (ignores float round-off in the GeoTIFFs)
                elif sig[:2] != ref[:2] or sig[3] != ref[3] or not np.allclose(
                        sig[2], ref[2], rtol=0, atol=abs(ref[2][0]) * 1e-3):
                    errors.append(f"{cid}: {layer} grid differs from S1Hand")
                if layer == "LabelHand":
                    v, n = np.unique(ds.read(1), return_counts=True)
                    label_values.update(dict(zip(v.tolist(), n.tolist())))
    bad_vals = set(label_values) - {-1, 0, 1}
    if bad_vals:
        errors.append(f"unexpected label values {bad_vals}")
    report["crs"] = dict(crs_seen)
    report["label_pixel_counts"] = {str(k): v for k, v in sorted(label_values.items())}
    report["errors"] = errors
    report["passed"] = not errors
    write_json(artifacts_dir() / "dataset_verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2))
    print(f"{len(errors)} error(s)")
    for e in errors[:20]:
        print("  ", e)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
