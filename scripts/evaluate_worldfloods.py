"""Independent external evaluation on the WorldFloods v2 TEST split (never used for training/selection).

Label mapping (from each scene's meta JSON "encoding_values"):
    gt band 1 (cloud): 0 invalid, 1 clear, 2 cloud
    gt band 2 (water): 0 invalid, 1 land,  2 water   ->  water=1, land=0, invalid=-1 (ignored)
Primary metric: pixels that are valid AND clear-sky (an optical-only model cannot see under cloud).
Secondary: all valid pixels (water under cloud counted as missed if not predicted).

Imagery: WorldFloods S2 GeoTIFF bands 1-13 = Sentinel-2 L1C B1..B12 incl. B8A (same order and x10000 scaling
as Sen1Floods11); bands 14-15 (QA60, cloud probability) are not used. Both datasets are 10 m.
Scenes are processed in overlapping 1024 px windows to bound RAM.

Models evaluated: every completed run whose modality needs only Sentinel-2 (s2, s2i, rgb), at 0.5 and at the
threshold each run selected on Sen1Floods11 validation. Baselines: NDWI>0, MNDWI>0.

    .venv\\Scripts\\python scripts\\evaluate_worldfloods.py
Writes artifacts/worldfloods_evaluation.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training.dataset import normalize, rgb_stretch  # noqa: E402
from training.infer import Predictor  # noqa: E402
from training.metrics import summarize  # noqa: E402
from training.utils import artifacts_dir, raw_dir, runs_dir, write_json  # noqa: E402

WIN, PAD = 1024, 64
S2_ONLY = {"s2", "s2i", "rgb"}


def cm(pred: np.ndarray, y: np.ndarray, m: np.ndarray) -> np.ndarray:
    p, t = pred[m], y[m] == 1
    return np.array([[(~p & ~t).sum(), (p & ~t).sum()], [(~p & t).sum(), (p & t).sum()]], np.int64)


def main() -> None:
    root = raw_dir().parent / "worldfloods_v2" / "test"
    scenes = sorted(p.stem for p in (root / "S2").glob("*.tif"))
    runs = []
    for d in sorted(runs_dir().iterdir()):
        ev = d / "evaluation.json"
        if d.name.startswith("smoke_") or not ev.exists():
            continue
        e = json.loads(ev.read_text())
        if e["modality"] in S2_ONLY:
            runs.append((d.name, e.get("val_selected_threshold", 0.5)))
    print("models:", runs, "scenes:", len(scenes), flush=True)
    preds = {r: Predictor(r) for r, _ in runs}
    keys = [f"{r}@0.5" for r, _ in runs] + [f"{r}@val{t}" for r, t in runs] + ["ndwi_gt0", "mndwi_gt0"]
    tot = {k: {"clear": np.zeros((2, 2), np.int64), "all": np.zeros((2, 2), np.int64)} for k in keys}
    per_scene = {}
    t0 = time.time()
    for sc in scenes:
        meta = json.loads((root / "meta" / f"{sc}.json").read_text())
        enc = meta.get("encoding_values")
        assert enc == [{"0": "invalid", "1": "clear", "2": "cloud"}, {"0": "invalid", "1": "land", "2": "water"}], enc
        sc_cm = {k: {"clear": np.zeros((2, 2), np.int64), "all": np.zeros((2, 2), np.int64)} for k in keys}
        with rasterio.open(root / "S2" / f"{sc}.tif") as s2ds, rasterio.open(root / "gt" / f"{sc}.tif") as gtds:
            assert s2ds.crs == gtds.crs and s2ds.transform == gtds.transform and s2ds.shape == gtds.shape
            H, W = s2ds.shape
            for r0 in range(0, H, WIN):
                for c0 in range(0, W, WIN):
                    rr0, cc0 = max(0, r0 - PAD), max(0, c0 - PAD)
                    rr1, cc1 = min(H, r0 + WIN + PAD), min(W, c0 + WIN + PAD)
                    win = Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
                    s2 = s2ds.read(list(range(1, 14)), window=win).astype(np.float32) / 10000.0
                    gt = gtds.read(window=win)
                    core = (slice(r0 - rr0, r0 - rr0 + min(WIN, H - r0)), slice(c0 - cc0, c0 - cc0 + min(WIN, W - c0)))
                    cloud, water = gt[0][core], gt[1][core]
                    y = np.where(water == 2, 1, 0)
                    valid = water > 0
                    nodata = (s2 == 0).all(0)
                    valid_core = valid & ~nodata[core]
                    clear = valid_core & (cloud == 1)
                    if not valid_core.any():
                        continue
                    with np.errstate(divide="ignore", invalid="ignore"):
                        nd = np.nan_to_num((s2[2] - s2[7]) / (s2[2] + s2[7]), nan=-1)[core] > 0
                        mnd = np.nan_to_num((s2[2] - s2[11]) / (s2[2] + s2[11]), nan=-1)[core] > 0
                    for k, m in (("ndwi_gt0", nd), ("mndwi_gt0", mnd)):
                        sc_cm[k]["clear"] += cm(m, y, clear)
                        sc_cm[k]["all"] += cm(m, y, valid_core)
                    for r, th in runs:
                        p = preds[r]
                        if p.modality == "rgb":
                            x = normalize("rgb", rgb=rgb_stretch(s2[[3, 2, 1]]))
                        else:
                            x = normalize(p.modality, s2=s2, stats=p.stats)
                        prob = p.predict(x)[core]
                        for k, t in ((f"{r}@0.5", 0.5), (f"{r}@val{th}", th)):
                            sc_cm[k]["clear"] += cm(prob > t, y, clear)
                            sc_cm[k]["all"] += cm(prob > t, y, valid_core)
        for k in keys:
            for m in ("clear", "all"):
                tot[k][m] += sc_cm[k][m]
        per_scene[sc] = {k: {m: summarize(v)["iou_water"] for m, v in d.items()} for k, d in sc_cm.items()}
        per_scene[sc]["_meta"] = {"s2_date": meta.get("s2_date"), "ems_code": meta.get("ems_code"),
                                  "event_type": meta.get("event type"), "cloud_probability": meta.get("cloud_probability"),
                                  "bounds_crs": meta.get("crs")}
        print(f"{sc:52s} " + " ".join(f"{k.split('@')[0][-12:]}@{k.split('@')[-1] if '@' in k else ''}="
                                      f"{per_scene[sc][k]['clear']:.3f}" for k in keys), f"{time.time() - t0:.0f}s",
              flush=True)
    out = {"dataset": "WorldFloods v2 test split (18 scenes, CC BY-NC 4.0)",
           "label_mapping": "water band: 2->water(1), 1->land(0), 0->ignored; primary = clear-sky valid pixels",
           "models": {r: {"val_selected_threshold": t} for r, t in runs},
           "totals": {k: {m: summarize(v) for m, v in d.items()} for k, d in tot.items()},
           "per_scene_iou_water": per_scene}
    write_json(artifacts_dir() / "worldfloods_evaluation.json", out)
    for k in keys:
        print(f"{k:40s} clear IoU={out['totals'][k]['clear']['iou_water']:.4f} "
              f"all-valid IoU={out['totals'][k]['all']['iou_water']:.4f}")


if __name__ == "__main__":
    main()
