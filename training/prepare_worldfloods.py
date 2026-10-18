"""Tile the WorldFloods v2 training subset into Sen1Floods11-compatible 512x512 chips (S2 only).

Label mapping (from each scene's meta "encoding_values"):
    gt band 2 (water): 2 -> 1 (water), 1 -> 0 (land), 0 -> -1 (invalid)
    gt band 1 (cloud): 2 (cloud) -> -1   an optical model cannot observe the surface under cloud
    S2 all-zero pixels -> -1 (no data)
Tiles: non-overlapping 512 px grid (scenes with a side of 256-511 px give smaller tiles; <256 px skipped); keep tiles with >= 50% valid pixels; up to TILES_PER_SCENE per scene,
water-bearing tiles (>= 1% water) first, then others, in a seeded random order. Imagery is stored exactly as
Sen1Floods11 chips (S2 TOA reflectance = DN / 10000, float16) and normalised later with the SAME Sen1Floods11
train statistics, so every model sees one consistent input scale. No Sentinel-1 is available for these tiles.

    .venv\\Scripts\\python -m training.prepare_worldfloods
Writes <processed>/wf_chips/*.npz (compressed) and <processed>/index_worldfloods.json (split "wf_train").
"""
from __future__ import annotations

import json
import sys

import numpy as np
import rasterio
from rasterio.windows import Window

from training.dataset import s2_to_rgb8
from training.utils import REPO, processed_dir, raw_dir, write_json

TILE = 512
MIN_SIDE = 256          # training uses 256 px crops; scenes with a side in [256, 512) become one smaller tile
TILES_PER_SCENE = 8
SEED = 42


def main() -> int:
    root = raw_dir().parent / "worldfloods_v2" / "train"
    subset = json.loads((REPO / "configs" / "worldfloods_train_subset.json").read_text())
    out_dir = processed_dir() / "wf_chips"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    chips, skipped = [], []
    for sc in subset["scenes"]:
        eid = sc["event_id"]
        s2p, gtp, mp = root / "S2" / f"{eid}.tif", root / "gt" / f"{eid}.tif", root / "meta" / f"{eid}.json"
        if not (s2p.exists() and gtp.exists() and mp.exists()):
            skipped.append((eid, "missing files"))
            continue
        meta = json.loads(mp.read_text())
        if meta.get("encoding_values") != [{"0": "invalid", "1": "clear", "2": "cloud"}, {"0": "invalid", "1": "land", "2": "water"}]:
            skipped.append((eid, "unexpected label encoding"))
            continue
        with rasterio.open(s2p) as s2ds, rasterio.open(gtp) as gtds:
            if s2ds.shape != gtds.shape or s2ds.transform != gtds.transform or s2ds.count < 13:
                skipped.append((eid, "grid/band mismatch"))
                continue
            H, W = s2ds.shape
            th, tw = min(TILE, H), min(TILE, W)
            if min(th, tw) < MIN_SIDE:
                skipped.append((eid, f"scene {H}x{W} smaller than {MIN_SIDE} px"))
                continue
            cands = []
            for r in range(0, H - th + 1, th):
                for c in range(0, W - tw + 1, tw):
                    gt = gtds.read(window=Window(c, r, tw, th))
                    lab = np.where(gt[1] == 2, 1, np.where(gt[1] == 1, 0, -1)).astype(np.int8)
                    lab[gt[0] == 2] = -1
                    valid = (lab != -1).mean()
                    if valid < 0.5:
                        continue
                    cands.append((r, c, float((lab == 1).sum() / max(1, (lab != -1).sum())), lab))
            water = [x for x in cands if x[2] >= 0.01]
            dry = [x for x in cands if x[2] < 0.01]
            rng.shuffle(water)
            rng.shuffle(dry)
            take = (water + dry)[:TILES_PER_SCENE]
            for r, c, wf, lab in take:
                s2 = s2ds.read(list(range(1, 14)), window=Window(c, r, tw, th)).astype(np.float32) / 10000.0
                nod = (s2 == 0).all(0)
                lab = lab.copy()
                lab[nod] = -1
                if (lab != -1).mean() < 0.5:
                    continue
                cid = f"WF_{eid}_{r}_{c}"
                np.savez_compressed(out_dir / f"{cid}.npz", s2=s2.astype(np.float16), rgb=s2_to_rgb8(s2), label=lab)
                tr = s2ds.window_transform(Window(c, r, tw, th))
                chips.append({"id": cid, "dataset": "worldfloods_v2", "event": sc["event"], "scene": eid, "split": "wf_train",
                              "crs": s2ds.crs.to_string(), "transform": list(tr)[:6], "bounds_wgs84": sc["bounds_wgs84"],
                              "height": th, "width": tw, "valid_fraction": float((lab != -1).mean()),
                              "water_fraction": float((lab == 1).sum() / max(1, (lab != -1).sum()))})
        print(f"{eid}: {sum(ch['scene'] == eid for ch in chips)} tiles", flush=True)
    write_json(processed_dir() / "index_worldfloods.json", {
        "dataset": "WorldFloods v2 training subset (S2 only)", "tile": TILE, "tiles_per_scene": TILES_PER_SCENE,
        "label_mapping": "water 2->1, land 1->0, invalid 0 / cloud / S2 no-data -> -1",
        "subset_rule": subset["rule"], "skipped": skipped, "chips": chips})
    lab_px = sum(c["valid_fraction"] for c in chips) * TILE * TILE
    print(f"{len(chips)} tiles from {len({c['scene'] for c in chips})} scenes; skipped {len(skipped)}; "
          f"mean water share {np.mean([c['water_fraction'] for c in chips]):.3f}; valid px {lab_px / 1e6:.1f} M")
    return 0


if __name__ == "__main__":
    sys.exit(main())
