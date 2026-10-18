"""Choose a reproducible, leakage-free subset of WorldFloods v2 TRAIN scenes for multi-dataset training.

Rules (deterministic, recorded in configs/worldfloods_train_subset.json):
  1. Start from the official WorldFloods v2 train split (dataset_metadata.csv).
  2. Exclude any scene whose footprint lies within BUFFER_KM of ANY Sen1Floods11 valid / test / Bolivia chip,
     so that no held-out evaluation area (or its immediate surroundings) is seen during training.
  3. Event diversity first: order scenes by (scene index within its EMS/UNOSAT event, size), so one scene per
     event is taken before any event contributes a second scene.
  4. At most --per-event scenes per event (default 1); stop at the download budget (default 5 GB).

    .venv\\Scripts\\python scripts\\select_worldfloods_subset.py [--budget-gb 20] [--buffer-km 10]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from pyproj import Transformer

REPO = Path(__file__).resolve().parents[1]
API = "https://huggingface.co/api/datasets/isp-uv-es/WorldFloodsv2/tree/main/train/S2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=5.0)
    ap.add_argument("--per-event", type=int, default=1, help="max scenes per flood event")
    ap.add_argument("--buffer-km", type=float, default=10.0)
    a = ap.parse_args()

    meta = list(csv.DictReader(open(REPO / "data/raw/worldfloods_v2/dataset_metadata.csv", encoding="utf-8")))
    train = [r for r in meta if r["split"] == "train"]
    with urllib.request.urlopen(API, timeout=60) as r:
        sizes = {Path(x["path"]).stem: x.get("lfs", {}).get("size", x.get("size", 0)) for x in json.load(r)}

    chips = json.loads((REPO / "data/processed/index.json").read_text())["chips"]
    held = [c for c in chips if c["split"] in ("valid", "test", "bolivia")]
    buf = a.buffer_km / 111.0  # degrees (conservative; >= true distance in longitude away from the equator)

    def wgs84_bounds(r):
        l, b, rr, t = json.loads(r["bounds"])
        tr = Transformer.from_crs(r["crs"], "EPSG:4326", always_xy=True)
        xs, ys = tr.transform([l, rr, l, rr], [b, b, t, t])
        return min(xs), min(ys), max(xs), max(ys)

    kept, excluded = [], []
    for r in train:
        l, b, rr, t = wgs84_bounds(r)
        hit = [c["id"] for c in held if not (c["bounds"][2] < l - buf or c["bounds"][0] > rr + buf
                                             or c["bounds"][3] < b - buf or c["bounds"][1] > t + buf)]
        (excluded if hit else kept).append((r, (l, b, rr, t), hit))

    by_event = defaultdict(list)
    for r, bb, _ in kept:
        ev = r["ems_code"] or r["event id"].split("_")[0]
        by_event[ev].append((r, bb))
    order = []
    for ev, items in by_event.items():
        for i, (r, bb) in enumerate(sorted(items, key=lambda x: (sizes.get(x[0]["event id"], 1e18), x[0]["event id"]))):
            order.append((i, sizes.get(r["event id"], 0), r["event id"], ev, bb, r))
    order.sort(key=lambda x: (x[0], x[1], x[2]))

    chosen, total = [], 0
    for rank, size, eid, ev, bb, r in order:
        if size <= 0 or rank >= a.per_event or total + size > a.budget_gb * 1e9:
            continue
        chosen.append({"event_id": eid, "event": ev, "source": r["source"], "s2_date": r["s2_date"], "bytes": size,
                       "bounds_wgs84": [round(v, 4) for v in bb]})
        total += size
    out = {
        "rule": f"official train split; exclude scenes within {a.buffer_km} km of any Sen1Floods11 valid/test/Bolivia "
                f"chip; at most {a.per_event} scene(s) per event, smallest first; budget {a.budget_gb} GB",
        "n_train_scenes": len(train), "n_excluded_near_heldout": len(excluded),
        "excluded": [{"event_id": r["event id"], "near_chips": hit[:5]} for r, _, hit in excluded],
        "n_events_available": len(by_event), "n_chosen": len(chosen),
        "n_events_chosen": len({c["event"] for c in chosen}), "bytes": total, "scenes": chosen}
    (REPO / "configs" / "worldfloods_train_subset.json").write_text(json.dumps(out, indent=1))
    print(f"train scenes {len(train)}; excluded near held-out {len(excluded)}; events {len(by_event)}; "
          f"chosen {len(chosen)} scenes from {out['n_events_chosen']} events, {total / 1e9:.1f} GB")
    for e in out["excluded"]:
        print("  excluded:", e["event_id"], "near", e["near_chips"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
