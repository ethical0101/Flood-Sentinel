"""Download (once) and cache the OSM drive network for every curated study area.

    .venv\\Scripts\\python scripts\\fetch_road_networks.py
Cached graphs live in $FLOOD_OSM_CACHE (default data/osm_cache) as GraphML + a JSON sidecar
with fetch date and attribution. Already-cached areas are skipped (safe to rerun).
Road data (c) OpenStreetMap contributors, ODbL 1.0.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.app.routing import graph_store  # noqa: E402


def main() -> int:
    areas = json.loads((REPO / "configs" / "study_areas.json").read_text())["areas"]
    index = {c["id"]: c for c in json.loads((REPO / "data" / "processed" / "index.json").read_text())["chips"]}
    failed = 0
    for a in areas:
        if graph_store.cached(a["id"]):
            print(f"{a['id']}: cached", graph_store.meta(a["id"]))
            continue
        bounds = index[a["chip_id"]]["bounds"]
        try:
            G = graph_store.fetch(a["id"], tuple(bounds))
            print(f"{a['id']}: fetched {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"{a['id']}: FAILED {type(e).__name__}: {e}")
        time.sleep(3)  # be gentle with the public Overpass API
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
