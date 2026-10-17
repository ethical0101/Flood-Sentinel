"""Choose a demonstrative default start/end pair for each curated study area.

For each area: 250 random node pairs (seed 0) from the largest strongly-connected component,
inside the raster footprint, at least 1.5 km apart; keep the pair whose risk-aware route avoids
the most predicted-flooded road while being at most 40% longer than the shortest route.
Selection uses the default model's prediction. Writes configs/study_areas.json in place.

    .venv\\Scripts\\python scripts\\select_demo_routes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.app.routing.planner import NoRouteError, RiskSettings, plan  # noqa: E402
from backend.app.services import areas  # noqa: E402


def main() -> None:
    path = REPO / "configs" / "study_areas.json"
    cfg = json.loads(path.read_text())
    rng = np.random.default_rng(0)
    for a in cfg["areas"]:
        G, *_ = areas.annotated_graph(a["id"], "model", None, 0.5, 1)
        l, b, r, t = areas.get_area(a["id"])["bounds"]
        comp = max(nx.strongly_connected_components(G), key=len)
        nodes = [n for n in comp if l < G.nodes[n]["x"] < r and b < G.nodes[n]["y"] < t]
        best = None
        for _ in range(250):
            u, v = rng.choice(nodes, 2, replace=False)
            p1, p2 = (G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])
            d = np.hypot((p1[0] - p2[0]) * 111e3, (p1[1] - p2[1]) * 111e3 * np.cos(np.radians(p1[0])))
            if d < 1500:
                continue
            try:
                res = plan(G, p1, p2, RiskSettings(alternatives=0))
            except NoRouteError:
                continue
            bd, rd = res["baseline"]["distance_m"], res["risk_aware"]["distance_m"]
            avoided = res["baseline"]["flooded_length_m"] - res["risk_aware"]["flooded_length_m"]
            score = avoided if rd <= 1.4 * bd else -1
            if best is None or score > best[0]:
                best = (score, p1, p2)
        _, p1, p2 = best
        a["default_start"] = [round(p1[0], 5), round(p1[1], 5)]
        a["default_end"] = [round(p2[0], 5), round(p2[1], 5)]
        print(a["id"], "avoided m:", round(best[0]))
    path.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
