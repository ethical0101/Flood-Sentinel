"""Evaluate risk-aware routing against the hand labels.

For every curated study area (held-out test chips) and N random origin-destination pairs:
  * plan on the MODEL-predicted flood layer (what the system would do in practice);
  * measure how much of each route is flooded according to the HAND LABEL (ground truth);
  * compare with the shortest route and with an "oracle" that plans on the hand label itself.
Also reports edge-level agreement of the flooded/not-flooded classification (model vs label).

    .venv\\Scripts\\python scripts\\evaluate_routing.py [--pairs 50]
Writes artifacts/routing_evaluation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.app.routing.planner import NoRouteError, RiskSettings, plan  # noqa: E402
from backend.app.services import areas  # noqa: E402
from training.utils import artifacts_dir, write_json  # noqa: E402


def gt_flooded(Gg, nodes: list[int]) -> float:
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        d = min(Gg.get_edge_data(u, v).values(), key=lambda e: e.get("length", 0))
        total += d.get("flooded_len_m", 0.0)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(123)
    settings = RiskSettings(alternatives=0)
    out = {"protocol": {
        "pairs_per_area": a.pairs, "seed": 123, "min_straight_line_m": 1000,
        "flood_threshold": 0.5, "buffer_px": 1, "settings": settings.__dict__,
        "note": "Routes planned on the model prediction; flooded length measured on the Sen1Floods11 hand label "
                "(same 5 m sampling, same buffer). Oracle = routes planned directly on the hand label."},
        "areas": {}}
    agg = {k: [] for k in ("base_gt", "risk_gt", "oracle_gt", "base_len", "risk_len", "oracle_len")}
    for area in areas.curated():
        aid = area["id"]
        Gm, sm, *_ = areas.annotated_graph(aid, "model", a.model, 0.5, 1)
        Gg, sg, *_ = areas.annotated_graph(aid, "ground_truth", None, 0.5, 1)
        # edge-level agreement on "has predicted flooding"
        tp = fp = fn = tn = 0
        for u, v, k, d in Gm.edges(keys=True, data=True):
            p, t = d["flooded_len_m"] > 0, Gg.edges[u, v, k]["flooded_len_m"] > 0
            tp += p and t; fp += p and not t; fn += (not p) and t; tn += (not p) and (not t)
        comp = max(nx.strongly_connected_components(Gm), key=len)
        l, b, r, t_ = areas.get_area(aid)["bounds"]
        nodes = [n for n in comp if l < Gm.nodes[n]["x"] < r and b < Gm.nodes[n]["y"] < t_]
        rows = []
        tries = 0
        while len(rows) < a.pairs and tries < a.pairs * 20 and len(nodes) > 1:
            tries += 1
            u, v = rng.choice(nodes, 2, replace=False)
            p1, p2 = (Gm.nodes[u]["y"], Gm.nodes[u]["x"]), (Gm.nodes[v]["y"], Gm.nodes[v]["x"])
            dist = np.hypot((p1[0] - p2[0]) * 111e3, (p1[1] - p2[1]) * 111e3 * np.cos(np.radians(p1[0])))
            if dist < 1000:
                continue
            try:
                rm = plan(Gm, p1, p2, settings)
                ro = plan(Gg, p1, p2, settings)
            except NoRouteError:
                continue
            row = {"base_len": rm["baseline"]["distance_m"], "risk_len": rm["risk_aware"]["distance_m"],
                   "oracle_len": ro["risk_aware"]["distance_m"],
                   "base_gt": gt_flooded(Gg, rm["baseline"]["nodes"]),
                   "risk_gt": gt_flooded(Gg, rm["risk_aware"]["nodes"]),
                   "oracle_gt": gt_flooded(Gg, ro["risk_aware"]["nodes"])}
            rows.append(row)
            for k_, v_ in row.items():
                agg[k_].append(v_)
        mean = {k: float(np.mean([r_[k] for r_ in rows])) if rows else None for k in agg}
        affected = [r_ for r_ in rows if r_["base_gt"] > 0]
        out["areas"][aid] = {
            "edges": sm["edges"], "flooded_road_km_model": sm["flooded_road_km"],
            "flooded_road_km_label": sg["flooded_road_km"],
            "edge_agreement": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                               "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
                               "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)},
            "n_pairs": len(rows), "n_pairs_shortest_route_crosses_labelled_flood": len(affected),
            "mean_gt_flooded_m": {"shortest": mean["base_gt"], "risk_aware_model": mean["risk_gt"],
                                  "risk_aware_oracle": mean["oracle_gt"]},
            "mean_length_m": {"shortest": mean["base_len"], "risk_aware_model": mean["risk_len"],
                              "risk_aware_oracle": mean["oracle_len"]},
        }
        print(aid, json.dumps(out["areas"][aid]["mean_gt_flooded_m"]), "pairs", len(rows), flush=True)
    n = len(agg["base_gt"])
    out["overall"] = {
        "n_pairs": n,
        "mean_gt_flooded_m": {"shortest": float(np.mean(agg["base_gt"])), "risk_aware_model": float(np.mean(agg["risk_gt"])),
                              "risk_aware_oracle": float(np.mean(agg["oracle_gt"]))},
        "mean_length_m": {"shortest": float(np.mean(agg["base_len"])), "risk_aware_model": float(np.mean(agg["risk_len"])),
                          "risk_aware_oracle": float(np.mean(agg["oracle_len"]))},
        "reduction_in_gt_flooded_length_vs_shortest": 1 - float(np.sum(agg["risk_gt"]) / max(1e-9, np.sum(agg["base_gt"]))),
        "mean_detour_pct": 100 * float(np.mean(np.array(agg["risk_len"]) / np.array(agg["base_len"]) - 1)),
        "pairs_where_model_route_has_more_gt_flood_than_shortest": int(np.sum(np.array(agg["risk_gt"]) > np.array(agg["base_gt"]) + 1e-6)),
    }
    write_json(artifacts_dir() / "routing_evaluation.json", out)
    print(json.dumps(out["overall"], indent=1))


if __name__ == "__main__":
    main()
