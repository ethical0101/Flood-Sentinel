"""Risk-aware route planning on a flood-annotated road graph (NetworkX Dijkstra).

Edge cost used by the risk-aware route (all terms in metres-equivalent):

    cost = length
         + flood_penalty   * flooded_len_m                 (predicted-flooded metres)
         + unknown_penalty * length * (1 - coverage)       (metres without flood information)
         + prob_penalty    * length * flood_prob_mean      (soft term: elevated but sub-threshold risk)

With the default ``flood_penalty = 10`` one predicted-flooded metre costs as much as eleven dry
metres, i.e. the planner accepts a detour of up to ~10x the flooded length it avoids.

Optional closure rule (``close_flooded = True``): edges whose predicted-flooded length is at
least ``closure_min_flooded_m`` are removed. This is a *user-selected conservative assumption*
(water depth, current and road elevation are unknown) - not a verified road closure.

The baseline route minimises length only, ignoring flood information.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np


@dataclass
class RiskSettings:
    flood_penalty: float = 10.0
    unknown_penalty: float = 0.5
    prob_penalty: float = 1.0
    close_flooded: bool = False
    closure_min_flooded_m: float = 20.0
    alternatives: int = 2

    def validate(self) -> None:
        for k in ("flood_penalty", "unknown_penalty", "prob_penalty", "closure_min_flooded_m"):
            v = getattr(self, k)
            if not (0 <= v <= 1000) or math.isnan(v):
                raise ValueError(f"{k} must be within [0, 1000]")
        if not 0 <= self.alternatives <= 3:
            raise ValueError("alternatives must be within [0, 3]")


class NoRouteError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def edge_cost(d: dict, s: RiskSettings) -> float:
    L = float(d.get("length", 0.0))
    cov = d.get("coverage", 0.0)
    pm = d.get("flood_prob_mean", 0.0)
    pm = 0.0 if pm is None or math.isnan(pm) else pm
    return (L + s.flood_penalty * d.get("flooded_len_m", 0.0) + s.unknown_penalty * L * (1 - cov)
            + s.prob_penalty * L * pm)


def is_closed(d: dict, s: RiskSettings) -> bool:
    return s.close_flooded and d.get("flooded_len_m", 0.0) >= s.closure_min_flooded_m


def simple_digraph(G, s: RiskSettings, use_risk: bool) -> nx.DiGraph:
    """Collapse parallel edges, keeping the cheapest; attach 'w' (routing weight) and 'key'."""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, k, d in G.edges(keys=True, data=True):
        if use_risk and is_closed(d, s):
            continue
        w = edge_cost(d, s) if use_risk else float(d.get("length", 0.0))
        if not H.has_edge(u, v) or w < H[u][v]["w"]:
            H.add_edge(u, v, w=w, key=k)
    return H


def nearest_node(G, lat: float, lon: float) -> tuple[int, float]:
    ids = np.fromiter(G.nodes, dtype=np.int64, count=G.number_of_nodes())
    ys = np.array([G.nodes[n]["y"] for n in ids])
    xs = np.array([G.nodes[n]["x"] for n in ids])
    p1, p2 = np.radians(lat), np.radians(ys)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(xs - lon) / 2) ** 2
    dist = 2 * 6371008.8 * np.arcsin(np.sqrt(a))
    i = int(dist.argmin())
    return int(ids[i]), float(dist[i])


def describe(G, H: nx.DiGraph, path: list[int]) -> dict:
    coords, segments = [], []
    dist = tt = flooded = unknown = 0.0
    max_p = 0.0
    n_flooded = 0
    by_class: dict[str, float] = {}
    affected: dict[str, float] = {}
    for u, v in zip(path[:-1], path[1:]):
        d = G.edges[u, v, H[u][v]["key"]]
        L = float(d.get("length", 0.0))
        geom = d.get("geometry")
        pts = [(y, x) for x, y in geom.coords] if geom is not None else \
            [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
        if geom is not None and (abs(pts[0][0] - G.nodes[u]["y"]) + abs(pts[0][1] - G.nodes[u]["x"])) > \
                (abs(pts[-1][0] - G.nodes[u]["y"]) + abs(pts[-1][1] - G.nodes[u]["x"])):
            pts = pts[::-1]  # geometry stored in the opposite direction
        coords.extend(pts if not coords else pts[1:])
        dist += L
        tt += float(d.get("travel_time", np.nan)) if d.get("travel_time") is not None else np.nan
        flooded += d.get("flooded_len_m", 0.0)
        unknown += L * (1 - d.get("coverage", 0.0))
        pm = d.get("flood_prob_max", 0.0)
        if pm is not None and not math.isnan(pm):
            max_p = max(max_p, pm)
        n_flooded += d.get("risk_class") in ("flooded", "partial")
        rc = d.get("risk_class") or "unknown"
        by_class[rc] = by_class.get(rc, 0.0) + L
        if d.get("flooded_len_m", 0.0) > 0:
            key = _name(d) or f"unnamed {_first(d.get('highway')) or 'road'}"
            affected[key] = affected.get(key, 0.0) + d.get("flooded_len_m", 0.0)
        segments.append({"coords": [[round(a, 6), round(b, 6)] for a, b in pts], "risk_class": d.get("risk_class"),
                         "flooded_len_m": round(d.get("flooded_len_m", 0.0), 1), "length_m": round(L, 1),
                         "name": _name(d), "highway": _first(d.get("highway"))})
    share = flooded / dist if dist else 0.0
    unknown_share = unknown / dist if dist else 0.0
    level = ("no predicted flooding" if flooded == 0 and unknown_share < 0.1 else
             "minor predicted flooding" if share < 0.05 else "substantial predicted flooding")
    return {"risk_level": level,
            "risk_level_rule": "none: 0 m flooded and <10% unknown; minor: <5% of length flooded; substantial: otherwise",
            "length_by_risk_class_m": {k: round(v, 1) for k, v in sorted(by_class.items())},
            "flood_affected_roads": [{"road": k, "flooded_m": round(v, 1)}
                                     for k, v in sorted(affected.items(), key=lambda kv: -kv[1])[:12]],
            "distance_m": round(dist, 1),
            "travel_time_s": None if math.isnan(tt) else round(tt, 1),
            "flooded_length_m": round(flooded, 1),
            "flooded_share": round(flooded / dist, 4) if dist else 0.0,
            "unknown_length_m": round(unknown, 1),
            "max_flood_prob": round(max_p, 3),
            "n_flood_affected_segments": int(n_flooded),
            "n_segments": len(segments),
            "coordinates": [[round(a, 6), round(b, 6)] for a, b in coords],
            "segments": segments, "nodes": [int(n) for n in path]}


def _first(v):
    return v[0] if isinstance(v, list) else v


def _name(d: dict) -> str | None:
    n = d.get("name")
    return ", ".join(n) if isinstance(n, list) else n


def _overlap(a: list[int], b: list[int]) -> float:
    ea, eb = set(zip(a[:-1], a[1:])), set(zip(b[:-1], b[1:]))
    return len(ea & eb) / max(1, min(len(ea), len(eb)))


def plan(G, start: tuple[float, float], end: tuple[float, float], s: RiskSettings | None = None,
         max_snap_m: float = 1000.0) -> dict:
    """start/end are (lat, lon). Returns baseline, risk-aware and alternative routes + explanation."""
    s = s or RiskSettings()
    s.validate()
    if G.number_of_nodes() < 2:
        raise NoRouteError("empty_graph", "The road network for this area has no routable roads.")
    o, od = nearest_node(G, *start)
    t, td = nearest_node(G, *end)
    warnings = []
    for label, dd in (("start", od), ("destination", td)):
        if dd > max_snap_m:
            raise NoRouteError("too_far_from_road", f"The {label} is {dd:.0f} m from the nearest road in this "
                                                    f"study area (limit {max_snap_m:.0f} m).")
        if dd > 150:
            warnings.append(f"The {label} was snapped to a road node {dd:.0f} m away.")
    if o == t:
        raise NoRouteError("same_node", "Start and destination snap to the same road node; pick points further apart.")

    base_g = simple_digraph(G, s, use_risk=False)
    try:
        base_path = nx.dijkstra_path(base_g, o, t, weight="w")
    except nx.NetworkXNoPath:
        raise NoRouteError("disconnected", "Start and destination lie in disconnected parts of the road network "
                                           "inside this study area (the area is clipped to the flood raster, so a "
                                           "connection may exist outside it).") from None
    baseline = describe(G, base_g, base_path)

    risk_g = simple_digraph(G, s, use_risk=True)
    closed = sum(1 for *_, d in G.edges(keys=True, data=True) if is_closed(d, s))
    fallback = False
    try:
        risk_path = nx.dijkstra_path(risk_g, o, t, weight="w")
    except nx.NetworkXNoPath:
        # every connection crosses a "closed" edge: fall back to penalties without closures
        fallback = True
        warnings.append("No route avoids every segment marked impassable; showing the least-risk route with "
                        "closures relaxed to penalties. Treat this as NOT safe.")
        risk_g = simple_digraph(G, RiskSettings(**{**asdict(s), "close_flooded": False}), use_risk=True)
        risk_path = nx.dijkstra_path(risk_g, o, t, weight="w")
    risk = describe(G, risk_g, risk_path)
    risk["cost"] = round(nx.path_weight(risk_g, risk_path, "w"), 1)

    alts, found = [], [base_path, risk_path]
    pen_g = risk_g.copy()
    for _ in range(s.alternatives * 3):
        if len(alts) >= s.alternatives:
            break
        for u, v in zip(found[-1][:-1], found[-1][1:]):
            pen_g[u][v]["w"] *= 1.6  # penalty method: discourage re-using the last route's edges
        try:
            p = nx.dijkstra_path(pen_g, o, t, weight="w")
        except nx.NetworkXNoPath:
            break
        found.append(p)
        if all(_overlap(p, q) < 0.8 for q in [base_path, risk_path] + [a["nodes"] for a in alts]):
            a = describe(G, risk_g, p)
            a["cost"] = round(nx.path_weight(risk_g, p, "w"), 1)
            alts.append(a)

    if fallback:
        status = "no_route_avoids_closures"
    elif risk["flooded_length_m"] > 0:
        status = "no_flood_free_route"
    else:
        status = "flood_free_route_found"
    return {"status": status,
            "status_note": {
                "flood_free_route_found": "The risk-aware route crosses no road predicted as flooded. This is NOT a "
                                          "guarantee of safety.",
                "no_flood_free_route": "Every route between these points crosses road predicted as flooded; the "
                                       "risk-aware route minimises it but is not flood-free.",
                "no_route_avoids_closures": "No route avoids all segments marked impassable; the route shown relaxes "
                                            "closures to penalties and must be treated as unsafe.",
            }[status],
            "start_snap_m": round(od, 1), "destination_snap_m": round(td, 1),
            "baseline": baseline, "risk_aware": risk, "alternatives": alts,
            "same_as_baseline": base_path == risk_path, "closure_fallback": fallback,
            "closed_edges": closed, "settings": asdict(s), "warnings": warnings,
            "explanation": explain(baseline, risk, base_path == risk_path, s, closed, fallback)}


def explain(base: dict, risk: dict, same: bool, s: RiskSettings, closed: int, fallback: bool) -> list[str]:
    out = []
    if same:
        if base["flooded_length_m"] > 0:
            out.append(f"The shortest route is also the least-risk route, but it still crosses "
                       f"{base['flooded_length_m']:.0f} m of road predicted as flooded: no lower-risk "
                       f"alternative exists in this network under the current penalties.")
        else:
            out.append("The shortest route crosses no road predicted as flooded, so it is also the risk-aware choice.")
    else:
        extra = risk["distance_m"] - base["distance_m"]
        avoided = base["flooded_length_m"] - risk["flooded_length_m"]
        pct = 100 * extra / base["distance_m"] if base["distance_m"] else 0
        out.append(f"The risk-aware route is {extra:+.0f} m ({pct:+.1f}%) compared with the shortest route and "
                   f"reduces predicted-flooded road from {base['flooded_length_m']:.0f} m to "
                   f"{risk['flooded_length_m']:.0f} m ({avoided:.0f} m avoided).")
        out.append(f"Each predicted-flooded metre is weighted {1 + s.flood_penalty:.0f}x a dry metre, so a detour "
                   f"is accepted whenever it is shorter than this trade-off.")
    if s.close_flooded:
        out.append(f"{closed} road segment(s) with >= {s.closure_min_flooded_m:.0f} m of predicted flooding were "
                   f"treated as impassable (a user-selected assumption, not a verified closure)."
                   + (" No route avoided all of them." if fallback else ""))
    if risk["unknown_length_m"] > 0:
        out.append(f"{risk['unknown_length_m']:.0f} m of the chosen route has no flood information "
                   f"(outside the raster or no-data pixels).")
    return out
