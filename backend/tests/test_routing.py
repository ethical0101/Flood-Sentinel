"""Routing tests on a small, explicitly constructed test graph (no network access needed).

Test graph (lon/lat, ~0.001 deg spacing ~ 111 m), two ways from A to D:

    A ---- B ---- D          top road  A-B-D  (short, B-D crosses the flooded block)
    |             |
    C ----------- E          bottom road A-C-E-D (longer, dry)

A separate component F-G tests disconnected networks.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from backend.app.routing.planner import NoRouteError, RiskSettings, plan
from backend.app.routing.risk import FloodLayer, annotate_graph, check_alignment, sample_lines

PX = 0.0001  # raster pixel size in degrees
NODES = {"A": (0.000, 0.002), "B": (0.002, 0.002), "D": (0.004, 0.002), "C": (0.000, 0.0005), "E": (0.004, 0.0005),
         "F": (0.0045, 0.0045), "G": (0.0049, 0.0049)}
IDS = {k: i for i, k in enumerate(NODES)}


def make_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph(crs="EPSG:4326")
    for k, (x, y) in NODES.items():
        G.add_node(IDS[k], x=x, y=y)
    for a, b in [("A", "B"), ("B", "D"), ("A", "C"), ("C", "E"), ("E", "D"), ("F", "G")]:
        (x1, y1), (x2, y2) = NODES[a], NODES[b]
        L = float(np.hypot(x2 - x1, y2 - y1) * 111_320)
        for u, v in ((a, b), (b, a)):
            G.add_edge(IDS[u], IDS[v], length=L, travel_time=L / 10.0,
                       geometry=LineString([NODES[u], NODES[v]]), highway="residential")
    return G


def make_layer(flood_cols=(25, 35), crs="EPSG:4326") -> FloodLayer:
    """50x50 raster over [0,0.005]^2; a vertical flooded band only across the top road between B and D."""
    prob = np.zeros((50, 50), np.float32)
    rows = slice(25, 35)  # y in (0.0025, 0.0015): covers the top road (y=0.002), not the bottom (y=0)
    prob[rows, flood_cols[0]:flood_cols[1]] = 0.95
    return FloodLayer(prob, crs, (PX, 0, 0.0, 0, -PX, 0.005), "model_prediction", "test layer")


def test_invalid_crs_and_degenerate_transform_rejected():
    with pytest.raises(ValueError):
        FloodLayer(np.zeros((4, 4), np.float32), "EPSG:999999", (1, 0, 0, 0, -1, 0), "x", "x")
    with pytest.raises(ValueError):
        FloodLayer(np.zeros((4, 4), np.float32), "EPSG:4326", (0, 0, 0, 0, 0, 0), "x", "x")


def test_alignment_report():
    layer = make_layer()
    assert check_alignment(layer, (0, 0, 0.005, 0.005))["graph_fraction_covered"] == pytest.approx(1.0)
    assert not check_alignment(layer, (1, 1, 1.1, 1.1))["ok"]


def test_sampling_outside_raster_is_unknown_not_dry():
    layer = make_layer()
    inside, outside = sample_lines(layer, [LineString([(0.001, 0.001), (0.002, 0.001)]),
                                           LineString([(0.01, 0.01), (0.02, 0.01)])], [111, 1113], buffer_px=0)
    assert np.isfinite(inside).all() and np.isnan(outside).all()


def test_projected_raster_alignment():
    """The same flood band expressed in UTM must flag the same road as the lon/lat version."""
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    x0, y1 = tr.transform(0.0, 0.005)
    x1, y0 = tr.transform(0.005, 0.0)
    w, h = 50, 50
    px, py = (x1 - x0) / w, (y1 - y0) / h
    prob = np.zeros((h, w), np.float32)
    prob[25:35, 25:35] = 0.95
    utm = FloodLayer(prob, "EPSG:32631", (px, 0, x0, 0, -py, y1), "model_prediction", "utm")
    G1, G2 = make_graph(), make_graph()
    annotate_graph(G1, make_layer(), buffer_px=0)
    annotate_graph(G2, utm, buffer_px=0)
    bd = (IDS["B"], IDS["D"], 0)
    assert G1.edges[bd]["flooded_len_m"] > 50
    assert G2.edges[bd]["flooded_len_m"] == pytest.approx(G1.edges[bd]["flooded_len_m"], rel=0.1)


def test_edge_risk_assignment():
    G = make_graph()
    stats = annotate_graph(G, make_layer(), buffer_px=0)
    bd = G.edges[IDS["B"], IDS["D"], 0]
    ce = G.edges[IDS["C"], IDS["E"], 0]
    assert bd["risk_class"] in ("partial", "flooded") and 80 < bd["flooded_len_m"] < 140
    assert ce["risk_class"] == "clear" and ce["flooded_len_m"] == 0
    assert stats["risk_class_counts"]["unknown"] == 0
    # buffering can only add flooded length
    G2 = make_graph()
    annotate_graph(G2, make_layer(), buffer_px=2)
    assert G2.edges[IDS["B"], IDS["D"], 0]["flooded_len_m"] >= bd["flooded_len_m"]


def test_risk_aware_route_avoids_flood_and_baseline_does_not():
    G = make_graph()
    annotate_graph(G, make_layer(), buffer_px=0)
    A, D = NODES["A"], NODES["D"]
    r = plan(G, (A[1], A[0]), (D[1], D[0]), RiskSettings(flood_penalty=10, alternatives=1))
    assert r["baseline"]["flooded_length_m"] > 0
    assert r["risk_aware"]["flooded_length_m"] == 0
    assert r["risk_aware"]["distance_m"] > r["baseline"]["distance_m"]
    assert not r["same_as_baseline"]
    assert r["baseline"]["travel_time_s"] == pytest.approx(r["baseline"]["distance_m"] / 10, rel=1e-2)
    assert any("avoided" in e for e in r["explanation"])


def test_zero_penalty_reproduces_shortest_route():
    G = make_graph()
    annotate_graph(G, make_layer(), buffer_px=0)
    A, D = NODES["A"], NODES["D"]
    r = plan(G, (A[1], A[0]), (D[1], D[0]), RiskSettings(flood_penalty=0, unknown_penalty=0, prob_penalty=0))
    assert r["same_as_baseline"]


def test_closure_rule_and_fallback():
    G = make_graph()
    annotate_graph(G, make_layer(), buffer_px=0)
    A, D = NODES["A"], NODES["D"]
    r = plan(G, (A[1], A[0]), (D[1], D[0]), RiskSettings(close_flooded=True, closure_min_flooded_m=10))
    assert r["closed_edges"] >= 2 and r["risk_aware"]["flooded_length_m"] == 0 and not r["closure_fallback"]
    # flood both roads: no route avoids closures -> explicit fallback with a warning
    layer = make_layer()
    layer.prob[:, 25:35] = 0.95
    G2 = make_graph()
    annotate_graph(G2, layer, buffer_px=0)
    r2 = plan(G2, (A[1], A[0]), (D[1], D[0]), RiskSettings(close_flooded=True, closure_min_flooded_m=10))
    assert r2["closure_fallback"] and r2["warnings"]


def test_disconnected_and_invalid_requests():
    G = make_graph()
    annotate_graph(G, make_layer(), buffer_px=0)
    A, F = NODES["A"], NODES["F"]
    with pytest.raises(NoRouteError) as e:
        plan(G, (A[1], A[0]), (F[1], F[0]))
    assert e.value.code == "disconnected"
    with pytest.raises(NoRouteError) as e:
        plan(G, (A[1], A[0]), (A[1] + 0.00001, A[0]))
    assert e.value.code == "same_node"
    with pytest.raises(NoRouteError) as e:
        plan(G, (A[1], A[0]), (1.0, 1.0))
    assert e.value.code == "too_far_from_road"
    with pytest.raises(ValueError):
        plan(G, (A[1], A[0]), (NODES["D"][1], NODES["D"][0]), RiskSettings(flood_penalty=-1))
