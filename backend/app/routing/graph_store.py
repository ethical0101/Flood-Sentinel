"""OpenStreetMap road graphs: fetch once with OSMnx (Overpass API), cache as GraphML.

Data: (c) OpenStreetMap contributors, available under the Open Database License (ODbL 1.0).
Requests go to the public Overpass API only when a graph is not cached yet; afterwards
routing works offline. One bbox per request, no tile scraping.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ATTRIBUTION = "Road network (c) OpenStreetMap contributors, ODbL 1.0"
FALLBACK_KMH = 40.0
_lock = threading.Lock()
_mem: dict[str, object] = {}


def cache_dir() -> Path:
    p = Path(os.environ.get("FLOOD_OSM_CACHE", REPO / "data" / "osm_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe(area_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", area_id):
        raise ValueError("invalid area id")
    return area_id


def cached(area_id: str) -> bool:
    return (cache_dir() / f"{_safe(area_id)}.graphml").exists()


def meta(area_id: str) -> dict | None:
    p = cache_dir() / f"{_safe(area_id)}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load(area_id: str, bbox_wgs84: tuple[float, float, float, float] | None = None, allow_download: bool = True):
    """Return a fresh copy of the OSMnx MultiDiGraph for an area (callers annotate it in place)."""
    import osmnx as ox

    area_id = _safe(area_id)
    with _lock:
        if area_id not in _mem:
            path = cache_dir() / f"{area_id}.graphml"
            if path.exists():
                _mem[area_id] = ox.load_graphml(path)
            else:
                if bbox_wgs84 is None:
                    raise FileNotFoundError(f"road graph for {area_id!r} is not cached")
                if not allow_download:
                    raise PermissionError("road graph not cached and OSM download is disabled "
                                          "(FLOOD_ALLOW_OSM_DOWNLOAD=false)")
                _mem[area_id] = fetch(area_id, bbox_wgs84)
        return _mem[area_id].copy()


def fetch(area_id: str, bbox_wgs84: tuple[float, float, float, float]):
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir() / "http")
    ox.settings.requests_timeout = 120
    left, bottom, right, top = bbox_wgs84
    G = None
    for attempt in range(4):  # Overpass asks clients to back off exponentially when at capacity
        try:
            G = ox.graph_from_bbox((left, bottom, right, top), network_type="drive", simplify=True,
                                   truncate_by_edge=True, retain_all=True)
            break
        except ox._errors.InsufficientResponseError:
            import networkx as nx

            G = nx.MultiDiGraph(crs="EPSG:4326")  # genuinely no drivable roads mapped here
            break
        except ox._errors.ResponseStatusCodeError:
            if attempt == 3:
                raise
            time.sleep(15 * 2 ** attempt)
    if G.number_of_edges():
        # km/h from OSM maxspeed where tagged, else the mean of that highway type in this graph,
        # else FALLBACK_KMH when no edge in the area carries a maxspeed tag at all
        G = ox.add_edge_speeds(G, fallback=FALLBACK_KMH)
        G = ox.add_edge_travel_times(G)  # free-flow seconds; flooding slowdown is NOT modelled
    ox.save_graphml(G, cache_dir() / f"{area_id}.graphml")
    (cache_dir() / f"{area_id}.json").write_text(json.dumps({
        "area_id": area_id, "bbox_wgs84": list(bbox_wgs84), "network_type": "drive",
        "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "OpenStreetMap via Overpass API (OSMnx " + ox.__version__ + ")",
        "attribution": ATTRIBUTION, "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}, indent=1))
    return G


def edges_geojson(G, limit: int = 20000) -> dict:
    """Road edges with risk attributes as GeoJSON (for map display)."""
    feats = []
    seen = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        if (v, u) in seen:  # draw two-way streets once
            continue
        seen.add((u, v))
        geom = d.get("geometry")
        coords = list(geom.coords) if geom is not None else [(G.nodes[u]["x"], G.nodes[u]["y"]),
                                                              (G.nodes[v]["x"], G.nodes[v]["y"])]
        feats.append({"type": "Feature",
                      "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in coords]},
                      "properties": {"risk_class": d.get("risk_class"),
                                     "flooded_len_m": round(d.get("flooded_len_m", 0.0), 1),
                                     "length_m": round(float(d.get("length", 0.0)), 1),
                                     "flood_prob_max": None if d.get("flood_prob_max") is None or
                                     d.get("flood_prob_max") != d.get("flood_prob_max") else round(d["flood_prob_max"], 3),
                                     "highway": d.get("highway") if not isinstance(d.get("highway"), list) else d["highway"][0],
                                     "name": d.get("name") if not isinstance(d.get("name"), list) else ", ".join(d["name"])}})
        if len(feats) >= limit:
            break
    return {"type": "FeatureCollection", "features": feats}
