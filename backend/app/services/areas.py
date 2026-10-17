"""Study areas for routing.

A study area is the footprint of a georeferenced flood raster. Two kinds exist:
  * curated areas (configs/study_areas.json): Sen1Floods11 *held-out* chips (test / Bolivia splits,
    never used for training or model selection) where OSM has a drivable road network;
  * "r_<result id>" areas created from any georeferenced upload that was segmented by a model.

Flood layer sources
  model         -> the trained model's prediction on the area's Sentinel imagery (real inference)
  ground_truth  -> the Sen1Floods11 hand label (reference only; clearly marked, not a prediction)
"""
from __future__ import annotations

import io
import json
import threading
from functools import lru_cache

import numpy as np
from PIL import Image

from backend.app.config import settings
from backend.app.routing import graph_store
from backend.app.routing.risk import FloodLayer, annotate_graph, check_alignment
from backend.app.services.figures import flood_rgba
from backend.app.services.models import registry
from backend.app.services.segmentation import get_result, results_dir

_lock = threading.Lock()
_layers: dict[tuple, FloodLayer] = {}
_graphs: dict[tuple, tuple] = {}


class AreaError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


@lru_cache(maxsize=1)
def _index() -> dict[str, dict]:
    p = settings.processed / "index.json"
    return {c["id"]: c for c in json.loads(p.read_text())["chips"]} if p.exists() else {}


def curated() -> list[dict]:
    if not settings.study_areas_file.exists():
        return []
    return json.loads(settings.study_areas_file.read_text())["areas"]


def get_area(area_id: str) -> dict:
    for a in curated():
        if a["id"] == area_id:
            chip = _index().get(a["chip_id"])
            if chip is None:
                raise AreaError(503, "Processed dataset not found; run training/prepare_data.py")
            return {**a, "kind": "curated", "bounds": chip["bounds"], "crs": chip["crs"],
                    "transform": chip["transform"], "split": chip["split"],
                    "water_fraction_label": chip["water_fraction"], "sources": ["model", "ground_truth"],
                    "road_graph_cached": graph_store.cached(area_id), "road_meta": graph_store.meta(area_id)}
    if area_id.startswith("r_"):
        r = get_result(area_id[2:])
        if r is None:
            raise AreaError(404, "Unknown result id")
        inp = r["input"]
        if not inp["georeferenced"]:
            raise AreaError(422, "This prediction has no georeferencing, so it cannot be aligned with roads. "
                                 "Upload a GeoTIFF with a CRS and geotransform.")
        layer = _result_layer(area_id[2:])
        l, b, rr, t = layer.bounds_wgs84()
        if max(rr - l, t - b) > settings.max_area_deg:
            raise AreaError(422, f"The raster footprint is larger than {settings.max_area_deg} degrees; "
                                 "crop it to a smaller area for routing.")
        return {"id": area_id, "kind": "upload", "name": f"Uploaded scene {area_id[2:10]}",
                "description": "Model prediction on an uploaded GeoTIFF", "bounds": [l, b, rr, t],
                "crs": inp["crs"], "sources": ["model"], "model": r["model"]["name"],
                "road_graph_cached": graph_store.cached(area_id), "road_meta": graph_store.meta(area_id)}
    raise AreaError(404, f"Unknown study area {area_id!r}")


def _result_layer(rid: str) -> FloodLayer:
    r = get_result(rid)
    prob = np.load(results_dir() / rid / "probability.npy").astype(np.float32)
    return FloodLayer(prob, r["input"]["crs"], tuple(r["input"]["transform"]),
                      "uploaded_prediction", f"{r['model']['name']} prediction on uploaded GeoTIFF")


def flood_layer(area_id: str, source: str, model: str | None = None) -> FloodLayer:
    area = get_area(area_id)
    if source not in area["sources"]:
        raise AreaError(422, f"source must be one of {area['sources']}")
    if area["kind"] == "upload":
        return _result_layer(area_id[2:])
    from training.dataset import load_chip

    chip_id = area["chip_id"]
    if source == "ground_truth":
        key = (area_id, "gt")
        if key not in _layers:
            label = load_chip(chip_id, settings.processed)["label"].astype(np.float32)
            label[label < 0] = np.nan
            _layers[key] = FloodLayer(label, area["crs"], tuple(area["transform"]), "ground_truth_label",
                                      "Sen1Floods11 hand label (reference, not a model prediction)")
        return _layers[key]
    from training.infer import SceneInput

    name = registry.choose(["s1s2", "s2", "s1", "rgb"], model)
    key = (area_id, "model", name)
    if key not in _layers:
        chip = load_chip(chip_id, settings.processed)
        scene = SceneInput("s1s2", s1=chip["s1"].astype(np.float32), s2=chip["s2"].astype(np.float32),
                           crs=area["crs"], transform=tuple(area["transform"]))
        from backend.app.services.segmentation import _infer_lock

        pred = registry.get(name)
        with _infer_lock:
            prob, _ = pred.run(scene)
        # pixels without valid imagery (no SAR and no optical data) are unknown, never "dry"
        nodata = ~np.isfinite(chip["s1"].astype(np.float32)).any(0) & (chip["s2"].astype(np.float32).sum(0) <= 0)
        prob[nodata] = np.nan
        _layers[key] = FloodLayer(prob.astype(np.float32), area["crs"], tuple(area["transform"]),
                                  "model_prediction", f"{name} prediction on Sentinel imagery of {chip_id} "
                                                      f"({area['split']} split, unseen during training)")
    return _layers[key]


def annotated_graph(area_id: str, source: str, model: str | None, threshold: float, buffer_px: int):
    area = get_area(area_id)
    layer = flood_layer(area_id, source, model)
    key = (area_id, source, model or "", round(threshold, 3), buffer_px, layer.description)
    with _lock:
        if key in _graphs:
            return _graphs[key]
    try:
        G = graph_store.load(area_id, tuple(area["bounds"]), settings.allow_osm_download)
    except PermissionError as e:
        raise AreaError(503, str(e)) from None
    except Exception as e:  # noqa: BLE001 - network / Overpass failures
        raise AreaError(502, f"Could not obtain the OpenStreetMap road network ({type(e).__name__}). "
                             "Check the internet connection or try again later.") from None
    alignment = check_alignment(layer, tuple(area["bounds"]))
    if not alignment["ok"]:
        raise AreaError(422, "The flood raster does not overlap the road network bbox.")
    stats = annotate_graph(G, layer, threshold=threshold, buffer_px=buffer_px)
    with _lock:
        _graphs[key] = (G, stats, alignment, layer)
        if len(_graphs) > 24:
            _graphs.pop(next(iter(_graphs)))
    return G, stats, alignment, layer


def overlay_png(area_id: str, source: str, model: str | None, threshold: float) -> bytes:
    layer = flood_layer(area_id, source, model)
    b = io.BytesIO()
    Image.fromarray(flood_rgba(layer.prob, threshold)).save(b, format="PNG")
    return b.getvalue()


def imagery_png(area_id: str) -> bytes:
    area = get_area(area_id)
    if area["kind"] == "upload":
        p = results_dir() / area_id[2:] / "input.png"
        return p.read_bytes()
    from training.dataset import load_chip

    b = io.BytesIO()
    Image.fromarray(np.transpose(load_chip(area["chip_id"], settings.processed)["rgb"], (1, 2, 0))).save(b, "PNG")
    return b.getvalue()
