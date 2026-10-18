"""FastAPI backend: flood segmentation + risk-aware routing.

    .venv\\Scripts\\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
Interactive API docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import io
import json
import logging
import math
import re
import time
import uuid
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend.app.config import settings
from backend.app.routing import graph_store
from backend.app.routing.planner import NoRouteError, RiskSettings, plan
from backend.app.services import areas, diagnostics, history, reports, segmentation, status, training_control
from backend.app.services.models import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("flood.api")
STARTED = time.time()
CHIP_RE = re.compile(r"^[A-Za-z\-]+_\d+$")
DISCLAIMER = ("Research prototype. Flood extents are model estimates from satellite imagery and routes are "
              "computed on OpenStreetMap data; neither is verified. Do not use for real evacuation or navigation "
              "decisions; follow official emergency services and verified road-closure information.")

def _finite(o):
    """Replace NaN/Inf (e.g. undefined precision of a constant predictor) with null for strict JSON."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(_finite(content))


app = FastAPI(title="Flood Segmentation & Safe-Route API", version=diagnostics.APP_VERSION,
              default_response_class=SafeJSONResponse,
              description="Sen1Floods11-trained water segmentation and flood-risk-aware routing on OSM roads. "
                          + DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                   expose_headers=["X-Request-ID", "Server-Timing", "Content-Disposition"],
                   allow_headers=["*"])


# ----------------------------------------------------------------------------- middleware / errors
@app.middleware("http")
async def timing(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        log.exception("unhandled error rid=%s %s %s", rid, request.method, request.url.path)
        response = JSONResponse({"error": "internal_error", "message": "Unexpected server error.",
                                 "request_id": rid}, status_code=500)
    ms = (time.perf_counter() - t0) * 1e3
    response.headers["X-Request-ID"] = rid
    response.headers["Server-Timing"] = f"app;dur={ms:.1f}"
    if request.url.path.startswith("/api") and response.status_code >= 400:
        diagnostics.record_error(request.method, request.url.path, response.status_code)
    if request.url.path.startswith("/api"):
        log.info("%s %s -> %s %.1fms rid=%s", request.method, request.url.path, response.status_code, ms, rid)
    return response


@app.exception_handler(segmentation.InputError)
async def input_error(_: Request, e: segmentation.InputError):
    return JSONResponse({"error": "invalid_input", "message": str(e)}, status_code=422)


@app.exception_handler(areas.AreaError)
async def area_error(_: Request, e: areas.AreaError):
    return JSONResponse({"error": "area_error", "message": e.message}, status_code=e.status)


@app.exception_handler(training_control.TrainingControlError)
async def training_error(_: Request, e: training_control.TrainingControlError):
    return JSONResponse({"error": "training_control", "message": e.message}, status_code=e.status)


@app.exception_handler(NoRouteError)
async def no_route(_: Request, e: NoRouteError):
    return JSONResponse({"error": e.code, "message": e.message}, status_code=422)


# ----------------------------------------------------------------------------- status
@app.get("/api/health", tags=["status"])
def health():
    return {"status": "ok", "uptime_s": round(time.time() - STARTED, 1), "version": app.version}


@app.get("/api/system", tags=["status"])
def system():
    return status.system()


@app.get("/api/model/status", tags=["status"])
def model_status():
    models = registry.available()
    import torch

    return {"ready": any(m["training_state"] == "completed" for m in models),
            "default_model": settings.default_model, "device_setting": settings.device,
            "cuda_available": torch.cuda.is_available(), "loaded": registry.loaded(), "models": models}


@app.get("/api/datasets/status", tags=["status"])
def datasets_status():
    return status.datasets()


@app.get("/api/training/status", tags=["training"])
def training_status():
    return status.training()


@app.get("/api/training/runs/{run}/log", tags=["training"])
def training_log(run: str, lines: int = Query(200, ge=1, le=2000)):
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", run):
        raise HTTPException(404, "unknown run")
    out = status.training_log(run, lines)
    if out is None:
        raise HTTPException(404, "unknown run")
    return {"run": run, "lines": out}


@app.get("/api/metrics", tags=["evaluation"])
def metrics():
    return status.metrics()


@app.get("/api/metrics/runs/{run}/images/{name}", tags=["evaluation"])
def run_image(run: str, name: str):
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", run):
        raise HTTPException(404, "not found")
    p = status.sample_image(run, name)
    if p is None:
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="image/png")


@app.get("/api/metrics/figures/{name}", tags=["evaluation"])
def figure(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}\.png", name):
        raise HTTPException(404, "not found")
    p = settings.artifacts / "figures" / name
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="image/png")


# ----------------------------------------------------------------------------- samples (held-out chips)
def _chip(chip_id: str) -> dict:
    if not CHIP_RE.fullmatch(chip_id):
        raise HTTPException(404, "unknown chip")
    c = areas._index().get(chip_id)
    if c is None or c["split"] not in ("test", "bolivia"):
        raise HTTPException(404, "unknown chip (only held-out test / Bolivia chips are offered as samples)")
    return c


@app.get("/api/samples", tags=["segmentation"])
def samples():
    """Held-out chips (test + Bolivia) that can be segmented directly or downloaded as GeoTIFF."""
    out = [{k: c[k] for k in ("id", "event", "split", "water_fraction", "valid_fraction", "bounds")}
           for c in areas._index().values() if c["split"] in ("test", "bolivia")]
    return {"samples": sorted(out, key=lambda c: (c["split"] != "test", c["id"]))}


@app.get("/api/samples/{chip_id}/preview.png", tags=["segmentation"])
def sample_preview(chip_id: str):
    from PIL import Image

    from training.dataset import load_chip

    _chip(chip_id)
    b = io.BytesIO()
    Image.fromarray(np.transpose(load_chip(chip_id, settings.processed)["rgb"], (1, 2, 0))).save(b, "PNG")
    return Response(b.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/samples/{chip_id}/download", tags=["segmentation"])
def sample_download(chip_id: str, layers: Literal["s1s2", "s1", "s2", "rgb"] = "s1s2"):
    """Original Sen1Floods11 S1/S2 GeoTIFF bands stacked into one file (georeferencing preserved)."""
    import rasterio
    from rasterio.io import MemoryFile

    _chip(chip_id)
    base = settings.raw / "sen1floods11" / "v1.1" / "data" / "flood_events" / "HandLabeled"
    with rasterio.open(base / "S1Hand" / f"{chip_id}_S1Hand.tif") as s1ds:
        s1, prof = s1ds.read().astype(np.float32), s1ds.profile
    with rasterio.open(base / "S2Hand" / f"{chip_id}_S2Hand.tif") as s2ds:
        s2 = s2ds.read().astype(np.float32)
    if layers == "rgb":
        from training.dataset import s2_to_rgb8

        arr, dtype = s2_to_rgb8(s2 / 10000.0), "uint8"
    else:
        arr = {"s1s2": np.concatenate([s1, s2]), "s1": s1, "s2": s2}[layers]
        dtype = "float32"
    prof = {"driver": "GTiff", "width": prof["width"], "height": prof["height"], "crs": prof["crs"],
            "transform": prof["transform"], "count": arr.shape[0], "dtype": dtype, "compress": "deflate"}
    with MemoryFile() as mf:
        with mf.open(**prof) as dst:
            dst.write(arr.astype(dtype))
        data = mf.read()
    return Response(data, media_type="image/tiff",
                    headers={"Content-Disposition": f'attachment; filename="{chip_id}_{layers}.tif"'})


# ----------------------------------------------------------------------------- segmentation
@app.post("/api/segmentation/predict", tags=["segmentation"])
async def predict(file: UploadFile = File(...), model: str | None = Form(None),
                  threshold: float = Form(0.5), tta: bool = Form(False)):
    limit = int(settings.max_upload_mb * 2**20)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise segmentation.InputError(f"File is larger than the {settings.max_upload_mb:.0f} MB limit.")
    scene = segmentation.read_scene(data, file.filename or "")
    return segmentation.run_prediction(scene, model or None, threshold, tta,
                                       source={"type": "upload", "filename": (file.filename or "")[:120]})


class SamplePredict(BaseModel):
    chip_id: str
    model: str | None = None
    threshold: float = Field(0.5, ge=0.05, le=0.95)
    tta: bool = False
    input: Literal["s1s2", "s1", "s2", "rgb"] = "s1s2"


@app.post("/api/segmentation/predict-sample", tags=["segmentation"])
def predict_sample(req: SamplePredict):
    from training.dataset import load_chip
    from training.infer import SceneInput

    c = _chip(req.chip_id)
    chip = load_chip(req.chip_id, settings.processed)
    s1, s2 = chip["s1"].astype(np.float32), chip["s2"].astype(np.float32)
    scene = {"s1s2": SceneInput("s1s2", s1=s1, s2=s2), "s1": SceneInput("s1", s1=s1),
             "s2": SceneInput("s2", s2=s2), "rgb": SceneInput("rgb", rgb=chip["rgb"])}[req.input]
    scene.crs, scene.transform, scene.bounds = c["crs"], tuple(c["transform"]), tuple(c["bounds"])
    return segmentation.run_prediction(scene, req.model, req.threshold, req.tta,
                                       source={"type": "sample", "chip_id": req.chip_id, "split": c["split"],
                                               "event": c["event"]},
                                       ground_truth=chip["label"])


@app.get("/api/segmentation/results/{rid}", tags=["segmentation"])
def get_result(rid: str):
    r = segmentation.get_result(rid)
    if r is None:
        raise HTTPException(404, "result not found")
    return r


@app.get("/api/segmentation/results/{rid}/files/{name}", tags=["segmentation"])
def get_result_file(rid: str, name: str, download: bool = False):
    p = segmentation.result_file(rid, name)
    if p is None:
        raise HTTPException(404, "file not found")
    headers = {"Content-Disposition": f'attachment; filename="flood_{rid[:8]}_{name}"'} if download else {}
    return FileResponse(p, media_type=segmentation.RESULT_FILES[name], headers=headers)


@app.get("/api/segmentation/recent", tags=["segmentation"])
def recent():
    return {"recent": segmentation.recent()}


# ----------------------------------------------------------------------------- routing
@app.get("/api/routes/areas", tags=["routing"])
def list_areas():
    out = []
    for a in areas.curated():
        try:
            out.append(areas.get_area(a["id"]))
        except areas.AreaError:
            continue
    return {"areas": out, "attribution": graph_store.ATTRIBUTION, "disclaimer": DISCLAIMER}


@app.get("/api/routes/areas/{area_id}", tags=["routing"])
def area_detail(area_id: str, source: Literal["model", "ground_truth"] = "model", model: str | None = None,
                threshold: float = Query(0.5, ge=0.05, le=0.95), buffer_px: int = Query(1, ge=0, le=3)):
    a = areas.get_area(area_id)
    G, stats, alignment, layer = areas.annotated_graph(area_id, source, model, threshold, buffer_px)
    return {"area": a, "layer": {"source": layer.source, "description": layer.description,
                                 "bounds_wgs84": layer.bounds_wgs84()},
            "risk_stats": stats, "alignment": alignment, "roads": graph_store.edges_geojson(G),
            "attribution": graph_store.ATTRIBUTION, "disclaimer": DISCLAIMER}


@app.get("/api/routes/areas/{area_id}/overlay.png", tags=["routing"])
def area_overlay(area_id: str, source: Literal["model", "ground_truth"] = "model", model: str | None = None,
                 threshold: float = Query(0.5, ge=0.05, le=0.95)):
    return Response(areas.overlay_png(area_id, source, model, threshold), media_type="image/png")


@app.get("/api/routes/areas/{area_id}/imagery.png", tags=["routing"])
def area_imagery(area_id: str):
    return Response(areas.imagery_png(area_id), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


class LatLon(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class PlanRequest(BaseModel):
    area_id: str = Field(max_length=80)
    source: Literal["model", "ground_truth"] = "model"
    model: str | None = None
    start: LatLon
    end: LatLon
    threshold: float = Field(0.5, ge=0.05, le=0.95)
    buffer_px: int = Field(1, ge=0, le=3)
    flood_penalty: float = Field(10.0, ge=0, le=1000)
    unknown_penalty: float = Field(0.5, ge=0, le=1000)
    close_flooded: bool = False
    closure_min_flooded_m: float = Field(20.0, ge=0, le=1000)
    alternatives: int = Field(2, ge=0, le=3)

    @field_validator("area_id")
    @classmethod
    def _aid(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", v):
            raise ValueError("invalid area id")
        return v


@app.post("/api/routes/plan", tags=["routing"])
def plan_route(req: PlanRequest):
    t0 = time.perf_counter()
    G, stats, alignment, layer = areas.annotated_graph(req.area_id, req.source, req.model, req.threshold,
                                                       req.buffer_px)
    s = RiskSettings(flood_penalty=req.flood_penalty, unknown_penalty=req.unknown_penalty,
                     close_flooded=req.close_flooded, closure_min_flooded_m=req.closure_min_flooded_m,
                     alternatives=req.alternatives)
    res = plan(G, (req.start.lat, req.start.lon), (req.end.lat, req.end.lon), s)
    res["flood_layer"] = {"source": layer.source, "description": layer.description,
                          "is_model_prediction": layer.source in ("model_prediction", "uploaded_prediction"),
                          "threshold": req.threshold, "buffer_px": req.buffer_px}
    res["risk_stats"] = stats
    res["alignment"] = alignment
    res["compute_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    res["attribution"] = graph_store.ATTRIBUTION
    res["disclaimer"] = DISCLAIMER
    res["travel_time_note"] = ("Free-flow estimate from OSM maxspeed or highway-type averages "
                               f"({graph_store.FALLBACK_KMH:.0f} km/h fallback); flood slowdown is not modelled.")
    for r in [res["baseline"], res["risk_aware"], *res["alternatives"]]:
        r.pop("nodes", None)
    a = areas.get_area(req.area_id)
    res["area_id"] = req.area_id
    res["area_name"] = a.get("name")
    res["road_meta"] = a.get("road_meta")
    res["imagery_event"] = a.get("event")
    res["start"], res["end"] = req.start.model_dump(), req.end.model_dump()
    res["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rid = reports.save_route(res)
    history.add(rid, "route", res["created"], f"{a.get('name')} route", req.model if req.source == "model" else None,
                req.source,
                {"status": res["status"], "baseline_m": res["baseline"]["distance_m"],
                 "risk_aware_m": res["risk_aware"]["distance_m"],
                 "baseline_flooded_m": res["baseline"]["flooded_length_m"],
                 "risk_aware_flooded_m": res["risk_aware"]["flooded_length_m"], "area_id": req.area_id,
                 "alternatives": len(res["alternatives"])},
                None, {k: v for k, v in req.model_dump().items() if k not in ("start", "end")})
    return res


@app.get("/api/routes/areas/{area_id}/search", tags=["routing"])
def area_search(area_id: str, q: str = Query(..., min_length=2, max_length=60)):
    """Offline place search: road names in the cached OSM graph of this study area."""
    a = areas.get_area(area_id)
    G, *_ = areas.annotated_graph(area_id, "model" if "model" in a["sources"] else a["sources"][0], None, 0.5, 1)
    ql = q.lower()
    hits: dict[str, list] = {}
    for u, v, d in G.edges(data=True):
        n = d.get("name")
        for name in (n if isinstance(n, list) else [n]):
            if name and ql in str(name).lower() and (str(name) in hits or len(hits) < 15):
                hits.setdefault(str(name), []).append(((G.nodes[u]["y"] + G.nodes[v]["y"]) / 2,
                                                       (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2))
    return {"query": q, "source": "OpenStreetMap road names in the cached study-area graph (offline)",
            "results": [{"name": k, "lat": round(sum(p[0] for p in v) / len(v), 6),
                         "lon": round(sum(p[1] for p in v) / len(v), 6), "segments": len(v)}
                        for k, v in sorted(hits.items())]}


@app.get("/api/routes/results/{rid}", tags=["routing"])
def route_result(rid: str):
    r = reports.get_route(rid)
    if r is None:
        raise HTTPException(404, "route result not found")
    return r


@app.get("/api/routes/results/{rid}/export", tags=["routing"])
def route_export(rid: str, format: Literal["json", "geojson", "html"] = "json"):
    r = reports.get_route(rid)
    if r is None:
        raise HTTPException(404, "route result not found")
    name = f"route_{rid[:8]}"
    if format == "geojson":
        return Response(json.dumps(reports.route_geojson(r), default=float), media_type="application/geo+json",
                        headers={"Content-Disposition": f'attachment; filename="{name}.geojson"'})
    if format == "html":
        return Response(_route_report_html(r), media_type="text/html",
                        headers={"Content-Disposition": f'attachment; filename="{name}_report.html"'})
    return Response(json.dumps(r, default=float, indent=1), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{name}.json"'})


def _route_report_html(r: dict) -> str:
    try:
        src = "ground_truth" if r["flood_layer"]["source"] == "ground_truth_label" else "model"
        ov = areas.overlay_png(r["area_id"], src, None, r["flood_layer"]["threshold"])
        bounds = list(areas.flood_layer(r["area_id"], src, None).bounds_wgs84())
    except Exception:  # noqa: BLE001 - area no longer available: report without the map
        ov, bounds = None, None
    return reports.route_report(r, ov, bounds)


# ----------------------------------------------------------------------------- history & reports
@app.get("/api/history", tags=["history"])
def history_list(kind: Literal["segmentation", "route"] | None = None, q: str | None = Query(None, max_length=60),
                 limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    out = history.list_(kind, limit, offset, q)
    for it in out["items"]:
        it["available"] = (segmentation.get_result(it["id"]) is not None if it["kind"] == "segmentation"
                           else reports.get_route(it["id"]) is not None)
    return out


@app.get("/api/history/export", tags=["history"])
def history_export(format: Literal["csv", "json"] = "csv", kind: Literal["segmentation", "route"] | None = None):
    body, mt = history.export(format, kind)
    return Response(body, media_type=mt,
                    headers={"Content-Disposition": f'attachment; filename="flood_analysis_history.{format}"'})


@app.get("/api/history/{hid}", tags=["history"])
def history_item(hid: str):
    it = history.get(hid)
    if it is None:
        raise HTTPException(404, "history item not found")
    it["detail"] = segmentation.get_result(hid) if it["kind"] == "segmentation" else reports.get_route(hid)
    it["available"] = it["detail"] is not None
    return it


@app.get("/api/reports/{hid}", tags=["history"])
def report(hid: str):
    """Self-contained HTML report (images embedded) for one segmentation or route analysis."""
    it = history.get(hid)
    if it is None:
        raise HTTPException(404, "history item not found")
    if it["kind"] == "segmentation":
        res = segmentation.get_result(hid)
        if res is None:
            raise HTTPException(410, "result files were pruned; only the history metadata remains")
        body = reports.segmentation_report(res, segmentation.results_dir() / hid)
    else:
        r = reports.get_route(hid)
        if r is None:
            raise HTTPException(410, "route result no longer available")
        body = _route_report_html(r)
    return Response(body, media_type="text/html",
                    headers={"Content-Disposition": f'attachment; filename="flood_report_{hid[:8]}.html"'})


# ----------------------------------------------------------------------------- experiments, diagnostics, control
@app.get("/api/experiments", tags=["evaluation"])
def experiments():
    from training.registry import build

    return {"experiments": build()}


@app.get("/api/evaluation/external", tags=["evaluation"])
def external_evaluation():
    w = status._read(settings.artifacts / "audit" / "weakness_analysis.json")
    weak = None
    if w:
        weak = {"data": w["data"], "valid_selected_threshold": w["model"].get("valid_selected_threshold"),
                "fn_rate_by_stratum": {s: w["model"][s]["fn_rate_by_stratum"] for s in ("valid", "test", "bolivia")},
                "ece": {s: w["model"][s]["ece_10bin"] for s in ("valid", "test", "bolivia")},
                "reliability": {s: w["model"][s]["reliability"] for s in ("valid", "test", "bolivia")}}
    return {"worldfloods": status._read(settings.artifacts / "worldfloods_evaluation.json"),
            "bootstrap": status._read(settings.artifacts / "v2_bootstrap.json"),
            "audit": status._read(settings.artifacts / "audit" / "reproduction.json"), "weakness_analysis": weak}


@app.get("/api/diagnostics", tags=["status"])
def run_diagnostics():
    return diagnostics.run_diagnostics()


@app.get("/api/warnings", tags=["status"])
def warnings_feed():
    return {"warnings": diagnostics.warnings()}


@app.get("/api/version", tags=["status"])
def version():
    return diagnostics.versions()


@app.get("/api/segmentation/formats", tags=["segmentation"])
def formats():
    return {"max_upload_mb": settings.max_upload_mb, "max_pixels": settings.max_image_pixels,
            "min_side_px": settings.min_image_side, "formats": [
                {"format": "GeoTIFF, 2 bands", "interpretation": "Sentinel-1 VV, VH backscatter in dB "
                                                               "(linear power is converted)",
                 "models": ["s1"], "georeferenced": "yes"},
                {"format": "GeoTIFF, 13 bands", "interpretation": "Sentinel-2 L1C B1..B12 incl. B8A, "
                                                                "reflectance x10000 or 0-1",
                 "models": ["s2", "s2i", "rgb"], "georeferenced": "yes"},
                {"format": "GeoTIFF, 15 bands", "interpretation": "Sentinel-1 (2 bands) followed by Sentinel-2 (13)",
                 "models": ["s1s2", "s1s2i", "s2", "s2i", "s1", "rgb"], "georeferenced": "yes"},
                {"format": "GeoTIFF / PNG / JPEG, 3-4 bands", "interpretation": "RGB (alpha ignored); per-image stretch",
                 "models": ["rgb"], "georeferenced": "GeoTIFF only"}],
            "notes": ["The band count decides the interpretation; band order must follow this table.",
                      "Only GeoTIFFs with a CRS and geotransform can be mapped, measured in km² or routed.",
                      "PNG/JPEG photos are outside the training distribution (10 m satellite imagery)."]}


class TrainStart(BaseModel):
    run: str = Field(max_length=80)
    confirm: str = Field(max_length=80)


@app.get("/api/training/presets", tags=["training"])
def training_presets():
    p = training_control.presets()
    return {"presets": [{"run": k, "overrides": v,
                         "state": (status._read(settings.artifacts / "runs" / k / "status.json") or {}).get("state", "new")}
                        for k, v in p.items()], "preflight": training_control.preflight()}


@app.post("/api/training/start", tags=["training"])
def training_start(req: TrainStart):
    return training_control.start(req.run, req.confirm)


@app.post("/api/training/stop", tags=["training"])
def training_stop(req: TrainStart):
    if req.confirm != req.run:
        raise HTTPException(400, "confirmation does not match the run name")
    return training_control.stop(req.run)


# ----------------------------------------------------------------------------- frontend (production build)
if settings.frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=settings.frontend_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        f = (settings.frontend_dist / path).resolve()
        if path and f.is_file() and f.is_relative_to(settings.frontend_dist.resolve()):
            return FileResponse(f)
        return FileResponse(settings.frontend_dist / "index.html")
