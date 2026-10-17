"""Segmentation inference + result storage.

Every result is written to outputs/predictions/<uuid>/ with a fixed set of file names; the
API only ever serves names from RESULT_FILES, so user input never becomes a filesystem path.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import shutil
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.config import settings
from backend.app.services.models import registry

RESULT_FILES = {
    "input.png": "image/png", "overlay.png": "image/png", "mask.png": "image/png",
    "probability.png": "image/png", "probability.tif": "image/tiff", "ground_truth.png": "image/png",
    "errors.png": "image/png",
}
_infer_lock = threading.Lock()  # one GPU inference at a time (4 GB laptop GPU)
_recent: deque = deque(maxlen=50)
Image.MAX_IMAGE_PIXELS = 64_000_000  # hard ceiling for the decoder (decompression-bomb guard)


class InputError(ValueError):
    """Invalid user input; message is safe to show to the user."""


def results_dir() -> Path:
    p = settings.outputs / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_upload(data: bytes, filename: str) -> str:
    name = (filename or "").lower()
    if not name.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
        raise InputError("Unsupported file type. Upload a GeoTIFF (.tif) or a PNG/JPEG image.")
    if len(data) == 0:
        raise InputError("The uploaded file is empty.")
    if len(data) > settings.max_upload_mb * 2**20:
        raise InputError(f"File is larger than the {settings.max_upload_mb:.0f} MB limit.")
    sig = data[:4]
    is_tif = sig in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
    is_png = data[:8] == b"\x89PNG\r\n\x1a\n"
    is_jpg = data[:3] == b"\xff\xd8\xff"
    if name.endswith((".tif", ".tiff")) and not is_tif:
        raise InputError("File extension is .tif but the content is not a TIFF.")
    if name.endswith(".png") and not is_png:
        raise InputError("File extension is .png but the content is not a PNG.")
    if name.endswith((".jpg", ".jpeg")) and not is_jpg:
        raise InputError("File extension is .jpg but the content is not a JPEG.")
    return "tif" if is_tif else "png" if is_png else "jpg"


def read_scene(data: bytes, filename: str):
    from training.infer import read_input

    validate_upload(data, filename)
    try:
        scene = read_input(data, filename)
    except ValueError as e:
        raise InputError(str(e)) from None
    except Exception:  # noqa: BLE001 - malformed/corrupt files
        raise InputError("The file could not be decoded as an image. It may be corrupt.") from None
    h, w = scene.shape
    if h * w > settings.max_image_pixels:
        raise InputError(f"Image is {w}x{h} px; the limit is {settings.max_image_pixels:,} pixels.")
    if min(h, w) < settings.min_image_side:
        raise InputError(f"Image is {w}x{h} px; both sides must be at least {settings.min_image_side} px.")
    return scene


def _png(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG", optimize=False)
    return b.getvalue()


def run_prediction(scene, model: str | None, threshold: float, tta: bool, source: dict,
                   ground_truth: np.ndarray | None = None) -> dict:
    from training.infer import overlay, prob_colormap, write_geotiff
    from training.metrics import confusion, summarize

    if not 0.05 <= threshold <= 0.95:
        raise InputError("threshold must be between 0.05 and 0.95")
    try:
        name = registry.choose(scene.available_modalities(), model)
    except (KeyError, ValueError, LookupError) as e:
        raise InputError(str(e).strip("'\"")) from None
    pred = registry.get(name)
    t0 = time.perf_counter()
    with _infer_lock:
        t_wait = time.perf_counter()
        prob, timing = pred.run(scene, tta=tta)
    mask = prob > threshold
    rid = uuid.uuid4().hex
    out = results_dir() / rid
    out.mkdir()
    rgb = scene.preview_rgb()
    (out / "input.png").write_bytes(_png(rgb))
    (out / "overlay.png").write_bytes(_png(overlay(rgb, mask)))
    (out / "mask.png").write_bytes(_png((mask * 255).astype(np.uint8)))
    (out / "probability.png").write_bytes(_png(prob_colormap(prob)))
    write_geotiff(out / "probability.tif", prob, scene)

    h, w = prob.shape
    area = None
    if scene.georeferenced:
        from pyproj import CRS, Geod

        crs = CRS.from_user_input(scene.crs)
        a, b, c, d, e, f = scene.transform
        if crs.is_geographic:
            lat = f + e * h / 2
            geod = Geod(ellps="WGS84")
            px_w = geod.inv(c, lat, c + a, lat)[2]
            px_h = geod.inv(c, lat, c, lat + e)[2]
            pixel_area = abs(px_w * px_h)
        else:
            pixel_area = abs(a * e - b * d) * (crs.axis_info[0].unit_conversion_factor ** 2)
        area = {"pixel_area_m2": round(pixel_area, 2), "water_area_km2": round(float(mask.sum()) * pixel_area / 1e6, 4),
                "scene_area_km2": round(h * w * pixel_area / 1e6, 4), "method": "pixel count x geodesic pixel area"}

    result = {
        "id": rid,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "kind": "model_prediction",
        "source": source,
        "model": {"name": name, "modality": pred.modality, "arch": pred.config["model"]["arch"],
                  "encoder": pred.config["model"]["encoder"], "checkpoint_epoch": pred.epoch,
                  "trained_on": pred.meta.get("trained_on")},
        "input": {"kind": scene.kind, "width": w, "height": h, "georeferenced": scene.georeferenced,
                  "crs": scene.crs, "bounds": scene.bounds, "transform": scene.transform, "notes": scene.notes,
                  "available_modalities": scene.available_modalities()},
        "settings": {"threshold": threshold, "tta": tta},
        "stats": {"water_fraction": round(float(mask.mean()), 5), "mean_probability": round(float(prob.mean()), 5),
                  "confident_pixels_fraction": round(float(((prob < 0.1) | (prob > 0.9)).mean()), 5)},
        "area": area,
        "timing": {**timing, "queue_ms": round((t_wait - t0) * 1e3, 1),
                   "total_ms": round((time.perf_counter() - t0) * 1e3, 1)},
        "files": ["input.png", "overlay.png", "mask.png", "probability.png", "probability.tif"],
        "ground_truth": None,
        "warnings": [],
    }
    if scene.kind == "rgb" and source.get("type") == "upload":
        result["warnings"].append(
            "RGB model trained on 10 m Sentinel-2 true-colour chips. Drone, aerial or phone photos are outside "
            "the training distribution; treat the mask as indicative only.")
    if not scene.georeferenced:
        result["warnings"].append("No georeferencing: the mask cannot be placed on a map or used for routing.")
    if ground_truth is not None:
        from backend.app.services.figures import error_rgb, label_rgb

        cm = confusion(mask.astype(np.int8), ground_truth)
        result["ground_truth"] = {"label": "Sen1Floods11 hand label (not a model output)",
                                  "metrics": summarize(cm)}
        (out / "ground_truth.png").write_bytes(_png(label_rgb(ground_truth)))
        (out / "errors.png").write_bytes(_png(error_rgb(mask, ground_truth)))
        result["files"] += ["ground_truth.png", "errors.png"]
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    np.save(out / "probability.npy", prob.astype(np.float16))
    _recent.appendleft({k: result[k] for k in ("id", "created", "source", "stats")} |
                       {"model": name, "total_ms": result["timing"]["total_ms"]})
    _prune()
    return result


def get_result(rid: str) -> dict | None:
    if not (len(rid) == 32 and all(c in "0123456789abcdef" for c in rid)):
        return None
    p = results_dir() / rid / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def result_file(rid: str, name: str) -> Path | None:
    if name not in RESULT_FILES or get_result(rid) is None:
        return None
    p = results_dir() / rid / name
    return p if p.exists() else None


def recent() -> list[dict]:
    return list(_recent)


def _prune() -> None:
    dirs = sorted((d for d in results_dir().iterdir() if d.is_dir() and len(d.name) == 32),
                  key=lambda d: d.stat().st_mtime)
    for d in dirs[:-settings.max_results_kept]:
        shutil.rmtree(d, ignore_errors=True)  # only our own uuid-named result folders
