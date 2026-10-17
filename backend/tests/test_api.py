"""API tests: health/readiness, input validation, end-to-end prediction on a real held-out chip,
and routing on a cached, attributed OpenStreetMap graph."""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import settings
from backend.app.main import app
from backend.app.routing import graph_store

client = TestClient(app)
HAS_MODEL = (settings.artifacts / "runs" / "unet_resnet34_s1s2" / "checkpoints" / "best.pt").exists()
HAS_DATA = (settings.processed / "index.json").exists() and (settings.raw / "sen1floods11" / "manifest.json").exists()
needs_model = pytest.mark.skipif(not (HAS_MODEL and HAS_DATA), reason="trained model / dataset not available")


def png_bytes(w=64, h=64) -> bytes:
    b = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(b, "PNG")
    return b.getvalue()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "X-Request-ID" in r.headers


def test_model_status_shape():
    r = client.get("/api/model/status")
    assert r.status_code == 200
    j = r.json()
    assert {"ready", "models", "default_model", "cuda_available"} <= set(j)
    for m in j["models"]:
        assert not m["name"].startswith("smoke_")


def test_status_endpoints_return_strict_json():
    for path in ("/api/system", "/api/datasets/status", "/api/training/status", "/api/metrics"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "NaN" not in r.text


@pytest.mark.parametrize("name,data,msg", [
    ("evil.exe", b"MZ....", "Unsupported file type"),
    ("fake.png", b"not a png at all", "not a PNG"),
    ("fake.tif", b"\x89PNG\r\n\x1a\n....", "not a TIFF"),
    ("empty.png", b"", "empty"),
])
def test_invalid_uploads_rejected(name, data, msg):
    r = client.post("/api/segmentation/predict", files={"file": (name, data)})
    assert r.status_code == 422
    assert msg in r.json()["message"]
    assert "Traceback" not in r.text and "F:\\" not in r.text


def test_corrupt_and_tiny_images_rejected():
    r = client.post("/api/segmentation/predict", files={"file": ("x.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)})
    assert r.status_code == 422 and "could not be decoded" in r.json()["message"]
    r = client.post("/api/segmentation/predict", files={"file": ("small.png", png_bytes(8, 8))})
    assert r.status_code == 422 and "at least" in r.json()["message"]


def test_threshold_validation():
    r = client.post("/api/segmentation/predict", files={"file": ("a.png", png_bytes())}, data={"threshold": "1.5"})
    assert r.status_code == 422


def test_result_paths_cannot_escape():
    assert client.get("/api/segmentation/results/..%2F..%2Fsecrets").status_code == 404
    assert client.get("/api/segmentation/results/" + "a" * 32 + "/files/result.json").status_code == 404
    assert client.get("/api/metrics/runs/unet_resnet34_s1s2/images/..%5C..%5Cconfig.yaml").status_code == 404
    assert client.get("/api/training/runs/..%2F..%2F/log").status_code == 404
    assert client.get("/api/samples/Spain_7370579_x/preview.png").status_code == 404


@needs_model
def test_end_to_end_geotiff_upload_prediction():
    """Upload a real held-out Sen1Floods11 chip as a 15-band GeoTIFF and validate the prediction."""
    tif = client.get("/api/samples/Spain_7370579/download")
    assert tif.status_code == 200 and tif.content[:2] in (b"II", b"MM")
    r = client.post("/api/segmentation/predict", files={"file": ("Spain_7370579_s1s2.tif", tif.content, "image/tiff")},
                    data={"threshold": "0.5"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["kind"] == "model_prediction" and j["model"]["name"] == "unet_resnet34_s1s2"
    assert j["input"]["georeferenced"] and j["input"]["crs"] == "EPSG:4326"
    assert (j["input"]["width"], j["input"]["height"]) == (512, 512)
    assert 0.05 < j["stats"]["water_fraction"] < 0.8
    assert j["area"]["water_area_km2"] > 0
    for f in ("overlay.png", "mask.png", "probability.tif"):
        assert client.get(f"/api/segmentation/results/{j['id']}/files/{f}").status_code == 200
    mask = Image.open(io.BytesIO(client.get(f"/api/segmentation/results/{j['id']}/files/mask.png").content))
    assert mask.size == (512, 512)
    assert client.get(f"/api/segmentation/results/{j['id']}").json()["id"] == j["id"]


HAS_RGB = (settings.artifacts / "runs" / "unet_resnet34_rgb" / "status.json").exists() and     '"completed"' in (settings.artifacts / "runs" / "unet_resnet34_rgb" / "status.json").read_text()


@needs_model
def test_png_upload_without_rgb_model_is_a_clean_error():
    if HAS_RGB:
        pytest.skip("RGB model exists")
    r = client.post("/api/segmentation/predict", files={"file": ("photo.png", png_bytes(100, 80))})
    assert r.status_code == 422 and "no trained model" in r.json()["message"]


@needs_model
@pytest.mark.skipif(not HAS_RGB, reason="RGB model not trained yet")
def test_rgb_png_upload_warns_about_domain_and_georeferencing():
    r = client.post("/api/segmentation/predict", files={"file": ("photo.png", png_bytes(100, 80))})
    assert r.status_code == 200
    j = r.json()
    assert j["model"]["modality"] == "rgb" or "rgb" in j["input"]["available_modalities"]
    assert not j["input"]["georeferenced"] and j["area"] is None
    assert any("georeferencing" in w for w in j["warnings"])


@needs_model
def test_sample_prediction_reports_ground_truth_separately():
    r = client.post("/api/segmentation/predict-sample", json={"chip_id": "Spain_7370579"})
    assert r.status_code == 200
    j = r.json()
    assert j["kind"] == "model_prediction"
    assert "not a model output" in j["ground_truth"]["label"]
    assert 0 <= j["ground_truth"]["metrics"]["iou_water"] <= 1
    assert client.post("/api/segmentation/predict-sample", json={"chip_id": "Ghana_103272"}).status_code in (404, 200)


@needs_model
@pytest.mark.skipif(not graph_store.cached("spain_7370579"), reason="OSM graph for spain_7370579 not cached")
def test_routing_integration_on_cached_osm_graph():
    """Real OSM drive network (c) OpenStreetMap contributors, ODbL; flood layer = model prediction."""
    areas = client.get("/api/routes/areas").json()
    assert "OpenStreetMap" in areas["attribution"]
    a = next(x for x in areas["areas"] if x["id"] == "spain_7370579")
    body = {"area_id": a["id"], "start": {"lat": a["default_start"][0], "lon": a["default_start"][1]},
            "end": {"lat": a["default_end"][0], "lon": a["default_end"][1]}}
    r = client.post("/api/routes/plan", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["flood_layer"]["is_model_prediction"] and j["alignment"]["ok"]
    assert j["risk_aware"]["flooded_length_m"] <= j["baseline"]["flooded_length_m"]
    assert j["risk_aware"]["distance_m"] >= j["baseline"]["distance_m"]
    assert len(j["risk_aware"]["coordinates"]) >= 2 and j["explanation"]
    gt = client.post("/api/routes/plan", json={**body, "source": "ground_truth"}).json()
    assert gt["flood_layer"]["source"] == "ground_truth_label" and not gt["flood_layer"]["is_model_prediction"]
    bad = client.post("/api/routes/plan", json={**body, "end": {"lat": 0.0, "lon": 0.0}})
    assert bad.status_code == 422 and bad.json()["error"] == "too_far_from_road"
    assert client.post("/api/routes/plan", json={**body, "area_id": "../../etc"}).status_code == 422
    assert client.post("/api/routes/plan", json={**body, "area_id": "nope"}).status_code == 404
