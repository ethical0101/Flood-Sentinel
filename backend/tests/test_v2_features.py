"""v2 features: history persistence, exports/reports, route status, training-control safeguards,
diagnostics, formats, external evaluation endpoint."""
from __future__ import annotations

import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app
from backend.app.routing import graph_store
from backend.app.services import history, reports, training_control

client = TestClient(app)
HAS_MODEL = (settings.artifacts / "runs" / "unet_resnet34_s1s2" / "checkpoints" / "best.pt").exists()
HAS_DATA = (settings.processed / "index.json").exists()
needs_model = pytest.mark.skipif(not (HAS_MODEL and HAS_DATA), reason="trained model / dataset not available")


@pytest.fixture()
def tmp_db(tmp_path):
    """Point the history database at a temp file for one test, then restore the real path."""
    original = settings.db_path
    object.__setattr__(settings, "db_path", tmp_path / "h.sqlite")  # frozen dataclass
    try:
        yield tmp_path / "h.sqlite"
    finally:
        object.__setattr__(settings, "db_path", original)


def test_history_roundtrip_and_export(tmp_db):
    history.add("a" * 32, "segmentation", "2026-10-18T01:00:00", "chip A", "m1", "sample",
                {"water_fraction": 0.2}, {"iou_water": 0.8}, {"threshold": 0.5})
    history.add("b" * 32, "route", "2026-10-18T02:00:00", "area route", None, "model",
                {"status": "no_flood_free_route", "risk_aware_m": 1200.0})
    out = history.list_()
    assert out["total"] == 2 and out["items"][0]["kind"] == "route"  # newest first
    assert history.list_("segmentation")["total"] == 1
    assert history.list_(q="chip")["items"][0]["id"] == "a" * 32
    assert history.get("a" * 32)["metrics"]["iou_water"] == 0.8
    assert history.counts()["route"]["count"] == 1
    body, mt = history.export("csv")
    rows = list(csv.DictReader(io.StringIO(body)))
    assert mt == "text/csv" and len(rows) == 2 and rows[1]["metrics.iou_water"] == "0.8"
    body, mt = history.export("json")
    assert len(json.loads(body)) == 2


def test_history_api_endpoints(tmp_db):
    history.add("c" * 32, "segmentation", "2026-10-18T03:00:00", "x", "m", "upload", {"water_fraction": 0.1})
    r = client.get("/api/history?kind=segmentation")
    assert r.status_code == 200 and r.json()["items"][0]["available"] is False  # no result folder on disk
    assert client.get("/api/history/export?format=csv").headers["content-type"].startswith("text/csv")
    assert client.get("/api/history/" + "f" * 32).status_code == 404
    assert client.get("/api/reports/" + "c" * 32).status_code == 410  # metadata only, files pruned
    assert client.get("/api/history?kind=bogus").status_code == 422


def test_route_store_rejects_bad_ids():
    assert reports.get_route("../../etc/passwd") is None
    assert reports.get_route("z" * 32) is None
    assert client.get("/api/routes/results/" + "0" * 32).status_code == 404


def test_training_control_safeguards(monkeypatch):
    with pytest.raises(training_control.TrainingControlError) as e:
        training_control.start("v2_r1_unet_s1s2_radiometric", "wrong")
    assert e.value.status == 400
    monkeypatch.setattr(training_control, "preflight", lambda run=None: {"ok": False, "active": [], "checks": [
        {"check": "No training run active", "ok": False, "detail": "x"}]})
    with pytest.raises(training_control.TrainingControlError) as e:
        training_control.start("v2_r1_unet_s1s2_radiometric", "v2_r1_unet_s1s2_radiometric")
    assert e.value.status == 409 and "No training run active" in e.value.message
    r = client.post("/api/training/start", json={"run": "rm -rf /", "confirm": "rm -rf /"})
    assert r.status_code == 409
    assert client.post("/api/training/stop", json={"run": "nope", "confirm": "nope"}).status_code == 409
    assert client.post("/api/training/stop", json={"run": "a", "confirm": "b"}).status_code == 400


def test_presets_are_fixed_and_known():
    p = training_control.presets()
    assert "unet_resnet34_s1s2" in p and "v2_r3_unet_s1s2i_radiometric" in p
    assert all(isinstance(v, list) for v in p.values())
    r = client.get("/api/training/presets")
    assert r.status_code == 200 and {"presets", "preflight"} <= set(r.json())


def test_diagnostics_warnings_formats_version():
    d = client.get("/api/diagnostics").json()
    assert d["summary"]["ok"] >= 1 and all(c["status"] in ("ok", "warn", "fail") for c in d["checks"])
    assert "works_offline" in d["offline"]
    w = client.get("/api/warnings").json()["warnings"]
    assert all(x["severity"] in ("critical", "warning", "info") for x in w)
    f = client.get("/api/segmentation/formats").json()
    assert any("15 bands" in x["format"] for x in f["formats"])
    v = client.get("/api/version").json()
    assert v["app"] == "2.0.0"
    assert client.get("/api/experiments").status_code == 200
    assert client.get("/api/evaluation/external").status_code == 200


def test_route_report_html_and_geojson_from_synthetic_result():
    r = {"id": "d" * 32, "status": "no_flood_free_route", "status_note": "n", "created": "t", "area_id": "x",
         "settings": {"flood_penalty": 10, "unknown_penalty": 0.5, "prob_penalty": 1, "close_flooded": False, "closure_min_flooded_m": 20},
         "flood_layer": {"description": "d", "is_model_prediction": True, "threshold": 0.5, "buffer_px": 1, "source": "model_prediction"},
         "attribution": "OSM", "travel_time_note": "free flow", "explanation": ["e1"], "warnings": [], "alternatives": [],
         "baseline": {}, "risk_aware": {}}
    route = {"coordinates": [[0.0, 0.0], [0.001, 0.001]], "distance_m": 150.0, "travel_time_s": 15.0, "flooded_length_m": 10.0,
             "flooded_share": 0.07, "unknown_length_m": 0.0, "max_flood_prob": 0.9, "risk_level": "substantial predicted flooding",
             "length_by_risk_class_m": {"clear": 140.0, "flooded": 10.0}, "flood_affected_roads": [{"road": "Main <St>", "flooded_m": 10.0}],
             "segments": [{"coords": [[0.0, 0.0], [0.001, 0.001]], "risk_class": "flooded", "flooded_len_m": 10.0, "length_m": 150.0, "name": "Main", "highway": "residential"}]}
    r["baseline"], r["risk_aware"] = dict(route), dict(route)
    html = reports.route_report(r, None, [0.0, 0.0, 0.002, 0.002])
    assert "<svg" in html and "no_flood_free_route" in html and "Main &lt;St&gt;" in html  # escaped
    gj = reports.route_geojson(r)
    assert gj["type"] == "FeatureCollection" and gj["features"][0]["geometry"]["coordinates"][1] == [0.001, 0.001]


@needs_model
def test_segmentation_v2_outputs_history_and_report(tmp_db):
    j = client.post("/api/segmentation/predict-sample", json={"chip_id": "Spain_7370579"}).json()
    for f in ("mask_rgba.png", "uncertainty.png", "model_vs_ndwi.png", "result.json"):
        assert f in j["files"] and client.get(f"/api/segmentation/results/{j['id']}/files/{f}").status_code == 200
    b = j["baseline"]
    assert b and 0 <= b["agreement_iou_with_model"] <= 1 and b["metrics_vs_ground_truth"]["iou_water"] > 0
    assert 0 <= j["uncertainty"]["mean"] <= 1
    h = client.get(f"/api/history/{j['id']}").json()
    assert h["kind"] == "segmentation" and h["available"] and h["metrics"]["iou_water"] == pytest.approx(j["ground_truth"]["metrics"]["iou_water"])
    rep = client.get(f"/api/reports/{j['id']}")
    assert rep.status_code == 200 and "Flood segmentation report" in rep.text and "data:image/png;base64" in rep.text


@needs_model
@pytest.mark.skipif(not graph_store.cached("spain_7370579"), reason="OSM graph not cached")
def test_route_plan_status_breakdown_persistence_and_exports(tmp_db):
    a = next(x for x in client.get("/api/routes/areas").json()["areas"] if x["id"] == "spain_7370579")
    body = {"area_id": a["id"], "start": {"lat": a["default_start"][0], "lon": a["default_start"][1]},
            "end": {"lat": a["default_end"][0], "lon": a["default_end"][1]}}
    j = client.post("/api/routes/plan", json=body).json()
    assert j["status"] in ("flood_free_route_found", "no_flood_free_route", "no_route_avoids_closures")
    ra = j["risk_aware"]
    assert abs(sum(ra["length_by_risk_class_m"].values()) - ra["distance_m"]) < 1.0
    assert ra["risk_level"] and isinstance(ra["flood_affected_roads"], list)
    assert client.get(f"/api/routes/results/{j['id']}").json()["id"] == j["id"]
    for fmt, ctype in (("json", "application/json"), ("geojson", "application/geo+json"), ("html", "text/html")):
        e = client.get(f"/api/routes/results/{j['id']}/export?format={fmt}")
        assert e.status_code == 200 and e.headers["content-type"].startswith(ctype)
    assert history.get(j["id"])["kind"] == "route"
    s = client.get(f"/api/routes/areas/{a['id']}/search?q=ca").json()
    assert "results" in s and all({"name", "lat", "lon"} <= set(x) for x in s["results"])
    assert client.get(f"/api/routes/areas/{a['id']}/search?q=a").status_code == 422  # min length
