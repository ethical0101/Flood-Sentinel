"""Route-result persistence, exports (JSON / GeoJSON / CSV) and self-contained HTML reports."""
from __future__ import annotations

import base64
import html
import json
import re
import uuid
from pathlib import Path

from backend.app.config import settings

ROUTE_ID = re.compile(r"^[0-9a-f]{32}$")
DISCLAIMER = ("Research prototype. Flood extents are model estimates from satellite imagery; routes are computed on "
              "OpenStreetMap data and are not verified. Do not use for real evacuation or navigation decisions.")


def routes_dir() -> Path:
    p = settings.outputs / "routes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_route(result: dict) -> str:
    rid = uuid.uuid4().hex
    result["id"] = rid
    (routes_dir() / f"{rid}.json").write_text(json.dumps(result, default=float))
    return rid


def get_route(rid: str) -> dict | None:
    if not ROUTE_ID.fullmatch(rid or ""):
        return None
    p = routes_dir() / f"{rid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def route_geojson(r: dict) -> dict:
    feats = []
    for name, route in [("shortest_baseline", r["baseline"]), ("risk_aware", r["risk_aware"]),
                        *[(f"alternative_{i + 1}", a) for i, a in enumerate(r["alternatives"])]]:
        feats.append({"type": "Feature", "geometry": {"type": "LineString",
                                                      "coordinates": [[lon, lat] for lat, lon in route["coordinates"]]},
                      "properties": {"route": name, **{k: route.get(k) for k in (
                          "distance_m", "travel_time_s", "flooded_length_m", "flooded_share", "unknown_length_m",
                          "max_flood_prob", "risk_level")}}})
        if name == "risk_aware":
            for seg in route["segments"]:
                if seg["risk_class"] in ("partial", "flooded", "unknown"):
                    feats.append({"type": "Feature",
                                  "geometry": {"type": "LineString", "coordinates": [[lo, la] for la, lo in seg["coords"]]},
                                  "properties": {"route": "risk_aware_segment", **{k: seg[k] for k in (
                                      "risk_class", "flooded_len_m", "length_m", "name", "highway")}}})
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"area_id": r.get("area_id"), "flood_layer": r.get("flood_layer"), "status": r.get("status"),
                           "attribution": r.get("attribution"), "disclaimer": DISCLAIMER}}


# ----------------------------------------------------------------------------- HTML reports
CSS = """body{font:13px/1.5 'Segoe UI',system-ui,sans-serif;color:#0f172a;max-width:980px;margin:24px auto;padding:0 16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:14px;margin:22px 0 6px;border-bottom:1px solid #e2e8f0;padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin:6px 0}td,th{border:1px solid #e2e8f0;padding:4px 8px;text-align:left}
th{background:#f8fafc;font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
.meta{color:#475569}.warn{background:#fffbeb;border:1px solid #fcd34d;padding:8px 10px;border-radius:6px;margin:6px 0}
.danger{background:#fef2f2;border:1px solid #fca5a5;padding:8px 10px;border-radius:6px;margin:6px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}figure{margin:0}
figure img{width:100%;border:1px solid #e2e8f0;border-radius:4px;image-rendering:pixelated}figcaption{font-size:11px;color:#475569}
.ref{color:#6d28d9}"""


def _e(v) -> str:
    return html.escape("–" if v is None else str(v))


def _f(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def segmentation_report(res: dict, folder: Path) -> str:
    m = res["model"]
    rows = [("Analysis id", res["id"]), ("Created", res["created"]), ("Source", json.dumps(res["source"])),
            ("Model", f"{m['name']} ({m['arch']} / {m['encoder']}, input {m['modality']}, checkpoint epoch {m['checkpoint_epoch']})"),
            ("Trained on", m.get("trained_on")), ("Input", f"{res['input']['width']}×{res['input']['height']} px, "
                                                           f"kind {res['input']['kind']}, CRS {res['input']['crs'] or 'none'}"),
            ("Threshold / TTA", f"{res['settings']['threshold']} / {res['settings']['tta']}"),
            ("Predicted water", f"{res['stats']['water_fraction'] * 100:.2f}% of pixels"),
            ("Water area", f"{res['area']['water_area_km2']:.3f} km² ({res['area']['method']})" if res.get("area") else
             "not available (no georeferencing)"),
            ("Latency", f"preprocess {res['timing']['preprocess_ms']} ms · model {res['timing']['model_ms']} ms · "
                        f"total {res['timing']['total_ms']} ms on {res['timing']['device']}")]
    figs = [(f, c) for f, c in [("input.png", "Input (display composite)"), ("overlay.png", "Model prediction overlay"),
                                ("probability.png", "Model P(water)"), ("uncertainty.png", "Decision uncertainty 1-|2P-1|"),
                                ("model_vs_ndwi.png", "Model vs NDWI: blue both, cyan model only, amber NDWI only"),
                                ("ground_truth.png", "Hand label (reference, not a model output)"),
                                ("errors.png", "Errors vs hand label: blue TP, red FP, amber FN")] if (folder / f).exists()]
    parts = [f"<h1>Flood segmentation report</h1><div class=meta>{_e(res['created'])} · generated by FloodRoute v2</div>",
             f"<div class=warn>{_e(DISCLAIMER)}</div><h2>Provenance and settings</h2><table>",
             *[f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in rows], "</table>"]
    if res.get("baseline"):
        b = res["baseline"]
        parts += ["<h2>Baseline comparison (same image)</h2><table>",
                  f"<tr><th>Baseline</th><td>{_e(b['name'])}</td></tr>",
                  f"<tr><th>Baseline water fraction</th><td>{b['water_fraction'] * 100:.2f}%</td></tr>",
                  f"<tr><th>Model–baseline agreement IoU</th><td>{_f(b['agreement_iou_with_model'])}</td></tr></table>"]
    if res.get("ground_truth"):
        g = res["ground_truth"]["metrics"]
        bg = (res.get("baseline") or {}).get("metrics_vs_ground_truth") or {}
        parts += ["<h2 class=ref>Agreement with hand label (this image only)</h2><table><tr><th>Metric</th><th>Model</th>"
                  "<th>NDWI baseline</th></tr>",
                  *[f"<tr><td>{k}</td><td class=n>{_f(g.get(k))}</td><td class=n>{_f(bg.get(k))}</td></tr>"
                    for k in ("iou_water", "f1_water", "precision_water", "recall_water", "pixel_accuracy")], "</table>"]
    if res.get("warnings"):
        parts += [f"<div class=warn>{_e(w)}</div>" for w in res["warnings"]]
    parts += ["<h2>Figures</h2><div class=grid>",
              *[f"<figure><img src='{_img(folder / f)}' alt='{_e(c)}'><figcaption>{_e(c)}</figcaption></figure>"
                for f, c in figs], "</div>"]
    return _page("Segmentation report " + res["id"][:8], "".join(parts))


def route_svg(r: dict, overlay_png: bytes | None, bounds: list[float]) -> str:
    l, b, rr, t = bounds
    W = 640
    H = max(200, int(W * (t - b) / max(1e-9, (rr - l))))

    def xy(lat, lon):
        return (lon - l) / (rr - l) * W, (t - lat) / (t - b) * H

    def path(coords):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(la, lo) for la, lo in coords))

    img = (f"<image href='data:image/png;base64,{base64.b64encode(overlay_png).decode()}' x='0' y='0' width='{W}' "
           f"height='{H}' preserveAspectRatio='none' opacity='0.8'/>") if overlay_png else ""
    lines = [f"<polyline points='{path(a['coordinates'])}' fill='none' stroke='#7c3aed' stroke-width='3' opacity='.8'/>"
             for a in r["alternatives"]]
    lines.append(f"<polyline points='{path(r['baseline']['coordinates'])}' fill='none' stroke='#0f172a' "
                 f"stroke-width='3' stroke-dasharray='7 6'/>")
    lines.append(f"<polyline points='{path(r['risk_aware']['coordinates'])}' fill='none' stroke='#059669' stroke-width='4'/>")
    s0, s1 = r["risk_aware"]["coordinates"][0], r["risk_aware"]["coordinates"][-1]
    (x0, y0), (x1, y1) = xy(*s0), xy(*s1)
    return (f"<svg viewBox='0 0 {W} {H}' width='100%' style='background:#eef2f6;border:1px solid #e2e8f0;border-radius:4px'>"
            f"{img}{''.join(lines)}<circle cx='{x0:.1f}' cy='{y0:.1f}' r='6' fill='#059669' stroke='#fff' stroke-width='2'/>"
            f"<circle cx='{x1:.1f}' cy='{y1:.1f}' r='6' fill='#dc2626' stroke='#fff' stroke-width='2'/></svg>"
            "<div class=meta>Blue: predicted water (flood layer) · green: risk-aware · dashed black: shortest · violet: "
            "alternatives · green dot start, red dot destination. Raster footprint shown; no basemap.</div>")


def route_report(r: dict, overlay_png: bytes | None, bounds: list[float] | None) -> str:
    routes = [("Shortest (baseline)", r["baseline"]), ("Risk-aware", r["risk_aware"]),
              *[(f"Alternative {i + 1}", a) for i, a in enumerate(r["alternatives"])]]
    cls = "danger" if r["status"] != "flood_free_route_found" else "warn"
    s = r["settings"]
    parts = [f"<h1>Flood-risk route report</h1><div class=meta>{_e(r.get('created'))} · area {_e(r.get('area_id'))} · "
             f"route id {_e(r['id'])}</div>", f"<div class=warn>{_e(DISCLAIMER)}</div>",
             f"<div class={cls}><b>Status: {_e(r['status'])}.</b> {_e(r['status_note'])}</div>",
             "<h2>Flood layer and assumptions</h2><table>",
             f"<tr><th>Flood layer</th><td>{_e(r['flood_layer']['description'])} "
             f"({'model prediction' if r['flood_layer']['is_model_prediction'] else 'hand label (reference)'})</td></tr>",
             f"<tr><th>Flood threshold / road buffer</th><td>{r['flood_layer']['threshold']} / {r['flood_layer']['buffer_px']} px</td></tr>",
             f"<tr><th>Penalties</th><td>flood {s['flood_penalty']}× · unknown {s['unknown_penalty']}× · probability "
             f"{s['prob_penalty']}× · closures {'on (≥ ' + str(s['closure_min_flooded_m']) + ' m flooded)' if s['close_flooded'] else 'off'}</td></tr>",
             f"<tr><th>Road data</th><td>{_e(r.get('attribution'))}; fetched {_e((r.get('road_meta') or {}).get('fetched_utc'))}</td></tr>",
             f"<tr><th>Travel time</th><td>{_e(r.get('travel_time_note'))}</td></tr></table>"]
    if bounds:
        parts += ["<h2>Map</h2>", route_svg(r, overlay_png, bounds)]
    parts += ["<h2>Route comparison</h2><table><tr><th>Route</th><th>Distance (m)</th><th>Est. time (s)</th>"
              "<th>Predicted-flooded (m)</th><th>Flooded share</th><th>Unknown (m)</th><th>Max P(water)</th><th>Risk level</th></tr>",
              *[f"<tr><td>{n}</td><td class=n>{x['distance_m']:.0f}</td><td class=n>{_e(x['travel_time_s'])}</td>"
                f"<td class=n>{x['flooded_length_m']:.0f}</td><td class=n>{x['flooded_share'] * 100:.1f}%</td>"
                f"<td class=n>{x['unknown_length_m']:.0f}</td><td class=n>{x['max_flood_prob']:.2f}</td><td>{_e(x['risk_level'])}</td></tr>"
                for n, x in routes], "</table>",
              "<h2>Risk-aware route: length by road risk class</h2><table>",
              *[f"<tr><td>{_e(k)}</td><td class=n>{v:.0f} m</td></tr>" for k, v in r["risk_aware"]["length_by_risk_class_m"].items()],
              "</table>"]
    if r["risk_aware"]["flood_affected_roads"]:
        parts += ["<h2>Flood-affected roads on the risk-aware route</h2><table><tr><th>Road</th><th>Predicted-flooded (m)</th></tr>",
                  *[f"<tr><td>{_e(a['road'])}</td><td class=n>{a['flooded_m']:.0f}</td></tr>"
                    for a in r["risk_aware"]["flood_affected_roads"]], "</table>"]
    parts += ["<h2>Explanation</h2><ul>", *[f"<li>{_e(e)}</li>" for e in r["explanation"]], "</ul>",
              *[f"<div class=warn>{_e(w)}</div>" for w in r.get("warnings", [])]]
    return _page("Route report " + r["id"][:8], "".join(parts))


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>{_e(title)}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
