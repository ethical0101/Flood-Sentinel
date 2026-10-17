"""Associate a georeferenced flood-probability raster with road-network edges.

Alignment
  * The raster carries its own CRS + affine transform (GeoTIFF / Sen1Floods11 index).
  * Road geometries come from OSMnx in EPSG:4326 (lon/lat).
  * Points are transformed to the raster CRS with pyproj when the CRSs differ; then the
    inverse affine transform maps them to (row, col). Points outside the raster or on
    no-data pixels are "unknown", never silently treated as dry.

Sampling
  * Every edge is densified to one sample per ``step_m`` metres (default 5 m = half a
    10 m pixel) along its true geometry.
  * Optionally the probability raster is max-filtered with a (2r+1)^2 window first
    (``buffer_px``), a conservative allowance for road width and for ~1 pixel
    co-registration error between OSM and the imagery.

Edge attributes written
  flood_prob_max, flood_prob_mean, flooded_frac, flooded_len_m, coverage, risk_class
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely.geometry import LineString

RISK_CLASSES = ("clear", "low", "partial", "flooded", "unknown")


@dataclass
class FloodLayer:
    """A probability-of-water raster plus georeferencing and provenance."""

    prob: np.ndarray            # (H,W) float32 in [0,1]; NaN = no data
    crs: str
    transform: tuple            # affine a,b,c,d,e,f
    source: str                 # "model_prediction" | "ground_truth_label" | "uploaded_prediction"
    description: str

    def __post_init__(self):
        if self.prob.ndim != 2:
            raise ValueError("flood layer must be a 2-D raster")
        try:
            CRS.from_user_input(self.crs)
        except Exception:  # noqa: BLE001 - pyproj raises CRSError for unparseable input
            raise ValueError(f"invalid CRS {self.crs!r}") from None
        self.affine = Affine(*self.transform)
        if self.affine.is_degenerate:
            raise ValueError("degenerate geotransform")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.prob.shape
        xs, ys = zip(*[self.affine @ (c, r) for c, r in ((0, 0), (w, 0), (0, h), (w, h))])
        return min(xs), min(ys), max(xs), max(ys)

    def bounds_wgs84(self) -> tuple[float, float, float, float]:
        l, b, r, t = self.bounds
        if CRS.from_user_input(self.crs).to_epsg() == 4326:
            return l, b, r, t
        tr = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        xs, ys = tr.transform([l, r, l, r], [b, b, t, t])
        return min(xs), min(ys), max(xs), max(ys)


def check_alignment(layer: FloodLayer, graph_bbox_wgs84: tuple[float, float, float, float]) -> dict:
    """Overlap between the road-graph bbox and the raster footprint (both in lon/lat)."""
    l1, b1, r1, t1 = layer.bounds_wgs84()
    l2, b2, r2, t2 = graph_bbox_wgs84
    iw, ih = max(0.0, min(r1, r2) - max(l1, l2)), max(0.0, min(t1, t2) - max(b1, b2))
    g_area = max((r2 - l2) * (t2 - b2), 1e-12)
    return {"raster_bounds_wgs84": [l1, b1, r1, t1], "graph_bounds_wgs84": [l2, b2, r2, t2],
            "graph_fraction_covered": round(iw * ih / g_area, 4), "raster_crs": layer.crs,
            "graph_crs": "EPSG:4326", "ok": iw * ih > 0}


def _max_filter(prob: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return prob
    from scipy.ndimage import maximum_filter

    filled = np.where(np.isfinite(prob), prob, -1.0)
    out = maximum_filter(filled, size=2 * r + 1, mode="nearest")
    out = np.where(out < 0, np.nan, out)
    out[~np.isfinite(prob)] = np.nan
    return out.astype(np.float32)


def sample_lines(layer: FloodLayer, lines: list[LineString], lengths_m: list[float], step_m: float = 5.0,
                 buffer_px: int = 1) -> list[np.ndarray]:
    """Return, for each line, the sampled probabilities (NaN where outside / no data)."""
    prob = _max_filter(layer.prob, buffer_px)
    h, w = prob.shape
    pts_per = [max(2, int(np.ceil(L / step_m)) + 1) for L in lengths_m]
    lon, lat = [], []
    for ln, n in zip(lines, pts_per):
        d = np.linspace(0, 1, n)
        coords = [ln.interpolate(t, normalized=True).coords[0] for t in d]
        lon.extend(c[0] for c in coords)
        lat.extend(c[1] for c in coords)
    x, y = np.asarray(lon), np.asarray(lat)
    if CRS.from_user_input(layer.crs).to_epsg() != 4326:
        x, y = Transformer.from_crs("EPSG:4326", layer.crs, always_xy=True).transform(x, y)
    inv = ~layer.affine
    cols, rows = inv @ (x, y)
    rows, cols = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    vals = np.full(x.shape, np.nan, np.float32)
    vals[inside] = prob[rows[inside], cols[inside]]
    out, i = [], 0
    for n in pts_per:
        out.append(vals[i:i + n])
        i += n
    return out


def classify(flooded_frac: float, coverage: float, max_prob: float) -> str:
    if coverage < 0.5:
        return "unknown"
    if flooded_frac >= 0.5:
        return "flooded"
    if flooded_frac > 0:
        return "partial"
    if max_prob >= 0.25:
        return "low"
    return "clear"


def annotate_graph(G, layer: FloodLayer, threshold: float = 0.5, step_m: float = 5.0, buffer_px: int = 1) -> dict:
    """Write flood attributes onto every edge of an OSMnx MultiDiGraph (in place)."""
    edges, lines, lengths = [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        geom = d.get("geometry") or LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                                                (G.nodes[v]["x"], G.nodes[v]["y"])])
        edges.append((u, v, k))
        lines.append(geom)
        lengths.append(float(d.get("length", 0.0)) or 1.0)
    samples = sample_lines(layer, lines, lengths, step_m, buffer_px)
    counts = dict.fromkeys(RISK_CLASSES, 0)
    flooded_total = 0.0
    for (u, v, k), s, L in zip(edges, samples, lengths):
        fin = s[np.isfinite(s)]
        cov = fin.size / s.size
        frac = float((fin >= threshold).mean()) if fin.size else 0.0
        mx = float(fin.max()) if fin.size else float("nan")
        d = G.edges[u, v, k]
        d["flood_prob_max"] = mx
        d["flood_prob_mean"] = float(fin.mean()) if fin.size else float("nan")
        d["flooded_frac"] = frac
        d["flooded_len_m"] = frac * L * cov
        d["coverage"] = cov
        d["risk_class"] = classify(frac, cov, mx if fin.size else 0.0)
        counts[d["risk_class"]] += 1
        flooded_total += d["flooded_len_m"]
    return {"edges": len(edges), "risk_class_counts": counts, "flooded_road_km": round(flooded_total / 1000, 3),
            "threshold": threshold, "step_m": step_m, "buffer_px": buffer_px}
