"""Inference: tiled sliding-window prediction + input readers for GeoTIFF / PNG / JPEG.

CLI (writes a probability GeoTIFF + PNG overlay next to the output prefix):
    python -m training.infer --run unet_resnet34_s1s2 --input chip_S1S2.tif --out outputs/pred
    python -m training.infer --run unet_resnet34_rgb  --input photo.png      --out outputs/pred

Supported inputs and how they are interpreted
    GeoTIFF, 2 bands      Sentinel-1 VV,VH. Values are expected in dB (Sen1Floods11 convention);
                          strictly-positive inputs with max < 10 are treated as linear power -> dB.
    GeoTIFF, 13 bands     Sentinel-2 L1C, bands B1..B12 incl. B8A (Sen1Floods11 order); values > 2
                          are assumed to be reflectance x 10000.
    GeoTIFF, 15 bands     S1 (2) + S2 (13) stacked in that order.
    3/4-band image        RGB (alpha ignored). PNG/JPEG carry no georeferencing.
Georeferencing (CRS + affine transform) is carried through only for GeoTIFF inputs.
"""
from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from training.dataset import MODALITIES, normalize, rgb_stretch
from training.model import load_for_inference
from training.utils import processed_dir, read_json, runs_dir


# ----------------------------------------------------------------------------- inputs
@dataclass
class SceneInput:
    kind: str                       # s1 | s2 | s1s2 | rgb
    s1: np.ndarray | None = None    # (2,H,W) dB
    s2: np.ndarray | None = None    # (13,H,W) reflectance 0-1
    rgb: np.ndarray | None = None   # (3,H,W) uint8 (already stretched)
    crs: str | None = None
    transform: tuple | None = None  # affine (a,b,c,d,e,f)
    bounds: tuple | None = None     # left,bottom,right,top in CRS units
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        a = next(v for v in (self.s1, self.s2, self.rgb) if v is not None)
        return a.shape[-2], a.shape[-1]

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None and self.transform is not None

    def available_modalities(self) -> list[str]:
        return {"s1s2": ["s1s2", "s1s2i", "s2", "s2i", "s1", "rgb"], "s2": ["s2", "s2i", "rgb"], "s1": ["s1"],
                "rgb": ["rgb"]}[self.kind]

    def preview_rgb(self) -> np.ndarray:
        """(H,W,3) uint8 for display. S1-only scenes use a VV/VH false-colour composite."""
        if self.rgb is not None:
            return np.transpose(self.rgb, (1, 2, 0))
        if self.s2 is not None:
            return np.transpose(rgb_stretch(self.s2[[3, 2, 1]]), (1, 2, 0))
        vv, vh = self.s1[0], self.s1[1]
        return np.transpose(rgb_stretch(np.stack([vv, vh, vv - vh])), (1, 2, 0))


def _s1_db(a: np.ndarray, notes: list[str]) -> np.ndarray:
    a = a.astype(np.float32)
    fin = a[np.isfinite(a)]
    if fin.size and fin.min() > 0 and fin.max() < 10:
        notes.append("S1 values looked like linear backscatter; converted to dB")
        a = 10 * np.log10(np.clip(a, 1e-6, None))
    return a


def _s2_refl(a: np.ndarray, notes: list[str]) -> np.ndarray:
    a = a.astype(np.float32)
    fin = a[np.isfinite(a)]
    if fin.size and np.nanpercentile(fin, 99) > 2:
        a = a / 10000.0
        notes.append("S2 values scaled by 1/10000 to reflectance")
    return a


def from_array(arr: np.ndarray, crs=None, transform=None, bounds=None) -> SceneInput:
    """arr: (C,H,W). Interprets channel count as documented in the module docstring."""
    c = arr.shape[0]
    notes: list[str] = []
    if c == 2:
        return SceneInput("s1", s1=_s1_db(arr, notes), crs=crs, transform=transform, bounds=bounds, notes=notes)
    if c == 13:
        return SceneInput("s2", s2=_s2_refl(arr, notes), crs=crs, transform=transform, bounds=bounds, notes=notes)
    if c == 15:
        return SceneInput("s1s2", s1=_s1_db(arr[:2], notes), s2=_s2_refl(arr[2:], notes), crs=crs,
                          transform=transform, bounds=bounds, notes=notes)
    if c in (3, 4):
        rgb = arr[:3]
        notes.append("RGB input: per-image 2-98% contrast stretch applied (same as training)")
        return SceneInput("rgb", rgb=rgb_stretch(rgb), crs=crs, transform=transform, bounds=bounds, notes=notes)
    if c == 1:
        raise ValueError("single-band images are not supported (need 2 SAR bands, 13/15 bands, or RGB)")
    raise ValueError(f"unsupported band count {c}; expected 2 (S1), 13 (S2), 15 (S1+S2) or 3 (RGB)")


def read_input(data: bytes, filename: str) -> SceneInput:
    name = filename.lower()
    if name.endswith((".tif", ".tiff")):
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(data) as mf, mf.open() as ds:
            arr = ds.read()
            if ds.nodata is not None:
                arr = arr.astype(np.float32)
                arr[arr == ds.nodata] = np.nan
            has_geo = ds.crs is not None and ds.transform != rasterio.Affine.identity()
            return from_array(arr, ds.crs.to_string() if has_geo else None,
                              tuple(ds.transform)[:6] if has_geo else None,
                              tuple(ds.bounds) if has_geo else None)
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    return from_array(np.transpose(np.asarray(img), (2, 0, 1)))


# ----------------------------------------------------------------------------- predictor
class Predictor:
    """Holds one trained model in memory and runs tiled inference."""

    def __init__(self, run: str | Path, device: str = "auto", checkpoint: str = "best.pt"):
        run_dir = Path(run) if Path(run).is_dir() else runs_dir() / str(run)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, ck = load_for_inference(run_dir / "checkpoints" / checkpoint, self.device)
        self.run_name = run_dir.name
        self.config = ck["config"]
        self.meta = ck["meta"]
        self.epoch = ck["epoch"]
        self.modality = self.config["data"]["modality"]
        stats_path = run_dir / "stats.json"
        self.stats = read_json(stats_path if stats_path.exists() else processed_dir() / "stats.json")
        self.amp = self.device.type == "cuda"

    def prepare(self, scene: SceneInput) -> np.ndarray:
        if self.modality not in scene.available_modalities():
            raise ValueError(f"model {self.run_name} needs {self.modality!r} input; "
                             f"this image provides {scene.available_modalities()}")
        rgb = scene.rgb
        if self.modality == "rgb" and rgb is None:
            rgb = rgb_stretch(scene.s2[[3, 2, 1]])
        return normalize(self.modality, s1=scene.s1, s2=scene.s2, rgb=rgb, stats=self.stats)

    @torch.no_grad()
    def predict(self, x: np.ndarray, tile: int = 512, overlap: int = 64, tta: bool = False) -> np.ndarray:
        """x: normalised (C,H,W) -> P(water) as float32 (H,W). Tiles are blended with a ramp window."""
        c, h, w = x.shape
        tile = min(tile, max(32, int(np.ceil(max(h, w) / 32) * 32)))
        stride = max(32, tile - overlap)
        ph = max(tile, int(np.ceil((h - tile) / stride)) * stride + tile) if h > tile else tile
        pw = max(tile, int(np.ceil((w - tile) / stride)) * stride + tile) if w > tile else tile
        xp = np.zeros((c, ph, pw), dtype=np.float32)
        xp[:, :h, :w] = x
        ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype(np.float32)
        win = np.minimum(np.outer(ramp, ramp) / max(1, overlap // 2), 1.0) if overlap else np.ones((tile, tile), np.float32)
        acc = np.zeros((ph, pw), dtype=np.float32)
        wsum = np.zeros((ph, pw), dtype=np.float32)
        for r in range(0, ph - tile + 1, stride):
            for col in range(0, pw - tile + 1, stride):
                t = torch.from_numpy(xp[None, :, r:r + tile, col:col + tile]).to(self.device)
                p = self._forward(t, tta)[0]
                acc[r:r + tile, col:col + tile] += p * win
                wsum[r:r + tile, col:col + tile] += win
        return (acc / np.maximum(wsum, 1e-6))[:h, :w]

    def _forward(self, t: torch.Tensor, tta: bool) -> np.ndarray:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            if not tta:
                return torch.sigmoid(self.model(t).float())[:, 0].cpu().numpy()
            outs = []
            for dims in ((), (-1,), (-2,), (-1, -2)):
                tt = torch.flip(t, dims) if dims else t
                o = torch.sigmoid(self.model(tt).float())
                outs.append(torch.flip(o, dims) if dims else o)
            return torch.stack(outs).mean(0)[:, 0].cpu().numpy()

    def run(self, scene: SceneInput, tta: bool = False) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        x = self.prepare(scene)
        t1 = time.perf_counter()
        prob = self.predict(x, tta=tta)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        return prob, {"preprocess_ms": round((t1 - t0) * 1e3, 1), "model_ms": round((t2 - t1) * 1e3, 1),
                      "device": str(self.device), "tta": tta}


# ----------------------------------------------------------------------------- outputs
def overlay(rgb: np.ndarray, mask: np.ndarray, color=(0, 140, 255), alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def prob_colormap(prob: np.ndarray) -> np.ndarray:
    """Probability -> RGBA uint8 (transparent below 0.05) using matplotlib's 'viridis'."""
    import matplotlib

    rgba = (matplotlib.colormaps["viridis"](prob) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(prob < 0.05, 0, (80 + 175 * prob)).astype(np.uint8)
    return rgba


def write_geotiff(path: Path, prob: np.ndarray, scene: SceneInput) -> None:
    import rasterio
    from rasterio.transform import Affine

    profile = dict(driver="GTiff", height=prob.shape[0], width=prob.shape[1], count=1, dtype="float32",
                   compress="deflate")
    if scene.georeferenced:
        profile.update(crs=scene.crs, transform=Affine(*scene.transform))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(prob.astype(np.float32), 1)
        dst.set_band_description(1, "P(water)")


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()
    pred = Predictor(a.run)
    scene = read_input(Path(a.input).read_bytes(), a.input)
    prob, timing = pred.run(scene, tta=a.tta)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(out.with_suffix(".prob.tif"), prob, scene)
    Image.fromarray(overlay(scene.preview_rgb(), prob > a.threshold)).save(out.with_suffix(".overlay.png"))
    Image.fromarray(((prob > a.threshold) * 255).astype(np.uint8)).save(out.with_suffix(".mask.png"))
    print({"run": pred.run_name, "modality": pred.modality, "shape": scene.shape,
           "water_fraction": float((prob > a.threshold).mean()), **timing, "notes": scene.notes})


if __name__ == "__main__":
    main()
