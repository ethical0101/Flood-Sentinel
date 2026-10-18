"""System, dataset, training and metrics status (all read from real files / live probes)."""
from __future__ import annotations

import json
import os
import platform
import shutil
import time
from pathlib import Path

from backend.app.config import settings


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


_size_cache: dict[str, tuple[float, int]] = {}


def dir_size_cached(p: Path, ttl: float = 300) -> int:
    k = str(p)
    if k in _size_cache and time.time() - _size_cache[k][0] < ttl:
        return _size_cache[k][1]
    v = _dir_size(p)
    _size_cache[k] = (time.time(), v)
    return v


def gpu_live() -> dict | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        name = pynvml.nvmlDeviceGetName(h)
        out = {"name": name.decode() if isinstance(name, bytes) else name,
               "memory_total_mib": mem.total // 2**20, "memory_used_mib": mem.used // 2**20,
               "utilization_pct": util.gpu, "temperature_c": pynvml.nvmlDeviceGetTemperature(h, 0),
               "driver": pynvml.nvmlSystemGetDriverVersion()}
        try:
            out["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1)
        except pynvml.NVMLError:
            pass
        return out
    except Exception:  # noqa: BLE001 - no NVIDIA driver / NVML unavailable
        return None


def system() -> dict:
    import psutil
    import torch

    vm = psutil.virtual_memory()
    du = shutil.disk_usage(settings.artifacts.anchor or "/")
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": {"logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
                "utilization_pct": psutil.cpu_percent(interval=None)},
        "ram": {"total_gib": round(vm.total / 2**30, 1), "available_gib": round(vm.available / 2**30, 1),
                "used_pct": vm.percent},
        "disk": {"drive": settings.artifacts.anchor, "total_gb": round(du.total / 1e9, 1),
                 "free_gb": round(du.free / 1e9, 1)},
        "torch": {"version": torch.__version__, "cuda_build": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available()},
        "gpu": gpu_live(),
        "preflight": _read(settings.artifacts / "system_info.json"),
    }


def datasets() -> dict:
    ver = _read(settings.artifacts / "dataset_verification.json")
    raw = settings.raw / "sen1floods11"
    manifest = _read(raw / "manifest.json")
    index = _read(settings.processed / "index.json")
    splits: dict[str, dict] = {}
    if index:
        for c in index["chips"]:
            s = splits.setdefault(c["split"], {"chips": 0, "events": set(), "water_fraction_sum": 0.0})
            s["chips"] += 1
            s["events"].add(c["event"])
            s["water_fraction_sum"] += c["water_fraction"]
        for s in splits.values():
            s["events"] = sorted(s["events"])
            s["mean_water_fraction"] = round(s.pop("water_fraction_sum") / s["chips"], 4)
    stats = _read(settings.processed / "stats.json")
    return {
        "name": "Sen1Floods11 v1.1 - hand-labelled subset",
        "source": "gs://sen1floods11 (public Google Cloud Storage bucket, Cloud to Street)",
        "license": "No explicit licence published: the GitHub repo has no LICENSE file and the bucket's STAC "
                   "collections declare \"proprietary\". Publicly released by Cloud to Street with a citation "
                   "request; used here for non-commercial research and not redistributed (see docs/data_card.md).",
        "citation": "Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11: a georeferenced "
                    "dataset to train and test deep learning flood algorithms for Sentinel-1. CVPR Workshops, 210-211.",
        "downloaded": manifest is not None,
        "downloaded_utc": manifest and manifest.get("downloaded_utc"),
        "n_files": manifest and manifest.get("n_objects"),
        "download_bytes": manifest and manifest.get("total_bytes"),
        "raw_bytes_on_disk": dir_size_cached(raw),
        "processed_bytes_on_disk": dir_size_cached(settings.processed),
        "verification": ver and {k: ver.get(k) for k in ("passed", "integrity_mode", "files_checked", "files_ok",
                                                         "unique_chips", "crs", "label_pixel_counts", "errors")},
        "processed": index is not None,
        "splits": splits,
        "train_water_fraction": stats and stats.get("train_pos_fraction"),
        "layers": ["S1Hand (VV,VH dB)", "S2Hand (13 bands)", "LabelHand", "S1OtsuLabelHand", "JRCWaterHand"],
        "channels": [
            {"name": "VV", "sensor": "Sentinel-1 GRD", "unit": "dB", "note": "clipped to [-50, 10] before standardisation"},
            {"name": "VH", "sensor": "Sentinel-1 GRD", "unit": "dB", "note": "clipped to [-50, 10] before standardisation"},
            *[{"name": b, "sensor": "Sentinel-2 L1C", "unit": "TOA reflectance x10000",
               "note": "divided by 10000, standardised with train statistics"}
              for b in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]],
            {"name": "NDWI", "sensor": "derived (v2 '*i' models)", "unit": "ratio [-1, 1]", "note": "(B3-B8)/(B3+B8)"},
            {"name": "MNDWI", "sensor": "derived (v2 '*i' models)", "unit": "ratio [-1, 1]", "note": "(B3-B11)/(B3+B11)"}],
        "label_definitions": [{"value": -1, "meaning": "no data / not labelled", "use": "ignored in loss and metrics"},
                              {"value": 0, "meaning": "not water", "use": "negative class"},
                              {"value": 1, "meaning": "water (flood and permanent)", "use": "positive class"}],
        "events": _events(index),
        "storage": {"raw_sen1floods11": dir_size_cached(raw), "processed_chips": dir_size_cached(settings.processed),
                    "raw_worldfloods_test": dir_size_cached(settings.raw / "worldfloods_v2"),
                    "osm_cache": dir_size_cached(settings.raw.parent / "osm_cache"),
                    "checkpoints": sum(dir_size_cached(d / "checkpoints") for d in (settings.artifacts / "runs").iterdir())
                    if (settings.artifacts / "runs").exists() else 0,
                    "analysis_outputs": dir_size_cached(settings.outputs)},
        "quality_warnings": _quality_warnings(),
        "external": _worldfloods(),
    }


def _events(index: dict | None) -> list[dict]:
    if not index:
        return []
    meta = _read(settings.raw / "sen1floods11" / "v1.1" / "Sen1Floods11_Metadata.geojson") or {"features": []}
    by_loc = {f["properties"].get("location"): f["properties"] for f in meta["features"]}
    ev: dict[str, dict] = {}
    for c in index["chips"]:
        e = ev.setdefault(c["event"], {"event": c["event"], "chips": {}, "lat": [], "lon": [], "water": []})
        e["chips"][c["split"]] = e["chips"].get(c["split"], 0) + 1
        l, b, r, t = c["bounds"]
        e["lat"].append((b + t) / 2)
        e["lon"].append((l + r) / 2)
        e["water"].append(c["water_fraction"])
    out = []
    for name, e in sorted(ev.items()):
        p = by_loc.get(name) or by_loc.get(name.replace("Mekong", "Cambodia")) or {}
        out.append({"event": name, "chips": e["chips"], "lat": round(sum(e["lat"]) / len(e["lat"]), 3),
                    "lon": round(sum(e["lon"]) / len(e["lon"]), 3), "mean_water_fraction": round(sum(e["water"]) / len(e["water"]), 4),
                    "s1_date": p.get("s1_date"), "s2_date": p.get("s2_date")})
    return out


def _quality_warnings() -> list[str]:
    out = []
    a = _read(settings.artifacts / "audit" / "reproduction.json")
    if a:
        t = a["geographic_proximity_to_train"]["test"]
        out.append(f"Test split is spatially close to training: {t['share_within_10km'] * 100:.0f}% of test chips lie "
                   f"within 10 km of a training chip and {t['chips_whose_footprint_overlaps_a_train_chip']} overlap one.")
    w = _read(settings.artifacts / "audit" / "weakness_analysis.json")
    if w:
        b = w["data"]["bolivia"]
        out.append(f"Bolivia has {b['label_share']['invalid'] * 100:.0f}% unlabelled pixels, "
                   f"{b['s1_nan_pixel_share'] * 100:.0f}% Sentinel-1 no-data and "
                   f"{b['s2_nodata_pixel_share'] * 100:.0f}% Sentinel-2 no-data (training: about 2%).")
        out.append(f"Normalisation statistics include {w['data']['stats_contamination']['train_s2_nodata_pixel_share'] * 100:.1f}% "
                   "Sentinel-2 no-data pixels (minor; unchanged for comparability).")
    out.append("Labels mark all surface water (flood + permanent); they do not separate flood water.")
    return out


def _worldfloods() -> dict | None:
    m = _read(settings.raw / "worldfloods_v2" / "manifest.json")
    if not m:
        return None
    return {"name": "WorldFloods v2 - test split only", "use": "external evaluation only (never training or selection)",
            "licence": m.get("licence"), "citation": m.get("citation"), "downloaded_utc": m.get("downloaded_utc"),
            "n_files": m.get("n_files"), "bytes": m.get("total_bytes"), "n_failed": m.get("n_failed"),
            "integrity": "SHA-256 of every LFS file verified at download",
            "scenes": len([k for k in m.get("files", {}) if k.startswith("test/S2/")]),
            "channels": "Sentinel-2 L1C, 13 bands (+ QA60 and cloud probability, unused); no Sentinel-1",
            "labels": "water band: 0 invalid, 1 land, 2 water; cloud band: 0 invalid, 1 clear, 2 cloud"}


def training() -> dict:
    runs_root = settings.artifacts / "runs"
    runs = []
    if runs_root.exists():
        for d in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime):
            st = _read(d / "status.json")
            if not st or d.name.startswith("smoke_"):
                continue
            if st.get("state") == "running" and not _pid_alive(st.get("pid", 0)):
                st["state"] = "stalled"
                st["note"] = "status says running but the training process is gone; resume it"
            ck = d / "checkpoints"
            runs.append({"name": d.name, **st,
                         "history": _read(d / "history.json") or [],
                         "checkpoints": [{"file": f.name, "mb": round(f.stat().st_size / 2**20, 1),
                                          "modified": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                                    time.localtime(f.stat().st_mtime))}
                                         for f in sorted(ck.glob("*.pt"))] if ck.exists() else [],
                         "resume_command": f".venv\\Scripts\\python -m training.train --resume --run {d.name}"})
    return {"runs": runs, "gpu": gpu_live()}


def training_log(run: str, lines: int = 200) -> list[str] | None:
    d = settings.artifacts / "runs" / run
    if not d.is_dir() or d.parent != settings.artifacts / "runs":
        return None
    p = d / "train.log"
    if not p.exists():
        return []
    return p.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def metrics() -> dict:
    runs_root = settings.artifacts / "runs"
    evals = {}
    if runs_root.exists():
        for d in sorted(runs_root.iterdir()):
            ev = _read(d / "evaluation.json")
            if ev and not d.name.startswith("smoke_"):
                ev["history"] = _read(d / "history.json") or []
                ev["sample_images"] = sorted(f.name for f in (d / "samples").glob("*.png")) if (d / "samples").exists() else []
                evals[d.name] = ev
    return {"summary": _read(settings.artifacts / "summary.json"),
            "baselines": _read(settings.artifacts / "baselines.json"), "runs": evals,
            "routing": _read(settings.artifacts / "routing_evaluation.json")}


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower()
    except Exception:  # noqa: BLE001
        return False


def sample_image(run: str, name: str) -> Path | None:
    d = settings.artifacts / "runs" / run
    if d.parent != settings.artifacts / "runs" or not name.endswith(".png") or "/" in name or "\\" in name \
            or ".." in name:
        return None
    for p in (d / "samples" / name, d / name):
        if p.exists() and p.parent.resolve().is_relative_to(d.resolve()):
            return p
    return None


def env_summary() -> dict:
    return {"FLOOD_DEVICE": settings.device, "osm_download_allowed": settings.allow_osm_download,
            "max_upload_mb": settings.max_upload_mb, "max_image_pixels": settings.max_image_pixels,
            "pid": os.getpid()}
