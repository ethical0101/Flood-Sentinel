"""System diagnostics, version information and data-quality / operational warnings.

Every warning is derived from real state (files on disk, NVML, psutil, request log); nothing is simulated.
"""
from __future__ import annotations

import datetime as dt
import os
import platform
import shutil
import sqlite3
import time
from collections import deque
from pathlib import Path

from backend.app.config import REPO, settings
from backend.app.services.status import _read

APP_VERSION = "2.0.0"
_errors: deque = deque(maxlen=200)  # (timestamp, method, path, status)


def record_error(method: str, path: str, status: int) -> None:
    _errors.append((time.time(), method, path, status))


def git_info() -> dict:
    head = REPO / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            name = ref.split("/", 2)[-1]
            sha_file = REPO / ".git" / ref[5:]
            sha = sha_file.read_text().strip() if sha_file.exists() else None
            return {"branch": name, "commit": sha[:7] if sha else None}
        return {"branch": None, "commit": ref[:7]}
    except OSError:
        return {"branch": None, "commit": None}


def versions() -> dict:
    import fastapi
    import numpy
    import torch

    out = {"app": APP_VERSION, **git_info(), "python": platform.python_version(), "torch": torch.__version__,
           "cuda_build": torch.version.cuda, "numpy": numpy.__version__, "fastapi": fastapi.__version__}
    for mod in ("segmentation_models_pytorch", "rasterio", "osmnx", "networkx", "pyproj"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def _check(name: str, ok: bool | None, detail: str, category: str, hint: str | None = None) -> dict:
    return {"check": name, "status": "ok" if ok else ("warn" if ok is None else "fail"), "detail": detail,
            "category": category, "hint": hint}


def run_diagnostics() -> dict:
    import psutil
    import torch

    from backend.app.routing import graph_store
    from backend.app.services import areas
    from backend.app.services.models import registry
    from backend.app.services.status import gpu_live
    from training.gpu_guard import foreign_gpu_processes

    t0 = time.perf_counter()
    c = []
    c.append(_check("PyTorch CUDA", torch.cuda.is_available(),
                    f"torch {torch.__version__}, CUDA build {torch.version.cuda}", "compute",
                    "Install the +cu130 wheel; see docs/setup_guide.md"))
    g = gpu_live()
    c.append(_check("NVIDIA telemetry (NVML)", g is not None, g["name"] if g else "unavailable", "compute"))
    foreign = foreign_gpu_processes()
    c.append(_check("GPU shared with other projects", None if foreign else True,
                    f"{len(foreign)} foreign GPU job(s)" if foreign else "none", "compute",
                    "Training runs in polite mode and waits; inference still works"))
    models = registry.available()
    done = [m for m in models if m["training_state"] == "completed"]
    c.append(_check("Trained models available", bool(done), f"{len(done)} completed, {len(models)} total", "models"))
    c.append(_check("Default model present", any(m["name"] == settings.default_model for m in done),
                    settings.default_model, "models"))
    unevaluated = [m["name"] for m in done if not m["evaluated"]]
    c.append(_check("All completed models evaluated", None if unevaluated else True,
                    ", ".join(unevaluated) or "yes", "models", "python -m training.evaluate --all"))
    ver = _read(settings.artifacts / "dataset_verification.json")
    c.append(_check("Sen1Floods11 integrity", bool(ver and ver.get("passed")),
                    f"{ver['files_ok']}/{ver['files_checked']} files ({ver['integrity_mode']})" if ver else "not verified",
                    "data", "python scripts/verify_datasets.py"))
    c.append(_check("Processed chips", (settings.processed / "index.json").exists(), str(settings.processed.name), "data",
                    "python -m training.prepare_data"))
    wf = _read(settings.raw / "worldfloods_v2" / "manifest.json")
    c.append(_check("WorldFloods v2 test split", None if not wf else wf.get("n_failed") == 0,
                    f"{wf['n_files']} files" if wf else "not downloaded (optional, evaluation only)", "data",
                    "python scripts/download_worldfloods_test.py"))
    cached = [a["id"] for a in areas.curated() if graph_store.cached(a["id"])]
    c.append(_check("OSM road graphs cached (offline routing)", len(cached) == len(areas.curated()),
                    f"{len(cached)}/{len(areas.curated())} study areas", "routing", "python scripts/fetch_road_networks.py"))
    try:
        with sqlite3.connect(settings.db_path, timeout=3) as con:
            con.execute("SELECT 1")
        c.append(_check("History database", True, Path(settings.db_path).name, "storage"))
    except sqlite3.Error as e:
        c.append(_check("History database", False, type(e).__name__, "storage"))
    try:
        p = settings.outputs / ".write_test"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ok")
        p.unlink()
        c.append(_check("Outputs folder writable", True, "outputs/", "storage"))
    except OSError:
        c.append(_check("Outputs folder writable", False, "outputs/", "storage"))
    du = shutil.disk_usage(settings.artifacts.anchor or "/")
    c.append(_check("Free disk", du.free / 1e9 > 10, f"{du.free / 1e9:.1f} GB on {settings.artifacts.anchor}", "storage"))
    vm = psutil.virtual_memory()
    c.append(_check("Free RAM", None if vm.available / 2**30 < 1.5 else True, f"{vm.available / 2**30:.1f} GiB available",
                    "compute", "Close other applications before training"))
    c.append(_check("Frontend production build", settings.frontend_dist.exists(), "frontend/dist", "app",
                    "cd frontend && npm run build"))
    c.append(_check("OSM downloads allowed", settings.allow_osm_download, str(settings.allow_osm_download), "routing"))
    summary = {s: sum(x["status"] == s for x in c) for s in ("ok", "warn", "fail")}
    return {"checked_at": dt.datetime.now().isoformat(timespec="seconds"), "duration_ms": round((time.perf_counter() - t0) * 1e3),
            "summary": summary, "checks": c, "versions": versions(),
            "offline": {
                "works_offline": ["segmentation inference (local checkpoints)", "evaluation pages (local artifacts)",
                                  "routing in cached study areas", "history and reports"],
                "needs_network": ["OSM basemap tiles (optional layer)", "road networks for new / uploaded areas (Overpass)",
                                  "initial dataset downloads"]}}


def warnings() -> list[dict]:
    import psutil

    from backend.app.routing import graph_store
    from backend.app.services import areas
    from backend.app.services.status import _pid_alive
    from training.gpu_guard import foreign_gpu_processes

    now = dt.datetime.now().isoformat(timespec="seconds")
    w = []

    def add(sev, src, msg):
        w.append({"severity": sev, "source": src, "message": msg, "observed": now})

    ver = _read(settings.artifacts / "dataset_verification.json")
    if not ver:
        add("warning", "data", "Sen1Floods11 has not been verified (run scripts/verify_datasets.py).")
    elif not ver.get("passed"):
        add("critical", "data", f"Dataset verification failed: {len(ver.get('errors', []))} error(s).")
    audit = _read(settings.artifacts / "audit" / "reproduction.json")
    if audit:
        t = audit["geographic_proximity_to_train"]["test"]
        add("info", "data quality", f"Sen1Floods11 test split is spatially close to training "
                                    f"({t['share_within_10km'] * 100:.0f}% of test chips within 10 km; "
                                    f"{t['chips_whose_footprint_overlaps_a_train_chip']} overlap). Use Bolivia / "
                                    f"WorldFloods for generalisation claims.")
    root = settings.artifacts / "runs"
    for d in sorted(root.iterdir()) if root.exists() else []:
        st = _read(d / "status.json") or {}
        s = st.get("state")
        if d.name.startswith("smoke_"):
            continue
        if s == "running" and not _pid_alive(st.get("pid", 0)):
            add("warning", "training", f"Run {d.name} stopped unexpectedly at epoch {st.get('epoch')}; resume it.")
        elif s == "paused":
            add("info", "training", f"Run {d.name} is paused at epoch {st.get('epoch')} ({st.get('note', '')}).")
        elif s == "failed":
            add("critical", "training", f"Run {d.name} failed: {st.get('error', 'see train.log')}")
    foreign = foreign_gpu_processes()
    if foreign:
        add("info", "gpu", f"GPU shared with {len(foreign)} job(s) from another project (pid {foreign[0]['pid']}); "
                           "training yields, timings may be slower.")
    vm = psutil.virtual_memory()
    if vm.available / 2**30 < 1.0:
        add("warning", "system", f"Low free RAM: {vm.available / 2**30:.2f} GiB available.")
    du = shutil.disk_usage(settings.artifacts.anchor or "/")
    if du.free / 1e9 < 10:
        add("warning", "system", f"Low disk space: {du.free / 1e9:.1f} GB free.")
    missing = [a["id"] for a in areas.curated() if not graph_store.cached(a["id"])]
    if missing:
        add("warning", "routing", f"Road graph not cached for: {', '.join(missing)} (needs network on first use).")
    recent = [e for e in _errors if time.time() - e[0] < 3600]
    if any(e[3] >= 500 for e in recent):
        add("critical", "api", f"{sum(e[3] >= 500 for e in recent)} server error(s) in the last hour.")
    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(w, key=lambda x: order[x["severity"]])


def gpu_processes_summary() -> dict:
    from training.gpu_guard import foreign_gpu_processes

    return {"foreign": foreign_gpu_processes(), "pid": os.getpid()}
