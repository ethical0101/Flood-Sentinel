"""Guarded training start/stop for the dashboard.

Only predefined experiment presets can be launched (no user-supplied arguments reach a shell).
Start is refused unless: the request repeats the run name as confirmation, no training run is active,
no other project's job is using the GPU, and VRAM/disk headroom is sufficient. Runs are launched with
--polite so they yield to other GPU work. Stop is graceful: a STOP file is honoured after the
current epoch, with the checkpoint saved.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

from backend.app.config import REPO, settings
from backend.app.services.status import _pid_alive, _read, gpu_live

MIN_FREE_VRAM_MIB = 1500
MIN_FREE_DISK_GB = 5


class TrainingControlError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def presets() -> dict[str, list[str]]:
    spec = importlib.util.spec_from_file_location("run_experiments", REPO / "scripts" / "run_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {**mod.EXPERIMENTS, **mod.EXPERIMENTS_V2}


def active_runs() -> list[dict]:
    out = []
    root = settings.artifacts / "runs"
    for d in root.iterdir() if root.exists() else []:
        st = _read(d / "status.json") or {}
        lock = d / "train.lock"
        pid = 0
        if lock.exists():
            try:
                pid = int(lock.read_text().strip() or 0)
            except ValueError:
                pid = 0
        if (st.get("state") == "running" and _pid_alive(st.get("pid", 0))) or (pid and _pid_alive(pid)):
            out.append({"run": d.name, "pid": st.get("pid") or pid, "epoch": st.get("epoch"), "epochs": st.get("epochs")})
    return out


def preflight(run: str | None = None) -> dict:
    from training.gpu_guard import foreign_gpu_processes

    gpu = gpu_live()
    free_vram = gpu["memory_total_mib"] - gpu["memory_used_mib"] if gpu else 0
    disk_free = shutil.disk_usage(settings.artifacts.anchor or "/").free / 1e9
    foreign = foreign_gpu_processes()
    active = active_runs()
    checks = [
        {"check": "CUDA GPU available", "ok": gpu is not None, "detail": gpu["name"] if gpu else "NVML unavailable"},
        {"check": "No training run active", "ok": not active, "detail": ", ".join(a["run"] for a in active) or "idle"},
        {"check": "No other project using the GPU", "ok": not foreign,
         "detail": "; ".join(f"pid {f['pid']}" for f in foreign) or "none"},
        {"check": f"Free VRAM >= {MIN_FREE_VRAM_MIB} MiB", "ok": free_vram >= MIN_FREE_VRAM_MIB, "detail": f"{free_vram} MiB"},
        {"check": f"Free disk >= {MIN_FREE_DISK_GB} GB", "ok": disk_free >= MIN_FREE_DISK_GB, "detail": f"{disk_free:.1f} GB"},
    ]
    if run is not None:
        p = presets()
        st = (_read(settings.artifacts / "runs" / run / "status.json") or {}).get("state")
        checks.append({"check": "Known experiment preset", "ok": run in p, "detail": run})
        checks.append({"check": "Run not already completed", "ok": st != "completed", "detail": st or "new"})
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "active": active}


def start(run: str, confirm: str) -> dict:
    if confirm != run:
        raise TrainingControlError(400, "Confirmation does not match the run name.")
    pf = preflight(run)
    if not pf["ok"]:
        failed = [c["check"] for c in pf["checks"] if not c["ok"]]
        raise TrainingControlError(409, "Refused: " + "; ".join(failed))
    run_dir = settings.artifacts / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "training.train", "--run", run, "--polite"]
    if (run_dir / "checkpoints" / "last.pt").exists():
        cmd.append("--resume")
    else:
        for o in presets()[run]:
            cmd += ["--set", o]
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # survives API restarts
    with open(run_dir / "console.log", "ab") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=flags, close_fds=True)
    return {"started": True, "run": run, "pid": proc.pid, "resume": "--resume" in cmd,
            "command": " ".join(["python", *cmd[1:]])}


def stop(run: str) -> dict:
    if run not in {a["run"] for a in active_runs()}:
        raise TrainingControlError(409, "That run is not active.")
    (settings.artifacts / "runs" / run / "STOP").write_text("stop requested from dashboard")
    return {"stop_requested": True, "run": run,
            "note": "Training stops after the current epoch finishes and its checkpoint is saved."}
