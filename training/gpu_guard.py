"""GPU courtesy: detect GPU compute processes that do not belong to this project.

Used so that long training runs yield the (single, 4 GB) GPU to other work on the machine
instead of competing with it. Foreign processes are only *observed*, never touched.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1]).lower()


def foreign_gpu_processes() -> list[dict]:
    """Compute processes on GPU 0 whose command line is not inside this repository."""
    try:
        import psutil
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h) + pynvml.nvmlDeviceGetGraphicsRunningProcesses(h)
    except Exception:  # noqa: BLE001 - no NVML: nothing to report
        return []
    me = os.getpid()
    out = []
    for p in procs:
        if p.pid in (me, 0):
            continue
        try:
            proc = psutil.Process(p.pid)
            cmd = " ".join(proc.cmdline()).lower()
            name = proc.name()
        except Exception:  # noqa: BLE001 - access denied / exited
            continue
        if "python" not in name.lower():
            continue  # desktop apps (browsers, compositors) hold small graphics contexts; ignore them
        if REPO in cmd:
            continue
        out.append({"pid": p.pid, "name": name, "cmd": cmd[-160:]})
    return out
