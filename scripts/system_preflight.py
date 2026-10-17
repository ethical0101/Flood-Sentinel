"""Collect verified hardware/software facts and write docs/system_preflight.md.

Run with the project interpreter so the PyTorch/CUDA section reflects the venv:
    .venv\\Scripts\\python scripts\\system_preflight.py
"""
from __future__ import annotations

import ctypes
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def ram_gb() -> tuple[float, float]:
    if os.name != "nt":
        return float("nan"), float("nan")

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30


def gpu() -> dict:
    q = "name,memory.total,memory.used,memory.free,driver_version,utilization.gpu,temperature.gpu"
    out = run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if not out:
        return {}
    v = [s.strip() for s in out.splitlines()[0].split(",")]
    return dict(zip(["name", "mem_total_mib", "mem_used_mib", "mem_free_mib", "driver", "util_pct", "temp_c"], v))


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    d = {"installed": True, "version": torch.__version__, "cuda_build": torch.version.cuda,
         "cuda_available": torch.cuda.is_available(), "cudnn": torch.backends.cudnn.version()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d.update(device=p.name, vram_gib=round(p.total_memory / 2**30, 2), capability=f"{p.major}.{p.minor}",
                 bf16=torch.cuda.is_bf16_supported())
    return d


def disks() -> list[dict]:
    out = []
    for letter in "CDEFGH":
        root = f"{letter}:\\"
        if os.path.exists(root):
            u = shutil.disk_usage(root)
            out.append({"drive": letter, "total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1)})
    return out


def main() -> None:
    total, avail = ram_gb()
    info = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ram_total_gib": round(total, 1),
        "ram_available_gib": round(avail, 1),
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "gpu": gpu(),
        "torch": torch_info(),
        "disks": disks(),
        "project_drive": str(REPO.drive),
        "data_dir": os.environ.get("FLOOD_DATA_DIR", str(REPO / "data" / "raw")),
        "tools": {
            "git": run(["git", "--version"]),
            "node": run(["node", "--version"]),
            "npm": run(["npm.cmd" if os.name == "nt" else "npm", "--version"]),
            "nvcc": run(["nvcc", "--version"]).splitlines()[-1] if run(["nvcc", "--version"]) else "",
        },
    }
    (REPO / "artifacts").mkdir(exist_ok=True)
    (REPO / "artifacts" / "system_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
