# System preflight report

Inspected on **2026-10-17 23:32 IST**, before any download or install. Re-measured at 2026-10-18 00:36 while training was running (`scripts/system_preflight.py` → `artifacts/system_info.json`).

## Project folder
* `F:\Disaster` was **empty** at the start (no files, not a git repository). `git init` was run there; no existing files had to be preserved.
* Project, datasets, caches and checkpoints all live on drive **F:** (`FLOOD_DATA_DIR` defaults to `F:\Disaster\data\raw`).

## Hardware (verified)
| Item | Value | Source |
|---|---|---|
| OS | Windows 11 Home Single Language 10.0.26300 | `Win32_OperatingSystem` |
| CPU | 12th Gen Intel Core i5-12500H, 12 cores / 16 threads | `Win32_Processor` |
| RAM | 15.7 GiB total; **only 1.8–2.8 GiB free** during the session (other apps) | `GlobalMemoryStatusEx` |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, compute capability 8.6, bf16 supported | `nvidia-smi`, `torch.cuda` |
| Driver | 610.74 (CUDA UMD 13.3) | `nvidia-smi` |
| GPU at start | 0 MiB used, 0% utilisation, 50 °C | `nvidia-smi` |
| CUDA toolkit | nvcc 13.1 (not needed; PyTorch ships its own runtime) | `nvcc --version` |

## Disk (free space at start, from `Get-PSDrive`, GiB)
| Drive | Free |
|---|---|
| C: | 59.7 GiB |
| D: | 44.3 GiB |
| E: | 16.9 GiB |
| **F: (project)** | **162.5 GiB** |

## Software found
| Tool | Version | Note |
|---|---|---|
| Python (default) | 3.15.0 | too new for some geospatial wheels; not used |
| Python 3.13 | 3.13.4 | **chosen**: has a CPU-only torch 2.13 globally, so a project venv was created |
| Python 3.11 | 3.11.0 | no torch |
| Git | 2.49.0 | |
| Node.js / npm | 20.20.2 / 10.8.2 | Vite 7 and Vitest 4 support Node 20 (Vite 8 / Vitest 5 / jsdom 30 need Node 22, so they were avoided) |

No Python had a CUDA build of PyTorch. Installed into `.venv`: **torch 2.14.1+cu130** and torchvision 0.29.1 (CUDA 13.0 wheels; the driver supports 13.3). Verified with an fp16 matmul on `cuda:0`, and cuDNN 9.24 is available.

## Storage estimate and actual use
| Item | Estimated before | Measured |
|---|---|---|
| Sen1Floods11 hand-labelled download | 1.75 GB (from bucket listing) | 1.7 GiB |
| Preprocessed `.npz` chips | ~4 GB (446 × 15 bands × 512² × fp16 + labels) | 4.0 GiB |
| OSM GraphML cache | < 50 MB | 7.5 MiB |
| Checkpoints | 5 runs × (≈95 MB best + ≈280 MB last) | ≈1.6 GB total (see artifacts) |
| Python venv (torch CUDA + geo stack) | 5–6 GB | (folder size not measured: enumerating it took too long on Windows) |
| node_modules | ~0.3 GB | not measured |
| **Total** | **≈ 12 GB** | well within the 162 GiB free on F: |

## Decisions made from these facts
* **Dataset:** the Sen1Floods11 hand-labelled subset (1.75 GB) instead of the 33 GB weakly-labelled set or multi-GB alternatives. See `dataset_report.md`.
* **Low free RAM (≈2 GiB):** no in-RAM dataset cache, at most 2 DataLoader workers, and per-chip lazy `.npz` loading with crop-before-normalise.
* **4 GB VRAM:** compact models (U-Net/ResNet-34, 24.5 M params; DeepLabV3+/MobileNetV3-L, 4.7 M), batch 8 with 256² crops and fp16 autocast. Measured peak allocation: 0.78–0.87 GiB, so no OOM handling was needed. GPU inference in the API is serialised.
* All training runs are sequential (never two GPU jobs at once), with a PID lock per run.
