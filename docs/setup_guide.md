# Setup guide (Windows 10/11, NVIDIA GPU)

Tested on Windows 11, Python 3.13.4, an RTX 3050 Laptop GPU (4 GB) with driver 610.74, and Node.js 20.20 / npm 10.8. PowerShell commands assume the repository root (`F:\Disaster`).

## 1. Python environment
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -m pip install -r backend\requirements.txt
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected output: `True NVIDIA GeForce RTX 3050 Laptop GPU`. If you see `False`, check `nvidia-smi` and the driver, and confirm the `+cu130` wheel was installed (`pip show torch`). Python 3.15 is too new for some geospatial wheels; use 3.13.

Optional: `copy .env.example .env` and edit paths. The defaults keep everything inside the repository.

## 2. System check
```powershell
.\.venv\Scripts\python scripts\system_preflight.py     # writes artifacts\system_info.json
```

## 3. Data (about 1.75 GB download, 4 GB after preprocessing)
```powershell
.\.venv\Scripts\python scripts\download_datasets.py --dry-run   # list what would be fetched
.\.venv\Scripts\python scripts\download_datasets.py             # resumable; safe to rerun
.\.venv\Scripts\python scripts\verify_datasets.py               # full MD5 + structure check
.\.venv\Scripts\python -m training.prepare_data
.\.venv\Scripts\python scripts\fetch_road_networks.py           # OSM graphs for the 6 study areas
```

## 4. Training and evaluation
```powershell
.\.venv\Scripts\python -m training.train --smoke                 # 2-epoch pipeline check (~1 min)
.\.venv\Scripts\python scripts\run_experiments.py                # 5 runs in sequence + evaluation
#   or a single run:
.\.venv\Scripts\python -m training.train --set data.modality=s1 --run my_s1_run
.\.venv\Scripts\python -m training.train --resume --run my_s1_run   # continue after an interruption
.\.venv\Scripts\python -m training.evaluate --all
```
A full 60-epoch run of U-Net/ResNet-34 takes about 15–20 minutes on the RTX 3050, and early stopping often ends it sooner. Peak GPU memory is about 0.85 GB at batch 8 with 256² crops.

If CUDA reports out of memory, apply these settings in order:
1. `--set train.batch_size=4`
2. `--set train.grad_accum=2`
3. `--set data.crop_size=192`
4. `--set data.num_workers=0`

## 5. Run the application
**Production-style (one process):**
```powershell
cd frontend; npm ci; npm run build; cd ..
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Then open http://127.0.0.1:8000 for the dashboard, or http://127.0.0.1:8000/docs for the API.

**Development (hot reload):**
```powershell
.\.venv\Scripts\python -m uvicorn backend.app.main:app --port 8000 --reload
cd frontend; npm run dev        # http://127.0.0.1:5173, proxies /api to :8000
```

## 6. Tests
```powershell
.\.venv\Scripts\python -m pytest            # backend + training (integration tests skip if data/models absent)
cd frontend; npm test; npm run build        # frontend unit tests + type-check + production build
```

## 7. Command-line inference
```powershell
.\.venv\Scripts\python -m training.infer --run unet_resnet34_s1s2 --input some_s1s2.tif --out outputs\cli\pred
```
This writes `pred.prob.tif` (georeferenced), `pred.mask.png` and `pred.overlay.png`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: torch` | You are not using `.venv\Scripts\python`. |
| `torch.cuda.is_available() == False` | Reinstall from the `cu130` index; update the NVIDIA driver. |
| Training refuses: "already being trained by PID …" | Another process holds `train.lock`. If no such process exists, delete `artifacts\runs\<run>\train.lock`. |
| "`last.pt` exists. Use --resume" | Resume, or pass a new `--run` name. Runs are never overwritten. |
| Route planner: "Could not obtain the OpenStreetMap road network" | Overpass is unreachable or busy. Cached areas still work; retry later. |
| PowerShell shows `NativeCommandError` around training output | PyTorch writes warnings to stderr; this is harmless. |
| Dashboard shows "API offline" | Start uvicorn on port 8000 (the Vite dev server proxies `/api` there). |
| Map has no basemap | Basemap tiles need internet. Imagery, flood layer, roads and routing work offline once cached. |
