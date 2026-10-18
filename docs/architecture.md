# System architecture

```
                    ┌──────────────────────────── offline / one-time ───────────────────────────┐
 gs://sen1floods11 ─┤ scripts/download_datasets.py → verify_datasets.py → training/prepare_data.py │
                    │        (MD5-verified GeoTIFFs)        (integrity)      (per-chip .npz + index) │
 Overpass API ──────┤ scripts/fetch_road_networks.py → data/osm_cache/*.graphml (+ attribution json) │
                    └────────────────────────────────────────────────────────────────────────────┘
                    ┌──────────────────────────── training (separate process) ───────────────────┐
                    │ training/train.py  →  artifacts/runs/<run>/{checkpoints/best.pt,last.pt,      │
                    │   (AMP, resumable,       history.json, status.json, train.log, config.yaml}   │
                    │    single-instance lock)                                                      │
                    │ training/evaluate.py → evaluation.json, samples/*.png, curves.png,            │
                    │                         artifacts/{baselines,summary}.json, figures/          │
                    └────────────────────────────────────────────────────────────────────────────┘
                    ┌──────────────────────────── serving ────────────────────────────────────────┐
  Browser (React) ──┤ FastAPI backend/app/main.py                                                 │
  Vite build or     │  ├─ services/models.py        ModelRegistry: discovers completed runs,      │
  dev proxy /api    │  │                            loads each Predictor once (GPU, fp16)         │
                    │  ├─ services/segmentation.py  upload validation → training/infer.py         │
                    │  │                            → outputs/predictions/<uuid>/ (png, GeoTIFF)  │
                    │  ├─ services/areas.py         study areas, flood layers (model / label /    │
                    │  │                            upload), annotated-graph cache                │
                    │  ├─ routing/graph_store.py    OSMnx fetch + GraphML cache                   │
                    │  ├─ routing/risk.py           raster ↔ road alignment, edge risk attributes │
                    │  ├─ routing/planner.py        Dijkstra baseline / risk-aware / alternatives │
                    │  └─ services/status.py        system, dataset, training, metrics (from disk)│
                    └────────────────────────────────────────────────────────────────────────────┘
```

## Components

### Training package (`training/`)
| File | Responsibility |
|---|---|
| `utils.py` | paths (`FLOOD_DATA_DIR`, `FLOOD_PROCESSED_DIR`, `FLOOD_ARTIFACTS_DIR`), config loading with `--set key=value` overrides, seeding, environment capture |
| `prepare_data.py` | GeoTIFF → `.npz` per chip; `index.json` keeps CRS/transform/bounds; `stats.json` holds per-band mean/std computed on train only |
| `dataset.py` | modality definitions; the single `normalize()` shared by training, evaluation and the API; `FloodDataset` (crop first, then normalise; flips/rot90; colour jitter for RGB) |
| `model.py` | `smp.Unet` / `smp.DeepLabV3Plus`, single-logit output; atomic checkpoint writes |
| `metrics.py` | confusion matrix with ignore label; IoU, Dice, F1, precision, recall, pixel accuracy, omission/commission; masked BCE + soft-Dice loss |
| `train.py` | AdamW, linear warmup + cosine LR (per step), fp16 autocast + GradScaler, gradient clipping, early stopping, `best.pt`/`last.pt`, `--resume`, PID lock against duplicate runs, live `status.json` |
| `evaluate.py` | held-out evaluation, per-event metrics, baselines, latency benchmark, figures |
| `infer.py` | `SceneInput` readers (GeoTIFF 2/13/15 bands, RGB PNG/JPEG), `Predictor` with ramp-blended sliding-window tiling and optional flip TTA, CLI |

### Backend (`backend/app/`)
* **Model lifecycle:** models load lazily on first use and stay in memory. A lock serialises GPU inference, because a 4 GB laptop GPU must not run concurrent forward passes for several users. Only runs whose `status.json` says `completed` are served.
* **Input safety:** extension and magic-byte checks, a 40 MB cap (`FLOOD_MAX_UPLOAD_MB`), a pixel cap (`FLOOD_MAX_IMAGE_PIXELS`), a decompression-bomb ceiling, and decode errors mapped to 422 with a user-safe message. Results are stored under random 32-hex IDs, and only whitelisted file names are served. Run/area IDs are regex-validated and resolved paths are checked to stay inside their folders.
* **Errors:** the middleware logs the full traceback server-side and returns `{"error","message","request_id"}` without stack traces or local paths. NaN/Inf are converted to `null` (strict JSON).
* **Observability:** every `/api` request is logged with status and duration, and the `X-Request-ID` and `Server-Timing` headers are set.

### Routing (`backend/app/routing/`)
1. **Graph:** `osmnx.graph_from_bbox(network_type="drive", truncate_by_edge=True)`. Speeds come from `maxspeed` or highway-type means, with a 40 km/h fallback; free-flow travel times are added. The graph is cached as GraphML.
2. **Alignment:** the raster carries CRS + affine transform. Road points (EPSG:4326) are reprojected with pyproj when the CRSs differ, then mapped to row/col with the inverse affine. Points outside the raster or on no-data pixels count as *unknown*, never as dry. Unit tests confirm a UTM raster and the equivalent lon/lat raster flag the same road.
3. **Risk:** each edge gets a sample every 5 m. An optional max-filter buffer (default 1 px = 10 m) is applied first. The attributes written are `flooded_frac`, `flooded_len_m`, `flood_prob_max/mean`, `coverage` and `risk_class` ∈ {clear, low, partial, flooded, unknown}.
4. **Cost:** `length + λ_flood·flooded_len + λ_unknown·length·(1−coverage) + λ_prob·length·mean_p`. Parallel edges are collapsed to the cheapest.
5. **Routes:** Dijkstra (NetworkX) on length (baseline) and on cost (risk-aware). Alternatives use the penalty method: the previous route's edges ×1.6 and the solver is re-run, keeping routes with < 80% edge overlap.
6. **Edge cases:** the start/end snap to the nearest node by haversine. The response distinguishes these explicit error codes: `too_far_from_road` (> 1 km), `same_node`, `disconnected`, `empty_graph`. If the closure rule leaves no path, the planner falls back to penalties only and flags it.

### Frontend (`frontend/`)
React 19 + TypeScript + Vite 7 + Tailwind 4, Recharts for charts, react-leaflet for maps, lucide icons.

Pages:
* Overview
* Segmentation Studio
* Safe-Route Planner
* Model Evaluation
* Data & Training
* Documentation

Every number comes from the API (`src/api.ts`). Ground-truth layers are labelled as references and are never presented as predictions.

## Data flow for the two main user actions

**Segment an uploaded GeoTIFF:** `POST /api/segmentation/predict` → validate → `read_input` (band count → modality; dB / reflectance scaling) → `ModelRegistry.choose` (default model if compatible) → `Predictor.run` (normalise with that run's train statistics → tiled fp16 inference) → overlay/mask/probability PNG + probability GeoTIFF with the source CRS → JSON with timings, area (geodesic pixel area), warnings.

**Plan a route:** `POST /api/routes/plan` → `areas.annotated_graph` (cached per area/source/model/threshold/buffer) → flood layer = model prediction on the chip's S1+S2 (or the hand label in reference mode, or an uploaded prediction) → `annotate_graph` → `plan()` → baseline, risk-aware, alternatives, explanation, alignment report, OSM attribution, disclaimer.

## v2 additions
| Module | Purpose |
|---|---|
| `training/dataset.py` | `s1s2i` / `s2i` modalities (NDWI and MNDWI index channels); `radiometric_augment` (per-chip S2 gain ± per-band jitter, S1 dB offset), applied to raw values before normalisation and to training data only |
| `training/gpu_guard.py` | detects GPU compute processes from other projects; used by `--polite` training, the suite runner and the training-start safeguards |
| `training/registry.py` | experiment registry (one row per run: data, model, loss, optimiser, seed, time, metrics, VRAM, checkpoint) |
| `training/evaluate.py` | adds a **validation-only** threshold sweep and reports every split at 0.5 and at the validation-selected threshold; records inference peak VRAM |
| `scripts/audit_reproduce.py`, `analyze_weaknesses.py` | independent re-verification from raw GeoTIFFs, leakage/proximity checks, error stratification, calibration |
| `scripts/evaluate_worldfloods.py` | external evaluation on the WorldFloods v2 test split (windowed, CRS-aware, explicit label mapping) |
| `backend/app/services/history.py` | SQLite analysis history (metadata + references), CSV/JSON export |
| `backend/app/services/reports.py` | route-result persistence; JSON / GeoJSON export; self-contained HTML reports (embedded images, SVG route map) |
| `backend/app/services/training_control.py` | guarded start/stop: fixed presets only, typed confirmation, preflight checks, graceful STOP file |
| `backend/app/services/diagnostics.py` | system diagnostics, version info, warnings derived from real state, API error log |
| `backend/app/routing/planner.py` | explicit route `status` (`flood_free_route_found` / `no_flood_free_route` / `no_route_avoids_closures`), per-route length by risk class, flood-affected roads, documented risk level |

New endpoints:
* `/api/history`, `/api/history/{id}`, `/api/history/export`, `/api/reports/{id}`
* `/api/routes/results/{id}` (+ `/export?format=json|geojson|html`), `/api/routes/areas/{id}/search`
* `/api/experiments`, `/api/evaluation/external`
* `/api/diagnostics`, `/api/warnings`, `/api/version`, `/api/segmentation/formats`
* `/api/training/presets`, `/api/training/start`, `/api/training/stop`

The frontend was rebuilt around a design system (`docs/ui_redesign_plan.md`): an `AppShell` with live status chips, and `Panel`, `Metric`, `DataTable` (sortable, CSV), `ChartFrame` (a "Data" drawer behind every chart), `InfoTip`, `Drawer` and more. There are eight code-split pages.

## Configuration

All settings come from environment variables (`.env.example`):
* data, artifact and cache folders
* upload and pixel limits
* device
* default model
* whether OSM downloads are allowed
* CORS origins

No secrets are needed.
