# Flood-Sentinel: Real-Time Flood Segmentation and Safe-Route Prediction for Disaster Response using Deep Learning

**Full project report.** All experiments were run on 2026-10-17/18 on one NVIDIA RTX 3050 Laptop GPU (4 GB). Every metric table in this report is generated from the recorded artifacts by `scripts/build_report_tables.py`; no number is copied by hand. Detailed per-experiment notes are in `docs/experimental_results.md`, and the audit is in `docs/upgrade_audit.md`.

---

## Abstract
Flood-Sentinel is an end-to-end, locally trained system. It segments surface water in Sentinel-1 (SAR) and Sentinel-2 (optical) satellite imagery, then uses the georeferenced prediction to plan flood-risk-aware routes on the real OpenStreetMap road network. It includes a FastAPI backend and a React dashboard.

Twelve segmentation models were trained on the hand-labelled Sen1Floods11 dataset. Two of them were also trained on a leakage-controlled subset of WorldFloods v2. Models were evaluated on three held-out sets:
* the Sen1Floods11 test split, which turns out to be spatially close to training;
* the unseen Bolivia event;
* the external WorldFloods v2 test split, unseen flood events labelled by another team.

**Segmentation results:**
* The default model, a U-Net/ResNet-34 on S1+S2, reaches water IoU **0.825** on test and **0.794** on Bolivia at a threshold selected on validation only.
* It beats every non-learned baseline on the test split. The NDWI index (0.832) remains best on Bolivia.
* On WorldFloods, the learned S2 model (0.781) clearly beats NDWI (0.533) and MNDWI (0.748).
* Controlled v2 experiments found no change that improves on v1 beyond seed noise under a pre-registered selection rule. Water-index input channels gave a significant +0.02 IoU on Bolivia, but it did not replicate on WorldFloods.
* Adding WorldFloods training data hurt overall. The cause was measured: its labels are mostly derived from other sensors and other dates.

**Routing results:** routes planned on the model's predictions carry **19.4% less hand-labelled flooded road** than shortest routes, for a 4.5% average detour. An oracle that plans on the hand labels reaches 21.4%.

**System:** model inference takes 13–17 ms per 512×512 chip. The software is covered by 65 Python tests and 18 frontend tests.

---

## 1. Introduction
### 1.1 Problem
During a flood, responders need to know where the water is and which roads avoid it. Sentinel-2 sees water clearly but not through cloud. Sentinel-1 radar sees through cloud but is noisier. Turning either into road-level guidance requires two things:
* an accurate segmentation model;
* a geospatially correct link between predicted pixels and road geometry.

### 1.2 Objectives
1. Train compact segmentation networks on authentic, hand-labelled flood imagery within 4 GB of VRAM.
2. Evaluate them rigorously: held-out splits, an unseen event, an external dataset, strong baselines and significance tests.
3. Align predictions with OpenStreetMap roads using each raster's CRS and geotransform, and score road segments.
4. Compare shortest and risk-aware routes (Dijkstra), explain the trade-off, and never present a route as guaranteed safe.
5. Deliver a usable dashboard, a documented API, reproducible scripts and honest documentation.

### 1.3 Contributions
* A reproducible pipeline: verified downloads, preprocessing, resumable AMP training, evaluation, an experiment registry and GPU-courteous scheduling.
* A **data-integrity audit** showing that the standard Sen1Floods11 test split is spatially correlated with training.
* **Controlled experiments**, each changing one factor, with a seed-noise reference and paired bootstrap tests, including negative results.
* **Routing evaluated against ground truth** (300 origin–destination pairs), not just demonstrated.
* A full web application: segmentation studio, route planner, evaluation lab, monitor, dataset explorer, history and reports.

---

## 2. Data
| Dataset | Role | Size used | Licence | Integrity |
|---|---|---|---|---|
| **Sen1Floods11 v1.1**, hand-labelled (Bonafilia et al., 2020) | train (252 chips) · validation (89) · test (90) · **Bolivia** (15, unseen event) | 446 chips of 512² at 10 m; S1 VV/VH + S2 13 bands; 1.75 GB | no explicit licence (STAC: "proprietary"); used non-commercially, not redistributed | MD5 of all 2,235 files |
| **WorldFloods v2** test split (Portalés-Julià et al., 2023) | **external evaluation only** | 18 S2 scenes, 3.08 GB | CC BY-NC 4.0 | SHA-256 of 74 files |
| **WorldFloods v2** training subset | multi-dataset training (S2 models) | 97 scenes → 303 tiles from 75 events, 4.85 GB | CC BY-NC 4.0 | SHA-256 of 291 files |
| **OpenStreetMap** drive network | routing | 6 study areas, 7.7 MB GraphML cache | ODbL 1.0 | fetched via OSMnx |

**Leakage controls:**
* The official Sen1Floods11 splits are used unchanged.
* Normalisation statistics come from the training split only.
* WorldFloods training scenes within 10 km of any Sen1Floods11 held-out chip are excluded (4 scenes).
* Bolivia and WorldFloods are never used for checkpoint, threshold or model selection.
* Routing study areas are test-split chips.

**Audit finding:** 42% of Sen1Floods11 test chips lie within 10 km of a training chip, and 4 test chips overlap one. Test IoU is therefore an *in-distribution* measure. Bolivia lies at least 1,216 km from any training chip.

---

## 3. System architecture
```
Sentinel-1/2 GeoTIFF, RGB image ─► preprocessing (train-split stats, optional NDWI/MNDWI) ─► U-Net / DeepLabV3+
      (PyTorch, fp16, tiled 512² inference) ─► georeferenced P(water) ─► mask, overlay, uncertainty, area, NDWI comparison
OpenStreetMap (OSMnx, cached GraphML) ─► road graph ─► CRS-aware 5 m sampling of P(water) ─► edge risk
      ─► Dijkstra: shortest vs flood-penalised (+ alternatives, explicit status) ─► explanation, exports, reports
FastAPI (models loaded once, serialised GPU inference, SQLite history) ◄──► React + TypeScript + Tailwind + Leaflet
Training: separate resumable process; --polite yields the GPU to other projects; guarded start/stop from the UI
```
Components and endpoints are listed in `docs/architecture.md`.

---

## 4. Methods
### 4.1 Preprocessing
* **S1:** VV/VH in dB, clipped to [−50, 10].
* **S2:** TOA reflectance (DN / 10⁴). Both are standardised per band with Sen1Floods11 training statistics. Non-finite values become 0.
* **Optional index channels:** NDWI = (B3 − B8)/(B3 + B8) and MNDWI = (B3 − B11)/(B3 + B11).
* **RGB models:** a per-image 2–98% stretch plus ImageNet normalisation.
* **WorldFloods tiles:** labels map to water 2→1, land 1→0, invalid/cloud/no-data → −1.

### 4.2 Models
* **U-Net** (Ronneberger et al., 2015) with a ResNet-34 encoder (He et al., 2016), 24.5 M parameters.
* **DeepLabV3+** (Chen et al., 2018) with MobileNetV3-Large (Howard et al., 2019), 4.7 M parameters.
* Both use ImageNet-initialised encoders and output a single P(water) logit (segmentation_models_pytorch 0.5.0).

### 4.3 Training
* Random 256² crops: 4 per Sen1Floods11 chip and 1 per WorldFloods tile per epoch. Flips and 90° rotations.
* Optional radiometric augmentation: S2 gain 0.75–1.25 with ±5% per-band jitter; S1 offset ±2 dB with ±0.5 dB per polarisation.
* Loss: masked BCE + soft Dice. Optimiser: AdamW (5e-4, wd 1e-4) with 2-epoch warmup then cosine decay.
* fp16 autocast, batch 8, gradient clipping at 5, at most 60 epochs, early stopping with patience 15 on validation IoU.
* Checkpoint and threshold are selected on Sen1Floods11 validation only. Seed 42, plus one seed-7 replicate.

### 4.4 Evaluation
* **Metric:** water-class IoU from confusion counts pooled over all valid pixels of a split, plus F1/Dice, precision, recall, mIoU, pixel accuracy, mean chip IoU, per-event IoU and confusion matrices. Pixels labelled −1 are ignored.
* **Threshold:** grid 0.10–0.90 (step 0.05), chosen on validation; every result is also reported at 0.5.
* **WorldFloods:** clear-sky valid pixels, processed in 1,024 px windows in each scene's UTM CRS.
* **Baselines:** NDWI > 0, MNDWI > 0, Sentinel-1 Otsu labels, JRC permanent water and a constant predictor.
* **Significance:** paired chip bootstrap (2,000 resamples). Seed noise is measured with a replicate run.
* **Latency:** batch 1, 512², fp16, CUDA-synchronised.

### 4.5 Routing
1. Edges are densified to one sample per 5 m and reprojected to the raster CRS. P(water) is sampled after a 1 px max-filter. Samples outside the raster or on no-data count as **unknown, never dry**.
2. Edge cost = length + 10 × flooded length + 0.5 × unknown length + length × mean P. All weights are user-adjustable.
3. Dijkstra computes the baseline (length) and the risk-aware (cost) routes; alternatives come from the penalty method. An optional closure rule treats edges with at least N m of predicted flooding as impassable.
4. Every result carries an explicit status: `flood_free_route_found`, `no_flood_free_route` or `no_route_avoids_closures`.
5. **Evaluation:** 300 random origin–destination pairs over 6 held-out areas. Routes are planned on the model prediction, and their flooded length is measured on the hand label against shortest and oracle routes.

---

## 5. Experiments
| Group | Runs | Question |
|---|---|---|
| v1 | 4 input modalities + 1 compact architecture | Which inputs and architectures work? |
| v2 | seed replicate; radiometric augmentation; index channels; both; S2-only | Can generalisation be improved, with controlled single-factor changes? |
| v2-MD | + WorldFloods data for S2 and S2+indices (paired with identical single-dataset recipes) | Does more, diverse training data help? |

Pre-registered selection rule: highest validation IoU at the validation-selected threshold, and only if it beats v1 by more than seed noise.

### 5.1 Training runs
<!-- TABLE:training -->
| Model | Change vs v1 | Data | Epochs (best) | Train time | Peak train VRAM | Params |
|---|---|---|---|---|---|---|
| `unet_resnet34_s1s2` | v1 default recipe | Sen1Floods11 | 46 (31) | 15.7 min | 867 MiB | 24.47 M |
| `unet_resnet34_s2` | input=s2 | Sen1Floods11 | 60 (54) | 19.6 min | 861 MiB | 24.47 M |
| `unet_resnet34_s1` | input=s1 | Sen1Floods11 | 59 (44) | 14.6 min | 828 MiB | 24.43 M |
| `unet_resnet34_rgb` | input=rgb | Sen1Floods11 | 60 (56) | 14.7 min | 830 MiB | 24.44 M |
| `deeplabv3plus_mnv3l_s1s2` | model=deeplabv3plus/tu-mobilenetv3_large_100 | Sen1Floods11 | 45 (30) | 13.6 min | 655 MiB | 4.71 M |
| `v2_r0b_unet_s1s2_seed7` | seed=7 | Sen1Floods11 | 46 (31) | 14.2 min | 1152 MiB | 24.47 M |
| `v2_r1_unet_s1s2_radiometric` | radiometric augmentation | Sen1Floods11 | 60 (59) | 19.5 min | 867 MiB | 24.47 M |
| `v2_r2_unet_s1s2i_indices` | input=s1s2i | Sen1Floods11 | 60 (59) | 20.4 min | 872 MiB | 24.48 M |
| `v2_r3_unet_s1s2i_radiometric` | input=s1s2i, radiometric augmentation | Sen1Floods11 | 60 (59) | 20.2 min | 872 MiB | 24.48 M |
| `v2_r4_unet_s2i_radiometric` | input=s2i, radiometric augmentation | Sen1Floods11 | 60 (53) | 19.5 min | 867 MiB | 24.47 M |
| `md1_unet_s2_worldfloods` | input=s2, + WorldFloods training data | S1F11 + WF subset | 41 (26) | 15.8 min | 861 MiB | 24.47 M |
| `md2_unet_s2i_radiometric_worldfloods` | input=s2i, radiometric augmentation, + WorldFloods training data | S1F11 + WF subset | 26 (11) | 12.9 min | 867 MiB | 24.47 M |
<!-- /TABLE -->

---

## 6. Results
### 6.1 All models and baselines (water IoU; models at their validation-selected threshold)
<!-- TABLE:main -->
| Model | Gen. | Input | Thr. | Val IoU | Test IoU | Test F1 | Bolivia IoU | Bolivia recall | WorldFloods IoU |
|---|---|---|---|---|---|---|---|---|---|
| `unet_resnet34_s1s2` | v1 | s1s2 | 0.4 | 0.8454 | 0.8253 | 0.9043 | 0.7939 | 0.826 | n/a (needs SAR) |
| `unet_resnet34_s2` | v1 | s2 | 0.45 | 0.8411 | 0.8159 | 0.8986 | 0.7740 | 0.810 | 0.7809 |
| `unet_resnet34_s1` | v1 | s1 | 0.4 | 0.6883 | 0.6659 | 0.7995 | 0.6270 | 0.709 | n/a (needs SAR) |
| `unet_resnet34_rgb` | v1 | rgb | 0.4 | 0.6276 | 0.6009 | 0.7507 | 0.5920 | 0.708 | 0.5636 |
| `deeplabv3plus_mnv3l_s1s2` | v1 | s1s2 | 0.6 | 0.8095 | 0.7886 | 0.8818 | 0.7874 | 0.862 | n/a (needs SAR) |
| `v2_r0b_unet_s1s2_seed7` | v2 | s1s2 | 0.45 | 0.8427 | 0.8254 | 0.9043 | 0.7695 | 0.801 | n/a (needs SAR) |
| `v2_r1_unet_s1s2_radiometric` | v2 | s1s2 | 0.4 | 0.8479 | 0.8227 | 0.9028 | 0.7892 | 0.841 | n/a (needs SAR) |
| `v2_r2_unet_s1s2i_indices` | v2 | s1s2i | 0.45 | 0.8471 | 0.8259 | 0.9046 | 0.8135 | 0.863 | n/a (needs SAR) |
| `v2_r3_unet_s1s2i_radiometric` | v2 | s1s2i | 0.4 | 0.8449 | 0.8265 | 0.9050 | 0.8173 | 0.862 | n/a (needs SAR) |
| `v2_r4_unet_s2i_radiometric` | v2 | s2i | 0.4 | 0.8360 | 0.8248 | 0.9040 | 0.8181 | 0.885 | 0.7668 |
| `md1_unet_s2_worldfloods` | v2-MD | s2 | 0.5 | 0.8382 | 0.8064 | 0.8928 | 0.8111 | 0.873 | 0.7608 |
| `md2_unet_s2i_radiometric_worldfloods` | v2-MD | s2i | 0.4 | 0.8010 | 0.7896 | 0.8824 | 0.7493 | 0.770 | 0.7419 |
| NDWI > 0 | baseline | – | fixed | – | 0.7578 | 0.8622 | 0.8320 | 0.888 | 0.5329 |
| MNDWI > 0 | baseline | – | fixed | – | 0.6504 | 0.7882 | 0.5514 | 0.970 | 0.7480 |
| S1 Otsu | baseline | – | fixed | – | 0.5458 | 0.7062 | 0.6972 | 0.867 | n/a |
| JRC permanent water | baseline | – | fixed | – | 0.3446 | 0.5126 | 0.0646 | 0.066 | n/a |
| constant 'not water' | baseline | – | fixed | – | 0.0000 | – | 0.0000 | 0.000 | n/a |
<!-- /TABLE -->

### 6.2 Significance (paired chip bootstrap, 2,000 resamples)
<!-- TABLE:bootstrap -->
| Comparison | Test ΔIoU [95% CI] | Bolivia ΔIoU [95% CI] |
|---|---|---|
| `v2_r0b_unet_s1s2_seed7` vs `unet_resnet34_s1s2` | +0.0001 [-0.011, +0.011] | -0.0244 [-0.102, +0.005] |
| `v2_r1_unet_s1s2_radiometric` vs `unet_resnet34_s1s2` | -0.0025 [-0.012, +0.006] | -0.0047 [-0.021, +0.013] |
| `v2_r2_unet_s1s2i_indices` vs `unet_resnet34_s1s2` | +0.0006 [-0.009, +0.007] | +0.0197 [+0.012, +0.035] |
| `v2_r3_unet_s1s2i_radiometric` vs `unet_resnet34_s1s2` | +0.0013 [-0.002, +0.005] | +0.0235 [+0.014, +0.041] |
| `v2_r4_unet_s2i_radiometric` vs `unet_resnet34_s1s2` | -0.0004 [-0.006, +0.005] | +0.0242 [+0.011, +0.043] |
| `md1_unet_s2_worldfloods` vs `unet_resnet34_s2` | -0.0095 [-0.015, -0.006] | +0.0371 [+0.008, +0.074] |
| `md2_unet_s2i_radiometric_worldfloods` vs `v2_r4_unet_s2i_radiometric` | -0.0353 [-0.059, -0.020] | -0.0688 [-0.123, -0.046] |
<!-- /TABLE -->
Seed replicate (`v2_r0b`) versus v1: validation ±0.0027, test +0.0001, Bolivia −0.024. Differences of this size are not attributable to the method.

### 6.3 Default model: confusion matrices (validation-selected threshold 0.40)
<!-- TABLE:confusion -->
| Split | TN | FP | FN | TP |
|---|---|---|---|---|
| Test | 17,791,167 | 160,099 | 316,289 | 2,249,812 |
| Bolivia | 2,394,541 | 18,410 | 79,145 | 375,719 |
<!-- /TABLE -->

Per-event test IoU:
<!-- TABLE:events -->
| Event | Ghana | India | Mekong | Nigeria | Pakistan | Paraguay | Somalia | Spain | Sri-Lanka | USA |
|---|---|---|---|---|---|---|---|---|---|---|
| Test IoU | 0.557 | 0.905 | 0.938 | 0.940 | 0.632 | 0.619 | 0.825 | 0.850 | 0.928 | 0.814 |
<!-- /TABLE -->

### 6.4 Inference latency (RTX 3050 Laptop, 1×C×512², fp16)
<!-- TABLE:latency -->
| Model | Model-only mean | p95 | End-to-end | Throughput | Inference peak VRAM |
|---|---|---|---|---|---|
| `unet_resnet34_s1s2` | 16.3 ms | 17.1 ms | 50.7 ms | 61 chips/s | 258 MiB |
| `unet_resnet34_s2` | 16.2 ms | 17.2 ms | 49.0 ms | 62 chips/s | 255 MiB |
| `unet_resnet34_s1` | 14.6 ms | 15.7 ms | 23.4 ms | 69 chips/s | 242 MiB |
| `unet_resnet34_rgb` | 16.2 ms | 27.6 ms | 23.4 ms | 62 chips/s | 243 MiB |
| `deeplabv3plus_mnv3l_s1s2` | 12.6 ms | 31.6 ms | 48.8 ms | 79 chips/s | 106 MiB |
| `v2_r0b_unet_s1s2_seed7` | 16.2 ms | 17.4 ms | 58.5 ms | 62 chips/s | 258 MiB |
| `v2_r1_unet_s1s2_radiometric` | 16.3 ms | 17.4 ms | 57.1 ms | 62 chips/s | 258 MiB |
| `v2_r2_unet_s1s2i_indices` | 16.3 ms | 17.3 ms | 63.2 ms | 61 chips/s | 260 MiB |
| `v2_r3_unet_s1s2i_radiometric` | 17.1 ms | 17.4 ms | 62.7 ms | 59 chips/s | 260 MiB |
| `v2_r4_unet_s2i_radiometric` | 16.7 ms | 18.0 ms | 58.1 ms | 60 chips/s | 258 MiB |
| `md1_unet_s2_worldfloods` | 16.5 ms | 17.8 ms | 68.0 ms | 61 chips/s | 255 MiB |
| `md2_unet_s2i_radiometric_worldfloods` | 17.7 ms | 21.2 ms | 81.4 ms | 57 chips/s | 258 MiB |
<!-- /TABLE -->

### 6.5 Routing evaluation against hand labels
<!-- TABLE:routing -->
| Area | Road edges | Flooded road model / label (km) | Edge agreement | Labelled flood per trip: shortest | risk-aware (model) | oracle (label) |
|---|---|---|---|---|---|---|
| spain_7370579 | 1659 | 58.9 / 58.4 | 96.3% | 647 m | 341 m | 341 m |
| spain_6095801 | 2039 | 9.1 / 8.7 | 98.8% | 21 m | 9 m | 10 m |
| spain_7387658 | 1637 | 86.9 / 102.6 | 91.3% | 1435 m | 1296 m | 1289 m |
| srilanka_713926 | 2410 | 2.5 / 2.4 | 99.8% | 44 m | 1 m | 1 m |
| srilanka_649970 | 412 | 1.6 / 3.0 | 93.7% | 109 m | 99 m | 48 m |
| india_747992 | 66 | 5.6 / 7.2 | 84.8% | 637 m | 586 m | 586 m |
| **All (300 pairs)** | | | | **482 m** | **389 m** | **379 m** |

Reduction vs shortest: **19.4%**; mean detour **+4.5%**; pairs worse than shortest: 3/300.
<!-- /TABLE -->

### 6.6 Figures
* Held-out IoU comparison: `artifacts/figures/iou_comparison.png`
* Training curves: `artifacts/runs/<run>/curves.png`
* Sample predictions (including failure cases): `artifacts/runs/<run>/samples/*.png`
* Dashboard screenshots: `docs/screenshots/*.png`

---

## 7. Discussion
1. **Learned models against indices.** On in-distribution test data, the fused S1+S2 U-Net beats the best index (NDWI) by 0.068 IoU (0.825 against 0.758). On the external WorldFloods set the S2 U-Net (0.781) beats MNDWI (0.748) and far exceeds NDWI (0.533). On Bolivia, NDWI (0.832) is still best. Which threshold method wins depends strongly on the event, which is why a single test split is not enough.
2. **Optical bands dominate.** S2 alone is close to S1+S2. SAR alone is clearly weaker (0.666), but it is the only option under cloud.
3. **Index channels** raised Bolivia recall and IoU (+0.020 to +0.024, chip-bootstrap significant). That is the same size as seed noise, and the gain did not transfer to WorldFloods. The idea came from the Bolivia error analysis, so WorldFloods is the deciding check, and it did not confirm the gain.
4. **Radiometric augmentation** alone changed nothing beyond noise. The Bolivia brightness shift is not the main limitation.
5. **More data is not automatically better.** WorldFloods masks are mostly derived from SAR or very-high-resolution sensors and other dates. Only 2 of 75 scenes have S2-derived masks, and label-vs-NDWI agreement is 0.58 against 0.65. Training on them reduced test and WorldFloods IoU, so for optical supervision label provenance matters as much as volume.
6. **Thresholds** chosen on validation (0.40–0.60) are a small, free and legitimate gain.
7. **Training budget:** R1–R3 reached their best validation IoU at epoch 59 of 60, so these runs may be limited by the 60-epoch cap. Longer schedules were not tested, which leaves room for small further gains.
8. **Routing:** model-planned routes capture about 90% of the oracle's reduction in flooded exposure. The remaining gap mirrors segmentation false negatives. Where floods cut every path, the system reports `no_flood_free_route` instead of implying safety.

## 8. Software engineering and quality
* **Backend:**
  * FastAPI with validation, size limits, magic-byte checks and NaN-safe JSON;
  * no stack traces or local paths in error responses, and request IDs on every response;
  * SQLite history, HTML/JSON/GeoJSON exports and diagnostics;
  * guarded training control: fixed presets, typed confirmation, VRAM, disk and foreign-GPU checks, graceful stop.
* **Frontend:**
  * a design system (tokens, Panel, Metric, DataTable with CSV, ChartFrame with a "view data" drawer);
  * eight code-split pages, all usable at 375 px;
  * an accessible tooltip and radiogroup pattern, with skeleton, empty and error states.
* **Tests:**
  * 65 Python tests: metrics, loss, checkpoints, preprocessing, augmentation invariants, CRS alignment, routing edge cases, API errors, history, reports, safeguards, end-to-end segmentation and routing;
  * 18 frontend tests: components, pages with a mocked API, and live integration.
* **Reproducibility:** pinned dependencies (`requirements-lock.txt`), seeds, configs saved per run, an experiment registry, resumable training, and verified downloads.

## 9. Limitations
* Labels mark all surface water, not flood specifically.
* A 10 m pixel is about the width of a road.
* The imagery dates from 2016–2019, but the roads are current.
* There is only one seed per configuration, apart from one replicate.
* Bolivia has only 15 chips.
* WorldFloods cannot test SAR models.
* Routing is confined to the ≈5×5 km rasters, and travel time is free-flow.
* It is a research prototype, not for real evacuation decisions.

Full list: `docs/limitations.md`.

## 10. Future work
* Multi-seed runs and confidence intervals that include seed variance.
* A clean-label WorldFloods subset (S2-derived or same-day masks) and SAR-labelled data for S1 models.
* Calibration-aware thresholds per region.
* DEM-based depth proxies and bridge awareness for routing.
* Live Sentinel-1 ingestion from the Copernicus Data Space.

## 11. Conclusion
Flood-Sentinel shows that a complete, honest pipeline, from public satellite data to risk-aware routing, runs on a 4 GB laptop GPU. It trains in 15–20 minutes per model and infers in milliseconds. The default model is strong in distribution, and its predictions measurably reduce flooded-road exposure in routing. The report documents where it does *not* generalise better than a simple index, and why neither augmentation, index channels nor extra cross-sensor training data closed that gap.

## References
* Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. *CVPR Workshops*, 210–211.
* Portalés-Julià, E., Mateo-García, G., Purcell, C., Gómez-Chova, L. (2023). Global flood extent segmentation in optical satellite images. *Scientific Reports*, 13, 20316.
* Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
* He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
* Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *ECCV*.
* Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV*.
* McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *Int. J. Remote Sensing*, 17(7), 1425–1432.
* Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. *Int. J. Remote Sensing*, 27(14), 3025–3033.
* Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126–139.
* Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269–271.
* OpenStreetMap contributors. Road network data, ODbL 1.0.

## Appendix A: Reproduction
```powershell
.\.venv\Scripts\python scripts\download_datasets.py; .\.venv\Scripts\python scripts\verify_datasets.py
.\.venv\Scripts\python -m training.prepare_data
.\.venv\Scripts\python scripts\run_experiments.py                    # v1
.\.venv\Scripts\python scripts\run_experiments.py --suite v2 --polite
.\.venv\Scripts\python scripts\download_worldfloods_test.py
.\.venv\Scripts\python scripts\select_worldfloods_subset.py; .\.venv\Scripts\python scripts\download_worldfloods_subset.py
.\.venv\Scripts\python scripts\run_multidataset.py                   # tiles, trains md1/md2, evaluates
.\.venv\Scripts\python -m training.evaluate --all --reuse; .\.venv\Scripts\python scripts\evaluate_worldfloods.py
.\.venv\Scripts\python scripts\bootstrap_compare.py; .\.venv\Scripts\python scripts\evaluate_routing.py
.\.venv\Scripts\python -m training.registry; .\.venv\Scripts\python scripts\build_report_tables.py
```

## Appendix B: Metric definitions
* IoU (water) = TP / (TP + FP + FN), pooled over the valid pixels of a split.
* F1 = Dice = 2TP / (2TP + FP + FN).
* Precision = TP / (TP + FP); recall = TP / (TP + FN).
* mIoU = mean of the water and not-water IoU.
* Mean chip IoU averages per-chip IoU, skipping chips with neither labelled nor predicted water.
* ECE uses 10 equal-width probability bins.
