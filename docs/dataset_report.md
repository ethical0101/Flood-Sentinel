# Dataset investigation and selection

Date of investigation and download: **2026-10-17/18** (local time, IST).

## Candidates investigated

| | FloodNet (Track 1) | **Sen1Floods11** | SpaceNet 8 |
|---|---|---|---|
| Source | github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021 → Google Drive folder | github.com/cloudtostreet/Sen1Floods11 → public GCS bucket `gs://sen1floods11` | spacenet.ai/sn8-challenge (AWS open data) |
| Imagery | UAV RGB photos, Hurricane Harvey, ~4000×3000 px | Sentinel-1 SAR (VV, VH) + Sentinel-2 (13 bands), 512×512 chips at 10 m | Very-high-resolution pre/post-event optical imagery |
| Labels | 10-class semantic masks, but only ~400 labelled training images (semi-supervised challenge) | 446 hand-labelled chips: water / not water / no data | Flooded roads and buildings (vector) |
| Georeferenced | **No** (photos, no CRS) | **Yes**: GeoTIFF, EPSG:4326 | Yes |
| Access | Google Drive folder (quota/rate-limited, manual) | Anonymous HTTPS (GCS JSON API), per-file MD5 | AWS S3 with tooling, tens of GB |
| Size checked | not measured (Drive folder) | 1.76 GB for the hand-labelled subset (listed via API) | not downloaded (large, AWS tooling) |
| Licence | No LICENSE file in repo (checked via GitHub API: 404) | No LICENSE file in repo; STAC collections say `"license": "proprietary"` | CC BY-SA 4.0 per SpaceNet (not verified in this session) |

## Decision: Sen1Floods11 hand-labelled subset only

Why it is the smallest scientifically defensible choice:

1. **Real annotations for the primary task.** Every chip has a hand-drawn water mask. Pixels labelled −1 are explicitly invalid and can be ignored.
2. **Georeferenced (EPSG:4326, 10 m).** This is what makes the routing part valid: a prediction from a Sen1Floods11-format GeoTIFF can be aligned with OpenStreetMap roads without inventing coordinates. FloodNet photos have no CRS and could not support capability B.
3. **Official, leakage-free splits.** The train / valid / test CSVs ship with the dataset, plus a separate Bolivia event that is held out for testing generalisation to an unseen region.
4. **Built-in baselines.** Sentinel-1 Otsu-threshold labels (`S1OtsuLabelHand`) and JRC permanent-water labels (`JRCWaterHand`) are included, so baselines need no extra data.
5. **Fits the hardware.** It is 1.76 GB raw and 4.0 GB after preprocessing, with 512² chips. F: had about 168 GB free.
6. **Accessible without accounts.** It downloads anonymously over HTTPS, and every file has a published MD5 hash.

The weakly-labelled subset (33 GB of Otsu-derived labels) was deliberately **not** downloaded. Its labels are machine-generated rather than hand-labelled, and it was not needed.

FloodNet and SpaceNet 8 were not downloaded. FloodNet lacks georeferencing and needs a quota-limited Drive download. SpaceNet 8 is large and needs AWS tooling. Both are listed as future work.

## Files downloaded

`scripts/download_datasets.py` lists the bucket through the GCS JSON API and downloads:

| Layer (`v1.1/data/flood_events/HandLabeled/…`) | Files | Size |
|---|---|---|
| S1Hand (VV, VH; dB; float32) | 446 | 729.5 MB |
| S2Hand (13 bands; uint16 = reflectance × 10⁴) | 446 | 1018.6 MB |
| LabelHand (−1 / 0 / 1; int16) | 446 | 2.6 MB |
| S1OtsuLabelHand | 446 | 1.7 MB |
| JRCWaterHand | 446 | 0.5 MB |
| `v1.1/splits/flood_handlabeled/*.csv` | 4 | <0.1 MB |
| `v1.1/Sen1Floods11_Metadata.geojson` | 1 | 14 KB |
| **Total** | **2,235** | **≈1.75 GB** |

Integrity: `scripts/verify_datasets.py` recomputed the MD5 of all 2,235 files against the bucket's published hashes, and **2,235 / 2,235 matched**. It also checked that every split chip has all five layers, that all 2,230 rasters are EPSG:4326 at 512×512, that each chip's layers share one grid (tolerance 10⁻³ px; one chip, `Mekong_98310`, differs by about 10⁻¹⁴° of float round-off), and that the label values are ⊂ {−1, 0, 1}. The report is in `artifacts/dataset_verification.json`; the manifest with per-file sizes and MD5s is in `data/raw/sen1floods11/manifest.json`.

## Split composition (official CSVs, unchanged)

| Split | Chips | Events | Use |
|---|---|---|---|
| train | 252 | 11 (all except Bolivia) | gradient updates; normalisation statistics |
| valid | 89 | 11 | checkpoint selection, early stopping |
| test | 90 | 11 | final evaluation only |
| bolivia | 15 | 1 | final evaluation only (unseen event/region) |

Label pixels across all chips: water 10.71 M, not-water 90.28 M, no-data 15.93 M. Water is 9.5% of valid training pixels.

## Storage estimate (measured)

| Item | Size |
|---|---|
| Raw download | 1.75 GB |
| Processed per-chip `.npz` | 4.0 GB |
| OSM road graph cache (6 areas) | < 10 MB |
| Checkpoints (5 runs × best + last) | ≈ 1.5 GB |
| Python venv (PyTorch CUDA etc.) | several GB (see system_preflight.md) |

## Road network data

OpenStreetMap drive networks were fetched through OSMnx/Overpass, one bounding-box request per study area, and cached as GraphML in `data/osm_cache/`. Choosing the study areas took one `out count` query per candidate held-out chip (19 queries). OSM data: © OpenStreetMap contributors, ODbL 1.0.

## v2 additions: WorldFloods v2 (Portalés-Julià et al., 2023)
* **Source:** https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2. Public, not gated, **CC BY-NC 4.0**. 509 Sentinel-2 L1C scenes with Copernicus EMS / UNOSAT / GLOFIMR flood masks (train 475, val 16, test 18), about 76 GB in total.
* **Downloaded:** only what was needed, each file SHA-256-verified against the Hugging Face LFS hash and resumable.
  * **Test split** (18 scenes, 3.08 GB, 74 files): external evaluation only (`scripts/download_worldfloods_test.py`).
  * **Training subset** (97 scenes, 4.85 GB, 291 files; `scripts/select_worldfloods_subset.py` → `configs/worldfloods_train_subset.json`, `scripts/download_worldfloods_subset.py`). The rule:
    * official train split only;
    * exclude scenes within 10 km of any Sen1Floods11 valid, test or Bolivia chip (4 excluded: 3 from EMSR388, the 2019 Spain event the test chips come from, and 1 from the Nigeria event);
    * one scene per flood event, smallest first;
    * a 5 GB budget, kept small on request.
  * The full 92.7 GB train split was deliberately not downloaded.
* **Label mapping** (from each scene's `encoding_values`): water 2→1, land 1→0, invalid 0→−1. Cloud (cloud band = 2) → −1, and S2 no-data → −1.
* **Processed:** 303 tiles (≤ 512², ≤ 8 per scene, water-bearing first) from 75 events, 832 MB compressed (`training/prepare_worldfloods.py` → `data/processed/wf_chips/`, `index_worldfloods.json`).
* **Caveat (measured):** most masks were mapped from other sensors and/or other dates than the paired S2 image. Only 2 of the 75 scenes used have S2-derived masks, and NDWI-vs-label agreement is 0.58 against 0.65 for Sen1Floods11. See `experimental_results.md` §A7.
* **Not used:** FloodNet (UAV RGB, no georeferencing, Drive download) and the Sen1Floods11 weakly-labelled set (algorithmic Otsu/index labels, 33 GB).
