# Data card: Sen1Floods11 (hand-labelled subset) as used in this project

| Field | Value |
|---|---|
| Name | Sen1Floods11 v1.1, `HandLabeled` subset |
| Creator | Cloud to Street (Public Benefit Corporation) |
| Original URL | https://github.com/cloudtostreet/Sen1Floods11 · data in `gs://sen1floods11/v1.1/` |
| Access used | Anonymous HTTPS: `https://storage.googleapis.com/sen1floods11/…` (GCS JSON API listing) |
| Downloaded | 2026-10-17 (manifest `downloaded_utc` holds the exact UTC timestamp) |
| Integrity | MD5 of all 2,235 files matched the bucket's `md5Hash` |
| Citation | Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* Proc. IEEE/CVF CVPR Workshops, pp. 210–211. |
| Licence | **Not explicitly stated.** The GitHub repository has no LICENSE file (GitHub API returns 404), and the STAC collections in `v1.1/catalog.zip` declare `"license": "proprietary"`. The README asks users to cite the paper. This project uses the data for non-commercial academic research, does not redistribute it (it stays in the git-ignored `data/` folder), and keeps the citation. Anyone wanting to redistribute it or use it commercially should ask the authors (support@cloudtostreet.info). |
| Underlying imagery | Copernicus Sentinel-1 GRD and Sentinel-2 L1C (ESA), with the preprocessing described by the dataset authors |

## Contents

* 446 chips of 512×512 px at 10 m ground resolution, in EPSG:4326, from 11 flood events: Bolivia, Ghana, India, Mekong (Cambodia), Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri Lanka, USA. The metadata also lists a Colombia event, but it has no hand-labelled chips.
* Event dates range from 2016-08-12 (India) to 2019-09-18 (Spain), per `Sen1Floods11_Metadata.geojson`.
* Layers per chip: S1 VV/VH (dB), S2 13 bands (reflectance × 10⁴), hand label, S1 Otsu label, JRC permanent water.

## Label semantics

* `1` = water visible on the image date. This covers **both** flood water and permanent water bodies; the label does not separate them.
* `0` = not water.
* `-1` = not labelled / invalid (for example clouds or image edges). These pixels are excluded from training loss and from every metric.

## Known biases and caveats

* Few chips per event (4 to 36), and geographically clustered. The test split shares events with training, so only the Bolivia split measures generalisation to a new event.
* Water makes up 9.5% of valid training pixels, so the classes are imbalanced. That is why Dice loss is used and why pixel accuracy alone is not reported as success.
* Sentinel-2 chips can be cloud-covered. Sentinel-1 misses water under vegetation and confuses smooth surfaces with water.
* At 10 m resolution, narrow flooded streets in towns are at or below the pixel size.

## How the data is used here

| Purpose | Data |
|---|---|
| Training | official train split (252 chips), random 256² crops |
| Model selection / early stopping | official valid split (89 chips) |
| Final evaluation | official test split (90) + Bolivia (15), evaluated once |
| Routing study areas | 6 *test-split* chips with mapped OSM roads |
| Normalisation statistics | train split only |

Synthetic or fabricated data is not used anywhere.

---

# Data card: WorldFloods v2 (as used in this project)
| Field | Value |
|---|---|
| Creator / citation | Portalés-Julià, E., Mateo-García, G., Purcell, C., Gómez-Chova, L. (2023). *Global flood extent segmentation in optical satellite images.* Scientific Reports 13, 20316. doi:10.1038/s41598-023-47595-7 |
| URL | https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2 |
| Licence | **CC BY-NC 4.0** (non-commercial, attribution) |
| Parts used | Test split, 18 scenes (evaluation only); training subset of 97 scenes / 75 contributing events (multi-dataset experiments) |
| Imagery | Sentinel-2 L1C, 13 bands, ×10⁴ reflectance, 10 m, UTM. Bands 14–15 (QA60, cloud probability) unused. No Sentinel-1. |
| Labels | Flood masks from Copernicus EMS (most), UNOSAT and GLOFIMR products, often derived from other satellites (SAR or very-high-resolution optical) and other dates. Water includes permanent water. Cloud band marks clouds. |
| Integrity | SHA-256 of every downloaded file matched the published LFS hash |
| Leakage control | Training scenes within 10 km of any Sen1Floods11 held-out chip excluded. The WorldFloods test split was never used for training or selection. |
| Redistribution | Raw data is not committed to git. Only small derived figures and metrics appear in the repository. |
