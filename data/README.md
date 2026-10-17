# data/

Large data lives here and is **excluded from git** (see `.gitignore`).

| Folder | Created by | Contents |
|---|---|---|
| `raw/sen1floods11/` | `scripts/download_datasets.py` | Original Sen1Floods11 v1.1 hand-labelled GeoTIFFs, split CSVs, event metadata, `manifest.json` (sizes + MD5) |
| `processed/` | `python -m training.prepare_data` | `chips/<id>.npz` (s1, s2, rgb, label, otsu, jrc), `index.json` (split, CRS, transform, bounds), `stats.json` (train-split normalisation) |
| `osm_cache/` | `scripts/fetch_road_networks.py` or first routing request | OSMnx GraphML road graphs + JSON sidecars with fetch date and ODbL attribution |

Override locations with `FLOOD_DATA_DIR`, `FLOOD_PROCESSED_DIR` and `FLOOD_OSM_CACHE`.
Re-running any of the scripts is safe: verified files are skipped and nothing is deleted.
