# outputs/

Runtime outputs of the API (git-ignored):

* `predictions/<uuid>/`: one folder per segmentation request, holding `result.json`, `input.png`, `overlay.png`, `mask.png`, `probability.png`, `probability.tif` (GeoTIFF with the source CRS when available) and `probability.npy`. Held-out samples also get `ground_truth.png` and `errors.png`.

The API keeps the newest `FLOOD_MAX_RESULTS` (default 300) result folders and removes only older folders that it created itself (UUID-named).
