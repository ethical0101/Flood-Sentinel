# Limitations and threats to validity

## Segmentation
1. **"Water", not "flood".** Sen1Floods11 hand labels mark all surface water on the image date. The model cannot separate flood water from rivers, lakes or reservoirs. The JRC permanent-water baseline shows that much of the labelled water is not permanent, but the network has no notion of "normal" water.
2. **Single-date snapshots.** Each prediction reflects one Sentinel acquisition. It says nothing about water depth, flow velocity, duration, or how the flood evolved before or after the image.
3. **Resolution.** At 10 m per pixel, roads and narrow streets are about one pixel wide. Water under trees or between buildings is often invisible to both sensors.
4. **Sensor failure modes.** Clouds and cloud shadows corrupt Sentinel-2. Sentinel-1 shows low backscatter on smooth non-water surfaces (tarmac, sand, dry riverbeds), and flooded vegetation gives high backscatter.
5. **Small, clustered dataset.** There are 446 chips from 11 events. The test split shares events with training (different chips), so its scores are optimistic compared with deployment on a new event. The 15-chip Bolivia split is the only true unseen-event test, and it is small, so its confidence intervals are wide.
6. **Fixed threshold.** Results use P > 0.5. No calibration study was done, and the probability is not a calibrated likelihood.
7. **RGB model and photos.** The RGB model is trained on 10 m Sentinel-2 true-colour chips. Drone, aerial or phone photos are out of distribution, and its output for them is indicative only. Such images also carry no geolocation.
8. **Single seed.** Each configuration was trained once (seed 42). The differences between close models (for example S1+S2 versus S2) may lie within seed-to-seed variance.

## Routing
1. **Temporal mismatch.** Imagery dates from 2016–2019; the OSM road network was downloaded in 2026. Roads built or changed in between are not reconciled.
2. **Water on a road ≠ impassable.** Depth, current, road elevation, embankments, bridges and culverts are unknown. Bridges over permanent water may be flagged as flooded.
3. **Alignment error.** OSM geometry and Sentinel pixels can be offset by around 10 m. The default 1-pixel max-filter buffer compensates conservatively, and is configurable.
4. **Clipped networks.** Each study area is a single ≈5 × 5 km chip. Routes outside the raster footprint are not considered, so "disconnected" can mean "connected only outside this area".
5. **Travel time** is a free-flow estimate from OSM `maxspeed` or highway-type means, with a 40 km/h fallback. Traffic, flood-induced slowdowns and closures are not modelled.
6. **Cost weights are policy choices.** The flood penalty (default 10×) and the optional closure rule are user assumptions, not learned or validated parameters.
7. **Default demo routes** were picked automatically to show a trade-off (`scripts/select_demo_routes.py`). They are not real evacuation requests.
8. **Uploaded scenes** need a GeoTIFF with a CRS and geotransform and a footprint ≤ 0.2°, plus internet access the first time to download its roads.

## System
* It is a single-user local prototype. GPU inference is serialised, there is no authentication, and it listens on localhost by default.
* OSM basemap tiles load from the public tile server and need internet. Everything else works offline once data is cached.

## Not a safety system
This is a research prototype. A "risk-aware" route is not a safe route. Do not use it for real evacuation or navigation decisions; rely on official emergency services and verified road-closure information.

## Future work
* Add FloodNet (UAV, multi-class "road flooded / not flooded") for very-high-resolution damage mapping. It would need georeferenced flight logs to be routable.
* Use SpaceNet 8 flooded-road labels to validate edge-level flood classification directly.
* Train on the Sen1Floods11 weakly-labelled set, add multi-seed runs and confidence intervals, and calibrate probabilities.
* Add a DEM (for example Copernicus GLO-30) for depth proxies, plus a permanent-water mask to separate flood water from normal water.
* Accept live Sentinel-1 scenes, for example by user-supplied download from the Copernicus Data Space.
