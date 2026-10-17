# Real-Time Flood Segmentation and Safe-Route Prediction for Disaster Response using Deep Learning

*Project report. All results were produced on 2026-10-18 on a single RTX 3050 Laptop GPU; numbers are reproduced from the artifacts listed in `docs/experimental_results.md`.*

## Abstract
We present an end-to-end prototype that segments surface water in Sentinel-1/Sentinel-2 satellite imagery and uses the georeferenced prediction to plan flood-risk-aware routes on OpenStreetMap roads. Five compact segmentation models were trained locally on the hand-labelled Sen1Floods11 dataset (446 chips, 11 flood events): a U-Net with a ResNet-34 encoder on SAR, optical, combined and RGB inputs, and a DeepLabV3+ with MobileNetV3-Large. On the official held-out test split, the combined-input U-Net reaches a water IoU of 0.823 (F1 0.903), compared with 0.758 for the strongest non-learned baseline (NDWI > 0). On an unseen flood event (Bolivia) the same model reaches 0.784, below the NDWI baseline (0.832). We report this as a limitation in generalisation. Inference takes 17 ms per 512×512 chip. For routing, predictions are sampled every 5 m along OSM road geometry after CRS-aware alignment, and Dijkstra's algorithm minimises a flood-penalised cost. Over 300 random origin–destination pairs in six held-out areas, routes planned on model predictions contain 19.4% less flooded road (measured on hand labels) than shortest routes, at a 4.5% mean detour. An oracle using the labels themselves achieves 21.4%.

## 1. Introduction
Flood response needs timely maps of inundation and of the roads that remain usable. Optical satellites (Sentinel-2) see water clearly but not through cloud; synthetic-aperture radar (Sentinel-1) works in any weather but is noisier. Deep segmentation networks can fuse both. However, a pixel mask alone does not tell a responder how to travel. Linking masks to road networks needs correct georeferencing, a policy for converting water into travel cost, and honest treatment of uncertainty. This project builds and evaluates that full chain on consumer hardware.

## 2. Data
**Sen1Floods11** (Bonafilia et al., 2020) provides 512×512 chips at 10 m resolution in EPSG:4326, with Sentinel-1 VV/VH backscatter, all 13 Sentinel-2 bands, and hand-drawn labels (water / not water / invalid). We used the 446 hand-labelled chips and the official train (252), validation (89) and test (90) splits. The 15-chip Bolivia event, which the authors hold out entirely, measures generalisation to an unseen event. The data was downloaded from the public bucket and all 2,235 files passed MD5 verification. Water makes up 9.5% of valid training pixels. FloodNet (UAV, not georeferenced) and SpaceNet 8 (large, needing AWS tooling) were investigated and not used (see `dataset_report.md`).

Road networks come from OpenStreetMap through OSMnx, one cached graph per study area. The six study areas are held-out test chips (in Spain, Sri Lanka and India) that contain a mapped drive network.

## 3. Methods

### 3.1 Segmentation
* **Input:** each band is standardised with train-split statistics (SAR clipped to [−50, 10] dB). RGB models use a per-image percentile stretch and ImageNet normalisation, which also apply unchanged to uploaded photos.
* **Models:** U-Net (Ronneberger et al., 2015) with a ResNet-34 encoder (He et al., 2016), and DeepLabV3+ (Chen et al., 2018) with MobileNetV3-Large (Howard et al., 2019). Both encoders are ImageNet-initialised, with a single P(water) output.
* **Training:** random 256² crops, flips and rotations. The loss is BCE plus soft Dice restricted to labelled pixels. Optimisation uses AdamW with warmup and cosine decay, fp16 mixed precision and batch 8, for up to 60 epochs with early stopping on validation IoU (patience 15) and seed 42. Peak GPU memory was at most 0.87 GiB.
* **Inference:** sliding-window tiling with ramp blending for arbitrary image sizes; optional flip test-time augmentation.

### 3.2 Evaluation protocol
The positive class is water, and pixels labelled −1 are ignored. Confusion matrices are pooled over each split, from which we report IoU, Dice/F1, precision, recall, mIoU, pixel accuracy and the mean per-chip IoU. The threshold is fixed at 0.5. Model selection uses the validation split only, and the test and Bolivia splits are evaluated once. Baselines:
* a constant predictor
* NDWI > 0 (McFeeters, 1996)
* MNDWI > 0 (Xu, 2006)
* the Sentinel-1 Otsu labels supplied with the dataset
* JRC permanent water

### 3.3 Flood-aware routing
For each road edge, sample points are placed every 5 m along its geometry and reprojected into the raster CRS. Each is mapped to a pixel through the inverse geotransform and read from a 3×3 max-filtered probability raster, which allows for road width and roughly 10 m misregistration. Points outside the raster or on no-data pixels count as unknown. The edge cost is:

c = L + λ_f·L_flooded + λ_u·L·(1−coverage) + λ_p·L·p̄

with defaults λ_f = 10, λ_u = 0.5 and λ_p = 1. The baseline route minimises L. The risk-aware route minimises c using Dijkstra. Alternatives come from the penalty method. An optional rule treats edges with at least N m of predicted flooding as impassable, and the planner falls back explicitly if no path remains. Disconnected networks, far-off snap points and identical endpoints return explained errors.

## 4. Experiments and results
**Segmentation** (pooled water IoU; full tables in `experimental_results.md`):

| Method | Test | Bolivia |
|---|---|---|
| U-Net ResNet-34, S1+S2 | **0.823** | 0.784 |
| U-Net ResNet-34, S2 | 0.814 | 0.770 |
| DeepLabV3+ MobileNetV3-L, S1+S2 | 0.792 | 0.790 |
| U-Net ResNet-34, S1 | 0.666 | 0.615 |
| U-Net ResNet-34, RGB | 0.599 | 0.591 |
| NDWI > 0 | 0.758 | **0.832** |
| S1 Otsu | 0.546 | 0.697 |
| MNDWI > 0 | 0.650 | 0.551 |
| JRC permanent water | 0.345 | 0.065 |

**Latency:** 13–17 ms per 512² chip (model only, fp16, batch 1); 45–53 ms end-to-end for the multispectral (13- and 15-band) models.

**Routing:** over 300 OD pairs in 6 areas, the mean hand-labelled flooded road per trip was 482 m on the shortest route, 389 m on the risk-aware route planned on the model, and 379 m on the oracle route planned on the labels. The mean detour was +4.5%, and 3 pairs got worse. Edge-level agreement between model and labels on "segment contains flooding" was 84.8–99.8% per area.

## 5. Discussion
* **Fusion versus single sensors.** Optical input dominates: adding SAR to S2 changes test IoU by less than 0.01, which is within what a single seed can resolve. SAR alone is substantially weaker but remains the only option under cloud. It beats the Otsu threshold on test, but not on Bolivia.
* **Generalisation.** Every network loses 3–5 IoU points from test to Bolivia, mainly through recall: the main model's precision is 0.962 but its recall only 0.809. NDWI, which has no learned parameters, does not suffer this shift and wins on Bolivia. A learned model is therefore not automatically more robust than a physically motivated index. Better robustness would need more events, weak labels, or index-aware inputs.
* **Compact model.** DeepLabV3+/MobileNetV3 is 5.2× smaller and somewhat faster, and its accuracy is close enough for edge deployment.
* **Routing value.** Model-based routing captures about 90% of the oracle's reduction in flooded exposure. The residual gap mirrors the segmentation false negatives. Benefits are largest where dry alternatives exist (Spain A: 647 → 341 m; Sri Lanka east: 44 → 1 m), and negligible where a flood blocks everything (Spain C, India). In those areas, the right output is a warning rather than a route.

## 6. Limitations
The labels describe water rather than flood depth or road passability. 10 m pixels are about the width of a road. The imagery dates from 2016–2019 while the roads are from 2026. Routing is confined to 5 km tiles. Travel times are free-flow. Each configuration has a single seed, and the out-of-event test is small. The system is a research prototype and must not be used for real evacuation decisions. See `limitations.md`.

## 7. Conclusion
A complete, reproducible pipeline from public satellite data to risk-aware routing runs on a 4 GB laptop GPU. It trains in about 15 minutes per model and infers in milliseconds. The fused U-Net outperforms classic water indices on held-out chips, but not on an unseen event, and routing on its predictions measurably reduces exposure to labelled flood water at a small cost in distance.

## References
* Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. *CVPR Workshops*, 210–211.
* Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126–139.
* Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *ECCV*.
* He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
* Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV*.
* McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425–1432.
* Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
* Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. *International Journal of Remote Sensing*, 27(14), 3025–3033.
* Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269–271.
* OpenStreetMap contributors. Road network data, Open Database License (ODbL) 1.0.
