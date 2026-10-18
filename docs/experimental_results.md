# Experimental results

All numbers below come from `artifacts/summary.json`, `artifacts/baselines.json`, `artifacts/runs/*/evaluation.json`, `artifacts/runs/*/history.json` and `artifacts/routing_evaluation.json`, all produced by the code in this repository on **2026-10-18**. Nothing was copied from the literature or adjusted by hand.

v2 adds the following machine-readable sources:
* `artifacts/experiments.{csv,json}` (registry)
* `artifacts/worldfloods_evaluation.json` (external test)
* `artifacts/v2_bootstrap.json` (paired chip bootstrap)
* `artifacts/audit/*.json` (audit)

---

# Part A: v2 upgrade (controlled experiments)

## A1. Verification of v1
Every v1 checkpoint was re-run on the **raw GeoTIFFs** with independent code (`scripts/audit_reproduce.py`). All held-out IoUs reproduce to within 0.0001. See `docs/upgrade_audit.md` for the leakage findings: the test split is spatially close to training, and only Bolivia is independent inside Sen1Floods11.

## A2. Protocol
* Same data, splits, epochs (≤ 60), early stopping (patience 15), optimiser and evaluation code as v1. **One factor changes per run.**
* Each run's decision threshold is **selected on validation only** (grid 0.10–0.90, step 0.05). Test and Bolivia are reported at that threshold, and also at 0.5 in `evaluation.json`.
* **Pre-registered selection rule:** the v2 default would be the run with the highest validation IoU, but only if it beats v1 by more than the seed noise measured by R0b. Otherwise v1 stays the default.
* Hardware: RTX 3050 Laptop (4 GB). Peak training memory 0.87 GB (1.15 GB for R0b, which resumed after a pause). The GPU was shared with another project, so runs paused and resumed via `--polite`. Wall-clock times include those pauses.

## A3. Results (IoU of the water class, each run at its validation-selected threshold)
| Run | Change vs v1 | Thr. | **Val IoU** (selection) | Test IoU | Bolivia IoU | Bolivia recall | Train time |
|---|---|---|---|---|---|---|---|
| `unet_resnet34_s1s2` (v1) | – (seed 42) | 0.40 | 0.8454 | 0.8253 | 0.7939 | 0.826 | 15.7 min |
| `v2_r0b_unet_s1s2_seed7` | seed 7 (noise reference) | 0.45 | 0.8427 | 0.8254 | 0.7695 | 0.801 | 14.3 min |
| `v2_r1_unet_s1s2_radiometric` | radiometric augmentation | 0.40 | **0.8479** | 0.8227 | 0.7892 | 0.841 | 19.5 min |
| `v2_r2_unet_s1s2i_indices` | + NDWI/MNDWI channels | 0.45 | 0.8471 | 0.8259 | 0.8135 | 0.863 | 20.4 min |
| `v2_r3_unet_s1s2i_radiometric` | both | 0.40 | 0.8449 | **0.8265** | 0.8173 | 0.862 | 20.2 min |
| `v2_r4_unet_s2i_radiometric` | both, S2 only (no SAR) | 0.40 | 0.8360 | 0.8248 | **0.8181** | **0.885** | 19.5 min |
| *NDWI > 0 baseline* | – | fixed | – | 0.7578 | 0.8320 | 0.888 | – |

v1 models evaluated again with validation-selected thresholds (previously reported at 0.5):

| Model | Thr. | Test IoU (0.5 → val thr.) | Bolivia IoU (0.5 → val thr.) |
|---|---|---|---|
| U-Net S1+S2 | 0.40 | 0.8233 → 0.8253 | 0.7836 → 0.7939 |
| U-Net S2 | 0.45 | 0.8143 → 0.8159 | 0.7696 → 0.7740 |
| U-Net S1 | 0.40 | 0.6658 → 0.6659 | 0.6154 → 0.6270 |
| U-Net RGB | 0.40 | 0.5990 → 0.6009 | 0.5911 → 0.5920 |
| DeepLabV3+ S1+S2 | 0.60 | 0.7921 → 0.7886 | 0.7896 → 0.7874 |

## A4. Significance (paired chip bootstrap vs v1, 2,000 resamples)
| Run | Test ΔIoU [95% CI] | Bolivia ΔIoU [95% CI] |
|---|---|---|
| R0b (seed 7) | +0.0001 [−0.011, +0.011] | −0.0244 [−0.102, +0.005] |
| R1 radiometric | −0.0025 [−0.012, +0.006] | −0.0047 [−0.021, +0.013] |
| R2 indices | +0.0006 [−0.009, +0.007] | **+0.0197 [+0.012, +0.035]** |
| R3 indices + radiometric | +0.0013 [−0.002, +0.005] | **+0.0235 [+0.014, +0.041]** |
| R4 S2 + indices + radiometric | −0.0004 [−0.006, +0.005] | **+0.0242 [+0.012, +0.043]** |

The bootstrap resamples chips, not training seeds. Retraining the unchanged v1 recipe with another seed moved Bolivia IoU by −0.024, the same order of magnitude as the index-channel gains.

## A5. External evaluation: WorldFloods v2 test split (18 scenes, never used for training or selection)
Clear-sky valid pixels (secondary: all valid pixels). Only models that need Sentinel-2 alone can run here.

| Method | Clear-sky IoU @0.5 | Clear-sky IoU @val thr. | All-valid IoU @val thr. |
|---|---|---|---|
| **v1 U-Net S2** (`unet_resnet34_s2`) | 0.7787 | **0.7809** (0.45) | 0.7728 |
| v2 R4 U-Net S2 + indices + radiometric | 0.7628 | 0.7668 (0.40) | 0.7573 |
| v1 U-Net RGB | 0.5480 | 0.5636 (0.40) | 0.5516 |
| MNDWI > 0 | 0.7480 | – | 0.7386 |
| NDWI > 0 | 0.5329 | – | 0.5230 |

## A6. Conclusions
1. **Model selection:** the best validation IoU (R1, 0.8479) beats v1 by only +0.0025, which is below the measured seed noise (0.0027). Under the pre-registered rule, **the v1 model remains the default.** No v2 run is a demonstrated improvement on the in-distribution test split: every test-split 95% interval spans zero.
2. **Validation-selected thresholds** are a small, legitimate gain at no cost. The v1 main model goes from Bolivia 0.784 to 0.794 and test 0.823 to 0.825.
3. **Generalisation, Bolivia:** every model with NDWI/MNDWI index channels (R2, R3, R4) scores 0.814–0.818 on Bolivia, against 0.770–0.794 for the two seeds of the v1 recipe. Recall rises (0.86–0.89 against 0.80–0.83), and the chip bootstrap excludes zero. Two cautions apply: the gain is comparable to seed-to-seed spread, and the hypothesis came from analysing Bolivia errors. This is **suggestive but not conclusive.** With index channels the networks now match the NDWI baseline's recall on Bolivia, but still not its IoU (0.818 against 0.832).
4. **Generalisation, WorldFloods:** the improvement **does not replicate** on the independent external test. The v2 S2-only recipe (R4) is 0.014 IoU *worse* than the v1 S2-only model. Both learned S2 models beat both index baselines there (0.78 against MNDWI 0.75 and NDWI 0.53). The NDWI advantage seen on Bolivia is specific to that event.
5. **Radiometric augmentation alone** (R1) changed neither test nor Bolivia beyond noise, so brightness invariance is not what limits Bolivia performance.
6. Net: v2 delivers a more honest and better-characterised model rather than a better one. The documented effect is validation-selected thresholds. The index-channel idea is a lead that needs multi-seed confirmation and more independent events.

## A7. Multi-dataset training (Sen1Floods11 + WorldFloods v2 training subset)
**Data:**
* The WorldFloods v2 training subset (`configs/worldfloods_train_subset.json`, `scripts/select_worldfloods_subset.py`): one scene per flood event, smallest first, a 5 GB budget, and every scene within 10 km of a Sen1Floods11 valid, test or Bolivia chip excluded (4 scenes).
* That gives 97 scenes (4.85 GB, SHA-256 verified). They were tiled into **303 tiles of up to 512² from 75 events**, with 68 M labelled pixels and 18.5% water (`training/prepare_worldfloods.py`). Cloud and no-data pixels are ignored.
* Two scenes were too small (< 256 px); the other 20 left over produced no tile with at least 50% labelled pixels after cloud masking.

**Training:** Sen1Floods11 chips give 4 random crops each per epoch and WorldFloods tiles 1 each (1,311 samples per epoch). The models are optical only, because WorldFloods has no Sentinel-1. Everything else (stats, optimiser, early stopping, selection on Sen1Floods11 validation) is unchanged. Each run is paired with the single-dataset run of the identical recipe.

| Run | Pair | Best epoch / run | Val IoU | Test IoU | Bolivia IoU | WorldFloods IoU (clear) |
|---|---|---|---|---|---|---|
| `unet_resnet34_s2` (v1) | – | 54 / 60 | 0.8411 | 0.8159 | 0.7740 | **0.7809** |
| `md1_unet_s2_worldfloods` | vs v1 S2 | 26 / 41 | 0.8382 | 0.8064 | **0.8111** | 0.7608 |
| `v2_r4_unet_s2i_radiometric` | – | 53 / 60 | 0.8360 | 0.8248 | 0.8181 | 0.7668 |
| `md2_unet_s2i_radiometric_worldfloods` | vs R4 | 11 / 26 | 0.8010 | 0.7896 | 0.7493 | 0.7419 |

All values are at the validation-selected threshold.

Paired chip bootstrap (multi-dataset minus single-dataset):

| Run | Test ΔIoU [95% CI] | Bolivia ΔIoU [95% CI] |
|---|---|---|
| md1 vs v1 S2 | **−0.0095 [−0.015, −0.006]** | **+0.0371 [+0.008, +0.074]** |
| md2 vs R4 | **−0.0353 [−0.059, −0.020]** | **−0.0688 [−0.123, −0.046]** |

**Why it did not help (measured):**
1. **Cross-sensor, cross-date labels.** Only 2 of the 75 scenes used have masks derived from the Sentinel-2 image itself. The rest come from Copernicus EMS / UNOSAT products made from Sentinel-1, COSMO-SkyMed, RADARSAT-2, TerraSAR-X, Pléiades, SPOT, Landsat-8 and others, often on a different date. The NDWI-vs-label IoU on the training tiles is **0.58 for WorldFloods against 0.65 for Sen1Floods11**, and about 0.4 for RADARSAT-2- and Landsat-8-derived labels. The network is being supervised with water it often cannot see in the S2 image.
2. **Early stopping on the Sen1Floods11 validation split** ended training sooner (best epoch 26 and 11, against 54 and 53), so the models also saw fewer updates on the target distribution.
3. md2, which combined the noisier data with radiometric augmentation and index channels, was the least stable.

**Conclusion:** with this data budget and label quality, adding WorldFloods training data is **not an improvement**. md1 gains on Bolivia (+0.037, significant) but loses on the test split and on WorldFloods itself, and md2 is worse everywhere. Neither run would be selected under the pre-registered rule, so **the default model is unchanged**. A clean-label subset (S2-derived or same-day very-high-resolution optical masks only) or a larger budget would be the next experiment.

---

# Part B: v1 results (unchanged; for reference)

## Setup
| | |
|---|---|
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), driver 610.74; Intel i5-12500H; 15.7 GiB RAM |
| Software | Python 3.13.4, PyTorch 2.14.1+cu130, cuDNN 9.24, segmentation_models_pytorch 0.5.0, timm 1.0.30 |
| Data | Sen1Floods11 v1.1 hand-labelled: train 252 / valid 89 / test 90 / Bolivia 15 chips (official splits) |
| Training | 4 random 256² crops per chip per epoch (1,008 samples), batch 8, AdamW 5e-4, warmup 2 epochs + cosine, BCE + Dice, fp16 AMP, up to 60 epochs, early-stopping patience 15 on validation IoU, seed 42, single run per configuration |
| Selection | `best.pt` = highest validation IoU (water); test and Bolivia evaluated once afterwards |
| Threshold | 0.5, fixed (not tuned) |
| Metric pooling | Confusion matrix accumulated over all valid pixels of a split; label −1 ignored; positive class = water |

## Training runs
| Run | Input | Params | Epochs run | Best epoch | Best val IoU | Wall time | Peak GPU mem |
|---|---|---|---|---|---|---|---|
| unet_resnet34_s1s2 | S1+S2 (15) | 24.47 M | 46 (early stop) | 31 | 0.8445 | 15.7 min | 867 MiB |
| unet_resnet34_s2 | S2 (13) | 24.47 M | 60 | 54 | 0.8409 | 19.6 min | 861 MiB |
| unet_resnet34_s1 | S1 (2) | 24.43 M | 59 (early stop) | 44 | 0.6871 | 14.6 min | 828 MiB |
| unet_resnet34_rgb | RGB (3) | 24.44 M | 60 | 56 | 0.6268 | 14.7 min | 830 MiB |
| deeplabv3plus_mnv3l_s1s2 | S1+S2 (15) | 4.71 M | 45 (early stop) | 30 | 0.8087 | 13.6 min | 655 MiB |

All five runs completed without errors. The total training time was about 78 minutes, and no out-of-memory mitigation was needed. The smoke run (`smoke_unet_resnet34_s1s2`, 2 epochs on 16 samples) only checked the pipeline; it is excluded from every result.

## Held-out segmentation results (water class, threshold 0.5)

### Test split (90 chips, events seen during training, chips unseen)
| Method | IoU | F1 / Dice | Precision | Recall | mIoU | Pixel acc. | Mean chip IoU |
|---|---|---|---|---|---|---|---|
| **U-Net ResNet-34, S1+S2** | **0.8233** | **0.9031** | 0.9412 | 0.8679 | **0.8986** | **0.9767** | **0.6152** |
| U-Net ResNet-34, S2 | 0.8143 | 0.8976 | 0.9447 | 0.8550 | 0.8935 | 0.9756 | 0.6072 |
| DeepLabV3+ MobileNetV3-L, S1+S2 | 0.7921 | 0.8840 | 0.9158 | 0.8543 | 0.8803 | 0.9720 | 0.5554 |
| U-Net ResNet-34, S1 | 0.6658 | 0.7994 | 0.8373 | 0.7648 | 0.8064 | 0.9520 | 0.3918 |
| U-Net ResNet-34, RGB | 0.5990 | 0.7492 | 0.8249 | 0.6862 | 0.7681 | 0.9425 | 0.3070 |
| *Baseline:* NDWI > 0 (S2) | 0.7578 | 0.8622 | 0.9226 | 0.8093 | 0.8609 | 0.9677 | – |
| *Baseline:* MNDWI > 0 (S2) | 0.6504 | 0.7882 | 0.6755 | 0.9460 | 0.7891 | 0.9364 | – |
| *Baseline:* S1 Otsu labels | 0.5458 | 0.7062 | 0.6811 | 0.7332 | 0.7309 | 0.9237 | – |
| *Baseline:* JRC permanent water | 0.3446 | 0.5126 | 0.9536 | 0.3505 | 0.6287 | 0.9166 | – |
| *Baseline:* constant "not water" | 0.0000 | undefined | undefined | 0.0000 | 0.4375 | 0.8749 | – |

### Bolivia (15 chips, an entire flood event never seen in training or validation)
| Method | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| *Baseline:* **NDWI > 0 (S2)** | **0.8320** | **0.9083** | 0.9291 | 0.8884 |
| DeepLabV3+ MobileNetV3-L, S1+S2 | 0.7896 | 0.8824 | 0.8895 | 0.8755 |
| U-Net ResNet-34, S1+S2 | 0.7836 | 0.8787 | **0.9621** | 0.8085 |
| U-Net ResNet-34, S2 | 0.7696 | 0.8698 | 0.9499 | 0.8022 |
| *Baseline:* S1 Otsu labels | 0.6972 | 0.8216 | 0.7809 | 0.8667 |
| U-Net ResNet-34, S1 | 0.6154 | 0.7619 | 0.8551 | 0.6870 |
| U-Net ResNet-34, RGB | 0.5911 | 0.7430 | 0.7960 | 0.6966 |
| *Baseline:* MNDWI > 0 | 0.5514 | 0.7108 | 0.5608 | 0.9703 |
| *Baseline:* JRC permanent water | 0.0646 | 0.1213 | 0.7712 | 0.0658 |

![IoU comparison](../artifacts/figures/iou_comparison.png)

### Confusion matrices: main model (U-Net ResNet-34, S1+S2)
| Split | TN | FP | FN | TP |
|---|---|---|---|---|
| Test | 17,812,104 | 139,162 | 338,897 | 2,227,204 |
| Bolivia | 2,398,465 | 14,486 | 87,096 | 367,768 |

Errors are dominated by **false negatives** (missed water), roughly 2.4× (test) and 6× (Bolivia) more common than false positives.

### Per-event test IoU (main model)
| Ghana | India | Mekong | Nigeria | Pakistan | Paraguay | Somalia | Spain | Sri Lanka | USA |
|---|---|---|---|---|---|---|---|---|---|
| 0.544 | 0.905 | 0.938 | 0.942 | 0.616 | 0.620 | 0.820 | 0.852 | 0.927 | 0.818 |

## Inference latency
Batch 1, input 1×C×512×512, fp16 autocast, CUDA-synchronised, RTX 3050 Laptop GPU, 30 timed iterations after 5 warm-up passes. "End-to-end" adds normalisation of an in-memory chip, tiling and copying the result back to the CPU. File decoding and PNG/GeoTIFF writing are excluded.

| Model | Model-only mean | p95 | End-to-end mean | Throughput (model only) |
|---|---|---|---|---|
| U-Net ResNet-34, S1+S2 | 17.1 ms | 25.2 ms | 46.3 ms | 58 chips/s |
| U-Net ResNet-34, S2 | 16.7 ms | 18.3 ms | 52.8 ms | 60 chips/s |
| U-Net ResNet-34, S1 | 14.9 ms | 19.0 ms | 22.9 ms | 67 chips/s |
| U-Net ResNet-34, RGB | 14.3 ms | 15.7 ms | 22.7 ms | 70 chips/s |
| DeepLabV3+ MobileNetV3-L, S1+S2 | 13.0 ms | 17.6 ms | 44.9 ms | 77 chips/s |

Through the API, a single 512² GeoTIFF upload measured 238 ms in total, including decoding, inference and writing five output files. The first request after start-up also loads the model.

## Routing evaluation against hand labels
`scripts/evaluate_routing.py`: 6 held-out study areas × 50 random origin–destination pairs (seed 123, ≥ 1 km apart, largest strongly connected component). Routes are planned on the **main model's prediction**. The flooded length of each route is then **measured on the hand label**, with the same 5 m sampling and 1 px buffer. The oracle plans directly on the hand label.

| Area | Edge agreement (flooded?) | Flooded road, model / label | Labelled flood per trip: shortest | Risk-aware (model) | Oracle |
|---|---|---|---|---|---|
| spain_7370579 | 96.3% | 58.9 / 58.4 km | 647 m | 341 m | 341 m |
| spain_6095801 | 98.8% | 9.1 / 8.7 km | 21 m | 9 m | 10 m |
| spain_7387658 | 91.3% | 86.9 / 102.6 km | 1,435 m | 1,296 m | 1,289 m |
| srilanka_713926 | 99.8% | 2.5 / 2.4 km | 44 m | 1 m | 1 m |
| srilanka_649970 | 93.7% | 1.6 / 3.0 km | 109 m | 99 m | 48 m |
| india_747992 | 84.8% | 5.6 / 7.2 km | 637 m | 586 m | 586 m |
| **All 300 pairs** | | | **482 m** | **389 m** | **379 m** |

* Planning on the model's prediction cut labelled-flooded road per trip by **19.4%** compared with the shortest route, for a mean detour of **+4.5%**. The oracle reaches 21.4%, so the model captures about 90% of the achievable reduction under this cost function.
* In 3 of 300 pairs the model-planned route crossed *more* labelled flood than the shortest route.
* In the heavily flooded areas (Spain C, India), little reduction is possible, because almost every connection crosses water.
* Edge-level agreement between model and label on "segment has flooding" ranges from 84.8% to 99.8%. Where the model under-segments (Spain C, Sri Lanka south, India) it reports *less* flooded road than the label, consistent with the false-negative bias above.

## Interpretation

1. **Learned models beat the baselines on the test split.** The S1+S2 U-Net reaches IoU 0.823 against 0.758 for the strongest baseline (NDWI > 0), and is better on every pooled metric.
2. **On the unseen Bolivia event, the simple NDWI threshold is best** (0.832 against 0.790 for the best model). The networks keep very high precision (0.96) but miss more water (recall 0.81), which suggests they under-segment under a domain shift. With only 15 Bolivia chips from one event this difference is not conclusive, but it means **no claim of better generalisation than a spectral index is supported**. This is the main negative result.
3. **Optical bands carry most of the signal.** S1+S2 and S2 alone differ by only 0.009 IoU on test, which is likely within seed variance (one seed per configuration). SAR alone is clearly weaker (0.666), although it still beats the Otsu SAR threshold on test (0.546). On Bolivia it does *not* (0.615 against 0.697).
4. **RGB alone is weakest (0.599).** Without NIR/SWIR it falls below even a simple NDWI. This bounds what the RGB upload path can deliver.
5. **The compact DeepLabV3+ trades a little accuracy for size.** With 5.2× fewer parameters it loses 3.1 IoU points on test, is about 1.3× faster on the GPU, and is marginally better on Bolivia (0.790 against 0.784).
6. **Chip-mean IoU (0.615) is far below pooled IoU (0.823).** Chips with little water, such as narrow rivers, score poorly (see `samples/USA_670826.png`, IoU 0.26, and `samples/Paraguay_913449.png`, IoU 0.32). Ghana, Pakistan and Paraguay are the weakest events.

## Known caveats of these results
* There is a single seed per configuration and no confidence intervals. Differences under about 0.01 IoU should not be over-interpreted.
* The test split shares flood events with training; Bolivia is the only out-of-event test, and it is small.
* No hyper-parameter search was performed. The same defaults were used for every run.
* The routing evaluation uses the hand label as ground truth for "flooded road". That label describes water on the image date, not road passability.
