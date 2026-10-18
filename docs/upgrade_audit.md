# Upgrade audit (v1 → v2)

Audit date: 2026-10-18. Branch `feature/v2-upgrade`, created from `main` @ `afc95b6` (clean working tree). Evidence files: `artifacts/audit/reproduction.json` and `artifacts/audit/weakness_analysis.json`, produced by `scripts/audit_reproduce.py` and `scripts/analyze_weaknesses.py`.

## 1. Environment at audit time
| Item | Finding |
|---|---|
| GPU | RTX 3050 Laptop, 4096 MiB, driver 610.74. At audit start, 375 MiB was held by the v1 API server, which I stopped. |
| Other GPU work | During the audit **another project's job** (`E:\RDD-YOLO`, a YOLO benchmark, PID 25264, 1.8 GB VRAM, 71% utilisation) started on the same GPU. It was left untouched; v2 training was queued to start only after it exited. |
| RAM | 15.7 GiB total, **1.0–1.8 GiB free** (DataLoader workers kept at 2; no in-RAM caching) |
| Disk F: | 150 GB free |
| v1 artifacts | All 6 run folders present (`best.pt`/`last.pt`, history, evaluation, samples). Checkpoints total about 1.9 GB. Raw Sen1Floods11 1.7 GB, processed 4.0 GB, OSM cache 7.5 MB. |
| Training device | `training/train.py` selects `cuda` when available and logs it. Every v1 `train.log` records `device=cuda amp=True`. |

## 2. Were the reported results reproducible?
**Yes.** `scripts/audit_reproduce.py` re-ran every `best.pt` on the **raw GeoTIFFs**, deliberately bypassing the `.npz` cache and `evaluate.py`, with an independent IoU implementation (boolean set arithmetic).

| Model | Test IoU (reported → reproduced) | Bolivia IoU (reported → reproduced) |
|---|---|---|
| U-Net ResNet-34 S1+S2 | 0.8233 → 0.8233 | 0.7836 → 0.7835 |
| U-Net ResNet-34 S2 | 0.8143 → 0.8143 | 0.7696 → 0.7696 |
| U-Net ResNet-34 S1 | 0.6658 → 0.6658 | 0.6154 → 0.6155 |
| U-Net ResNet-34 RGB | 0.5990 → 0.5990 | 0.5911 → 0.5911 |
| DeepLabV3+ MobileNetV3 S1+S2 | 0.7921 → 0.7921 | 0.7896 → 0.7896 |
| NDWI > 0 (from raw digital numbers) | 0.7578 → 0.7578 | 0.8320 → 0.8320 |

The differences of 0.0001 come from the float16 storage in the `.npz` cache against float32 raw data.

**IoU definition (verified in `training/metrics.py`):** IoU of the *water* class, TP / (TP + FP + FN), with confusion counts **pooled over all valid pixels of a split**. Pixels labelled −1 are excluded. "mIoU" is the mean of the water and not-water IoU.

## 3. Split integrity and leakage
| Check | Result |
|---|---|
| Chip IDs shared between splits | 0 |
| Identical S2 content across chips (SHA-1 of arrays) | 0 groups |
| Bolivia chips in train/valid | 0 |
| Bolivia used for model selection in v1? | **No.** Checkpoints were chosen on validation IoU (`train.py`) and the threshold was fixed at 0.5. Bolivia was evaluated once per model, after training. The v1 *report* discussed Bolivia results, and **the v2 experiment ideas below were motivated by the Bolivia error analysis**. That is a mild design leak, so v2 adds an untouched external test set (§6). |
| Geographic proximity, test ↔ train | **Not independent.** Median distance from a test chip to its nearest training chip is 10.2 km (min 4.0 km); **42% of test chips lie within 10 km of a training chip, and 4 test chips' footprints overlap a training chip** (chips are 5.1 km wide). Same events and same acquisition dates. |
| Geographic proximity, valid ↔ train | Median 11.2 km; 36% within 10 km; 3 overlapping footprints |
| Geographic proximity, Bolivia ↔ train | ≥ 1,216 km; no overlap. **Genuinely independent.** |

**Consequence:** test-split IoU measures *in-distribution, spatially autocorrelated* performance. It must not be presented as generalisation. Bolivia (15 chips, 1 event) is the only independent test inside Sen1Floods11, and it is small.

## 4. Baseline audit
* NDWI = (B3 − B8)/(B3 + B8) > 0 and MNDWI = (B3 − B11)/(B3 + B11) > 0 use **only Sentinel-2 bands the S2/S1+S2 models also receive**, so NDWI uses no information hidden from the model. Its threshold (0) is the textbook value, not tuned on any split.
* S1 Otsu labels and JRC permanent water are layers shipped with the dataset. Otsu thresholds were derived per chip by the dataset authors, so it is a per-image adaptive method.
* Caveat: Sen1Floods11 hand labels were drawn by analysts looking at the Sentinel imagery. Spectral-index baselines may share some of the labellers' visual cues, so the baseline is not independent of the labels.

## 5. Weakness analysis (main model, v1)
| # | Issue | Evidence | Proposed experiment | Expected benefit |
|---|---|---|---|---|
| W1 | **Radiometric domain shift on Bolivia** | Mean normalised S2 input on Bolivia: −0.10 to −0.68 std in every band (B4 −0.36, B11 −0.46, B9 −0.68); test/valid stay within ±0.06. NDWI, a ratio insensitive to multiplicative brightness, is the only method that does *better* on Bolivia than on test. | **R1:** physically motivated radiometric augmentation (per-chip S2 gain 0.75–1.25 with ±5% per-band jitter; S1 ±2 dB offset with ±0.5 dB per polarisation) | Invariance to brightness and calibration differences, so higher Bolivia recall |
| W2 | **Misses optically obvious water under shift** | Where NDWI > 0, the model misses 2.6% of water on test but **10.5% on Bolivia** (42k pixels, about half of all Bolivia FNs). NDWI beats the model on **10 of 13** Bolivia chips that have more than 500 water pixels, but only on 15 of 78 test chips. | **R2:** add NDWI and MNDWI as two extra input channels | A shift-invariant spectral cue at the input |
| W3 | Recall deficit / threshold | Bolivia precision 0.962 against recall 0.809. A validation-only threshold sweep selects 0.40, which raises valid/test/Bolivia IoU to 0.845/0.825/0.794. Calibration is good (ECE 0.009 / 0.015 / 0.026). | Use the **validation-selected threshold** for every v2 model; report both 0.5 and the selected value | Small, legitimate gain without touching the held-out data |
| W4 | Water missed where SAR is bright | FN rate where VH ≥ −20 dB: 0.51 on test and 0.63 on Bolivia (against 0.04 and 0.12 where VH < −20 dB). This is flooded vegetation, wind-roughened water or label noise. | Not addressed by a cheap experiment. Documented as a limitation. | – |
| W5 | Cloud | On test, FN rate is 0.25 under the cloud proxy against 0.12 in clear sky. On Bolivia only 884 water pixels are cloudy, so cloud is **not** the Bolivia problem. | Not prioritised | – |
| W6 | Normalisation statistics include S2 no-data | 2.0% of training pixels are S2 all-zero fill, and they were included in mean/std | Not changed: a small effect, and changing the statistics would confound the controlled comparison | – |
| W7 | Class imbalance | Water is 9.5% of valid training pixels (11.0% valid, 12.5% test, 15.9% Bolivia). Already handled by BCE + Dice; the remaining recall deficit is covered by W3. | – | – |
| W8 | No-data handling | Bolivia has 10.3% SAR NaN, 9.4% S2 no-data and 27% unlabelled pixels, against about 2% in train. NaN is set to 0 after standardisation (the training mean). | Monitored. Not changed in v2. | – |
| W9 | Single seed | Each v1 configuration was trained once, so model differences under about 0.01 IoU are uninterpretable | **R0b:** the v1 recipe with seed 7, to measure seed-to-seed noise | Honest effect sizes |
| W10 | Test split not spatially independent | See §3 | **External test set:** WorldFloods v2 test split (§6) | A credible generalisation estimate |

## 6. Additional authentic data
**WorldFloods v2** (Portalés-Julià et al., *Scientific Reports* 13, 20316, 2023), Hugging Face `isp-uv-es/WorldFloodsv2`. It is public and not gated, licensed **CC BY-NC 4.0** (stated in the README and the dataset tags), and contains 509 Sentinel-2 scenes in total (≈ 76 GB).

* **Only the test split was downloaded** (74 files, 3.08 GB: S2, ground truth, JRC permanent water, metadata). Each LFS file is verified by SHA-256 (`scripts/download_worldfloods_test.py`).
* Its events are Copernicus EMS activations labelled by a different team from Sen1Floods11.
* **Use:** evaluation only. It is never used for training, threshold selection or choosing between models.
* **Compatibility:** Sentinel-2 only, with no Sentinel-1. It can therefore evaluate the S2-only models (v1 `unet_resnet34_s2`, v2 R4 `s2i`) and the NDWI/MNDWI baselines, but **not** the S1+S2 models. Label mapping and resolution handling are documented in `experimental_results.md`.
* FloodNet (UAV RGB) was not re-considered for v2. It would need a separate pipeline and does not address any weakness in this table.

## 7. Pre-registered v2 protocol
1. Every run uses identical data, splits, epochs, early stopping and evaluation code. Only the listed factor changes.
2. **Selection rule (fixed before training):** the v2 default model is the run with the highest *validation* IoU at its *validation-selected* threshold. If no v2 run beats the v1 baseline by more than the seed-noise range measured by R0b, v1 stays the default.
3. Test, Bolivia and WorldFloods results are reported for every run, including runs that get worse.

## 8. Outcome of the v2 experiments
Full tables are in `experimental_results.md`, Part A.

| Hypothesis | Experiment | Outcome |
|---|---|---|
| W1: brightness shift limits Bolivia | R1 radiometric augmentation | **Not supported.** Bolivia 0.789 vs 0.794 (v1); bootstrap interval spans 0 |
| W2: shift-invariant spectral cue helps | R2 / R3 / R4 index channels | **Supported on Bolivia** (+0.020 to +0.024 IoU, recall +0.04 to +0.06; chip-bootstrap 95% CI excludes 0), but the gain is the same size as seed spread (−0.024), and **not replicated on WorldFloods**, where the S2+indices recipe is 0.014 worse than v1 S2 |
| W3: threshold | validation-selected threshold, all runs | **Supported:** small consistent gains (v1 main model: test +0.002, Bolivia +0.010) |
| W9: seed noise | R0b | Validation ±0.0027; test ±0.0001; Bolivia −0.024 (large) |

**Default model:** unchanged (`unet_resnet34_s1s2`). The dashboard threshold slider still defaults to 0.5; the validation-selected value for each model (0.40 for the default) is reported in the evaluation lab and can be set manually. No v2 run beat v1 on validation by more than seed noise (pre-registered rule, §7). The v2 runs remain available for comparison in the dashboard.

**Integrity notes:**
* Bolivia and WorldFloods were never used to select a checkpoint, a threshold or the default model.
* The v2 *hypotheses* were informed by the Bolivia error analysis. That is why the WorldFloods result is the deciding check, and it did not confirm the improvement.
