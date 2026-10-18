"""Phase-1 weakness analysis for the v1 main model (evidence for docs/upgrade_audit.md).

Computes, per split (train / valid / test / bolivia):
  * class balance, invalid-label share, S1 NaN share, S2 all-zero (no-data) share
  * per-band shift of the normalised inputs relative to train (mean z-score)
  * whether S2 no-data zeros contaminate the normalisation statistics
For the main model (best.pt) on valid / test / bolivia:
  * threshold sweep (selection on VALID only) and calibration (ECE, 10 bins)
  * false-negative analysis stratified by NDWI sign, cloud proxy, JRC permanent water, S1 VH
  * per-chip IoU of model vs NDWI (where does NDWI win?)

    .venv\\Scripts\\python scripts\\analyze_weaknesses.py
Writes artifacts/audit/weakness_analysis.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training.dataset import chip_input, load_chip, load_index  # noqa: E402
from training.infer import Predictor  # noqa: E402
from training.utils import artifacts_dir, processed_dir, write_json  # noqa: E402

THRESHOLDS = np.round(np.arange(0.1, 0.91, 0.05), 2)


def ndwi(s2):
    g, n = s2[2].astype(np.float32), s2[7].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (g - n) / (g + n)


def main() -> None:
    idx = load_index()
    stats = json.loads((processed_dir() / "stats.json").read_text())
    by = defaultdict(list)
    for c in idx:
        by[c["split"]].append(c["id"])
    out: dict = {"data": {}, "model": {}}

    # ------------------------------------------------------------ data-level statistics
    s2_zero_train = 0
    s2_px_train = 0
    for split, ids in by.items():
        lab = np.zeros(3, np.int64)
        s1nan = s2zero = px = 0
        zsum = np.zeros(15)
        zn = 0
        for cid in ids:
            ch = load_chip(cid)
            lab += np.array([(ch["label"] == v).sum() for v in (-1, 0, 1)])
            s1 = ch["s1"].astype(np.float32)
            s2 = ch["s2"].astype(np.float32)
            s1nan += int((~np.isfinite(s1)).any(0).sum())
            zero = (s2 == 0).all(0)
            s2zero += int(zero.sum())
            px += zero.size
            x = chip_input(ch, "s1s2", stats)
            ok = np.isfinite(s1).all(0) & ~zero
            if ok.any():
                zsum += x[:, ok].mean(1)
                zn += 1
            if split == "train":
                s2_zero_train += int(zero.sum())
                s2_px_train += zero.size
        out["data"][split] = {
            "chips": len(ids),
            "label_share": {"invalid": lab[0] / lab.sum(), "not_water": lab[1] / lab.sum(), "water": lab[2] / lab.sum()},
            "water_share_of_valid": lab[2] / max(1, lab[1] + lab[2]),
            "s1_nan_pixel_share": s1nan / px,
            "s2_nodata_pixel_share": s2zero / px,
            "mean_normalised_input_per_band (valid px)": dict(zip(
                ["VV", "VH", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
                np.round(zsum / max(zn, 1), 3).tolist())),
        }
    out["data"]["stats_contamination"] = {
        "train_s2_nodata_pixel_share": s2_zero_train / s2_px_train,
        "note": "v1 stats.json included S2 all-zero no-data pixels in mean/std; share shown here",
    }

    # ------------------------------------------------------------ model-level analysis
    pred = Predictor("unet_resnet34_s1s2")
    for split in ("valid", "test", "bolivia"):
        cms = {t: np.zeros(4, np.int64) for t in THRESHOLDS}
        bins = np.zeros((10, 2))  # sum prob, sum label per confidence bin
        bin_n = np.zeros(10)
        strata = defaultdict(lambda: np.zeros(2, np.int64))  # [FN, water px]
        chip_rows = []
        for cid in by[split]:
            ch = load_chip(cid)
            p = pred.predict(chip_input(ch, "s1s2", pred.stats))
            y = ch["label"]
            v = y != -1
            t = y == 1
            for th in THRESHOLDS:
                pp = p > th
                cms[th] += [(pp & t & v).sum(), (pp & ~t & v).sum(), (~pp & t & v).sum(), (~pp & ~t & v).sum()]
            b = np.clip((p[v] * 10).astype(int), 0, 9)
            np.add.at(bins[:, 0], b, p[v])
            np.add.at(bins[:, 1], b, t[v])
            np.add.at(bin_n, b, 1)
            s2 = ch["s2"].astype(np.float32)
            nd = ndwi(s2)
            cloud = (s2[1] > 0.18) | (s2[10] > 0.012)  # blue-band brightness or cirrus proxy
            jrc = ch["jrc"] == 1
            vh = ch["s1"][1].astype(np.float32)
            fn = (p <= 0.5) & t & v
            w = t & v
            for name, m in {"ndwi>0": nd > 0, "ndwi<=0": ~(nd > 0), "cloud_proxy": cloud, "clear": ~cloud,
                            "jrc_permanent": jrc, "not_jrc (flood only)": ~jrc,
                            "vh<-20dB (dark SAR)": vh < -20, "vh>=-20dB": ~(vh < -20)}.items():
                strata[name] += [(fn & m).sum(), (w & m).sum()]
            pi = ((p > 0.5) & t & v).sum() / max(1, ((p > 0.5) | t)[v].sum())
            ni = ((nd > 0) & t & v).sum() / max(1, ((nd > 0) | t)[v].sum())
            chip_rows.append({"chip": cid, "water_px": int(w.sum()), "model_iou": float(pi), "ndwi_iou": float(ni),
                              "cloud_share": float(cloud[v].mean()) if v.any() else None})
        sweep = {}
        for th, (tp, fp, fn_, tn) in cms.items():
            sweep[float(th)] = {"iou": tp / max(1, tp + fp + fn_), "precision": tp / max(1, tp + fp),
                                "recall": tp / max(1, tp + fn_)}
        ece = float(np.sum(np.abs(bins[:, 0] - bins[:, 1])) / max(1, bin_n.sum()))
        rel = [{"bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "n": int(bin_n[i]),
                "mean_p": bins[i, 0] / bin_n[i] if bin_n[i] else None,
                "water_rate": bins[i, 1] / bin_n[i] if bin_n[i] else None} for i in range(10)]
        rows = [r for r in chip_rows if r["water_px"] > 500]
        out["model"][split] = {
            "threshold_sweep": sweep, "ece_10bin": ece, "reliability": rel,
            "fn_rate_by_stratum": {k: {"fn_rate": int(a) / max(1, int(b)), "water_px": int(b)} for k, (a, b) in strata.items()},
            "chips_with_water_gt500px": len(rows),
            "chips_where_ndwi_beats_model": sum(r["ndwi_iou"] > r["model_iou"] for r in rows),
            "per_chip": chip_rows,
        }
        print(split, "ECE", round(ece, 4), "FN-rate", {k: round(int(a) / max(1, int(b)), 3) for k, (a, b) in strata.items()},
              "ndwi>model chips", out["model"][split]["chips_where_ndwi_beats_model"], "/", len(rows), flush=True)
    best_th = max(out["model"]["valid"]["threshold_sweep"].items(), key=lambda kv: kv[1]["iou"])[0]
    out["model"]["valid_selected_threshold"] = best_th
    out["model"]["at_valid_selected_threshold"] = {s: out["model"][s]["threshold_sweep"][best_th] for s in ("valid", "test", "bolivia")}
    out["model"]["at_0.5"] = {s: out["model"][s]["threshold_sweep"][0.5] for s in ("valid", "test", "bolivia")}
    write_json(artifacts_dir() / "audit" / "weakness_analysis.json", out)
    print("valid-selected threshold", best_th, json.dumps(out["model"]["at_valid_selected_threshold"], indent=1))
    print(json.dumps({s: {k: v for k, v in d.items() if k != "mean_normalised_input_per_band (valid px)"}
                      for s, d in out["data"].items() if s != "stats_contamination"}, indent=1, default=float))
    print(json.dumps({s: d["mean_normalised_input_per_band (valid px)"] for s, d in out["data"].items()
                      if s != "stats_contamination"}, default=float))
    print(out["data"]["stats_contamination"])


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
