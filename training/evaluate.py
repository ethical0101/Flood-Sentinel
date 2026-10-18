"""Final evaluation on held-out splits + baselines + latency + figures.

    python -m training.evaluate --all                 # every completed run + baselines + summary
    python -m training.evaluate --run unet_resnet34_s1s2
    python -m training.evaluate --baselines-only

Protocol
  * The checkpoint evaluated is ``best.pt``, chosen during training on the *valid* split only.
  * Decision threshold is fixed at 0.5 (not tuned on test data).
  * ``test``    = official Sen1Floods11 hand-labelled test split (same 10 events as training,
                  disjoint chips).
  * ``bolivia`` = all 15 Bolivia chips; this event is never seen in training/validation, so it
                  measures generalisation to an unseen flood event / region.
  * Metrics pool all valid pixels of a split (Sen1Floods11 "total IoU"); mean per-chip IoU is
    reported alongside. Pixels labelled -1 are ignored.

Outputs: artifacts/runs/<run>/evaluation.json, curves.png, samples/*.png,
         artifacts/baselines.json, artifacts/summary.json, artifacts/figures/*.png
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from training.dataset import chip_input, load_chip, load_index  # noqa: E402
from training.infer import Predictor  # noqa: E402
from training.metrics import chip_iou, confusion, summarize  # noqa: E402
from training.utils import artifacts_dir, read_json, runs_dir, write_json  # noqa: E402

EVAL_SPLITS = ["valid", "test", "bolivia"]
SAMPLE_CHIPS_N = 6


def chips_by_split() -> dict[str, list[dict]]:
    out = defaultdict(list)
    for c in load_index():
        out[c["split"]].append(c)
    return out


def sample_chips(test: list[dict]) -> list[str]:
    """Deterministic, diverse test examples: spread over water fraction, one per event where possible."""
    ranked = sorted((c for c in test if c["valid_fraction"] > 0.9), key=lambda c: c["water_fraction"])
    picks, events = [], set()
    for q in np.linspace(0.35, 0.98, SAMPLE_CHIPS_N * 3):
        c = ranked[int(q * (len(ranked) - 1))]
        if c["event"] not in events and c["id"] not in picks:
            picks.append(c["id"])
            events.add(c["event"])
        if len(picks) == SAMPLE_CHIPS_N:
            break
    return picks


# ----------------------------------------------------------------------------- figures
def error_map(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    img = np.full(label.shape + (3,), 235, np.uint8)
    img[(pred == 1) & (label == 1)] = (30, 110, 220)   # TP blue
    img[(pred == 1) & (label == 0)] = (230, 80, 50)    # FP red-orange
    img[(pred == 0) & (label == 1)] = (250, 190, 40)   # FN amber
    img[label == -1] = (120, 120, 120)                  # ignored
    return img


def mask_img(m: np.ndarray, label: np.ndarray | None = None) -> np.ndarray:
    img = np.full(m.shape + (3,), 245, np.uint8)
    img[m == 1] = (30, 110, 220)
    if label is not None:
        img[label == -1] = (120, 120, 120)
    return img


def save_sample(path: Path, chip: dict, pred: np.ndarray, prob: np.ndarray, title: str) -> None:
    from training.infer import SceneInput

    rgb = np.transpose(chip["rgb"], (1, 2, 0))
    vv = SceneInput("s1", s1=chip["s1"].astype(np.float32)).preview_rgb()
    panels = [(rgb, "Sentinel-2 RGB"), (vv, "Sentinel-1 false colour"),
              (mask_img(chip["label"], chip["label"]), "Ground truth (hand label)"),
              (prob, "Model P(water)"), (error_map(pred, chip["label"]), "Errors: TP blue/FP red/FN amber")]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im, cmap="viridis", vmin=0, vmax=1) if im.ndim == 2 else ax.imshow(im)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_curves(run_dir: Path) -> None:
    hist = read_json(run_dir / "history.json")
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="valid")
    ax[0].set_title("BCE + Dice loss"); ax[0].legend()
    ax[1].plot(ep, [h["val_iou_water"] for h in hist], label="IoU (water)")
    ax[1].plot(ep, [h["val_f1_water"] for h in hist], label="F1 (water)")
    ax[1].set_title("Validation metrics"); ax[1].legend()
    ax[2].plot(ep, [h["lr"] for h in hist]); ax[2].set_title("Learning rate")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def benchmark(pred: Predictor, chip: dict, iters: int = 30) -> dict:
    x = chip_input(chip, pred.modality, pred.stats)
    xt = torch.from_numpy(x[None]).to(pred.device)
    sync = torch.cuda.synchronize if pred.device.type == "cuda" else (lambda: None)
    for _ in range(5):
        pred._forward(xt, False)
    sync()
    model_ms = []
    for _ in range(iters):
        t = time.perf_counter(); pred._forward(xt, False); sync(); model_ms.append((time.perf_counter() - t) * 1e3)
    e2e = []
    for _ in range(10):
        t = time.perf_counter()
        pred.predict(chip_input(chip, pred.modality, pred.stats)); sync()
        e2e.append((time.perf_counter() - t) * 1e3)
    return {"input": f"1x{x.shape[0]}x{x.shape[1]}x{x.shape[2]}", "batch_size": 1, "device": str(pred.device),
            "gpu": torch.cuda.get_device_name(0) if pred.device.type == "cuda" else None,
            "precision": "fp16 autocast" if pred.amp else "fp32",
            "model_only_ms_mean": float(np.mean(model_ms)), "model_only_ms_p50": float(np.median(model_ms)),
            "model_only_ms_p95": float(np.percentile(model_ms, 95)),
            "end_to_end_ms_mean": float(np.mean(e2e)),
            "end_to_end_includes": "normalisation of an in-memory chip + tiled inference + sigmoid + copy to CPU "
                                   "(excludes file decoding)",
            "throughput_fps_model_only": float(1000 / np.mean(model_ms))}


def evaluate_run(run: str, threshold: float = 0.5, tta: bool = False) -> dict:
    run_dir = runs_dir() / run
    pred = Predictor(run_dir)
    splits = chips_by_split()
    samples = set(sample_chips(splits["test"]))
    result = {"run": run, "checkpoint": "best.pt", "checkpoint_epoch": pred.epoch, "threshold": threshold,
              "tta": tta, "modality": pred.modality, "config": pred.config, "meta": pred.meta, "splits": {},
              "per_event": {}}
    grid = [round(float(t), 2) for t in np.arange(0.1, 0.91, 0.05)]
    if pred.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cms_grid: dict[str, dict[float, np.ndarray]] = {}
    per_chip: dict[str, list] = {}
    for split in EVAL_SPLITS:
        cm = np.zeros((2, 2), np.int64)
        ious, ev_cm = [], defaultdict(lambda: np.zeros((2, 2), np.int64))
        cms_grid[split] = {t: np.zeros((2, 2), np.int64) for t in grid}
        per_chip[split] = []
        for c in splits[split]:
            chip = load_chip(c["id"])
            prob = pred.predict(chip_input(chip, pred.modality, pred.stats), tta=tta)
            p = (prob > threshold).astype(np.int8)
            cc = confusion(p, chip["label"])
            cm += cc
            ev_cm[c["event"]] += cc
            ious.append(chip_iou(cc))
            for t in grid:
                cms_grid[split][t] += confusion((prob > t).astype(np.int8), chip["label"])
            per_chip[split].append({"chip": c["id"], "event": c["event"], "probs": prob.astype(np.float16)})
            if split == "test" and c["id"] in samples:
                save_sample(run_dir / "samples" / f"{c['id']}.png", chip, p, prob,
                            f"{run} | {c['id']} (test split) | IoU={chip_iou(cc):.3f}")
        result["splits"][split] = summarize(cm, ious)
        if split != "valid":
            result["per_event"][split] = {e: summarize(v) for e, v in sorted(ev_cm.items())}
        print(f"  {run:28s} {split:8s} IoU={result['splits'][split]['iou_water']:.4f} "
              f"F1={result['splits'][split]['f1_water']:.4f}", flush=True)
    # threshold chosen on VALIDATION ONLY, then applied unchanged to test / Bolivia
    sweep = {s: {t: summarize(m)["iou_water"] for t, m in g.items()} for s, g in cms_grid.items()}
    best_t = max(grid, key=lambda t: (sweep["valid"][t], -abs(t - 0.5)))
    result["threshold_sweep_iou"] = {s: {str(t): v for t, v in d.items()} for s, d in sweep.items()}
    result["val_selected_threshold"] = best_t
    result["splits_at_val_threshold"] = {}
    result["per_event_at_val_threshold"] = {}
    for split in EVAL_SPLITS:
        result["splits_at_val_threshold"][split] = summarize(cms_grid[split][best_t])
        if split != "valid":
            ev = defaultdict(lambda: np.zeros((2, 2), np.int64))
            for row in per_chip[split]:
                lab = load_chip(row["chip"])["label"]
                ev[row["event"]] += confusion((row["probs"] > best_t).astype(np.int8), lab)
            result["per_event_at_val_threshold"][split] = {e: summarize(v) for e, v in sorted(ev.items())}
    result["inference_peak_gpu_mem_mib"] = (round(torch.cuda.max_memory_allocated() / 2**20)
                                            if pred.device.type == "cuda" else None)
    print(f"  {run:28s} val-selected threshold {best_t}: test IoU="
          f"{result['splits_at_val_threshold']['test']['iou_water']:.4f} bolivia IoU="
          f"{result['splits_at_val_threshold']['bolivia']['iou_water']:.4f}", flush=True)
    result["latency"] = benchmark(pred, load_chip(splits["test"][0]["id"]))
    result["sample_chips"] = sorted(samples)
    write_json(run_dir / "evaluation.json", result)
    plot_curves(run_dir)
    del pred
    torch.cuda.empty_cache()
    return result


def evaluate_baselines() -> dict:
    """Non-learned reference methods, evaluated with the same protocol."""

    def nd(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (a - b) / (a + b)

    methods = {
        "all_not_water": ("Constant predictor: every pixel = not water", lambda c: np.zeros_like(c["label"])),
        "s1_otsu_threshold": ("Sentinel-1 Otsu threshold labels shipped with Sen1Floods11 (S1OtsuLabelHand)",
                              lambda c: (c["otsu"] == 1).astype(np.int8)),
        "s2_ndwi_gt0": ("Sentinel-2 NDWI (B3-B8)/(B3+B8) > 0 (McFeeters, 1996)",
                        lambda c: (nd(c["s2"][2].astype(np.float32), c["s2"][7].astype(np.float32)) > 0).astype(np.int8)),
        "s2_mndwi_gt0": ("Sentinel-2 MNDWI (B3-B11)/(B3+B11) > 0 (Xu, 2006)",
                         lambda c: (nd(c["s2"][2].astype(np.float32), c["s2"][11].astype(np.float32)) > 0).astype(np.int8)),
        "jrc_permanent_water": ("JRC Global Surface Water layer shipped with the dataset (pre-event water only)",
                                lambda c: (c["jrc"] == 1).astype(np.int8)),
    }
    splits = chips_by_split()
    out = {}
    for name, (desc, fn) in methods.items():
        out[name] = {"description": desc, "splits": {}}
        for split in EVAL_SPLITS:
            cm = np.zeros((2, 2), np.int64)
            ious = []
            for c in splits[split]:
                chip = load_chip(c["id"])
                cc = confusion(fn(chip), chip["label"])
                cm += cc
                ious.append(chip_iou(cc))
            out[name]["splits"][split] = summarize(cm, ious)
        print(f"  baseline {name:22s} test IoU={out[name]['splits']['test']['iou_water']:.4f} "
              f"bolivia IoU={out[name]['splits']['bolivia']['iou_water']:.4f}", flush=True)
    write_json(artifacts_dir() / "baselines.json", out)
    return out


def summary_figures(runs: dict, baselines: dict) -> None:
    fig_dir = artifacts_dir() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = list(runs) + list(baselines)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.38
    for k, split in enumerate(["test", "bolivia"]):
        vals = [(runs.get(n) or baselines.get(n))["splits"][split]["iou_water"] for n in names]
        ax.bar(np.arange(len(names)) + (k - 0.5) * w, vals, w, label=split)
    ax.set_xticks(np.arange(len(names)), names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("IoU (water)")
    ax.set_title("Held-out IoU: trained models vs baselines (threshold 0.5)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "iou_comparison.png", dpi=100); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    a = ap.parse_args()

    if a.baselines_only:
        evaluate_baselines()
        return
    runs = list(a.run)
    if a.all:
        runs = sorted(p.name for p in runs_dir().iterdir()
                      if (p / "checkpoints" / "best.pt").exists() and not p.name.startswith("smoke_")
                      and read_json(p / "status.json").get("state") == "completed")
    results = {r: evaluate_run(r, a.threshold, a.tta) for r in runs}
    if a.all:
        baselines = read_json(artifacts_dir() / "baselines.json") if a.skip_baselines else evaluate_baselines()
        summary = {
            "protocol": "best.pt chosen on valid; metrics at threshold 0.5 and at the threshold selected on valid only; pooled-pixel metrics; label -1 ignored",
            "runs": {r: {"modality": v["modality"], "arch": v["config"]["model"]["arch"],
                         "encoder": v["config"]["model"]["encoder"], "epoch": v["checkpoint_epoch"],
                         "n_params": v["meta"]["n_params"], "splits": v["splits"],
                         "val_selected_threshold": v["val_selected_threshold"],
                         "splits_at_val_threshold": v["splits_at_val_threshold"],
                         "latency_ms": v["latency"]["model_only_ms_mean"]} for r, v in results.items()},
            "baselines": {k: {"description": v["description"], "splits": v["splits"]} for k, v in baselines.items()},
        }
        write_json(artifacts_dir() / "summary.json", summary)
        summary_figures(summary["runs"], summary["baselines"])


if __name__ == "__main__":
    main()
