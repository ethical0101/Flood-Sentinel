"""Fill the <!-- TABLE:name --> blocks in docs/report.md from recorded artifacts (no hand-copied numbers).

    .venv\\Scripts\\python scripts\\build_report_tables.py
Tables: main (all models + baselines on all evaluation sets), training (per-run config/time/VRAM),
latency, routing, bootstrap, events (per-event test IoU of the default model), worldfloods.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
A = REPO / "artifacts"


def rj(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def f(v, d=4):
    return "–" if v is None or v != v else f"{v:.{d}f}"


def build() -> dict[str, str]:
    exps = {e["experiment_id"]: e for e in rj(A / "experiments.json")}
    wf = rj(A / "worldfloods_evaluation.json") or {"totals": {}, "models": {}}
    base = rj(A / "baselines.json")
    routing = rj(A / "routing_evaluation.json")
    boot = rj(A / "v2_bootstrap.json")
    order = ["unet_resnet34_s1s2", "unet_resnet34_s2", "unet_resnet34_s1", "unet_resnet34_rgb", "deeplabv3plus_mnv3l_s1s2",
             "v2_r0b_unet_s1s2_seed7", "v2_r1_unet_s1s2_radiometric", "v2_r2_unet_s1s2i_indices", "v2_r3_unet_s1s2i_radiometric",
             "v2_r4_unet_s2i_radiometric", "md1_unet_s2_worldfloods", "md2_unet_s2i_radiometric_worldfloods"]
    gen = {r: ("v1" if not r.startswith(("v2_", "md")) else "v2" if r.startswith("v2_") else "v2-MD") for r in order}
    t = {}

    def wfiou(r):
        th = wf["models"].get(r, {}).get("val_selected_threshold")
        k = f"{r}@val{th}" if th is not None else None
        return wf["totals"].get(k, {}).get("clear", {}).get("iou_water") if k else None

    rows = ["| Model | Gen. | Input | Thr. | Val IoU | Test IoU | Test F1 | Bolivia IoU | Bolivia recall | WorldFloods IoU |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in order:
        e = exps.get(r)
        if not e:
            continue
        ev = rj(A / "runs" / r / "evaluation.json")
        sv = ev["splits_at_val_threshold"]
        rows.append(f"| `{r}` | {gen[r]} | {e['input_modality']} | {e['val_selected_threshold']} | {f(e['val_iou@val_thr'])} | "
                    f"{f(sv['test']['iou_water'])} | {f(sv['test']['f1_water'])} | {f(sv['bolivia']['iou_water'])} | "
                    f"{f(sv['bolivia']['recall_water'], 3)} | {f(wfiou(r)) if e['input_modality'] in ('s2', 's2i', 'rgb') else 'n/a (needs SAR)'} |")
    for k, name in [("s2_ndwi_gt0", "NDWI > 0"), ("s2_mndwi_gt0", "MNDWI > 0"), ("s1_otsu_threshold", "S1 Otsu"),
                    ("jrc_permanent_water", "JRC permanent water"), ("all_not_water", "constant 'not water'")]:
        b = base[k]["splits"]
        wk = {"s2_ndwi_gt0": "ndwi_gt0", "s2_mndwi_gt0": "mndwi_gt0"}.get(k)
        rows.append(f"| {name} | baseline | – | fixed | – | {f(b['test']['iou_water'])} | {f(b['test']['f1_water'])} | "
                    f"{f(b['bolivia']['iou_water'])} | {f(b['bolivia']['recall_water'], 3)} | "
                    f"{f(wf['totals'].get(wk, {}).get('clear', {}).get('iou_water')) if wk else 'n/a'} |")
    t["main"] = "\n".join(rows)

    rows = ["| Model | Change vs v1 | Data | Epochs (best) | Train time | Peak train VRAM | Params |", "|---|---|---|---|---|---|---|"]
    for r in order:
        e = exps.get(r)
        if not e:
            continue
        ev = rj(A / "runs" / r / "evaluation.json")
        rows.append(f"| `{r}` | {e['change_vs_v1']} | {'S1F11 + WF subset' if 'WorldFloods' in e['dataset'] else 'Sen1Floods11'} | "
                    f"{e['epochs_run']} ({e['best_epoch']}) | {e['training_time_s'] / 60:.1f} min | {e['peak_train_gpu_mem_mib']} MiB | "
                    f"{ev['meta']['n_params'] / 1e6:.2f} M |")
    t["training"] = "\n".join(rows)

    rows = ["| Model | Model-only mean | p95 | End-to-end | Throughput | Inference peak VRAM |", "|---|---|---|---|---|---|"]
    for r in order:
        ev = rj(A / "runs" / r / "evaluation.json")
        if not ev:
            continue
        L = ev["latency"]
        rows.append(f"| `{r}` | {L['model_only_ms_mean']:.1f} ms | {L['model_only_ms_p95']:.1f} ms | {L['end_to_end_ms_mean']:.1f} ms | "
                    f"{L['throughput_fps_model_only']:.0f} chips/s | {ev.get('inference_peak_gpu_mem_mib', '–')} MiB |")
    t["latency"] = "\n".join(rows)

    if routing:
        rows = ["| Area | Road edges | Flooded road model / label (km) | Edge agreement | Labelled flood per trip: shortest | risk-aware (model) | oracle (label) |",
                "|---|---|---|---|---|---|---|"]
        for k, a in routing["areas"].items():
            rows.append(f"| {k} | {a['edges']} | {a['flooded_road_km_model']:.1f} / {a['flooded_road_km_label']:.1f} | "
                        f"{a['edge_agreement']['accuracy'] * 100:.1f}% | {a['mean_gt_flooded_m']['shortest']:.0f} m | "
                        f"{a['mean_gt_flooded_m']['risk_aware_model']:.0f} m | {a['mean_gt_flooded_m']['risk_aware_oracle']:.0f} m |")
        o = routing["overall"]
        rows.append(f"| **All ({o['n_pairs']} pairs)** | | | | **{o['mean_gt_flooded_m']['shortest']:.0f} m** | "
                    f"**{o['mean_gt_flooded_m']['risk_aware_model']:.0f} m** | **{o['mean_gt_flooded_m']['risk_aware_oracle']:.0f} m** |")
        t["routing"] = "\n".join(rows) + (f"\n\nReduction vs shortest: **{o['reduction_in_gt_flooded_length_vs_shortest'] * 100:.1f}%**; "
                                          f"mean detour **+{o['mean_detour_pct']:.1f}%**; pairs worse than shortest: "
                                          f"{o['pairs_where_model_route_has_more_gt_flood_than_shortest']}/{o['n_pairs']}.")

    if boot:
        rows = ["| Comparison | Test ΔIoU [95% CI] | Bolivia ΔIoU [95% CI] |", "|---|---|---|"]
        for r in boot["splits"]["test"]["comparisons"]:
            c1, c2 = boot["splits"]["test"]["comparisons"][r], boot["splits"]["bolivia"]["comparisons"][r]
            rows.append(f"| `{r}` vs `{boot['baseline']}` | {c1['delta_iou']:+.4f} [{c1['ci95'][0]:+.3f}, {c1['ci95'][1]:+.3f}] | "
                        f"{c2['delta_iou']:+.4f} [{c2['ci95'][0]:+.3f}, {c2['ci95'][1]:+.3f}] |")
        for r, c1 in boot["splits"]["test"].get("multi_dataset", {}).items():
            c2 = boot["splits"]["bolivia"]["multi_dataset"][r]
            rows.append(f"| `{r}` vs `{c1['reference']}` | {c1['delta_iou']:+.4f} [{c1['ci95'][0]:+.3f}, {c1['ci95'][1]:+.3f}] | "
                        f"{c2['delta_iou']:+.4f} [{c2['ci95'][0]:+.3f}, {c2['ci95'][1]:+.3f}] |")
        t["bootstrap"] = "\n".join(rows)

    ev = rj(A / "runs" / "unet_resnet34_s1s2" / "evaluation.json")
    pe = ev["per_event_at_val_threshold"]["test"]
    t["events"] = ("| Event | " + " | ".join(pe) + " |\n|---|" + "---|" * len(pe) + "\n| Test IoU | "
                   + " | ".join(f"{m['iou_water']:.3f}" for m in pe.values()) + " |")
    cm = ev["splits_at_val_threshold"]["test"]["confusion_matrix"]
    cb = ev["splits_at_val_threshold"]["bolivia"]["confusion_matrix"]
    t["confusion"] = ("| Split | TN | FP | FN | TP |\n|---|---|---|---|---|\n"
                      f"| Test | {cm[0][0]:,} | {cm[0][1]:,} | {cm[1][0]:,} | {cm[1][1]:,} |\n"
                      f"| Bolivia | {cb[0][0]:,} | {cb[0][1]:,} | {cb[1][0]:,} | {cb[1][1]:,} |")
    return t


def main() -> int:
    p = REPO / "docs" / "report.md"
    s = p.read_text(encoding="utf-8")
    for name, table in build().items():
        s = re.sub(rf"<!-- TABLE:{name} -->.*?<!-- /TABLE -->", f"<!-- TABLE:{name} -->\n{table}\n<!-- /TABLE -->", s, flags=re.S)
    p.write_text(s, encoding="utf-8")
    print("report tables updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
