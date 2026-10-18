"""Train a flood/water segmentation model on Sen1Floods11 (hand-labelled).

    python -m training.train                                   # defaults from training/config.yaml
    python -m training.train --set data.modality=s1 --set train.epochs=40
    python -m training.train --run unet_resnet34_s1s2 --resume  # continue an interrupted run
    python -m training.train --smoke                           # 2 epochs on a few chips

Outputs (artifacts/runs/<run_name>/):
    config.yaml, environment.json, history.json (per-epoch metrics), train.log,
    status.json (live progress, read by the dashboard), checkpoints/{last,best}.pt
Model selection uses the validation split only; the test and Bolivia splits are never
touched here (see training/evaluate.py).
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from training.dataset import FloodDataset, worker_init
from training.metrics import chip_iou, confusion, masked_bce_dice, summarize
from training.model import build_model, count_params, save_checkpoint
from training.utils import (environment_info, load_config, processed_dir, read_json, runs_dir, seed_everything,
                            write_json)


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@torch.no_grad()
def validate(model, loader, device, amp: bool, threshold: float, cfg: dict) -> dict:
    model.eval()
    cm = np.zeros((2, 2), dtype=np.int64)
    ious, losses = [], []
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
        losses.append(masked_bce_dice(logits, y, bce_weight=cfg["train"]["bce_weight"],
                                      dice_weight=cfg["train"]["dice_weight"]).item())
        pred = (torch.sigmoid(logits.float())[:, 0] > threshold).long()
        for p, t in zip(pred, y):
            c = confusion(p, t)
            cm += c
            ious.append(chip_iou(c))
    m = summarize(cm, ious)
    m["loss"] = float(np.mean(losses))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--run", default=None, help="run name (required with --resume)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny 2-epoch run to check the pipeline")
    ap.add_argument("--polite", action="store_true",
                    help="yield the GPU: after any epoch, pause (exit 75) if another project's GPU job is running")
    args = ap.parse_args()

    if args.resume:
        if not args.run:
            ap.error("--resume needs --run <name>")
        run_dir = runs_dir() / args.run
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    else:
        overrides = list(args.set)
        if args.run:
            overrides.append(f"run_name={args.run}")
        if args.smoke:
            overrides += ["train.epochs=2", "data.crops_per_chip=1", "data.num_workers=0"]
        cfg = load_config(args.config, overrides)
        if args.smoke:
            cfg["run_name"] = "smoke_" + cfg["run_name"]
        run_dir = runs_dir() / cfg["run_name"]
    ck_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- single-instance guard
    lock = run_dir / "train.lock"
    if lock.exists():
        try:
            other = int(lock.read_text().strip() or 0)
        except ValueError:
            other = 0
        if pid_alive(other):
            print(f"run {cfg['run_name']} is already being trained by PID {other}; refusing to start a duplicate")
            return 3
    lock.write_text(str(os.getpid()))

    if not args.resume and (ck_dir / "last.pt").exists() and not args.smoke:
        print(f"{ck_dir / 'last.pt'} exists. Use --resume --run {cfg['run_name']} or choose another run name.")
        lock.unlink(missing_ok=True)
        return 4

    log = Logger(run_dir / "train.log")
    try:
        return train(cfg, run_dir, ck_dir, args, log)
    except KeyboardInterrupt:
        log("interrupted by user; resume with: python -m training.train --resume --run " + cfg["run_name"])
        _status(run_dir, state="interrupted")
        return 130
    except Exception as e:  # noqa: BLE001
        log(f"FAILED: {type(e).__name__}: {e}")
        _status(run_dir, state="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        lock.unlink(missing_ok=True)


def _status(run_dir: Path, **kw) -> None:
    p = run_dir / "status.json"
    s = read_json(p) if p.exists() else {}
    s.update(kw, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(p, s)


def train(cfg: dict, run_dir: Path, ck_dir: Path, args, log: Logger) -> int:
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log("WARNING: CUDA not available - training on CPU will be very slow")
    torch.backends.cudnn.benchmark = True
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    d, t = cfg["data"], cfg["train"]

    train_ds = FloodDataset("train", d["modality"], d["crop_size"], d["crops_per_chip"], d["augment"],
                            seed=cfg["seed"], radiometric=d.get("radiometric"))
    val_ds = FloodDataset("valid", d["modality"])
    if args.smoke:
        train_ds = Subset(train_ds, range(min(16, len(train_ds))))
        val_ds = Subset(val_ds, range(min(6, len(val_ds))))
    g = torch.Generator().manual_seed(cfg["seed"])
    nw = int(d["num_workers"])
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True, generator=g,
                          num_workers=nw, worker_init_fn=worker_init if nw else None,
                          persistent_workers=nw > 0, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    m = cfg["model"]
    model = build_model(m["arch"], m["encoder"], d["modality"], m["encoder_weights"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps_per_epoch = max(1, len(train_dl) // t["grad_accum"])
    total, warm = t["epochs"] * steps_per_epoch, t["warmup_epochs"] * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    hist_path = run_dir / "history.json"
    start_epoch, best, history, bad_epochs = 1, -1.0, [], 0
    if args.resume and (ck_dir / "last.pt").exists():
        ck = torch.load(ck_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        history = read_json(hist_path) if hist_path.exists() else []
        history = [h for h in history if h["epoch"] < start_epoch]
        bad_epochs = ck["meta"].get("bad_epochs", 0)
        log(f"resumed from epoch {ck['epoch']} (best {t['monitor']}={best:.4f})")
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json(run_dir / "environment.json", environment_info())
        # the run keeps its own copy of the train-split normalisation stats used for its inputs
        shutil.copyfile(processed_dir() / "stats.json", run_dir / "stats.json")

    meta_base = {
        "pretrained": f"{m['encoder']} encoder initialised from {m['encoder_weights']} weights"
                      if m["encoder_weights"] else "trained from scratch",
        "trained_on": "Sen1Floods11 v1.1 HandLabeled, official train split",
        "selected_on": "official valid split",
        "n_params": count_params(model),
        "modality": d["modality"],
    }
    log(f"run={cfg['run_name']} device={device} amp={amp} params={meta_base['n_params']/1e6:.2f}M "
        f"train_samples={len(train_ds)} val_chips={len(val_ds)} steps/epoch={steps_per_epoch}")
    _status(run_dir, state="running", run=cfg["run_name"], pid=os.getpid(), epoch=start_epoch - 1,
            epochs=t["epochs"], best=best, monitor=t["monitor"], started=time.strftime("%Y-%m-%dT%H:%M:%S"))

    monitor_key = t["monitor"].removeprefix("val_")
    t0_run = time.time()
    for epoch in range(start_epoch, t["epochs"] + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0, losses = time.time(), []
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = masked_bce_dice(logits, y, bce_weight=t["bce_weight"], dice_weight=t["dice_weight"],
                                   pos_weight=t.get("pos_weight"))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {i}")
            scaler.scale(loss / t["grad_accum"]).backward()
            if (i + 1) % t["grad_accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            losses.append(loss.item())
            if i % 25 == 0 or i + 1 == len(train_dl):  # batch-level progress for the dashboard
                _status(run_dir, state="running", pid=os.getpid(), phase="train", current_epoch=epoch,
                        step=i + 1, steps=len(train_dl), running_loss=float(np.mean(losses[-25:])))
        train_time = time.time() - t0
        val = validate(model, val_dl, device, amp, t["threshold"], cfg)
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"],
               "epoch_time_s": round(time.time() - t0, 1), "train_time_s": round(train_time, 1),
               "peak_gpu_mem_mib": round(peak), **{f"val_{k}": v for k, v in val.items() if k != "confusion_matrix"}}
        history.append(rec)
        write_json(hist_path, history)

        score = val[monitor_key]
        improved = score > best
        if improved:
            best, bad_epochs = score, 0
        else:
            bad_epochs += 1
        meta = {**meta_base, "bad_epochs": bad_epochs, "val_metrics": val, "epoch": epoch}
        save_checkpoint(ck_dir / "last.pt", model, opt, sched, scaler, epoch, best, cfg, meta)
        if improved:
            save_checkpoint(ck_dir / "best.pt", model, None, None, None, epoch, best, cfg, meta)
        log(f"epoch {epoch:3d}/{t['epochs']} loss={rec['train_loss']:.4f} val_loss={val['loss']:.4f} "
            f"val_iou={val['iou_water']:.4f} val_f1={val['f1_water']:.4f} lr={rec['lr']:.2e} "
            f"{rec['epoch_time_s']}s peak={rec['peak_gpu_mem_mib']}MiB{' *best*' if improved else ''}")
        elapsed = time.time() - t0_run
        done = epoch - start_epoch + 1
        _status(run_dir, state="running", epoch=epoch, best=best, best_epoch=epoch if improved else
                read_json(run_dir / "status.json").get("best_epoch"),
                last=rec, eta_s=round(elapsed / done * (t["epochs"] - epoch)))
        if bad_epochs >= t["early_stopping_patience"]:
            log(f"early stopping: no improvement in {bad_epochs} epochs")
            break
        stop_file = run_dir / "STOP"
        if stop_file.exists():
            stop_file.unlink(missing_ok=True)
            log(f"stop requested; checkpoint saved at epoch {epoch}. Resume with: python -m training.train "
                f"--resume --run {cfg['run_name']}")
            _status(run_dir, state="stopped", note="stopped on request after a completed epoch")
            return 0
        if getattr(args, "polite", False):
            from training.gpu_guard import foreign_gpu_processes

            other = foreign_gpu_processes()
            if other:
                log(f"pausing: another GPU job is running (pid {other[0]['pid']}); checkpoint saved at epoch {epoch}")
                _status(run_dir, state="paused", note=f"yielded GPU to pid {other[0]['pid']}")
                return 75

    _status(run_dir, state="completed", finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_time_s=round(time.time() - t0_run))
    log(f"finished. best {t['monitor']}={best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
