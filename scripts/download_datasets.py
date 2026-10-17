"""Reproducible, resumable downloader for the Sen1Floods11 hand-labelled subset.

Source: public Google Cloud Storage bucket ``gs://sen1floods11`` (v1.1), accessed over
plain HTTPS through the GCS JSON API, so no gsutil / Google account is required.

Safety properties
-----------------
* Files that already exist and match the bucket's MD5 hash are skipped (safe to rerun).
* Partial downloads are written to ``*.part`` and resumed with HTTP Range requests.
* A file is only moved into place after its MD5 matches the published hash.
* Existing files with a *mismatching* hash are never overwritten silently: they are
  reported and left untouched unless ``--repair`` is passed (then the bad copy is
  renamed to ``*.corrupt-<timestamp>`` rather than deleted).

Usage
-----
    python scripts/download_datasets.py                 # default subset
    python scripts/download_datasets.py --dry-run       # list what would be fetched
    python scripts/download_datasets.py --layers S1Hand LabelHand

The destination is ``$FLOOD_DATA_DIR/sen1floods11`` (default ``<repo>/data/raw``).
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "sen1floods11"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"
HANDLABELED = "v1.1/data/flood_events/HandLabeled/"
DEFAULT_LAYERS = ["S1Hand", "S2Hand", "LabelHand", "S1OtsuLabelHand", "JRCWaterHand"]
EXTRA_PREFIXES = ["v1.1/splits/flood_handlabeled/", "v1.1/Sen1Floods11_Metadata.geojson"]

REPO = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("FLOOD_DATA_DIR", REPO / "data" / "raw")) / "sen1floods11"


def list_objects(prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=60) as r:
            d = json.load(r)
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return items


def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def fetch(obj: dict, root: Path, repair: bool, retries: int = 5) -> tuple[str, str]:
    rel = obj["name"]
    dest = root / rel
    size, want = int(obj["size"]), obj["md5Hash"]
    if dest.exists():
        if dest.stat().st_size == size and md5_b64(dest) == want:
            return rel, "skipped (verified)"
        if not repair:
            return rel, "MISMATCH (existing file left untouched; rerun with --repair)"
        stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        dest.rename(dest.with_name(dest.name + f".corrupt-{stamp}"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = MEDIA + urllib.parse.quote(rel)
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                part.unlink()
                have = 0
            if have < size:
                req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                    if have and r.status != 206:  # server ignored Range -> restart
                        f.seek(0)
                        f.truncate()
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            if md5_b64(part) != want:
                part.unlink()
                raise IOError("md5 mismatch after download")
            os.replace(part, dest)
            return rel, "downloaded"
        except Exception as e:  # noqa: BLE001 - network errors are retried
            if attempt == retries:
                return rel, f"FAILED: {e}"
            time.sleep(2 * attempt)
    return rel, "FAILED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repair", action="store_true", help="re-download files whose hash mismatches")
    args = ap.parse_args()

    root = data_root()
    objs = []
    for layer in args.layers:
        objs += list_objects(f"{HANDLABELED}{layer}/")
    for p in EXTRA_PREFIXES:
        objs += list_objects(p)
    total = sum(int(o["size"]) for o in objs)
    print(f"Destination: {root}")
    print(f"{len(objs)} objects, {total / 1e9:.2f} GB")
    if args.dry_run:
        return 0

    results: dict[str, str] = {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, (rel, status) in enumerate(ex.map(lambda o: fetch(o, root, args.repair), objs), 1):
            results[rel] = status
            if not status.startswith(("skipped", "downloaded")) or i % 100 == 0 or i == len(objs):
                print(f"[{i}/{len(objs)}] {rel}: {status}", flush=True)

    bad = {k: v for k, v in results.items() if not v.startswith(("skipped", "downloaded"))}
    manifest = {
        "dataset": "Sen1Floods11 v1.1 (hand-labelled subset)",
        "source": f"gs://{BUCKET}/ via {MEDIA}",
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "layers": args.layers,
        "n_objects": len(objs),
        "total_bytes": total,
        "n_failed": len(bad),
        "files": {o["name"]: {"size": int(o["size"]), "md5": o["md5Hash"]} for o in objs},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Done. failed={len(bad)}; manifest written to {root / 'manifest.json'}")
    for k, v in bad.items():
        print("  ", k, v)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
