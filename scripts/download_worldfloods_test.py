"""Download ONLY the test split of WorldFloods v2 (independent evaluation set).

Source : https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2 (public, not gated)
Licence: CC BY-NC 4.0. Cite Portales-Julia et al., Scientific Reports 13, 20316 (2023).
Use    : evaluation only, never training or model selection.

* Lists files through the Hugging Face tree API; every LFS file carries a SHA-256 ("lfs.oid").
* Resumable (HTTP Range into *.part), verified against SHA-256 before being moved into place.
* Existing verified files are skipped; mismatching files are never overwritten silently.

    .venv\\Scripts\\python scripts\\download_worldfloods_test.py [--dry-run]
Destination: $FLOOD_DATA_DIR/worldfloods_v2/test (default data/raw/worldfloods_v2/test)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
API = "https://huggingface.co/api/datasets/isp-uv-es/WorldFloodsv2/tree/main/"
RESOLVE = "https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2/resolve/main/"
FOLDERS = ["test/S2", "test/gt", "test/PERMANENTWATERJRC", "test/meta"]
EXTRA = ["dataset_metadata.csv", "README.md"]


def root() -> Path:
    return Path(os.environ.get("FLOOD_DATA_DIR", REPO / "data" / "raw")) / "worldfloods_v2"


def ls(path: str) -> list[dict]:
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return [x for x in json.load(r) if x["type"] == "file"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(item: dict, dest_root: Path) -> str:
    rel = item["path"]
    dest = dest_root / rel
    want = item.get("lfs", {}).get("oid")  # sha256 for LFS files
    size = item.get("lfs", {}).get("size", item.get("size"))
    if dest.exists():
        if (want and sha256(dest) == want) or (not want and dest.stat().st_size == size):
            return "skipped (verified)"
        return "MISMATCH (left untouched)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(5):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have < size:
                req = urllib.request.Request(RESOLVE + rel, headers={"Range": f"bytes={have}-"} if have else {})
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                    if have and r.status != 206:
                        f.seek(0)
                        f.truncate()
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            if want and sha256(part) != want:
                part.unlink()
                raise IOError("sha256 mismatch")
            os.replace(part, dest)
            return "downloaded"
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                return f"FAILED: {e}"
            time.sleep(3 * (attempt + 1))
    return "FAILED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    items = [i for f in FOLDERS for i in ls(f)] + [i for i in ls("") if i["path"] in EXTRA]
    total = sum(i.get("lfs", {}).get("size", i.get("size", 0)) for i in items)
    print(f"{len(items)} files, {total / 1e9:.2f} GB -> {root()}")
    if a.dry_run:
        return 0
    res = {}
    for n, it in enumerate(items, 1):
        res[it["path"]] = fetch(it, root())
        print(f"[{n}/{len(items)}] {it['path']}: {res[it['path']]}", flush=True)
    bad = {k: v for k, v in res.items() if not v.startswith(("skipped", "downloaded"))}
    (root()).mkdir(parents=True, exist_ok=True)
    (root() / "manifest.json").write_text(json.dumps({
        "dataset": "WorldFloods v2 (test split only)", "source": RESOLVE, "licence": "CC BY-NC 4.0",
        "citation": "Portales-Julia, Mateo-Garcia, Purcell, Gomez-Chova (2023). Global flood extent segmentation in "
                    "optical satellite images. Scientific Reports 13, 20316. doi:10.1038/s41598-023-47595-7",
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(items), "total_bytes": total, "n_failed": len(bad),
        "files": {i["path"]: {"size": i.get("lfs", {}).get("size", i.get("size")),
                              "sha256": i.get("lfs", {}).get("oid")} for i in items}}, indent=1))
    print("failed:", len(bad), bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
