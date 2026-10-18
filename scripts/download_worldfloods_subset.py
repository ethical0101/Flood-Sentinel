"""Download the WorldFloods v2 training subset (configs/worldfloods_train_subset.json); val split optional.

Same guarantees as download_worldfloods_test.py: resumable, SHA-256 verified against the Hugging Face LFS
hash, never overwrites a mismatching file. Licence CC BY-NC 4.0 (Portales-Julia et al., 2023).

    .venv\\Scripts\\python scripts\\download_worldfloods_subset.py [--workers 3]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from download_worldfloods_test import fetch, ls, root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--with-val", action="store_true", help="also download the 2.5 GB val split (not needed)")
    a = ap.parse_args()
    subset = json.loads((REPO / "configs" / "worldfloods_train_subset.json").read_text())
    want = {s["event_id"] for s in subset["scenes"]}
    items = []
    for folder in ("train/S2", "train/gt", "train/meta"):
        items += [i for i in ls(folder) if Path(i["path"]).stem in want]
    if a.with_val:
        for folder in ("val/S2", "val/gt", "val/meta", "val/PERMANENTWATERJRC"):
            items += ls(folder)
    total = sum(i.get("lfs", {}).get("size", i.get("size", 0)) for i in items)
    print(f"{len(items)} files, {total / 1e9:.2f} GB -> {root()}", flush=True)
    res = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, it, root()): it["path"] for it in items}
        for n, f in enumerate(cf.as_completed(futs), 1):
            res[futs[f]] = f.result()
            if n % 20 == 0 or not res[futs[f]].startswith(("skipped", "downloaded")):
                print(f"[{n}/{len(items)}] {futs[f]}: {res[futs[f]]}", flush=True)
    bad = {k: v for k, v in res.items() if not v.startswith(("skipped", "downloaded"))}
    (root() / "manifest_train_val.json").write_text(json.dumps({
        "dataset": "WorldFloods v2 - training subset", "licence": "CC BY-NC 4.0",
        "subset_rule": subset["rule"], "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(items), "total_bytes": total, "n_failed": len(bad),
        "files": {i["path"]: {"size": i.get("lfs", {}).get("size", i.get("size")), "sha256": i.get("lfs", {}).get("oid")}
                  for i in items}}, indent=1))
    print("done; failed:", len(bad), list(bad.items())[:5])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
