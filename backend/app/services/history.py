"""Persistent analysis history (SQLite, standard library only).

Stores metadata and *references* (result / route ids) for every segmentation and route analysis.
Images, masks and route geometry stay on disk under outputs/; the database never embeds them.
No user-identifying data is stored: uploaded file names are truncated, and no IP address or
client information is recorded.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from pathlib import Path

from backend.app.config import settings

_lock = threading.Lock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id         TEXT PRIMARY KEY,           -- result id (segmentation) or route id
    kind       TEXT NOT NULL CHECK (kind IN ('segmentation', 'route')),
    created    TEXT NOT NULL,
    title      TEXT NOT NULL,
    model      TEXT,
    source     TEXT,                        -- upload | sample | curated area | uploaded area
    summary    TEXT NOT NULL,               -- JSON: headline numbers shown in lists
    metrics    TEXT,                        -- JSON: ground-truth metrics when available
    settings   TEXT,                        -- JSON: threshold, penalties, ...
    app_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_kind ON analyses(kind);
"""
APP_VERSION = "2.0.0"


def db_path() -> Path:
    p = Path(settings.db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(db_path(), timeout=10)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def add(id: str, kind: str, created: str, title: str, model: str | None, source: str | None,
        summary: dict, metrics: dict | None = None, settings_: dict | None = None) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (id, kind, created, title[:160], model, source, json.dumps(summary, default=float),
                   json.dumps(metrics, default=float) if metrics else None,
                   json.dumps(settings_, default=float) if settings_ else None, APP_VERSION))


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for k in ("summary", "metrics", "settings"):
        d[k] = json.loads(d[k]) if d[k] else None
    return d


def list_(kind: str | None = None, limit: int = 50, offset: int = 0, q: str | None = None) -> dict:
    where, args = [], []
    if kind:
        where.append("kind = ?")
        args.append(kind)
    if q:
        where.append("(title LIKE ? OR model LIKE ? OR source LIKE ?)")
        args += [f"%{q}%"] * 3
    w = f"WHERE {' AND '.join(where)}" if where else ""
    with _lock, _conn() as c:
        total = c.execute(f"SELECT COUNT(*) FROM analyses {w}", args).fetchone()[0]
        rows = c.execute(f"SELECT * FROM analyses {w} ORDER BY created DESC LIMIT ? OFFSET ?",
                         [*args, limit, offset]).fetchall()
    return {"total": total, "items": [_row(r) for r in rows]}


def get(id: str) -> dict | None:
    with _lock, _conn() as c:
        r = c.execute("SELECT * FROM analyses WHERE id = ?", (id,)).fetchone()
    return _row(r) if r else None


def counts() -> dict:
    with _lock, _conn() as c:
        rows = c.execute("SELECT kind, COUNT(*) n, MAX(created) last FROM analyses GROUP BY kind").fetchall()
    return {r["kind"]: {"count": r["n"], "last": r["last"]} for r in rows}


def export(fmt: str, kind: str | None = None) -> tuple[str, str]:
    items = list_(kind, limit=100000)["items"]
    if fmt == "json":
        return json.dumps(items, indent=1, default=float), "application/json"
    buf = io.StringIO()
    cols = ["id", "kind", "created", "title", "model", "source", "app_version"]
    summary_keys = sorted({k for i in items for k in (i["summary"] or {})})
    metric_keys = ["iou_water", "f1_water", "precision_water", "recall_water"]
    w = csv.writer(buf)
    w.writerow(cols + [f"summary.{k}" for k in summary_keys] + [f"metrics.{k}" for k in metric_keys])
    for i in items:
        w.writerow([i[c] for c in cols] + [(i["summary"] or {}).get(k) for k in summary_keys]
                   + [(i["metrics"] or {}).get(k) for k in metric_keys])
    return buf.getvalue(), "text/csv"
