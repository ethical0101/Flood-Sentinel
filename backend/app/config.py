"""Backend settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    p = REPO / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    artifacts: Path = Path(os.environ.get("FLOOD_ARTIFACTS_DIR", REPO / "artifacts"))
    processed: Path = Path(os.environ.get("FLOOD_PROCESSED_DIR", REPO / "data" / "processed"))
    raw: Path = Path(os.environ.get("FLOOD_DATA_DIR", REPO / "data" / "raw"))
    outputs: Path = Path(os.environ.get("FLOOD_OUTPUTS_DIR", REPO / "outputs"))
    study_areas_file: Path = REPO / "configs" / "study_areas.json"
    max_upload_mb: float = float(os.environ.get("FLOOD_MAX_UPLOAD_MB", 40))
    max_image_pixels: int = int(os.environ.get("FLOOD_MAX_IMAGE_PIXELS", 2048 * 2048))
    min_image_side: int = 32
    device: str = os.environ.get("FLOOD_DEVICE", "auto")
    default_model: str = os.environ.get("FLOOD_DEFAULT_MODEL", "unet_resnet34_s1s2")
    allow_osm_download: bool = _bool("FLOOD_ALLOW_OSM_DOWNLOAD", True)
    max_area_deg: float = 0.2  # largest bbox side (degrees) for an uploaded study area's OSM request
    max_results_kept: int = int(os.environ.get("FLOOD_MAX_RESULTS", 300))
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in os.environ.get("FLOOD_CORS_ORIGINS",
                                          "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()])
    frontend_dist: Path = REPO / "frontend" / "dist"
    db_path: Path = Path(os.environ.get("FLOOD_DB_PATH", REPO / "outputs" / "history.sqlite"))


settings = Settings()
