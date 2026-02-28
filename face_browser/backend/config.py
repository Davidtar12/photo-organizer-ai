from __future__ import annotations

import os
from pathlib import Path


class Config:
    """Runtime configuration for the face browser backend."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    # Source library (read-only operations except deletions handled via trash).
    ORGANIZED_DIR = Path(os.environ.get(
        "FACE_BROWSER_ORGANIZED_DIR",
        r"C:\\Users\\david\\OneDrive\\Documents\\Pictures\\Organized",
    ))

    FACE_TRASH_DIR = Path(os.environ.get(
        "FACE_BROWSER_TRASH_DIR",
        r"C:\\Users\\david\\OneDrive\\Documents\\Pictures\\duplicates_face_trash",
    ))

    # SQLite DB lives under project data directory by default.
    DB_PATH = Path(os.environ.get(
        "FACE_BROWSER_DB_PATH",
        str(DATA_DIR / "face_index.db"),
    ))

    # Thumbnail cache stored in AppData for fast local access.
    THUMB_CACHE_DIR = Path(
        os.environ.get(
            "FACE_BROWSER_THUMB_CACHE",
            os.path.join(
                os.getenv("LOCALAPPDATA", PROJECT_ROOT),
                "FaceBrowser",
                "thumbs",
            ),
        )
    )

    # Embeddings cache (serialized numpy arrays).
    EMBEDDING_CACHE_DIR = Path(
        os.environ.get(
            "FACE_BROWSER_EMBED_CACHE",
            os.path.join(
                os.getenv("LOCALAPPDATA", PROJECT_ROOT),
                "FaceBrowser",
                "embeddings",
            ),
        )
    )

    # DeepFace settings.
    FACE_MODEL_NAME = os.environ.get("FACE_BROWSER_MODEL", "ArcFace")
    DETECTOR_BACKEND = os.environ.get("FACE_BROWSER_DETECTOR", "opencv")  # Changed from retinaface to opencv for 3x speed
    BATCH_SIZE = int(os.environ.get("FACE_BROWSER_BATCH_SIZE", "32"))  # Increased batch size for GPU
    MAX_RAM_GB = float(os.environ.get("FACE_BROWSER_MAX_RAM_GB", "16"))
    MAX_VRAM_GB = float(os.environ.get("FACE_BROWSER_MAX_VRAM_GB", "6"))

    # Files considered for indexing.
    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
        ".webp",
    }

    # Logging / progress intervals.
    PROGRESS_EVERY = int(os.environ.get("FACE_BROWSER_PROGRESS_EVERY", "10"))  # Changed from 200 to 10 for more frequent updates


def ensure_directories(cfg: Config = Config()) -> None:
    """Create required directories if they do not exist."""

    for path in [cfg.DATA_DIR, cfg.FACE_TRASH_DIR, cfg.THUMB_CACHE_DIR, cfg.EMBEDDING_CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
