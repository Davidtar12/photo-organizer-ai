from __future__ import annotations

import platform
from pathlib import Path

from flask import Blueprint, jsonify

from config import Config
from database import session_scope
from models import FaceEmbedding, MediaFile, PersonCluster, TaskProgress
from schemas import TaskProgressSchema, PersonClusterSchema
from sqlalchemy import func, select

bp = Blueprint("system", __name__, url_prefix="/api")
progress_schema = TaskProgressSchema(many=True)
cluster_schema = PersonClusterSchema()


@bp.get("/health")
def healthcheck():
    with session_scope() as session:
        media_count = session.query(MediaFile).count()
        face_count = session.query(FaceEmbedding).count()
        cluster_count = session.query(PersonCluster).count()

    result = {
        "status": "ok",
        "media_count": media_count,
        "face_count": face_count,
        "cluster_count": cluster_count,
        "db_path": str(Config.DB_PATH),
        "python": platform.python_version(),
    }
    return jsonify(result)


@bp.get("/progress")
def get_progress():
    with session_scope() as session:
        tasks = session.query(TaskProgress).all()
        return jsonify(progress_schema.dump(tasks))


@bp.get("/paths")
def get_paths():
    return jsonify(
        {
            "organized_dir": str(Config.ORGANIZED_DIR),
            "trash_dir": str(Config.FACE_TRASH_DIR),
            "thumb_cache": str(Config.THUMB_CACHE_DIR),
            "embedding_cache": str(Config.EMBEDDING_CACHE_DIR),
            "db_path": str(Config.DB_PATH),
            "db_exists": Path(Config.DB_PATH).exists(),
        }
    )
