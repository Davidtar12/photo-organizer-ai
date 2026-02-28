from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

# faiss is optional; endpoints that need it should guard against None
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency not required for basic endpoints
    faiss = None
import numpy as np
from flask import Blueprint, jsonify, request, send_file, redirect, url_for
from sqlalchemy import func, select

from config import Config
from database import session_scope
from models import DeleteLog, MediaFile, MergeEvent, PetCluster, PetEmbedding
from schemas import PetEmbeddingSchema, PetClusterSchema

bp = Blueprint("pets", __name__, url_prefix="/api/pets")

cluster_schema = PetClusterSchema()
pet_schema = PetEmbeddingSchema(many=True)

# In-memory centroid cache for Active Learning suggestions
# Similar to persons, but for pets
CENTROID_CACHE = {
    "built": 0.0,
    "min_size": 3,
    "dimension": 512,
    "cluster_ids": None,
    "cluster_sizes": None,
    "centroids": None,
    "faiss_index": None,
    "num_clusters_total": 0,
}

# --- Optional: load pets model vocab to map class index (e.g., 'Max') ---
_PET_LABELS = None
try:
    # Lazy import fastai only to read labels; inference not required here
    from fastai.vision.all import load_learner
    _pets_model_path = Path(__file__).parent.parent / 'models' / 'pets.pkl'
    if _pets_model_path.exists():
        _learner = load_learner(_pets_model_path)
        _PET_LABELS = list(getattr(getattr(_learner, 'dls', None), 'vocab', []) or [])
except Exception:
    _PET_LABELS = None

def _get_label_index(label: str) -> int | None:
    if not _PET_LABELS:
        return None
    try:
        return _PET_LABELS.index(label)
    except ValueError:
        return None

def _decode_embedding(pet: PetEmbedding):
    """Decode stored float32 probs from bytes -> np.ndarray of shape (embedding_dim,)."""
    try:
        arr = np.frombuffer(pet.embedding, dtype=np.float32)
        if pet.embedding_dim and arr.size != pet.embedding_dim:
            # Best effort: trim or pad
            arr = arr[: pet.embedding_dim]
        return arr
    except Exception:
        return None

@bp.get("/")
def list_pet_clusters():
    """Return all pet clusters with pet counts."""

    with session_scope() as session:
        stmt = (
            select(
                PetCluster,
                func.count(PetEmbedding.id).label("pet_count"),
            )
            .outerjoin(PetEmbedding, PetEmbedding.cluster_id == PetCluster.id)
            .group_by(PetCluster.id)
            .order_by(func.count(PetEmbedding.id).desc())
        )
        results = session.execute(stmt).all()

        clusters = []
        for cluster, pet_count in results:
            data = cluster_schema.dump(cluster)
            data["count"] = pet_count
            clusters.append(data)

        return jsonify({"clusters": clusters})

@bp.get("/by-label")
def list_pets_by_label():
    """Return pet detections filtered by cluster_id or label probability.

    Query params:
      - cluster_id: optional (returns all pets in this cluster)
      - label: optional (e.g., 'Max')
      - min_prob: float, default 0.8
      - species: optional filter ('dog' or 'cat')
      - limit, offset: pagination
    """
    cluster_id = request.args.get('cluster_id', type=int)
    label = request.args.get('label', type=str)
    species = request.args.get('species', type=str)
    # Filter out 'null' string which comes from JavaScript
    if species == 'null':
        species = None
    limit = request.args.get('limit', 5000, type=int)
    offset = request.args.get('offset', 0, type=int)

    with session_scope() as session:
        # If cluster_id is provided, return all pets in that cluster
        if cluster_id is not None:
            q = session.query(PetEmbedding).filter(PetEmbedding.cluster_id == cluster_id)
            if species:
                q = q.filter(PetEmbedding.species == species)
            
            pets = q.order_by(PetEmbedding.id).limit(limit).offset(offset).all()
            return jsonify({
                "cluster_id": cluster_id,
                "species": species,
                "total": len(pets),
                "pets": pet_schema.dump(pets),
            })

        # Otherwise, use label-based filtering (original logic)
        if not label:
            return jsonify({"error": "either cluster_id or label is required"}), 400

        min_prob = request.args.get('min_prob', 0.1, type=float)
        label_idx = _get_label_index(label)
        if label_idx is None:
            return jsonify({"error": f"label '{label}' not found in pets model vocab", "labels": _PET_LABELS or []}), 400

        q = session.query(PetEmbedding).order_by(PetEmbedding.id)
        if species:
            q = q.filter(PetEmbedding.species == species)

        # Fetch all candidates and score them in Python to allow sorting by probability
        all_candidates = q.all()
        
        scored_pets = []
        single_pet_schema = PetEmbeddingSchema()
        for p in all_candidates:
            probs = _decode_embedding(p)
            if probs is None or probs.size <= label_idx:
                continue
            
            prob = float(probs[label_idx])
            pet_data = single_pet_schema.dump(p)
            pet_data['probability'] = prob  # Add probability to the response
            scored_pets.append(pet_data)
            
        # Sort by probability descending
        scored_pets.sort(key=lambda x: x['probability'], reverse=True)

        # Now filter by min_prob and handle pagination
        matched = [p for p in scored_pets if p['probability'] >= min_prob]
        paginated_matches = matched[offset : offset + limit]
        
        return jsonify({
            "label": label,
            "min_prob": min_prob,
            "limit": limit,
            "offset": offset,
            "total_matches": len(matched),
            "pets": paginated_matches,
        })

@bp.get("/max")
def list_pets_max():
    """Convenience endpoint to fetch pets predicted as 'Max' with default threshold."""
    # Set 'Max' as the label and forward to the main by-label endpoint
    request.args = request.args.copy()
    request.args.setdefault('label', 'Max')
    return list_pets_by_label()

@bp.get("/<int:cluster_id>")
def get_pet_cluster(cluster_id: int):
    with session_scope() as session:
        cluster = session.get(PetCluster, cluster_id)
        if not cluster:
            return jsonify({"error": "Pet cluster not found"}), 404

        pets = (
            session.query(PetEmbedding)
            .filter(PetEmbedding.cluster_id == cluster_id)
            .order_by(PetEmbedding.id)
            .all()
        )

        return jsonify(
            {
                "cluster": cluster_schema.dump(cluster),
                "pets": pet_schema.dump(pets),
            }
        )

@bp.patch("/<int:cluster_id>")
def update_pet_cluster(cluster_id: int):
    payload = request.get_json(force=True)
    allowed_fields = {"display_name", "description", "is_hidden"}

    with session_scope() as session:
        cluster = session.get(PetCluster, cluster_id)
        if not cluster:
            return jsonify({"error": "Pet cluster not found"}), 404

        for key, value in payload.items():
            if key in allowed_fields:
                setattr(cluster, key, value)

        session.add(cluster)
        session.flush()

        data = cluster_schema.dump(cluster)
        pet_count = (
            session.query(PetEmbedding)
            .filter(PetEmbedding.cluster_id == cluster_id)
            .count()
        )
        data["pet_count"] = pet_count
        return jsonify(data)

@bp.get("/<int:cluster_id>/cover-image")
def get_pet_cluster_cover_image(cluster_id: int):
    """Return a redirect to a cover image for the pet cluster."""
    with session_scope() as session:
        cluster = session.get(PetCluster, cluster_id)
        if cluster and cluster.primary_pet_id:
            return redirect(url_for('pets.get_pet_thumbnail', pet_id=cluster.primary_pet_id))

        pet_with_thumb = session.query(PetEmbedding).filter(
            PetEmbedding.cluster_id == cluster_id,
            PetEmbedding.thumbnail_path.isnot(None)
        ).first()

        if pet_with_thumb:
            return redirect(url_for('pets.get_pet_thumbnail', pet_id=pet_with_thumb.id))

        pet_with_media = session.query(PetEmbedding).filter(
            PetEmbedding.cluster_id == cluster_id,
            PetEmbedding.media_id != None
        ).first()

        if pet_with_media:
            return redirect(url_for('pets.get_media_image', media_id=pet_with_media.media_id))

        return '', 404

@bp.get("/pet/<int:pet_id>/thumbnail")
def get_pet_thumbnail(pet_id: int):
    """Serve pet thumbnail image."""
    with session_scope() as session:
        pet = session.get(PetEmbedding, pet_id)
        if not pet:
            return jsonify({"error": "Pet not found"}), 404
        
        if pet.thumbnail_path:
            thumb_path = Path(pet.thumbnail_path)
            if thumb_path.exists():
                return send_file(str(thumb_path), mimetype='image/jpeg')
        
        if pet.media:
            media_path = Path(pet.media.path)
            if media_path.exists():
                ext = media_path.suffix.lower()
                mimetype_map = {
                    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                    '.png': 'image/png', '.gif': 'image/gif',
                    '.webp': 'image/webp', '.bmp': 'image/bmp'
                }
                mimetype = mimetype_map.get(ext, 'image/jpeg')
                return send_file(str(media_path), mimetype=mimetype)
        
        return jsonify({"error": "Thumbnail file not found"}), 404

@bp.get("/media/<int:media_id>/image")
def get_media_image(media_id: int):
    """Serve full media file."""
    with session_scope() as session:
        media = session.get(MediaFile, media_id)
        if not media:
            return jsonify({"error": "Media not found"}), 404
        
        media_path = Path(media.path)
        if not media_path.exists():
            return jsonify({"error": "Media file not found"}), 404
        
        ext = media_path.suffix.lower()
        mimetype_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.webp': 'image/webp', '.bmp': 'image/bmp'
        }
        mimetype = mimetype_map.get(ext, 'image/jpeg')
        
        return send_file(str(media_path), mimetype=mimetype)

@bp.delete("/media/<int:media_id>")
def delete_media(media_id: int):
    """Move media file to trash."""
    with session_scope() as session:
        media = session.get(MediaFile, media_id)
        if not media:
            return jsonify({"error": "Media not found"}), 404
        
        source_path = Path(media.path)
        if not source_path.exists():
            return jsonify({"error": "Media file not found"}), 404
        
        relative_path = source_path.relative_to(Config.ORGANIZED_DIR)
        trash_path = Config.FACE_TRASH_DIR / relative_path
        trash_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(source_path), str(trash_path))
        
        delete_log = DeleteLog(
            media_id=media.id,
            original_path=str(source_path),
            trash_path=str(trash_path),
            session_id=request.remote_addr or "api"
        )
        session.add(delete_log)
        
        media.is_missing = True
        session.add(media)
        session.flush()
        
        return jsonify({
            "status": "deleted",
            "media_id": media_id,
            "trash_path": str(trash_path)
        })

@bp.get("/unclustered")
def list_unclustered_pets():
    """Return all unclustered pets (cluster_id is NULL)."""
    with session_scope() as session:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        total = session.query(PetEmbedding).filter(
            PetEmbedding.cluster_id.is_(None)
        ).count()
        
        pets = session.query(PetEmbedding).filter(
            PetEmbedding.cluster_id.is_(None)
        ).order_by(PetEmbedding.id).limit(limit).offset(offset).all()
        
        return jsonify({
            "total": total,
            "limit": limit,
            "offset": offset,
            "pets": pet_schema.dump(pets)
        })