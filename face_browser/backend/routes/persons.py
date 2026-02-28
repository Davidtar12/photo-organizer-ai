from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

import faiss
import numpy as np
from flask import Blueprint, jsonify, request, send_file, redirect, url_for
from sqlalchemy import func, select

from config import Config
from database import session_scope
from models import DeleteLog, FaceEmbedding, MediaFile, MergeEvent, PersonCluster
from schemas import FaceEmbeddingSchema, PersonClusterSchema
from .utils import is_pet_face

bp = Blueprint("persons", __name__, url_prefix="/api/persons")

cluster_schema = PersonClusterSchema()
face_schema = FaceEmbeddingSchema(many=True)

# In-memory centroid cache for Active Learning suggestions
# Stores normalized centroid matrix, corresponding cluster IDs and sizes,
# and a FAISS index for fast lookups. Rebuilt on demand.
CENTROID_CACHE = {
    "built": 0.0,
    "min_size": 3,
    "dimension": 512,
    "cluster_ids": None,
    "cluster_sizes": None,
    "centroids": None,        # np.ndarray [N, D]
    "faiss_index": None,      # faiss.IndexFlatIP over centroids
    "num_clusters_total": 0,
}


def _rebuild_centroid_cache(session, *, min_size: int = 3) -> dict:
    """Compute centroids for existing clusters and build FAISS index.

    This function reads all clustered faces once, computes per-cluster centroids,
    normalizes them, and builds an IndexFlatIP for cosine similarity.
    """
    from collections import defaultdict
    import time

    results = session.query(
        FaceEmbedding.cluster_id,
        FaceEmbedding.embedding
    ).filter(
        FaceEmbedding.cluster_id.isnot(None)
    ).all()

    if not results:
        # Reset cache
        CENTROID_CACHE.update({
            "built": time.time(),
            "min_size": min_size,
            "cluster_ids": np.array([], dtype=np.int64),
            "cluster_sizes": {},
            "centroids": np.zeros((0, CENTROID_CACHE["dimension"]), dtype=np.float32),
            "faiss_index": faiss.IndexFlatIP(CENTROID_CACHE["dimension"]),
            "num_clusters_total": 0,
        })
        return CENTROID_CACHE

    cluster_embeddings = defaultdict(list)
    for cluster_id, emb_bytes in results:
        vec = np.frombuffer(emb_bytes, dtype=np.float32)
        cluster_embeddings[cluster_id].append(vec)

    centroids = []
    cluster_ids = []
    cluster_sizes = {}
    for cid, vecs in cluster_embeddings.items():
        if len(vecs) < min_size:
            continue
        centroid = np.mean(np.vstack(vecs), axis=0).astype(np.float32)
        # Normalize to unit length
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)
        cluster_ids.append(cid)
        cluster_sizes[cid] = len(vecs)

    # Build FAISS index
    if len(centroids) == 0:
        centroid_mat = np.zeros((0, CENTROID_CACHE["dimension"]), dtype=np.float32)
    else:
        centroid_mat = np.vstack(centroids).astype(np.float32)

    index = faiss.IndexFlatIP(CENTROID_CACHE["dimension"])
    if centroid_mat.shape[0] > 0:
        index.add(centroid_mat)

    CENTROID_CACHE.update({
        "built": time.time(),
        "min_size": min_size,
        "cluster_ids": np.array(cluster_ids, dtype=np.int64),
        "cluster_sizes": cluster_sizes,
        "centroids": centroid_mat,
        "faiss_index": index,
        "num_clusters_total": len(cluster_embeddings),  # total clusters before min_size filter
    })

    return CENTROID_CACHE


def _ensure_centroid_cache(session) -> dict:
    """Ensure the centroid cache is built; build it on first use."""
    if CENTROID_CACHE.get("faiss_index") is None:
        return _rebuild_centroid_cache(session, min_size=CENTROID_CACHE.get("min_size", 3))
    return CENTROID_CACHE


@bp.get("/")
def list_clusters():
    """Return all clusters with face counts."""

    with session_scope() as session:
        stmt = (
            select(
                PersonCluster,
                func.count(FaceEmbedding.id).label("face_count"),
            )
            .outerjoin(FaceEmbedding, FaceEmbedding.cluster_id == PersonCluster.id)
            .group_by(PersonCluster.id)
            .order_by(func.count(FaceEmbedding.id).desc())
        )
        results = session.execute(stmt).all()

        payload = []
        for cluster, face_count in results:
            # Check if any face in cluster is pet
            faces = session.query(FaceEmbedding).filter(FaceEmbedding.cluster_id == cluster.id).limit(5).all()  # check first 5
            is_pet_cluster = any(is_pet_face(face.thumbnail_path) for face in faces if face.thumbnail_path)
            if not is_pet_cluster:
                data = cluster_schema.dump(cluster)
                data["face_count"] = face_count
                payload.append(data)

        return jsonify(payload)


@bp.get("/<int:cluster_id>")
def get_cluster(cluster_id: int):
    with session_scope() as session:
        cluster = session.get(PersonCluster, cluster_id)
        if not cluster:
            return jsonify({"error": "Cluster not found"}), 404

        faces = (
            session.query(FaceEmbedding)
            .filter(FaceEmbedding.cluster_id == cluster_id)
            .order_by(FaceEmbedding.id)
            .all()
        )

        return jsonify(
            {
                "cluster": cluster_schema.dump(cluster),
                "faces": face_schema.dump(faces),
            }
        )


@bp.patch("/<int:cluster_id>")
def update_cluster(cluster_id: int):
    payload = request.get_json(force=True)
    allowed_fields = {"display_name", "description", "is_hidden"}

    with session_scope() as session:
        cluster = session.get(PersonCluster, cluster_id)
        if not cluster:
            return jsonify({"error": "Cluster not found"}), 404

        for key, value in payload.items():
            if key in allowed_fields:
                setattr(cluster, key, value)

        session.add(cluster)
        session.flush()

        data = cluster_schema.dump(cluster)
        face_count = (
            session.query(FaceEmbedding)
            .filter(FaceEmbedding.cluster_id == cluster_id)
            .count()
        )
        data["face_count"] = face_count
        return jsonify(data)


@bp.post("/merge")
def merge_clusters():
    payload = request.get_json(force=True)
    target_cluster_id = payload.get("target_cluster_id")
    source_cluster_ids: List[int] = payload.get("source_cluster_ids", [])
    initiated_by = payload.get("initiated_by", "manual")

    if not target_cluster_id or not source_cluster_ids:
        return jsonify({"error": "target_cluster_id and source_cluster_ids required"}), 400

    if target_cluster_id in source_cluster_ids:
        return jsonify({"error": "target cannot be part of source list"}), 400

    with session_scope() as session:
        target_cluster = session.get(PersonCluster, target_cluster_id)
        if not target_cluster:
            return jsonify({"error": "Target cluster not found"}), 404

        updated_faces = 0
        for source_id in source_cluster_ids:
            source_cluster = session.get(PersonCluster, source_id)
            if not source_cluster:
                continue

            updated_faces += (
                session.query(FaceEmbedding)
                .filter(FaceEmbedding.cluster_id == source_id)
                .update({FaceEmbedding.cluster_id: target_cluster_id})
            )

            merge_event = MergeEvent(
                source_cluster_id=source_id,
                target_cluster_id=target_cluster_id,
                initiated_by=initiated_by,
                notes=payload.get("notes"),
            )
            session.add(merge_event)
            session.delete(source_cluster)

        session.add(target_cluster)
        session.flush()

        response = cluster_schema.dump(target_cluster)
        response["face_count"] = (
            session.query(FaceEmbedding)
            .filter(FaceEmbedding.cluster_id == target_cluster_id)
            .count()
        )
        response["updated_faces"] = updated_faces
        return jsonify(response)


@bp.get("/<int:cluster_id>/cover-image")
def get_cluster_cover_image(cluster_id: int):
    """Return a redirect to a cover image for the cluster."""
    with session_scope() as session:
        # Prioritize the primary face if set
        cluster = session.get(PersonCluster, cluster_id)
        if cluster and cluster.primary_face_id:
            return redirect(url_for('persons.get_face_thumbnail', face_id=cluster.primary_face_id))

        # Fallback: find the first face with a thumbnail
        face_with_thumb = session.query(FaceEmbedding).filter(
            FaceEmbedding.cluster_id == cluster_id,
            FaceEmbedding.thumbnail_path != None
        ).first()

        if face_with_thumb:
            return redirect(url_for('persons.get_face_thumbnail', face_id=face_with_thumb.id))

        # Fallback 2: find the first face with any media
        face_with_media = session.query(FaceEmbedding).filter(
            FaceEmbedding.cluster_id == cluster_id,
            FaceEmbedding.media_id != None
        ).first()

        if face_with_media:
            return redirect(url_for('persons.get_media_image', media_id=face_with_media.media_id))

        # If no image can be found, return 404
        return '', 404


@bp.get("/face/<int:face_id>/thumbnail")
def get_face_thumbnail(face_id: int):
    """Serve face thumbnail image."""
    with session_scope() as session:
        face = session.get(FaceEmbedding, face_id)
        if not face:
            return jsonify({"error": "Face not found"}), 404
        
        # Try thumbnail first
        if face.thumbnail_path:
            thumb_path = Path(face.thumbnail_path)
            if thumb_path.exists():
                return send_file(str(thumb_path), mimetype='image/jpeg')
        
        # Fallback to media file if thumbnail doesn't exist
        if face.media:
            media_path = Path(face.media.path)
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
        
        # Determine mimetype
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
        
        # Create trash path maintaining directory structure
        relative_path = source_path.relative_to(Config.ORGANIZED_DIR)
        trash_path = Config.FACE_TRASH_DIR / relative_path
        trash_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file to trash
        shutil.move(str(source_path), str(trash_path))
        
        # Log deletion
        delete_log = DeleteLog(
            media_id=media.id,
            original_path=str(source_path),
            trash_path=str(trash_path),
            session_id=request.remote_addr or "api"
        )
        session.add(delete_log)
        
        # Mark media as missing
        media.is_missing = True
        session.add(media)
        session.flush()
        
        return jsonify({
            "status": "deleted",
            "media_id": media_id,
            "trash_path": str(trash_path)
        })


@bp.get("/unclustered")
def list_unclustered_faces():
    """Return all unclustered faces (cluster_id is NULL)."""
    with session_scope() as session:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Get total count
        total = session.query(FaceEmbedding).filter(
            FaceEmbedding.cluster_id.is_(None)
        ).count()
        
        # Get page of results
        faces = session.query(FaceEmbedding).filter(
            FaceEmbedding.cluster_id.is_(None)
        ).order_by(FaceEmbedding.id).limit(limit).offset(offset).all()
        
        return jsonify({
            "total": total,
            "limit": limit,
            "offset": offset,
            "faces": face_schema.dump(faces)
        })


@bp.get("/face/<int:face_id>/suggestions")
def get_face_suggestions(face_id: int):
    """
    ACTIVE LEARNING API: Get top 5 cluster suggestions for an unclustered face.
    
    This is the "fourth pass" - human-in-the-loop for hard cases.
    Uses FAISS to find the 5 nearest cluster centroids.
    
    Returns:
        {
            "face_id": int,
            "suggestions": [
                {
                    "cluster_id": int,
                    "similarity": float,  # cosine similarity (0-1, higher=closer)
                    "person_name": str,
                    "face_count": int,
                    "sample_face_id": int  # for showing preview thumbnail
                },
                ...
            ],
            "num_clusters_total": int
        }
    """
    with session_scope() as session:
        # Step 1: Load the target face's embedding
        face = session.get(FaceEmbedding, face_id)
        if not face:
            return jsonify({"error": "Face not found"}), 404
        
        if not face.embedding:
            return jsonify({"error": "Face has no embedding"}), 400
        
        query_emb = np.frombuffer(face.embedding, dtype=np.float32)
        if query_emb.size != 512:  # AdaFace dimension
            return jsonify({"error": f"Invalid embedding dimension: {query_emb.size}"}), 400
        
        # Normalize (should already be normalized, but ensure)
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Step 2: Use cached cluster centroids and FAISS index
        cache = _ensure_centroid_cache(session)
        centroid_mat = cache.get("centroids")  # [N, D]
        cluster_ids_arr = cache.get("cluster_ids")  # [N]
        cluster_sizes = cache.get("cluster_sizes") or {}
        index = cache.get("faiss_index")

        if centroid_mat is None or index is None or centroid_mat.shape[0] == 0:
            return jsonify({
                "face_id": face_id,
                "suggestions": [],
                "message": "No clusters available yet. Run clustering first.",
                "num_clusters_total": cache.get("num_clusters_total", 0)
            }), 200

        # Step 3: Search for top K=5 closest centroids
        k = min(5, centroid_mat.shape[0])
        similarities, indices = index.search(query_emb.reshape(1, -1), k)
        
        # Step 5: Build suggestions response
        suggestions = []
        for i in range(k):
            idx = indices[0][i]
            cluster_id = int(cluster_ids_arr[idx])
            similarity = float(similarities[0][i])  # Cosine similarity (0-1)
            
            # Get cluster info
            cluster = session.get(PersonCluster, cluster_id)
            person_name = cluster.display_name if cluster and cluster.display_name else f"Person {cluster_id}"
            
            # Get a sample face from this cluster for thumbnail preview
            sample_face = session.query(FaceEmbedding).filter(
                FaceEmbedding.cluster_id == cluster_id,
                FaceEmbedding.thumbnail_path.isnot(None)
            ).first()
            
            suggestions.append({
                "cluster_id": int(cluster_id),
                "similarity": round(similarity, 4),
                "person_name": person_name,
                "face_count": cluster_sizes[cluster_id],
                "sample_face_id": sample_face.id if sample_face else None,
                "distance": round(1.0 - similarity, 4)  # L2 distance approximation
            })
        
        # Sort by similarity descending (already sorted by FAISS, but ensure)
        suggestions.sort(key=lambda x: x['similarity'], reverse=True)
        
        return jsonify({
            "face_id": face_id,
            "suggestions": suggestions,
            "num_clusters_total": int(cache.get("num_clusters_total", centroid_mat.shape[0])),
            "threshold_recommendation": {
                "high_confidence": 0.85,  # >0.85 similarity = very likely same person
                "moderate": 0.75,         # 0.75-0.85 = probably same person
                "low": 0.65               # 0.65-0.75 = maybe same person
            }
        })


@bp.post("/rebuild-cache")
def rebuild_centroid_cache_endpoint():
    """Rebuild the in-memory centroid cache for suggestions.

    Body (optional): { "min_size": int }
    """
    payload = request.get_json(silent=True) or {}
    min_size = int(payload.get("min_size", CENTROID_CACHE.get("min_size", 3)))
    with session_scope() as session:
        cache = _rebuild_centroid_cache(session, min_size=min_size)
        return jsonify({
            "status": "rebuilt",
            "centroids": int(cache["centroids"].shape[0]) if cache.get("centroids") is not None else 0,
            "dimension": int(cache.get("dimension", 512)),
            "min_size": int(cache.get("min_size", 3)),
            "num_clusters_total": int(cache.get("num_clusters_total", 0)),
            "built": float(cache.get("built", 0.0))
        })


@bp.post("/face/<int:face_id>/assign")
def assign_face_to_cluster(face_id: int):
    """
    Assign a single face to a cluster (for active learning workflow).
    
    Body:
        {
            "cluster_id": int,  # Existing cluster to assign to
            "create_new": bool  # If true, create new cluster for this face
        }
    """
    payload = request.get_json(force=True)
    cluster_id = payload.get("cluster_id")
    create_new = payload.get("create_new", False)
    
    with session_scope() as session:
        face = session.get(FaceEmbedding, face_id)
        if not face:
            return jsonify({"error": "Face not found"}), 404
        
        if create_new:
            # Create a new cluster for this face
            new_cluster = PersonCluster(
                display_name=None,  # User can name it later
                description="Created from active learning"
            )
            session.add(new_cluster)
            session.flush()
            
            face.cluster_id = new_cluster.id
            session.add(face)
            session.flush()
            
            return jsonify({
                "status": "created_new",
                "face_id": face_id,
                "cluster_id": new_cluster.id,
                "message": "Created new cluster for this face"
            })
        else:
            # Assign to existing cluster
            if not cluster_id:
                return jsonify({"error": "cluster_id required when create_new=false"}), 400
            
            cluster = session.get(PersonCluster, cluster_id)
            if not cluster:
                return jsonify({"error": "Cluster not found"}), 404
            
            face.cluster_id = cluster_id
            session.add(face)
            session.flush()
            
            return jsonify({
                "status": "assigned",
                "face_id": face_id,
                "cluster_id": cluster_id,
                "message": f"Assigned face to cluster {cluster_id}"
            })
