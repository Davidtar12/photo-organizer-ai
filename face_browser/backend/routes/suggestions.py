"""API routes for active learning suggestions."""

from __future__ import annotations

import logging
from typing import Any, Dict

from flask import Blueprint, jsonify, request
from sqlalchemy import select

from database import session_scope
from models import ClusterSuggestion, FaceEmbedding, PersonCluster

logger = logging.getLogger(__name__)

bp = Blueprint("suggestions", __name__, url_prefix="/api/suggestions")


@bp.route("/pending", methods=["GET"])
def get_pending_suggestions():
    """Get pending suggestions for review."""
    limit = request.args.get("limit", 20, type=int)
    
    with session_scope() as session:
        suggestions = session.execute(
            select(ClusterSuggestion)
            .where(ClusterSuggestion.status == 'pending')
            .order_by(ClusterSuggestion.distance.desc())  # Most uncertain first
            .limit(limit)
        ).scalars().all()
        
        result = []
        for suggestion in suggestions:
            face = session.get(FaceEmbedding, suggestion.face_id)
            cluster = session.get(PersonCluster, suggestion.suggested_cluster_id)
            
            if not face or not cluster:
                continue
            
            # Get sample faces from the suggested cluster
            cluster_faces = session.execute(
                select(FaceEmbedding)
                .where(FaceEmbedding.cluster_id == cluster.id)
                .limit(5)
            ).scalars().all()
            
            result.append({
                "id": suggestion.id,
                "face": {
                    "id": face.id,
                    "media_id": face.media_id,
                    "thumbnail_path": face.thumbnail_path,
                    "bbox": face.bbox
                },
                "suggested_cluster": {
                    "id": cluster.id,
                    "display_name": cluster.display_name or f"Person {cluster.id}",
                    "sample_faces": [
                        {
                            "id": cf.id,
                            "thumbnail_path": cf.thumbnail_path
                        }
                        for cf in cluster_faces
                    ]
                },
                "distance": suggestion.distance,
                "reason": suggestion.reason
            })
        
        return jsonify({
            "suggestions": result,
            "total": len(result)
        })


@bp.route("/<int:suggestion_id>/accept", methods=["POST"])
def accept_suggestion(suggestion_id: int):
    """Accept a suggestion: assign face to suggested cluster."""
    with session_scope() as session:
        suggestion = session.get(ClusterSuggestion, suggestion_id)
        
        if not suggestion:
            return jsonify({"error": "Suggestion not found"}), 404
        
        face = session.get(FaceEmbedding, suggestion.face_id)
        
        if not face:
            return jsonify({"error": "Face not found"}), 404
        
        # Assign face to suggested cluster
        old_cluster_id = face.cluster_id
        face.cluster_id = suggestion.suggested_cluster_id
        
        # Mark suggestion as accepted
        suggestion.status = 'accepted'
        from datetime import datetime
        suggestion.reviewed_at = datetime.utcnow()
        
        session.add(face)
        session.add(suggestion)
        session.commit()
        
        logger.info(
            f"Accepted suggestion {suggestion_id}: Face {face.id} "
            f"moved from cluster {old_cluster_id} to {suggestion.suggested_cluster_id}"
        )
        
        return jsonify({
            "success": True,
            "message": "Face assigned to cluster"
        })


@bp.route("/<int:suggestion_id>/reject", methods=["POST"])
def reject_suggestion(suggestion_id: int):
    """Reject a suggestion: keep face where it is (or create new cluster)."""
    with session_scope() as session:
        suggestion = session.get(ClusterSuggestion, suggestion_id)
        
        if not suggestion:
            return jsonify({"error": "Suggestion not found"}), 404
        
        face = session.get(FaceEmbedding, suggestion.face_id)
        
        if not face:
            return jsonify({"error": "Face not found"}), 404
        
        # If face is noise and we're rejecting, optionally create a new cluster
        create_new = request.json.get("create_new_cluster", False) if request.json else False
        
        if create_new and face.cluster_id is None:
            # Create new single-person cluster
            new_cluster = PersonCluster(
                display_name=None  # User can name it later
            )
            session.add(new_cluster)
            session.flush()
            
            face.cluster_id = new_cluster.id
            session.add(face)
            
            logger.info(
                f"Rejected suggestion {suggestion_id}: Created new cluster {new_cluster.id} for face {face.id}"
            )
        else:
            logger.info(
                f"Rejected suggestion {suggestion_id}: Kept face {face.id} in current state"
            )
        
        # Mark suggestion as rejected
        suggestion.status = 'rejected'
        from datetime import datetime
        suggestion.reviewed_at = datetime.utcnow()
        
        session.add(suggestion)
        session.commit()
        
        return jsonify({
            "success": True,
            "message": "Suggestion rejected"
        })


@bp.route("/<int:suggestion_id>/skip", methods=["POST"])
def skip_suggestion(suggestion_id: int):
    """Skip a suggestion for now."""
    with session_scope() as session:
        suggestion = session.get(ClusterSuggestion, suggestion_id)
        
        if not suggestion:
            return jsonify({"error": "Suggestion not found"}), 404
        
        suggestion.status = 'skipped'
        from datetime import datetime
        suggestion.reviewed_at = datetime.utcnow()
        
        session.add(suggestion)
        session.commit()
        
        return jsonify({
            "success": True,
            "message": "Suggestion skipped"
        })


@bp.route("/stats", methods=["GET"])
def get_stats():
    """Get statistics about suggestions."""
    with session_scope() as session:
        total_pending = session.execute(
            select(ClusterSuggestion).where(ClusterSuggestion.status == 'pending')
        ).scalars().all()
        
        total_accepted = session.execute(
            select(ClusterSuggestion).where(ClusterSuggestion.status == 'accepted')
        ).scalars().all()
        
        total_rejected = session.execute(
            select(ClusterSuggestion).where(ClusterSuggestion.status == 'rejected')
        ).scalars().all()
        
        return jsonify({
            "pending": len(total_pending),
            "accepted": len(total_accepted),
            "rejected": len(total_rejected),
            "total": len(total_pending) + len(total_accepted) + len(total_rejected)
        })
