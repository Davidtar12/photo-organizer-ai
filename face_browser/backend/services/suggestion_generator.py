"""Active Learning: Generate face clustering suggestions for human review.

This service identifies uncertain clustering decisions and creates ClusterSuggestion
records for review. It runs after automatic clustering to find:
1. Boundary faces: faces close to the distance threshold
2. High-variance clusters: faces that might not belong together
3. Orphan faces: noise points that might actually belong to a cluster
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sqlalchemy import and_, delete, select

from config import Config
from database import session_scope
from models import ClusterSuggestion, FaceEmbedding, PersonCluster, TaskProgress

logger = logging.getLogger(__name__)


class SuggestionGenerator:
    """Generate clustering suggestions for active learning."""

    def __init__(self, cfg: Config = Config()) -> None:
        self.cfg = cfg
        # Threshold range for boundary detection
        self.lower_threshold = 0.20  # Below this = confident match
        self.upper_threshold = 0.30  # Above this = confident non-match
        # Faces in between are uncertain

    def generate_suggestions(self, max_suggestions: int = 100) -> int:
        """
        Generate suggestions for uncertain face assignments.
        
        Returns:
            Number of suggestions created
        """
        logger.info("="*70)
        logger.info("GENERATING ACTIVE LEARNING SUGGESTIONS")
        logger.info("="*70)
        
        with session_scope() as session:
            # Clear old pending suggestions
            session.execute(
                delete(ClusterSuggestion).where(ClusterSuggestion.status == 'pending')
            )
            session.commit()
            
            suggestions_created = 0
            
            # 1. Find boundary faces (noise points near clusters)
            logger.info("Finding boundary faces (noise near clusters)...")
            noise_suggestions = self._find_boundary_noise_faces(session, max_suggestions)
            suggestions_created += len(noise_suggestions)
            self._update_progress("suggestions", total_items=max_suggestions, processed=suggestions_created, message=f"Boundary: {len(noise_suggestions)}")
            
            # 2. Find cluster outliers (faces that might not belong)
            logger.info("Finding potential cluster outliers...")
            outlier_suggestions = self._find_cluster_outliers(session, max_suggestions - suggestions_created)
            suggestions_created += len(outlier_suggestions)
            self._update_progress("suggestions", total_items=max_suggestions, processed=suggestions_created, message=f"Outliers: {len(outlier_suggestions)}")
            
            logger.info(f"Created {suggestions_created} suggestions for review")
            logger.info("="*70)
            
            return suggestions_created

    def _find_boundary_noise_faces(self, session, limit: int) -> List[ClusterSuggestion]:
        """
        Find noise faces that are close to cluster boundaries.
        
        These are faces marked as noise but have a nearest neighbor
        in a cluster that's just above the threshold.
        """
        suggestions = []
        
        # Get all noise faces (cluster_id is None)
        noise_faces = session.execute(
            select(FaceEmbedding).where(FaceEmbedding.cluster_id.is_(None))
        ).scalars().all()
        
        if not noise_faces:
            logger.info("No noise faces found")
            return suggestions
        
        # Get all clustered faces
        clustered_faces = session.execute(
            select(FaceEmbedding).where(FaceEmbedding.cluster_id.isnot(None))
        ).scalars().all()
        
        if not clustered_faces:
            logger.info("No clustered faces found")
            return suggestions
        
        # Build embedding arrays
        noise_embeddings = []
        noise_ids = []
        for face in noise_faces[:1000]:  # Limit to avoid memory issues
            emb = np.frombuffer(face.embedding, dtype=np.float32)
            noise_embeddings.append(emb)
            noise_ids.append(face.id)
        
        clustered_embeddings = []
        clustered_face_map = {}
        for face in clustered_faces[:5000]:  # Limit search space
            emb = np.frombuffer(face.embedding, dtype=np.float32)
            clustered_embeddings.append(emb)
            clustered_face_map[len(clustered_embeddings) - 1] = face
        
        if not noise_embeddings or not clustered_embeddings:
            return suggestions
        
        noise_matrix = np.array(noise_embeddings, dtype=np.float32)
        clustered_matrix = np.array(clustered_embeddings, dtype=np.float32)
        
        # Normalize for cosine distance
        noise_norms = np.linalg.norm(noise_matrix, axis=1, keepdims=True)
        noise_matrix = noise_matrix / (noise_norms + 1e-8)
        
        clustered_norms = np.linalg.norm(clustered_matrix, axis=1, keepdims=True)
        clustered_matrix = clustered_matrix / (clustered_norms + 1e-8)
        
        # Find nearest clustered face for each noise face
        for i, noise_id in enumerate(noise_ids):
            if len(suggestions) >= limit:
                break
            
            # Compute cosine distance to all clustered faces
            similarities = np.dot(noise_matrix[i:i+1], clustered_matrix.T)[0]
            distances = 1.0 - similarities
            
            # Find closest match
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            # If in uncertain range, create suggestion
            if self.lower_threshold <= min_dist <= self.upper_threshold:
                closest_face = clustered_face_map[min_idx]
                
                suggestion = ClusterSuggestion(
                    face_id=noise_id,
                    suggested_cluster_id=closest_face.cluster_id,
                    distance=float(min_dist),
                    reason='boundary_noise',
                    status='pending'
                )
                session.add(suggestion)
                suggestions.append(suggestion)
        
        session.commit()
        logger.info(f"Created {len(suggestions)} boundary noise suggestions")
        return suggestions

    def _find_cluster_outliers(self, session, limit: int) -> List[ClusterSuggestion]:
        """
        Find faces in clusters that might be outliers.
        
        These are faces that are unusually far from their cluster centroid.
        """
        suggestions = []
        
        # Get clusters with at least 5 faces
        clusters = session.execute(
            select(PersonCluster).where(PersonCluster.is_hidden == False)
        ).scalars().all()
        
        for cluster in clusters:
            if len(suggestions) >= limit:
                break
            
            faces = session.execute(
                select(FaceEmbedding).where(FaceEmbedding.cluster_id == cluster.id)
            ).scalars().all()
            
            if len(faces) < 5:
                continue
            
            # Build embedding matrix
            embeddings = []
            face_ids = []
            for face in faces:
                emb = np.frombuffer(face.embedding, dtype=np.float32)
                embeddings.append(emb)
                face_ids.append(face.id)
            
            matrix = np.array(embeddings, dtype=np.float32)
            
            # Normalize
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / (norms + 1e-8)
            
            # Compute centroid
            centroid = np.mean(matrix, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            
            # Compute distance from each face to centroid
            distances = 1.0 - np.dot(matrix, centroid)
            
            # Find outliers (faces > 75th percentile distance)
            threshold = np.percentile(distances, 75)
            
            for i, dist in enumerate(distances):
                if len(suggestions) >= limit:
                    break
                
                if dist > threshold and dist > self.lower_threshold:
                    # This face might not belong - suggest reviewing it
                    # We'll suggest the same cluster, but user can reject to split it out
                    suggestion = ClusterSuggestion(
                        face_id=face_ids[i],
                        suggested_cluster_id=cluster.id,
                        distance=float(dist),
                        reason='cluster_outlier',
                        status='pending'
                    )
                    session.add(suggestion)
                    suggestions.append(suggestion)
        
        session.commit()
        logger.info(f"Created {len(suggestions)} cluster outlier suggestions")
        return suggestions

    def _update_progress(self, task_name: str, *, total_items: int, processed: int, message: str) -> None:
        """Record progress into TaskProgress table for UI visibility."""
        with session_scope() as session:
            task = session.query(TaskProgress).filter_by(task_name=task_name).one_or_none()
            if not task:
                task = TaskProgress(task_name=task_name)
            task.total_items = total_items
            task.processed_items = processed
            task.last_message = message
            session.add(task)


if __name__ == "__main__":  # pragma: no cover
    from logging_config import setup_logging
    setup_logging()
    generator = SuggestionGenerator()
    count = generator.generate_suggestions(max_suggestions=100)
    logger.info(f"Generated {count} suggestions")
