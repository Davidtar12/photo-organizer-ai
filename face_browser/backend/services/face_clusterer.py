from __future__ import annotations

import logging
from typing import List, Optional

import faiss
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sqlalchemy import select, update

from config import Config
from database import session_scope
from models import FaceEmbedding, PersonCluster, TaskProgress
from services.faiss_index import FaissIndex

logger = logging.getLogger(__name__)


class FaceClusterer:
    """Cluster face embeddings into person groups using FAISS kNN + connected components."""

    def __init__(self, cfg: Config = Config(), min_cluster_size: int = 2) -> None:
        self.cfg = cfg
        self.min_cluster_size = min_cluster_size
        self.faiss_index = FaissIndex(cfg)

    def cluster_all(self, distance_threshold: float = 0.315) -> None:
        """
        PASS 1: Cluster all unclustered faces using FAISS kNN + connected components.
        This creates high-purity "seed" clusters using a strict threshold.
        
        Args:
            distance_threshold: L2 distance threshold for clustering (default 0.315 for strict/high-purity)
        """
        logger.info("="*70)
        logger.info("STARTING CLUSTERING - PASS 1 (Seed Clusters)")
        logger.info("="*70)
        logger.info("Algorithm: FAISS kNN + Connected Components")
        logger.info("  → L2-normalize embeddings")
        logger.info("  → Build k-NN graph (k=50)")
        logger.info("  → Filter edges by distance threshold: %.3f", distance_threshold)
        logger.info("  → Find connected components = clusters")
        logger.info("  → Drop clusters with < %s faces", self.min_cluster_size)

        all_embeddings = []
        all_face_ids = []

        with session_scope() as session:
            # Get all faces without a cluster
            query = session.query(FaceEmbedding.id, FaceEmbedding.embedding).filter(
                FaceEmbedding.cluster_id.is_(None)
            )
            total = session.query(FaceEmbedding).filter(FaceEmbedding.cluster_id.is_(None)).count()
            
            logger.info("STEP 1/5: Loading embeddings from database...")
            logger.info("Total unclustered faces: %s", total)
            # Progress: initialize task
            self._update_progress("face_cluster", total_items=total, processed=0, message="Loading embeddings")

            for idx, (face_id, emb_bytes) in enumerate(query, 1):
                embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                all_embeddings.append(embedding)
                all_face_ids.append(face_id)

                if idx % 5000 == 0:
                    progress = (idx / total * 100)
                    logger.info("  → Loaded %s/%s embeddings (%.1f%%)", idx, total, progress)
                    self._update_progress("face_cluster", total_items=total, processed=idx, message=f"Loaded {idx}/{total} embeddings")

        if len(all_embeddings) < self.min_cluster_size:
            logger.warning("Not enough faces to cluster (%s < %s)", len(all_embeddings), self.min_cluster_size)
            return

        logger.info("STEP 2/5: L2-normalizing embeddings...")
        embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
        logger.info("  → Original matrix shape: %s embeddings × %s dimensions", 
                   embeddings_matrix.shape[0], embeddings_matrix.shape[1])
        
        # L2 normalize: critical for ArcFace embeddings!
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        embeddings_matrix = embeddings_matrix / norms
        logger.info("  ✓ Embeddings normalized (unit vectors)")
        
        logger.info("STEP 3/5: Building k-NN graph with FAISS...")
        logger.info("  ⏱️  Estimated time: 10-30 seconds for ~77k faces")
        logger.info("  ⚙️  Using GPU (FAISS GpuIndexFlatIP)")
        
        import time
        start_time = time.time()
        
        # Build FAISS GPU index for Inner Product (= cosine similarity on normalized vectors)
        d = embeddings_matrix.shape[1]
        
        try:
            # Try GPU first
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, d)
            logger.info("  ✓ Using GPU acceleration")
        except Exception as e:
            # Fallback to CPU if GPU fails
            logger.warning("  ⚠️  GPU unavailable (%s), falling back to CPU", str(e))
            index = faiss.IndexFlatIP(d)
        
        index.add(embeddings_matrix)
        
        # Find k nearest neighbors for each face
        k = 50  # Check 50 nearest neighbors
        distances, indices = index.search(embeddings_matrix, k)
        
        elapsed = time.time() - start_time
        logger.info("  ✓ k-NN graph built in %.1f seconds", elapsed)
        logger.info("  → Found %s neighbors per face (k=%s)", k, k)
        
        logger.info("STEP 4/5: Filtering edges and finding connected components...")
        # Convert cosine similarity to cosine distance: distance = 1 - similarity
        cosine_distances = 1.0 - distances
        
        logger.info("  → Distance threshold: %.3f (L2 distance on normalized vectors)", distance_threshold)
        
        # Build sparse adjacency matrix
        n = len(all_face_ids)
        rows, cols, data = [], [], []
        
        for i in range(n):
            for j_idx in range(k):
                j = indices[i, j_idx]
                dist = cosine_distances[i, j_idx]
                
                # Keep edge if below threshold and not self-loop
                if dist < distance_threshold and i != j:
                    rows.append(i)
                    cols.append(j)
                    data.append(1)  # Binary: connected or not
        
        # Create sparse matrix and find connected components
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
        n_components, cluster_labels = connected_components(
            csgraph=adj_matrix, 
            directed=False, 
            return_labels=True
        )
        
        logger.info("  ✓ Found %s connected components (raw clusters)", n_components)
        
        # Filter out small clusters (noise)
        cluster_sizes = np.bincount(cluster_labels)
        valid_clusters = np.where(cluster_sizes >= self.min_cluster_size)[0]
        
        # Remap labels: valid clusters get new IDs, small clusters get -1 (noise)
        label_remap = np.full(n_components, -1, dtype=int)
        label_remap[valid_clusters] = np.arange(len(valid_clusters))
        cluster_labels = label_remap[cluster_labels]
        
        num_clusters = len(valid_clusters)
        num_noise = sum(1 for lbl in cluster_labels if lbl == -1)
        
        elapsed_total = time.time() - start_time
        logger.info("  ✓ Clustering completed in %.1f seconds (%.1f minutes)", elapsed_total, elapsed_total / 60)

        unique_labels = set(cluster_labels)
        num_clusters = len([lbl for lbl in unique_labels if lbl != -1])
        num_noise = sum(1 for lbl in cluster_labels if lbl == -1)

        logger.info("STEP 5/5: Assigning faces to clusters...")
        logger.info("  → Found %s distinct people", num_clusters)
        logger.info("  → %s faces marked as noise/outliers (< %s faces)", num_noise, self.min_cluster_size)

        # Assign faces to clusters in DB
        self._assign_clusters(all_face_ids, cluster_labels)
        self._update_progress("face_cluster", total_items=len(all_face_ids), processed=len(all_face_ids), message="Assignment complete")

        # Rebuild Faiss index to include newly clustered faces
        logger.info("STEP 6/6: Rebuilding search index...")
        self.faiss_index.build()
        
        logger.info("="*70)
        logger.info("CLUSTERING COMPLETE!")
        logger.info("="*70)

    def _assign_clusters(self, face_ids: List[int], cluster_labels: np.ndarray) -> None:
        """Assign face IDs to PersonCluster records based on clustering labels."""
        total_faces = len(face_ids)
        clustered_count = sum(1 for lbl in cluster_labels if lbl != -1)
        
        logger.info("Assigning %s faces to clusters (skipping %s noise points)...", 
                   clustered_count, total_faces - clustered_count)
        
        with session_scope() as session:
            # Map cluster label -> PersonCluster ID
            cluster_map = {}
            assigned = 0
            last_progress = 0

            for idx, (face_id, label) in enumerate(zip(face_ids, cluster_labels), 1):
                if label == -1:
                    # Noise point - skip
                    continue

                if label not in cluster_map:
                    # Create new PersonCluster
                    new_cluster = PersonCluster(display_name=f"Person {label}")
                    session.add(new_cluster)
                    session.flush()
                    cluster_map[label] = new_cluster.id
                    logger.info("Created cluster %s/%s (ID: %s)", 
                               len(cluster_map), len(set(cluster_labels) - {-1}), new_cluster.id)

                # Assign face to cluster
                face = session.get(FaceEmbedding, face_id)
                if face:
                    face.cluster_id = cluster_map[label]
                    session.add(face)
                    assigned += 1
                
                # Progress logging every 10%
                progress = int((idx / total_faces) * 100)
                if progress >= last_progress + 10:
                    logger.info("Assignment progress: %s/%s faces (%s%%)", 
                               idx, total_faces, progress)
                    last_progress = progress
                    # Update task progress roughly every 10%
                    self._update_progress("face_cluster", total_items=total_faces, processed=idx, message=f"Assign {idx}/{total_faces}")

            logger.info("Assignment complete: %s faces assigned to %s clusters", 
                       assigned, len(cluster_map))

    def _update_progress(self, task_name: str, *, total_items: Optional[int], processed: int, message: str) -> None:
        """Record progress into TaskProgress table for UI visibility."""
        with session_scope() as session:
            task = session.query(TaskProgress).filter_by(task_name=task_name).one_or_none()
            if not task:
                task = TaskProgress(task_name=task_name)
            task.total_items = total_items
            task.processed_items = processed
            task.last_message = message
            session.add(task)

    def assign_noise_to_centroids(self, distance_threshold: float = 0.4, batch_size: int = 5000) -> int:
        """
        PASS 2: Assign 'noise' faces (cluster_id=NULL) to the closest existing cluster
        if they are within the distance_threshold.
        
        Returns the number of faces successfully assigned.
        """
        logger.info("="*70)
        logger.info("STARTING CLUSTERING - PASS 2 (Centroid Assignment)")
        logger.info("="*70)
        logger.info("Strategy: Assign unclustered faces to nearest cluster centroids")
        logger.info("  → Distance threshold: %.2f (L2 distance on normalized vectors)", distance_threshold)
        
        centroid_embeddings = []
        cluster_ids = []

        with session_scope() as session:
            # STEP 1: Calculate centroids in Python (robust, portable)
            logger.info("STEP 1/4: Calculating centroids for all existing clusters...")
            
            results = session.query(FaceEmbedding.cluster_id, FaceEmbedding.embedding).filter(
                FaceEmbedding.cluster_id.isnot(None)
            ).all()

            if not results:
                logger.warning("No clusters found to build centroids. Skipping Pass 2.")
                return 0

            # Group embeddings by cluster
            centroids_dict = {}
            for cluster_id, emb_bytes in results:
                vec = np.frombuffer(emb_bytes, dtype=np.float32)
                centroids_dict.setdefault(cluster_id, []).append(vec)
            
            # Filter small clusters and compute means
            min_size = 3
            for cluster_id, vecs in centroids_dict.items():
                if len(vecs) < min_size:
                    continue  # Skip tiny clusters
                centroid = np.mean(np.vstack(vecs), axis=0)
                centroid_embeddings.append(centroid)
                cluster_ids.append(cluster_id)
            
            logger.info("  ✓ Calculated %s cluster centroids (min_size=%s)", len(cluster_ids), min_size)

            if not centroid_embeddings:
                logger.warning("No valid centroids after filtering. Skipping Pass 2.")
                return 0

            # STEP 2: Build temporary FAISS index for centroids
            logger.info("STEP 2/4: Building temporary centroid search index...")
            embeddings_matrix = np.array(centroid_embeddings, dtype=np.float32)
            dimension = embeddings_matrix.shape[1]
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_matrix)
            
            centroid_index = faiss.IndexFlatL2(dimension)
            centroid_index.add(embeddings_matrix)
            logger.info("  ✓ Centroid index ready (%s centroids, %s dims)", len(cluster_ids), dimension)

            # STEP 3: Count 'noise' faces and process in streaming batches
            logger.info("STEP 3/4: Processing 'noise' faces (cluster_id is NULL) in batches of %s...", batch_size)
            total_noise = session.query(FaceEmbedding).filter(
                FaceEmbedding.cluster_id.is_(None)
            ).count()
            logger.info("  ✓ Found %s 'noise' faces to process", total_noise)

            if total_noise == 0:
                logger.info("No unclustered faces to assign. Pass 2 complete.")
                return 0

            # STEP 4: Search and assign in batches (vectorized)
            logger.info("STEP 4/4: Searching for matches and assigning (vectorized batches)...")
            assignments = 0
            processed = 0
            last_log_pct = -1

            # Iterate with offset/limit to keep memory bounded
            for offset in range(0, total_noise, batch_size):
                rows = session.query(FaceEmbedding.id, FaceEmbedding.embedding).filter(
                    FaceEmbedding.cluster_id.is_(None)
                ).order_by(FaceEmbedding.id).limit(batch_size).offset(offset).all()

                if not rows:
                    break

                face_ids_batch = [row[0] for row in rows]
                embs_batch = np.frombuffer(b"".join(row[1] for row in rows), dtype=np.float32)
                # Guard: handle potential ragged arrays
                try:
                    embs_batch = embs_batch.reshape(len(rows), -1)
                except Exception:
                    # Fallback: slower path if any embedding size is inconsistent
                    embs_batch = np.vstack([np.frombuffer(row[1], dtype=np.float32) for row in rows])

                faiss.normalize_L2(embs_batch)

                # Search nearest centroid for the whole batch
                distances, indices = centroid_index.search(embs_batch, 1)

                # Prepare DB updates for those under threshold
                to_assign = []
                for i, dist in enumerate(distances[:, 0]):
                    if dist < distance_threshold:
                        cid = cluster_ids[indices[i, 0]]
                        to_assign.append((face_ids_batch[i], cid))

                # Commit in smaller chunks to keep transactions fast
                if to_assign:
                    for i in range(0, len(to_assign), 2000):
                        chunk = to_assign[i:i+2000]
                        for fid, cid in chunk:
                            session.execute(
                                update(FaceEmbedding)
                                .where(FaceEmbedding.id == fid)
                                .values(cluster_id=cid)
                            )
                        session.commit()

                assignments += len(to_assign)
                processed += len(rows)

                pct = int((processed / total_noise) * 100)
                if pct >= last_log_pct + 5:
                    logger.info("  → Processed %s/%s (%.0f%%), assigned so far: %s", processed, total_noise, pct, assignments)
                    last_log_pct = pct

            reduction_pct = (assignments / total_noise) * 100 if total_noise > 0 else 0
            logger.info("="*70)
            logger.info("PASS 2 COMPLETE!")
            logger.info("="*70)
            logger.info("  ✓ Assigned %s / %s noise faces (%.1f%% reduction)", 
                       assignments, total_noise, reduction_pct)
            logger.info("  → Remaining unclustered: %s", total_noise - assignments)
            
            return assignments

    def merge_similar_clusters(self, distance_threshold: float = 0.35) -> int:
        """
        PASS 3: Automatically merge clusters that are very similar (likely the same person).
        This fixes "split errors" where one person was divided into multiple clusters.
        
        Returns the number of clusters merged.
        """
        logger.info("="*70)
        logger.info("STARTING CLUSTERING - PASS 3 (Cluster Merging)")
        logger.info("="*70)
        logger.info("Strategy: Merge clusters with similar centroids")
        logger.info("  → Distance threshold: %.2f (L2 distance on normalized centroids)", distance_threshold)
        
        with session_scope() as session:
            # STEP 1: Get all clusters and compute their centroids
            logger.info("STEP 1/4: Computing centroids for all clusters...")
            
            results = session.query(FaceEmbedding.cluster_id, FaceEmbedding.embedding).filter(
                FaceEmbedding.cluster_id.isnot(None)
            ).all()

            if not results:
                logger.warning("No clusters found. Skipping Pass 3.")
                return 0

            # Group embeddings by cluster
            centroids_dict = {}
            for cluster_id, emb_bytes in results:
                vec = np.frombuffer(emb_bytes, dtype=np.float32)
                centroids_dict.setdefault(cluster_id, []).append(vec)
            
            # Compute centroids only for clusters with enough faces
            min_size = 3
            centroid_embeddings = []
            cluster_ids = []
            cluster_sizes = []
            
            for cluster_id, vecs in centroids_dict.items():
                if len(vecs) < min_size:
                    continue
                centroid = np.mean(np.vstack(vecs), axis=0)
                centroid_embeddings.append(centroid)
                cluster_ids.append(cluster_id)
                cluster_sizes.append(len(vecs))
            
            num_clusters = len(cluster_ids)
            logger.info("  ✓ Computed %s cluster centroids (min_size=%s)", num_clusters, min_size)

            if num_clusters < 2:
                logger.info("Not enough clusters to merge. Skipping Pass 3.")
                return 0

            # STEP 2: Build k-NN graph between cluster centroids
            logger.info("STEP 2/4: Building centroid similarity graph...")
            embeddings_matrix = np.array(centroid_embeddings, dtype=np.float32)
            dimension = embeddings_matrix.shape[1]
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_matrix)
            
            # Use Inner Product index (cosine similarity on normalized vectors)
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_matrix)
            
            # Find k nearest cluster centroids (k=10 should be enough)
            k = min(10, num_clusters)
            similarities, indices = index.search(embeddings_matrix, k)
            
            # Convert to distances: distance = 1 - similarity
            distances = 1.0 - similarities
            
            logger.info("  ✓ Found %s nearest neighbors per cluster centroid", k)

            # STEP 3: Build merge graph (clusters to merge)
            logger.info("STEP 3/4: Finding clusters to merge (distance < %.2f)...", distance_threshold)
            
            # Build adjacency list for merging
            merge_graph = {i: set() for i in range(num_clusters)}
            
            for i in range(num_clusters):
                for j_idx in range(k):
                    j = indices[i, j_idx]
                    dist = distances[i, j_idx]
                    
                    # Skip self-loops and edges above threshold
                    if i == j or dist >= distance_threshold:
                        continue
                    
                    # Add bidirectional edge
                    merge_graph[i].add(j)
                    merge_graph[j].add(i)
            
            # Find connected components in merge graph (these are merge groups)
            visited = set()
            merge_groups = []
            
            def dfs(node, group):
                """Depth-first search to find connected components."""
                visited.add(node)
                group.append(node)
                for neighbor in merge_graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor, group)
            
            for i in range(num_clusters):
                if i not in visited:
                    group = []
                    dfs(i, group)
                    if len(group) > 1:  # Only keep groups with 2+ clusters
                        merge_groups.append(group)
            
            num_merge_groups = len(merge_groups)
            total_clusters_to_merge = sum(len(g) for g in merge_groups)
            
            logger.info("  ✓ Found %s merge groups (total %s clusters to merge)", 
                       num_merge_groups, total_clusters_to_merge)

            if num_merge_groups == 0:
                logger.info("No similar clusters found to merge. Pass 3 complete.")
                return 0

            # STEP 4: Perform the merges
            logger.info("STEP 4/4: Merging clusters...")
            merges_performed = 0
            
            for group_idx, group in enumerate(merge_groups, 1):
                # Get the actual cluster IDs
                group_cluster_ids = [cluster_ids[i] for i in group]
                group_cluster_sizes = [cluster_sizes[i] for i in group]
                
                # Keep the largest cluster as the target
                max_idx = group_cluster_sizes.index(max(group_cluster_sizes))
                target_cluster_id = group_cluster_ids[max_idx]
                
                # Merge all others into the target
                for i, source_cluster_id in enumerate(group_cluster_ids):
                    if i == max_idx:
                        continue  # Skip the target itself
                    
                    # Update all faces in source cluster to point to target
                    session.execute(
                        update(FaceEmbedding)
                        .where(FaceEmbedding.cluster_id == source_cluster_id)
                        .values(cluster_id=target_cluster_id)
                    )
                    
                    # Delete the now-empty source cluster
                    session.query(PersonCluster).filter(
                        PersonCluster.id == source_cluster_id
                    ).delete()
                    
                    merges_performed += 1
                
                logger.info("  → Merge group %s/%s: Merged %s clusters into cluster %s (%s faces)", 
                           group_idx, num_merge_groups, len(group), target_cluster_id, sum(group_cluster_sizes))
            
            session.commit()
            
            logger.info("="*70)
            logger.info("PASS 3 COMPLETE!")
            logger.info("="*70)
            logger.info("  ✓ Merged %s clusters into %s groups", merges_performed, num_merge_groups)
            logger.info("  → Reduced cluster count by %s", merges_performed)
            
            return merges_performed


if __name__ == "__main__":  # pragma: no cover
    from logging_config import setup_logging
    setup_logging()
    clusterer = FaceClusterer(min_cluster_size=2)
    clusterer.cluster_all()
