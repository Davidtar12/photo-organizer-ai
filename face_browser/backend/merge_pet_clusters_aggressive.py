"""
Aggressive Multi-Pass Cluster Merging
Uses multiple strategies to achieve maximum consolidation while maintaining purity
"""

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, PetCluster
from config import Config
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from collections import defaultdict
from tqdm import tqdm

# Configuration - AGGRESSIVE settings
CENTROID_SIMILARITY_THRESHOLD = 0.80  # Lower = more merging
TIGHT_CLUSTER_THRESHOLD = 0.90  # High intra-cluster similarity = safe to merge
MAX_MERGED_CLUSTER_SIZE = 100  # Allow larger clusters
MIN_SAMPLES_FOR_STATS = 3  # Need at least 3 samples to compute variance
MAX_PASSES = 10  # Maximum number of refinement passes

# Database setup
DATABASE_URL = f'sqlite:///{Config.DB_PATH}'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


def get_cluster_data(session, species_name):
    """Get all cluster data including embeddings"""
    pets = session.query(PetEmbedding).filter(
        PetEmbedding.species == species_name,
        PetEmbedding.cluster_id.isnot(None)
    ).all()
    
    cluster_data = defaultdict(list)
    for pet in pets:
        embedding = np.frombuffer(pet.embedding, dtype=np.float32)
        cluster_data[pet.cluster_id].append({
            'pet_id': pet.id,
            'embedding': embedding
        })
    
    return cluster_data


def calculate_cluster_statistics(cluster_data):
    """Calculate centroid and intra-cluster cohesion for each cluster"""
    stats = {}
    
    for cluster_id, members in cluster_data.items():
        embeddings = np.array([m['embedding'] for m in members])
        
        # Centroid
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        # Intra-cluster similarity (cohesion)
        if len(members) >= 2:
            pairwise_sim = cosine_similarity(embeddings)
            # Average similarity between all pairs (excluding diagonal)
            mask = ~np.eye(len(members), dtype=bool)
            avg_similarity = pairwise_sim[mask].mean()
        else:
            avg_similarity = 1.0  # Single member clusters are perfectly cohesive
        
        # Variance (spread)
        distances_to_centroid = cosine_distances(embeddings, centroid.reshape(1, -1)).flatten()
        variance = distances_to_centroid.var()
        
        stats[cluster_id] = {
            'centroid': centroid,
            'size': len(members),
            'cohesion': avg_similarity,
            'variance': variance,
            'tight': avg_similarity >= TIGHT_CLUSTER_THRESHOLD
        }
    
    return stats


def find_mergeable_pairs_centroid(stats, threshold):
    """Find pairs based on centroid similarity"""
    cluster_ids = list(stats.keys())
    centroids = np.array([stats[cid]['centroid'] for cid in cluster_ids])
    
    similarity_matrix = cosine_similarity(centroids)
    
    merge_pairs = []
    n = len(cluster_ids)
    
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                cid1, cid2 = cluster_ids[i], cluster_ids[j]
                size1, size2 = stats[cid1]['size'], stats[cid2]['size']
                
                if size1 + size2 <= MAX_MERGED_CLUSTER_SIZE:
                    merge_pairs.append((cid1, cid2, similarity_matrix[i, j]))
    
    return merge_pairs


def find_mergeable_pairs_tight_clusters(stats):
    """
    Merge very tight clusters (high internal cohesion) even if centroids are slightly different
    Rationale: If both clusters are very tight, they're likely the same dog
    """
    cluster_ids = [cid for cid, s in stats.items() if s['tight'] and s['size'] >= 2]
    
    if len(cluster_ids) < 2:
        return []
    
    centroids = np.array([stats[cid]['centroid'] for cid in cluster_ids])
    similarity_matrix = cosine_similarity(centroids)
    
    # For tight clusters, use a slightly lower threshold
    threshold = CENTROID_SIMILARITY_THRESHOLD - 0.05
    
    merge_pairs = []
    n = len(cluster_ids)
    
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                cid1, cid2 = cluster_ids[i], cluster_ids[j]
                size1, size2 = stats[cid1]['size'], stats[cid2]['size']
                
                if size1 + size2 <= MAX_MERGED_CLUSTER_SIZE:
                    merge_pairs.append((cid1, cid2, similarity_matrix[i, j]))
    
    return merge_pairs


def union_find_merge(merge_pairs, all_cluster_ids):
    """Use Union-Find to handle transitive merges"""
    parent = {cid: cid for cid in all_cluster_ids}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Apply all merges
    for cid1, cid2, _ in merge_pairs:
        union(cid1, cid2)
    
    # Group by parent
    merge_groups = defaultdict(list)
    for cid in all_cluster_ids:
        root = find(cid)
        merge_groups[root].append(cid)
    
    # Filter to only groups with merges
    merge_groups = {k: v for k, v in merge_groups.items() if len(v) > 1}
    
    return merge_groups


def apply_merges(session, merge_groups, species_name, cluster_data):
    """Apply merges to database"""
    pets_reassigned = 0
    clusters_merged = 0
    
    for root_cluster, cluster_group in tqdm(merge_groups.items(), desc=f"Merging {species_name}s"):
        clusters_to_merge = [c for c in cluster_group if c != root_cluster]
        
        for cluster_id in clusters_to_merge:
            # Get all pet IDs in this cluster
            pet_ids = [m['pet_id'] for m in cluster_data[cluster_id]]
            
            # Reassign to root
            for pet_id in pet_ids:
                pet = session.query(PetEmbedding).get(pet_id)
                if pet:
                    pet.cluster_id = root_cluster
                    pets_reassigned += 1
            
            # Delete cluster
            cluster = session.query(PetCluster).get(cluster_id)
            if cluster:
                session.delete(cluster)
            
            clusters_merged += 1
    
    session.commit()
    
    return clusters_merged, pets_reassigned


def multi_pass_merge(session, species_name, max_passes=3):
    """
    Iteratively merge clusters over multiple passes
    Each pass recalculates statistics and finds new merge opportunities
    """
    print(f"\n{'='*70}")
    print(f"MULTI-PASS AGGRESSIVE MERGING: {species_name.upper()}")
    print(f"{'='*70}")
    
    total_merged = 0
    total_reassigned = 0
    
    for pass_num in range(1, max_passes + 1):
        print(f"\n--- Pass {pass_num}/{max_passes} ---")
        
        # Get current cluster data
        cluster_data = get_cluster_data(session, species_name)
        
        if not cluster_data:
            print("No clusters found!")
            break
        
        print(f"Current clusters: {len(cluster_data)}")
        
        # Calculate statistics
        stats = calculate_cluster_statistics(cluster_data)
        
        # Count tight clusters
        tight_count = sum(1 for s in stats.values() if s['tight'])
        print(f"Tight clusters (cohesion ≥ {TIGHT_CLUSTER_THRESHOLD}): {tight_count}")
        
        # Strategy 1: Centroid-based merging
        print(f"\nStrategy 1: Centroid similarity (threshold={CENTROID_SIMILARITY_THRESHOLD})")
        centroid_pairs = find_mergeable_pairs_centroid(stats, CENTROID_SIMILARITY_THRESHOLD)
        print(f"  Found {len(centroid_pairs)} merge pairs")
        
        # Strategy 2: Tight cluster merging
        print(f"\nStrategy 2: Tight cluster merging (threshold={CENTROID_SIMILARITY_THRESHOLD - 0.05})")
        tight_pairs = find_mergeable_pairs_tight_clusters(stats)
        print(f"  Found {len(tight_pairs)} tight cluster pairs")
        
        # Combine all merge pairs
        all_pairs = centroid_pairs + tight_pairs
        
        if not all_pairs:
            print(f"\nNo merges found in pass {pass_num}. Stopping.")
            break
        
        # Sort by similarity (merge most similar first)
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nTotal merge pairs: {len(all_pairs)}")
        
        # Apply Union-Find to handle transitive merges
        merge_groups = union_find_merge(all_pairs, list(cluster_data.keys()))
        
        print(f"Merge groups: {len(merge_groups)}")
        
        if not merge_groups:
            print(f"\nNo merge groups in pass {pass_num}. Stopping.")
            break
        
        # Show examples
        print(f"\nExample merge groups:")
        for i, (root, group) in enumerate(list(merge_groups.items())[:3]):
            sizes = [stats[cid]['size'] for cid in group]
            cohesions = [stats[cid]['cohesion'] for cid in group]
            print(f"  Group {i+1}: {len(group)} clusters, sizes={sizes}, cohesions={[f'{c:.3f}' for c in cohesions]}")
        
        # Apply merges
        merged, reassigned = apply_merges(session, merge_groups, species_name, cluster_data)
        
        print(f"\nPass {pass_num} results:")
        print(f"  Clusters merged: {merged}")
        print(f"  Pets reassigned: {reassigned}")
        
        total_merged += merged
        total_reassigned += reassigned
    
    # Final statistics
    cluster_data = get_cluster_data(session, species_name)
    if cluster_data:
        sizes = [len(members) for members in cluster_data.values()]
        print(f"\n{'='*70}")
        print(f"FINAL STATISTICS FOR {species_name.upper()}")
        print(f"{'='*70}")
        print(f"Total clusters: {len(cluster_data)}")
        print(f"Largest cluster: {max(sizes)}")
        print(f"Average cluster size: {np.mean(sizes):.1f}")
        print(f"Median cluster size: {np.median(sizes):.1f}")
        print(f"\nTotal clusters merged: {total_merged}")
        print(f"Total pets reassigned: {total_reassigned}")


def main():
    print("="*70)
    print("AGGRESSIVE MULTI-PASS CLUSTER MERGING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Centroid similarity threshold: {CENTROID_SIMILARITY_THRESHOLD}")
    print(f"  Tight cluster threshold: {TIGHT_CLUSTER_THRESHOLD}")
    print(f"  Max merged cluster size: {MAX_MERGED_CLUSTER_SIZE}")
    print(f"  Max passes: {MAX_PASSES}")
    
    session = Session()
    
    try:
        # Process dogs
        multi_pass_merge(session, 'dog', max_passes=MAX_PASSES)
        
        # Process cats
        multi_pass_merge(session, 'cat', max_passes=MAX_PASSES)
        
        print("\n" + "="*70)
        print("AGGRESSIVE MERGING COMPLETE!")
        print("="*70)
        print("\n✅ Maximum consolidation achieved!")
        print("✅ Refresh the web interface to see the results")
        
    finally:
        session.close()


if __name__ == '__main__':
    main()
