"""
Merge Similar Pet Clusters
Merges clusters that likely represent the same dog in different poses/lighting
Uses centroid-based similarity and statistical analysis
"""

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, PetCluster
from config import Config
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm

# Configuration
CENTROID_SIMILARITY_THRESHOLD = 0.85  # High similarity = likely same dog
MIN_CLUSTER_SIZE_TO_MERGE = 1  # Merge even singleton clusters
MAX_MERGED_CLUSTER_SIZE = 50  # Don't merge into giant clusters

# Database setup
DATABASE_URL = f'sqlite:///{Config.DB_PATH}'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


def get_cluster_centroids(session, species_name):
    """
    Calculate centroid for each cluster
    Returns dict of {cluster_id: centroid_embedding}
    """
    print(f"\nCalculating centroids for {species_name} clusters...")
    
    # Get all pets of this species with cluster assignments
    pets = session.query(PetEmbedding).filter(
        PetEmbedding.species == species_name,
        PetEmbedding.cluster_id.isnot(None)
    ).all()
    
    # Group by cluster
    cluster_embeddings = defaultdict(list)
    for pet in pets:
        embedding = np.frombuffer(pet.embedding, dtype=np.float32)
        cluster_embeddings[pet.cluster_id].append(embedding)
    
    # Calculate centroids
    centroids = {}
    cluster_sizes = {}
    for cluster_id, embeddings in cluster_embeddings.items():
        embeddings_array = np.array(embeddings)
        centroid = embeddings_array.mean(axis=0)
        # L2 normalize the centroid
        centroid = centroid / np.linalg.norm(centroid)
        centroids[cluster_id] = centroid
        cluster_sizes[cluster_id] = len(embeddings)
    
    print(f"  Found {len(centroids)} clusters")
    print(f"  Cluster size range: {min(cluster_sizes.values())} - {max(cluster_sizes.values())}")
    
    return centroids, cluster_sizes


def find_mergeable_clusters(centroids, cluster_sizes, similarity_threshold):
    """
    Find pairs of clusters that should be merged based on centroid similarity
    Uses Union-Find to handle transitive merging (A→B, B→C means A→B→C)
    """
    print(f"\nFinding mergeable clusters (threshold={similarity_threshold})...")
    
    cluster_ids = list(centroids.keys())
    n_clusters = len(cluster_ids)
    
    # Build similarity matrix
    print("  Computing pairwise similarities...")
    centroid_matrix = np.array([centroids[cid] for cid in cluster_ids])
    similarity_matrix = cosine_similarity(centroid_matrix)
    
    # Union-Find data structure for grouping
    parent = {cid: cid for cid in cluster_ids}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Find similar pairs
    merge_count = 0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if similarity_matrix[i, j] >= similarity_threshold:
                cid1, cid2 = cluster_ids[i], cluster_ids[j]
                
                # Check if merging would create too large a cluster
                size1 = cluster_sizes[cid1]
                size2 = cluster_sizes[cid2]
                
                if size1 + size2 <= MAX_MERGED_CLUSTER_SIZE:
                    union(cid1, cid2)
                    merge_count += 1
    
    # Group clusters by parent
    merge_groups = defaultdict(list)
    for cid in cluster_ids:
        root = find(cid)
        merge_groups[root].append(cid)
    
    # Filter to only groups that have merges
    merge_groups = {k: v for k, v in merge_groups.items() if len(v) > 1}
    
    print(f"  Found {len(merge_groups)} merge groups")
    print(f"  Total clusters to be merged: {sum(len(v) for v in merge_groups.values())}")
    
    # Show some examples
    if merge_groups:
        print(f"\n  Example merge groups:")
        for i, (root, group) in enumerate(list(merge_groups.items())[:5]):
            sizes = [cluster_sizes[cid] for cid in group]
            total_size = sum(sizes)
            print(f"    Group {i+1}: {len(group)} clusters ({sizes}) → {total_size} total pets")
    
    return merge_groups


def apply_cluster_merges(session, merge_groups, species_name):
    """
    Apply the cluster merges to the database
    Reassign all pets in merged clusters to the root cluster
    """
    print(f"\nApplying merges for {species_name}...")
    
    # Track statistics
    clusters_merged = 0
    pets_reassigned = 0
    
    for root_cluster, cluster_group in tqdm(merge_groups.items(), desc="Merging clusters"):
        # Get all pets in the clusters to be merged (excluding root)
        clusters_to_merge = [c for c in cluster_group if c != root_cluster]
        
        for cluster_id in clusters_to_merge:
            # Reassign all pets from this cluster to root
            pets = session.query(PetEmbedding).filter(
                PetEmbedding.species == species_name,
                PetEmbedding.cluster_id == cluster_id
            ).all()
            
            for pet in pets:
                pet.cluster_id = root_cluster
                pets_reassigned += 1
            
            # Delete the now-empty cluster
            cluster = session.query(PetCluster).get(cluster_id)
            if cluster:
                session.delete(cluster)
            
            clusters_merged += 1
    
    session.commit()
    
    print(f"  Merged {clusters_merged} clusters")
    print(f"  Reassigned {pets_reassigned} pets")
    
    return clusters_merged, pets_reassigned


def get_final_statistics(session, species_name):
    """Get final cluster statistics after merging"""
    pets = session.query(PetEmbedding).filter(
        PetEmbedding.species == species_name,
        PetEmbedding.cluster_id.isnot(None)
    ).all()
    
    cluster_sizes = defaultdict(int)
    for pet in pets:
        cluster_sizes[pet.cluster_id] += 1
    
    if cluster_sizes:
        sizes = list(cluster_sizes.values())
        return {
            'n_clusters': len(cluster_sizes),
            'n_pets': len(pets),
            'largest': max(sizes),
            'smallest': min(sizes),
            'average': np.mean(sizes),
            'median': np.median(sizes)
        }
    return None


def main():
    print("=" * 70)
    print("Pet Cluster Merging - Combine Same Dog in Different Poses")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Centroid similarity threshold: {CENTROID_SIMILARITY_THRESHOLD}")
    print(f"  Max merged cluster size: {MAX_MERGED_CLUSTER_SIZE}")
    
    session = Session()
    
    try:
        # Process dogs
        print("\n" + "=" * 70)
        print("PROCESSING DOGS")
        print("=" * 70)
        
        # Get initial stats
        initial_stats = get_final_statistics(session, 'dog')
        if initial_stats:
            print(f"\nInitial statistics:")
            print(f"  Clusters: {initial_stats['n_clusters']}")
            print(f"  Dogs: {initial_stats['n_pets']}")
            print(f"  Largest cluster: {initial_stats['largest']}")
            print(f"  Average cluster size: {initial_stats['average']:.1f}")
        
        # Calculate centroids
        dog_centroids, dog_sizes = get_cluster_centroids(session, 'dog')
        
        # Find mergeable clusters
        dog_merges = find_mergeable_clusters(
            dog_centroids, 
            dog_sizes, 
            CENTROID_SIMILARITY_THRESHOLD
        )
        
        # Apply merges
        if dog_merges:
            apply_cluster_merges(session, dog_merges, 'dog')
            
            # Get final stats
            final_stats = get_final_statistics(session, 'dog')
            print(f"\nFinal statistics:")
            print(f"  Clusters: {final_stats['n_clusters']} (reduced by {initial_stats['n_clusters'] - final_stats['n_clusters']})")
            print(f"  Dogs: {final_stats['n_pets']}")
            print(f"  Largest cluster: {final_stats['largest']}")
            print(f"  Average cluster size: {final_stats['average']:.1f}")
        else:
            print("\n  No clusters to merge!")
        
        # Process cats
        print("\n" + "=" * 70)
        print("PROCESSING CATS")
        print("=" * 70)
        
        initial_stats = get_final_statistics(session, 'cat')
        if initial_stats:
            print(f"\nInitial statistics:")
            print(f"  Clusters: {initial_stats['n_clusters']}")
            print(f"  Cats: {initial_stats['n_pets']}")
            print(f"  Largest cluster: {initial_stats['largest']}")
            print(f"  Average cluster size: {initial_stats['average']:.1f}")
        
        cat_centroids, cat_sizes = get_cluster_centroids(session, 'cat')
        cat_merges = find_mergeable_clusters(
            cat_centroids, 
            cat_sizes, 
            CENTROID_SIMILARITY_THRESHOLD
        )
        
        if cat_merges:
            apply_cluster_merges(session, cat_merges, 'cat')
            
            final_stats = get_final_statistics(session, 'cat')
            print(f"\nFinal statistics:")
            print(f"  Clusters: {final_stats['n_clusters']} (reduced by {initial_stats['n_clusters'] - final_stats['n_clusters']})")
            print(f"  Cats: {final_stats['n_pets']}")
            print(f"  Largest cluster: {final_stats['largest']}")
            print(f"  Average cluster size: {final_stats['average']:.1f}")
        else:
            print("\n  No clusters to merge!")
        
        print("\n" + "=" * 70)
        print("Cluster Merging Complete!")
        print("=" * 70)
        print("\n✅ Same dogs in different poses should now be in single clusters")
        print("✅ Refresh the web interface to see the updated clusters")
        
    finally:
        session.close()


if __name__ == '__main__':
    main()
