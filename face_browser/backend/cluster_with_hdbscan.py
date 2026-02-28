"""
Cluster Dogs Using HDBSCAN with ResNet50 Embeddings
HDBSCAN is better than DBSCAN for varying cluster densities
"""

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, PetCluster
from config import Config
import hdbscan
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration - MUCH stricter to avoid mixing breeds
HDBSCAN_MIN_CLUSTER_SIZE = 2  # Allow very small clusters (individual dogs)
HDBSCAN_MIN_SAMPLES = 1  # Very aggressive, create more clusters
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.15  # Stricter distance threshold
SOFT_CLUSTERING_THRESHOLD = 0.15  # Only merge noise if VERY similar

# Database setup
DATABASE_URL = f'sqlite:///{Config.DB_PATH}'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


def get_embeddings_by_species(session, species_name):
    """Get all embeddings for a specific species"""
    pets = session.query(PetEmbedding).filter(
        PetEmbedding.species == species_name
    ).all()
    
    if not pets:
        return [], []
    
    # Extract IDs and embeddings
    pet_ids = [pet.id for pet in pets]
    embeddings = np.array([np.frombuffer(pet.embedding, dtype=np.float32) for pet in pets])
    
    return pet_ids, embeddings


def cluster_species(pet_ids, embeddings, species, cluster_id_offset=0):
    """
    Cluster pets using HDBSCAN
    
    Args:
        pet_ids: List of pet IDs
        embeddings: numpy array of embeddings (N x D)
        species: 'dog' or 'cat'
        cluster_id_offset: Offset for cluster IDs (for cats to avoid conflicts)
    
    Returns:
        dict mapping pet_id to cluster_id
    """
    print(f"Found {len(pet_ids)} {species} embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Running HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES})...")
    
    # For cosine distance, we need to precompute distance matrix
    # Or use sklearn's implementation that supports cosine via pairwise_distances
    from sklearn.metrics.pairwise import pairwise_distances
    
    # Compute cosine distance matrix (1 - cosine_similarity)
    print("  Computing cosine distance matrix...")
    distance_matrix = pairwise_distances(embeddings, metric='cosine').astype(np.float64)
    
    # Run HDBSCAN with stricter parameters to avoid mixing breeds
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='precomputed',  # Use precomputed distance matrix
        cluster_selection_method='eom',  # Excess of Mass - better for separating similar items
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,  # Stricter merging threshold
        prediction_data=False  # Can't use with precomputed
    )
    
    labels = clusterer.fit_predict(distance_matrix)
    
    # Assign noise points to nearest cluster (soft clustering)
    noise_mask = labels == -1
    if noise_mask.any() and len(set(labels)) > 1:
        print(f"  Assigning {noise_mask.sum()} noise points to nearest clusters...")
        # For each noise point, find nearest cluster
        from sklearn.metrics.pairwise import cosine_distances
        
        # Get cluster centroids
        unique_labels = sorted(set(labels) - {-1})
        centroids = []
        for label in unique_labels:
            cluster_mask = labels == label
            centroid = embeddings[cluster_mask].mean(axis=0)
            centroids.append(centroid)
        
        if centroids:
            centroids = np.array(centroids)
            noise_embeddings = embeddings[noise_mask]
            
            # Find nearest cluster for each noise point
            distances = cosine_distances(noise_embeddings, centroids)
            nearest_clusters = distances.argmin(axis=1)
            
            # Assign noise points to nearest cluster (only if distance is reasonable)
            distance_threshold = SOFT_CLUSTERING_THRESHOLD  # Much stricter - only assign if VERY similar
            assigned_count = 0
            for idx, (cluster_idx, min_dist) in enumerate(zip(nearest_clusters, distances.min(axis=1))):
                if min_dist < distance_threshold:
                    noise_indices = np.where(noise_mask)[0]
                    labels[noise_indices[idx]] = unique_labels[cluster_idx]
                    assigned_count += 1
            print(f"  Assigned {assigned_count}/{noise_mask.sum()} noise points (threshold={distance_threshold})")
    
    # Create mapping (HDBSCAN uses -1 for noise)
    pet_to_cluster = {}
    for pet_id, cluster_label in zip(pet_ids, labels):
        if cluster_label == -1:
            # Noise point - leave as None
            pet_to_cluster[pet_id] = None
        else:
            pet_to_cluster[pet_id] = int(cluster_label) + cluster_id_offset
    
    # Print statistics
    print(f"\nClustering Results:")
    cluster_counts = Counter([l for l in labels if l != -1])
    noise_count = sum(1 for l in labels if l == -1)
    
    print(f"  Total clusters: {len(cluster_counts)}")
    print(f"  Noise points (unclustered): {noise_count}")
    print(f"  Clustered {species}s: {len(pet_ids) - noise_count}")
    
    if cluster_counts:
        print(f"\nCluster size distribution:")
        sizes = list(cluster_counts.values())
        print(f"  Largest cluster: {max(sizes)} {species}s")
        print(f"  Smallest cluster: {min(sizes)} {species}s")
        print(f"  Average cluster size: {np.mean(sizes):.1f}")
        print(f"  Median cluster size: {np.median(sizes):.1f}")
        
        # Show top 10 largest clusters
        print(f"\nTop 10 largest clusters:")
        for cluster_id, count in cluster_counts.most_common(10):
            print(f"  Cluster {cluster_id}: {count} {species}s")
    
    return pet_to_cluster, labels


def visualize_clusters(embeddings, labels, species, cluster_id_offset=0):
    """Create a 2D visualization of clusters using t-SNE"""
    from sklearn.manifold import TSNE
    
    print(f"\nCreating visualization for {species}...")
    
    # Use t-SNE for 2D projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Separate noise points
    noise_mask = labels == -1
    cluster_mask = ~noise_mask
    
    # Plot clustered points
    if cluster_mask.any():
        scatter = plt.scatter(
            embeddings_2d[cluster_mask, 0],
            embeddings_2d[cluster_mask, 1],
            c=labels[cluster_mask],
            cmap='tab20',
            alpha=0.6,
            s=30,
            label='Clustered'
        )
        plt.colorbar(scatter, label='Cluster ID')
    
    # Plot noise points in gray
    if noise_mask.any():
        plt.scatter(
            embeddings_2d[noise_mask, 0],
            embeddings_2d[noise_mask, 1],
            c='gray',
            alpha=0.3,
            s=20,
            label='Noise'
        )
    
    plt.title(f'{species.capitalize()} Clusters (HDBSCAN with ResNet50)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    
    output_path = f'{species}_clusters_hdbscan.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {output_path}")


def save_clusters_to_db(session, pet_to_cluster):
    """Save cluster assignments to database"""
    print("\nUpdating database...")
    
    # Update each pet's cluster_id
    for pet_id, cluster_id in tqdm(pet_to_cluster.items(), desc="Saving clusters"):
        pet = session.query(PetEmbedding).get(pet_id)
        if pet:
            pet.cluster_id = cluster_id
    
    session.commit()
    
    # Create PetCluster entries (only for non-None clusters)
    cluster_ids = set(cid for cid in pet_to_cluster.values() if cid is not None)
    for cluster_id in cluster_ids:
        # Check if cluster already exists
        existing = session.query(PetCluster).get(cluster_id)
        if not existing:
            cluster = PetCluster(id=cluster_id)
            session.add(cluster)
    
    session.commit()
    print(f"Database updated with {len(cluster_ids)} clusters")


def main():
    print("=" * 60)
    print("Pet Clustering with HDBSCAN (ResNet50 Embeddings)")
    print("=" * 60)
    
    session = Session()
    
    try:
        # Clear existing clusters
        session.query(PetCluster).delete()
        session.query(PetEmbedding).update({PetEmbedding.cluster_id: None})
        session.commit()
        
        # Cluster dogs
        print("\n" + "=" * 60)
        print("Clustering dogs with HDBSCAN...")
        print("=" * 60)
        
        dog_ids, dog_embeddings = get_embeddings_by_species(session, 'dog')
        if len(dog_ids) > 0:
            dog_to_cluster, dog_labels = cluster_species(dog_ids, dog_embeddings, 'dog', cluster_id_offset=0)
            save_clusters_to_db(session, dog_to_cluster)
            visualize_clusters(dog_embeddings, dog_labels, 'dog', cluster_id_offset=0)
        
        # Cluster cats with offset to avoid ID conflicts
        print("\n" + "=" * 60)
        print("Clustering cats with HDBSCAN...")
        print("=" * 60)
        
        cat_ids, cat_embeddings = get_embeddings_by_species(session, 'cat')
        if len(cat_ids) > 0:
            cat_to_cluster, cat_labels = cluster_species(cat_ids, cat_embeddings, 'cat', cluster_id_offset=10000)
            save_clusters_to_db(session, cat_to_cluster)
            visualize_clusters(cat_embeddings, cat_labels, 'cat', cluster_id_offset=10000)
        
        print("\n" + "=" * 60)
        print("Clustering Complete!")
        print("=" * 60)
        
        # Print final statistics
        n_dog_clusters = len(set(c for c in dog_to_cluster.values() if c is not None)) if dog_ids else 0
        n_cat_clusters = len(set(c for c in cat_to_cluster.values() if c is not None)) if cat_ids else 0
        
        print(f"\nFinal Statistics:")
        print(f"  Dog clusters: {n_dog_clusters}")
        print(f"  Cat clusters: {n_cat_clusters}")
        print(f"  Total clusters: {n_dog_clusters + n_cat_clusters}")
        print(f"\nYou can now view the clusters in the web interface!")
        print(f"ResNet50 should provide better individual dog separation.")
        
    finally:
        session.close()


if __name__ == '__main__':
    main()
