"""
Strict Agglomerative Clustering to avoid mixing breeds
Uses very conservative distance thresholds
"""

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, PetCluster
from config import Config
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# VERY strict threshold - only cluster if almost identical
# With ViT embeddings, we can be even stricter since they separate better
DISTANCE_THRESHOLD = 0.20  # cosine distance - ultra conservative for ViT
LINKAGE = 'average'  # average linkage - conservative

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
    Cluster pets using Agglomerative Clustering with strict distance threshold
    """
    print(f"Found {len(pet_ids)} {species} embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Running Agglomerative Clustering (distance_threshold={DISTANCE_THRESHOLD}, linkage={LINKAGE})...")
    
    # Compute cosine distance matrix
    print("  Computing cosine distance matrix...")
    distance_matrix = cosine_distances(embeddings)
    
    # Run Agglomerative Clustering
    # n_clusters=None means use distance_threshold to determine number of clusters
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        linkage=LINKAGE,
        metric='precomputed'
    )
    
    labels = clusterer.fit_predict(distance_matrix)
    
    # Create mapping
    pet_to_cluster = {}
    for pet_id, cluster_label in zip(pet_ids, labels):
        pet_to_cluster[pet_id] = int(cluster_label) + cluster_id_offset
    
    # Print statistics
    print(f"\nClustering Results:")
    cluster_counts = Counter(labels)
    
    print(f"  Total clusters: {len(cluster_counts)}")
    print(f"  All {species}s clustered: {len(pet_ids)}")
    
    if cluster_counts:
        print(f"\nCluster size distribution:")
        sizes = list(cluster_counts.values())
        print(f"  Largest cluster: {max(sizes)} {species}s")
        print(f"  Smallest cluster: {min(sizes)} {species}s")
        print(f"  Average cluster size: {np.mean(sizes):.1f}")
        print(f"  Median cluster size: {np.median(sizes):.1f}")
        
        # Show distribution
        size_dist = Counter(sizes)
        print(f"\n  Cluster size frequency:")
        for size in sorted(size_dist.keys()):
            count = size_dist[size]
            print(f"    {count} clusters with {size} {species}(s)")
        
        # Show top 10 largest clusters
        print(f"\nTop 10 largest clusters:")
        for cluster_id, count in cluster_counts.most_common(10):
            print(f"  Cluster {cluster_id}: {count} {species}s")
    
    return pet_to_cluster, labels


def visualize_clusters(embeddings, labels, species):
    """Create a 2D visualization of clusters using t-SNE"""
    from sklearn.manifold import TSNE
    
    print(f"\nCreating visualization for {species}...")
    
    # Use t-SNE for 2D projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(16, 12))
    
    # Get unique labels
    unique_labels = sorted(set(labels))
    
    # Use a colormap
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cluster
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            alpha=0.6,
            s=30,
            label=f'Cluster {label} ({mask.sum()})'
        )
    
    plt.title(f'{species.capitalize()} Clusters (Agglomerative, threshold={DISTANCE_THRESHOLD})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Only show legend if not too many clusters
    if len(unique_labels) <= 50:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    output_path = f'{species}_clusters_agglomerative.png'
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
    
    # Create PetCluster entries
    cluster_ids = set(pet_to_cluster.values())
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
    print("Strict Agglomerative Clustering")
    print(f"Distance Threshold: {DISTANCE_THRESHOLD} (cosine)")
    print("=" * 60)
    
    session = Session()
    
    try:
        # Clear existing clusters
        session.query(PetCluster).delete()
        session.query(PetEmbedding).update({PetEmbedding.cluster_id: None})
        session.commit()
        
        # Cluster dogs
        print("\n" + "=" * 60)
        print("Clustering dogs...")
        print("=" * 60)
        
        dog_ids, dog_embeddings = get_embeddings_by_species(session, 'dog')
        if len(dog_ids) > 0:
            dog_to_cluster, dog_labels = cluster_species(dog_ids, dog_embeddings, 'dog', cluster_id_offset=0)
            save_clusters_to_db(session, dog_to_cluster)
            visualize_clusters(dog_embeddings, dog_labels, 'dog')
        
        # Cluster cats with offset to avoid ID conflicts
        print("\n" + "=" * 60)
        print("Clustering cats...")
        print("=" * 60)
        
        cat_ids, cat_embeddings = get_embeddings_by_species(session, 'cat')
        if len(cat_ids) > 0:
            cat_to_cluster, cat_labels = cluster_species(cat_ids, cat_embeddings, 'cat', cluster_id_offset=10000)
            save_clusters_to_db(session, cat_to_cluster)
            visualize_clusters(cat_embeddings, cat_labels, 'cat')
        
        print("\n" + "=" * 60)
        print("Clustering Complete!")
        print("=" * 60)
        
        # Print final statistics
        n_dog_clusters = len(set(dog_to_cluster.values())) if dog_ids else 0
        n_cat_clusters = len(set(cat_to_cluster.values())) if cat_ids else 0
        
        print(f"\nFinal Statistics:")
        print(f"  Dog clusters: {n_dog_clusters}")
        print(f"  Cat clusters: {n_cat_clusters}")
        print(f"  Total clusters: {n_dog_clusters + n_cat_clusters}")
        print(f"\nNOTE: With distance_threshold={DISTANCE_THRESHOLD}, we're being very")
        print(f"      conservative. Only very similar images are grouped together.")
        print(f"      This should minimize mixing different breeds, but may create")
        print(f"      many small clusters for the same dog in different poses.")
        
    finally:
        session.close()


if __name__ == '__main__':
    main()
