"""
Cluster Pets Using Hierarchical Agglomerative Clustering
Uses a fixed number of clusters instead of DBSCAN's density-based approach
"""

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, PetCluster
from config import Config
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
# Aim for ~50-100 dog clusters to get individual dog separation
NUM_DOG_CLUSTERS = 80
NUM_CAT_CLUSTERS = 50
LINKAGE = 'ward'  # minimizes variance within clusters

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


def cluster_species(pet_ids, embeddings, n_clusters, cluster_id_offset=0):
    """
    Cluster pets using Agglomerative Clustering
    
    Args:
        pet_ids: List of pet IDs
        embeddings: numpy array of embeddings (N x D)
        n_clusters: Number of clusters to create
        cluster_id_offset: Offset for cluster IDs (for cats to avoid conflicts)
    
    Returns:
        dict mapping pet_id to cluster_id
    """
    print(f"Found {len(pet_ids)} pet embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Creating {n_clusters} clusters using Agglomerative Clustering (linkage={LINKAGE})...")
    
    # Run Agglomerative Clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=LINKAGE
    )
    labels = clustering.fit_predict(embeddings)
    
    # Create mapping
    pet_to_cluster = {}
    for pet_id, cluster_label in zip(pet_ids, labels):
        pet_to_cluster[pet_id] = int(cluster_label) + cluster_id_offset
    
    # Print statistics
    print(f"\nClustering Results:")
    cluster_counts = Counter(labels)
    print(f"  Total clusters: {len(cluster_counts)}")
    print(f"  Clustered pets: {len(pet_ids)}")
    
    print(f"\nCluster size distribution:")
    sizes = list(cluster_counts.values())
    print(f"  Largest cluster: {max(sizes)} pets")
    print(f"  Smallest cluster: {min(sizes)} pets")
    print(f"  Average cluster size: {np.mean(sizes):.1f}")
    print(f"  Median cluster size: {np.median(sizes):.1f}")
    
    # Show top 10 largest clusters
    print(f"\nTop 10 largest clusters:")
    for cluster_id, count in cluster_counts.most_common(10):
        print(f"  Cluster {cluster_id}: {count} pets")
    
    return pet_to_cluster


def visualize_clusters(embeddings, labels, species, cluster_id_offset=0):
    """Create a 2D visualization of clusters using t-SNE"""
    from sklearn.manifold import TSNE
    
    print(f"\nCreating visualization for {species}...")
    
    # Use t-SNE for 2D projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Adjust labels for offset
    adjusted_labels = labels - cluster_id_offset
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=adjusted_labels,
        cmap='tab20',
        alpha=0.6,
        s=20
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'{species.capitalize()} Clusters (Agglomerative)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
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
    print("Pet Clustering with Agglomerative Clustering")
    print("=" * 60)
    
    session = Session()
    
    try:
        # Clear existing clusters
        session.query(PetCluster).delete()
        session.query(PetEmbedding).update({PetEmbedding.cluster_id: None})
        session.commit()
        
        # Cluster dogs
        print("\n" + "=" * 60)
        print("Clustering dogs with agglomerative clustering...")
        print("=" * 60)
        
        dog_ids, dog_embeddings = get_embeddings_by_species(session, 'dog')
        if len(dog_ids) > 0:
            # Adjust number of clusters if we have fewer dogs than requested
            n_dog_clusters = min(NUM_DOG_CLUSTERS, len(dog_ids))
            dog_to_cluster = cluster_species(dog_ids, dog_embeddings, n_dog_clusters, cluster_id_offset=0)
            save_clusters_to_db(session, dog_to_cluster)
            
            # Get labels for visualization
            dog_labels = np.array([dog_to_cluster[pid] for pid in dog_ids])
            visualize_clusters(dog_embeddings, dog_labels, 'dog', cluster_id_offset=0)
        
        # Cluster cats with offset to avoid ID conflicts
        print("\n" + "=" * 60)
        print("Clustering cats with agglomerative clustering...")
        print("=" * 60)
        
        cat_ids, cat_embeddings = get_embeddings_by_species(session, 'cat')
        if len(cat_ids) > 0:
            # Adjust number of clusters if we have fewer cats than requested
            n_cat_clusters = min(NUM_CAT_CLUSTERS, len(cat_ids))
            cat_to_cluster = cluster_species(cat_ids, cat_embeddings, n_cat_clusters, cluster_id_offset=10000)
            save_clusters_to_db(session, cat_to_cluster)
            
            # Get labels for visualization
            cat_labels = np.array([cat_to_cluster[pid] for pid in cat_ids])
            visualize_clusters(cat_embeddings, cat_labels, 'cat', cluster_id_offset=10000)
        
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
        print(f"\nYou can now view the clusters in the web interface!")
        print(f"Max should now have his own separate cluster.")
        
    finally:
        session.close()


if __name__ == '__main__':
    main()
