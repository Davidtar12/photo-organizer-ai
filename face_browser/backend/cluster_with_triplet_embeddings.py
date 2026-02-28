"""
Cluster Pets Using Triplet Loss Embeddings
Re-cluster all pets using the learned embeddings from triplet loss training
"""

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, PetCluster
from config import Config
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
# Lower eps for finer granularity (separate individual dogs)
DBSCAN_EPS_DOGS = 0.015  # Even stricter for individual dog separation
DBSCAN_EPS_CATS = 0.015  # Even stricter for individual cat separation
DBSCAN_MIN_SAMPLES = 2

# Database setup - use the same database as the rest of the app
DATABASE_URL = f'sqlite:///{Config.DB_PATH}'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


def get_embeddings_by_species(session, species_name):
    """Get all embeddings for a specific species"""
    pets = session.query(PetEmbedding).filter(
        PetEmbedding.species == species_name
    ).all()
    
    embeddings = []
    pet_ids = []
    
    for pet in pets:
        if pet.embedding:
            # Convert bytes back to numpy array
            emb = np.frombuffer(pet.embedding, dtype=np.float32)
            embeddings.append(emb)
            pet_ids.append(pet.id)
    
    return np.array(embeddings), pet_ids


def cluster_species(session, species_name, eps, cluster_id_offset=0):
    """Cluster pets of a specific species"""
    print(f"\n{'='*60}")
    print(f"Clustering {species_name}s with triplet loss embeddings...")
    print(f"{'='*60}")
    
    # Get embeddings
    embeddings, pet_ids = get_embeddings_by_species(session, species_name)
    
    if len(embeddings) == 0:
        print(f"No {species_name} embeddings found!")
        return
    
    print(f"Found {len(embeddings)} {species_name} embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Cluster with DBSCAN
    print(f"Running DBSCAN (eps={eps}, min_samples={DBSCAN_MIN_SAMPLES})...")
    clusterer = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    
    # Analyze results
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\nClustering Results:")
    print(f"  Total clusters: {n_clusters}")
    print(f"  Noise points (unclustered): {n_noise}")
    print(f"  Clustered {species_name}s: {len(embeddings) - n_noise}")
    
    # Show cluster size distribution
    counter = Counter(labels)
    if -1 in counter:
        del counter[-1]  # Remove noise
    
    if counter:
        print(f"\nCluster size distribution:")
        sizes = sorted(counter.values(), reverse=True)
        print(f"  Largest cluster: {sizes[0]} {species_name}s")
        print(f"  Smallest cluster: {sizes[-1]} {species_name}s")
        print(f"  Average cluster size: {np.mean(sizes):.1f}")
        print(f"  Median cluster size: {np.median(sizes):.1f}")
        
        # Show top 10 clusters
        print(f"\nTop 10 largest clusters:")
        for i, (label, count) in enumerate(counter.most_common(10)):
            print(f"  Cluster {label}: {count} {species_name}s")
    
    # Update database
    print(f"\nUpdating database...")
    
    # Clear existing clusters for this species
    session.query(PetCluster).filter(PetCluster.species == species_name).delete()
    session.commit()
    
    # Create new clusters
    for pet_id, label in tqdm(zip(pet_ids, labels), total=len(pet_ids), desc="Saving clusters"):
        if label != -1:  # Skip noise
            # Update pet embedding cluster_id
            pet = session.query(PetEmbedding).filter(PetEmbedding.id == pet_id).first()
            if pet:
                # Offset cluster ID to avoid conflicts between species
                actual_cluster_id = int(label) + cluster_id_offset
                
                # Find or create cluster
                cluster = session.query(PetCluster).filter(
                    PetCluster.id == actual_cluster_id,
                    PetCluster.species == species_name
                ).first()
                
                if not cluster:
                    cluster = PetCluster(
                        id=actual_cluster_id,
                        species=species_name
                    )
                    session.add(cluster)
                
                pet.cluster_id = actual_cluster_id
    
    session.commit()
    print(f"Database updated with {n_clusters} {species_name} clusters")
    
    return n_clusters, n_noise, labels, embeddings


def visualize_clusters(labels, embeddings, species_name):
    """Create visualization of cluster distribution"""
    from sklearn.decomposition import PCA
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c='gray', marker='x', alpha=0.3, label='Noise')
        else:
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[color], alpha=0.6, edgecolors='k', linewidth=0.5)
    
    plt.title(f'{species_name.capitalize()} Clusters (Triplet Loss Embeddings)\nPCA Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend(['Noise'], loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f'{species_name}_clusters_triplet.png'
    plt.savefig(filename, dpi=150)
    print(f"Visualization saved: {filename}")
    plt.close()


def main():
    """Main clustering function"""
    session = Session()
    
    try:
        print("="*60)
        print("Pet Clustering with Triplet Loss Embeddings")
        print("="*60)
        
        # Cluster dogs
        dog_results = cluster_species(session, species_name='dog', eps=DBSCAN_EPS_DOGS, cluster_id_offset=0)
        if dog_results:
            n_clusters, n_noise, labels, embeddings = dog_results
            visualize_clusters(labels, embeddings, species_name='dog')
        
        # Cluster cats (offset by 10000 to avoid ID conflicts)
        cat_results = cluster_species(session, species_name='cat', eps=DBSCAN_EPS_CATS, cluster_id_offset=10000)
        if cat_results:
            n_clusters, n_noise, labels, embeddings = cat_results
            visualize_clusters(labels, embeddings, species_name='cat')
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"Clustering Complete!")
        print(f"{'='*60}")
        
        total_dog_clusters = session.query(PetCluster).filter(PetCluster.species == 'dog').count()
        total_cat_clusters = session.query(PetCluster).filter(PetCluster.species == 'cat').count()
        
        print(f"\nFinal Statistics:")
        print(f"  Dog clusters: {total_dog_clusters}")
        print(f"  Cat clusters: {total_cat_clusters}")
        print(f"  Total clusters: {total_dog_clusters + total_cat_clusters}")
        
        print("\nYou can now view the clusters in the web interface!")
        print("The clustering should now separate individual dogs like Max.")
        
    finally:
        session.close()


if __name__ == '__main__':
    main()
