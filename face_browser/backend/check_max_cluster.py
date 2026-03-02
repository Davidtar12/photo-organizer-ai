"""
Check which cluster Max images are in
"""

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, MediaFile
from config import Config
from collections import Counter
import os

# Database setup
DATABASE_URL = f'sqlite:///{Config.DB_PATH}'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def main():
    session = Session()
    
    # Get all dog embeddings
    dogs = session.query(PetEmbedding).filter(PetEmbedding.species == 'dog').all()
    
    # Find Max images (from the training folder)
    max_folder = r'C:\Users\USERNAME\Downloads\Max'
    max_clusters = []
    other_clusters = []
    
    for dog in dogs:
        # Get the media path
        media = session.query(MediaFile).get(dog.media_id)
        if media:
            media_path = media.path
            # Check if this is from Max folder
            if max_folder.lower() in media_path.lower():
                if dog.cluster_id is not None:
                    max_clusters.append(dog.cluster_id)
            else:
                if dog.cluster_id is not None:
                    other_clusters.append(dog.cluster_id)
    
    print(f"Max images found: {len(max_clusters)}")
    print(f"Other dog images: {len(other_clusters)}")
    
    if max_clusters:
        print(f"\nMax cluster distribution:")
        max_counter = Counter(max_clusters)
        for cluster_id, count in max_counter.most_common(10):
            print(f"  Cluster {cluster_id}: {count} Max images")
        
        # Check if Max is mostly in one cluster
        most_common_cluster, max_count = max_counter.most_common(1)[0]
        print(f"\nMax concentration: {max_count}/{len(max_clusters)} ({100*max_count/len(max_clusters):.1f}%) in cluster {most_common_cluster}")
        
        # Check how many other dogs are in Max's main cluster
        other_in_max_cluster = other_clusters.count(most_common_cluster)
        print(f"Other dogs in Max's cluster {most_common_cluster}: {other_in_max_cluster}")
    
    session.close()

if __name__ == '__main__':
    main()
