"""Quick script to cluster cats."""
from cluster_dogs import DogClusterer
from database import session_scope
from models import PetEmbedding, PetCluster
from sqlalchemy import select, delete
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

clusterer = DogClusterer()

with session_scope() as session:
    cats = session.execute(select(PetEmbedding).where(PetEmbedding.species == 'cat').order_by(PetEmbedding.id)).scalars().all()
    logger.info(f'Found {len(cats)} cats')
    
    embeddings = []
    valid_cats = []
    
    for i, cat in enumerate(cats):
        if i % 50 == 0:
            logger.info(f'Extracting: {i}/{len(cats)}')
        if cat.thumbnail_path and Path(cat.thumbnail_path).exists():
            emb = clusterer.extract_embedding(cat.thumbnail_path)
            if emb is not None and np.any(emb != 0):
                embeddings.append(emb)
                valid_cats.append(cat)
    
    X = np.array(embeddings, dtype=np.float32)
    logger.info(f"Running DBSCAN on {len(X)} cats...")
    clustering = DBSCAN(eps=0.08, min_samples=2, metric='cosine', n_jobs=-1)
    labels = clustering.fit_predict(X)
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    logger.info(f'✓ Found {n_clusters} cat clusters')
    
    session.execute(delete(PetCluster).where(PetCluster.species == 'cat'))
    session.flush()
    
    cluster_map = {}
    for label in unique_labels:
        if label != -1:
            c = PetCluster(species='cat', display_name=f'Cat {label + 1}')
            session.add(c)
            session.flush()
            cluster_map[label] = c.id
    
    for cat, label in zip(valid_cats, labels):
        cat.cluster_id = cluster_map.get(label)
    
    session.commit()
    logger.info(f'✓ Cat clustering complete: {n_clusters} clusters created')
