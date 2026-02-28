"""
Dog clustering using ResNet embeddings + Faiss + DBSCAN.
Similar to face clustering but for dog detections.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import DBSCAN
from sqlalchemy import delete, select

from config import Config
from database import session_scope
from models import PetCluster, PetEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DogClusterer:
    """Cluster dog detections using ResNet embeddings."""
    
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        
        # Load pretrained ResNet for feature extraction
        logger.info("Loading ResNet50 for dog embeddings...")
        self.model = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"✓ ResNet50 loaded on {self.device}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_embedding(self, img_path: str) -> np.ndarray:
        """Extract 2048-dim ResNet embedding from an image."""
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(img_tensor)
            
            # Flatten and normalize
            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
            return embedding
        except Exception as e:
            logger.error(f"Failed to extract embedding from {img_path}: {e}")
            return np.zeros(2048, dtype=np.float32)
    
    def cluster_dogs(self, eps: float = 0.35, min_samples: int = 2):
        """Cluster all dog detections using DBSCAN.
        
        Args:
            eps: Maximum distance between samples to be in same cluster (0.3-0.5 typical)
            min_samples: Minimum number of samples to form a cluster
        """
        logger.info("🐕 Starting dog clustering...")
        
        with session_scope() as session:
            # Get all dog detections (species='dog')
            stmt = select(PetEmbedding).where(PetEmbedding.species == 'dog').order_by(PetEmbedding.id)
            dogs = session.execute(stmt).scalars().all()
            
            if len(dogs) == 0:
                logger.warning("No dog detections found. Run pet detection first.")
                return
            
            logger.info(f"Found {len(dogs)} dog detections")
            
            # Extract ResNet embeddings for all dogs
            embeddings = []
            valid_dogs = []
            
            for i, dog in enumerate(dogs):
                if i % 50 == 0:
                    logger.info(f"Extracting embeddings: {i}/{len(dogs)}")
                
                # Use thumbnail if available, otherwise skip
                if not dog.thumbnail_path or not Path(dog.thumbnail_path).exists():
                    logger.warning(f"Skipping dog {dog.id}: no thumbnail")
                    continue
                
                emb = self.extract_embedding(dog.thumbnail_path)
                if emb is not None and np.any(emb != 0):
                    embeddings.append(emb)
                    valid_dogs.append(dog)
            
            if len(embeddings) == 0:
                logger.error("No valid embeddings extracted. Aborting.")
                return
            
            logger.info(f"Extracted {len(embeddings)} valid embeddings")
            
            # Convert to numpy array
            X = np.array(embeddings, dtype=np.float32)
            
            # Run DBSCAN clustering
            logger.info(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
            labels = clustering.fit_predict(X)
            
            # Count clusters
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(labels).count(-1)
            
            logger.info(f"✓ Found {n_clusters} dog clusters ({n_noise} noise points)")
            
            # Clear existing dog clusters
            session.execute(delete(PetCluster).where(PetCluster.species == 'dog'))
            session.flush()
            
            # Create PetCluster entries and assign dogs
            cluster_map = {}
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise points
                
                cluster = PetCluster(
                    species='dog',
                    display_name=f"Dog {label + 1}"
                )
                session.add(cluster)
                session.flush()
                cluster_map[label] = cluster.id
            
            # Assign each dog to its cluster
            for dog, label in zip(valid_dogs, labels):
                if label == -1:
                    dog.cluster_id = None  # Unclustered
                else:
                    dog.cluster_id = cluster_map[label]
            
            session.commit()
            logger.info(f"✓ Dog clustering complete: {n_clusters} clusters created")


if __name__ == "__main__":
    clusterer = DogClusterer()
    # Use very tight clustering for individual dogs: eps=0.08-0.12 for maximum separation
    clusterer.cluster_dogs(eps=0.08, min_samples=2)
