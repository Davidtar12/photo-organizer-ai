"""
CLIP-based pet clustering with semantic understanding.
Superior to ResNet because it understands visual+text semantics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from sqlalchemy import delete, select

try:
    import open_clip
except ImportError:
    print("Installing open-clip-torch...")
    import subprocess
    subprocess.check_call(["pip", "install", "open-clip-torch"])
    import open_clip

from config import Config
from database import session_scope
from models import PetCluster, PetEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPPetClusterer:
    """Cluster pets using CLIP embeddings for semantic understanding."""
    
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        
        # Load CLIP model (ViT-B-32 is good balance of speed/quality)
        logger.info("Loading CLIP ViT-B-32 model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='openai'
        )
        self.model.to(self.device).eval()
        logger.info(f"✓ CLIP loaded on {self.device}")
    
    def extract_embedding(self, img_path: str) -> np.ndarray:
        """Extract CLIP image embedding (512-dim semantic vector)."""
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(img_tensor)
                # L2 normalize for cosine similarity
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to extract CLIP embedding from {img_path}: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def cluster_species(self, species: str, eps: float = 0.15, min_samples: int = 2):
        """
        Cluster pets of a specific species using CLIP embeddings.
        
        Args:
            species: 'dog' or 'cat'
            eps: Maximum distance for clustering (0.10-0.20 typical for CLIP)
                 Lower = more clusters (finer separation)
            min_samples: Minimum samples to form a cluster
        """
        logger.info(f"🐾 Starting CLIP clustering for {species}...")
        
        with session_scope() as session:
            # Get all detections of this species
            stmt = select(PetEmbedding).where(
                PetEmbedding.species == species
            ).order_by(PetEmbedding.id)
            pets = session.execute(stmt).scalars().all()
            
            if len(pets) == 0:
                logger.warning(f"No {species} detections found.")
                return
            
            logger.info(f"Found {len(pets)} {species} detections")
            
            # Extract CLIP embeddings
            embeddings = []
            valid_pets = []
            
            for i, pet in enumerate(pets):
                if i % 50 == 0:
                    logger.info(f"Extracting CLIP embeddings: {i}/{len(pets)}")
                
                if not pet.thumbnail_path or not Path(pet.thumbnail_path).exists():
                    logger.warning(f"Skipping pet {pet.id}: no thumbnail")
                    continue
                
                emb = self.extract_embedding(pet.thumbnail_path)
                if emb is not None and np.any(emb != 0):
                    embeddings.append(emb)
                    valid_pets.append(pet)
            
            if len(embeddings) == 0:
                logger.error("No valid embeddings extracted.")
                return
            
            logger.info(f"Extracted {len(embeddings)} valid CLIP embeddings")
            
            # Convert to numpy array
            X = np.array(embeddings, dtype=np.float32)
            
            # Run DBSCAN clustering with cosine distance
            logger.info(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
            clustering = DBSCAN(
                eps=eps, 
                min_samples=min_samples, 
                metric='cosine', 
                n_jobs=-1
            )
            labels = clustering.fit_predict(X)
            
            # Count clusters
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(labels).count(-1)
            
            logger.info(f"✓ Found {n_clusters} {species} clusters ({n_noise} noise points)")
            
            # Clear existing clusters for this species
            session.execute(delete(PetCluster).where(PetCluster.species == species))
            session.flush()
            
            # Create PetCluster entries
            cluster_map = {}
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise
                
                cluster = PetCluster(
                    species=species,
                    display_name=f"{species.capitalize()} {label + 1}"
                )
                session.add(cluster)
                session.flush()
                cluster_map[label] = cluster.id
            
            # Assign pets to clusters
            for pet, label in zip(valid_pets, labels):
                if label == -1:
                    pet.cluster_id = None  # Unclustered (noise)
                else:
                    pet.cluster_id = cluster_map[label]
            
            session.commit()
            logger.info(f"✓ {species.capitalize()} clustering complete: {n_clusters} clusters created")


if __name__ == "__main__":
    clusterer = CLIPPetClusterer()
    
    # Cluster dogs with tight parameters for individual separation
    # CLIP embeddings are more semantic, so can use slightly higher eps than ResNet
    clusterer.cluster_species('dog', eps=0.12, min_samples=2)
    
    # Cluster cats
    clusterer.cluster_species('cat', eps=0.12, min_samples=2)
    
    logger.info("✅ All pet clustering complete!")
