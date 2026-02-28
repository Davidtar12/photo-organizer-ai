"""
EfficientFormer-L1 Vision Transformer + Metric Learning
State-of-the-art approach for dog re-identification
Uses ViT attention mechanism to capture global features (fur patterns, face structure)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, MediaFile
from config import Config
from tqdm import tqdm
import timm

# Configuration
EMBEDDING_SIZE = 512  # Higher dimensional space for better separation
BATCH_SIZE = 32  # Process multiple images at once for speed


class EfficientFormerEmbedder(nn.Module):
    """
    Vision Transformer-based embedding model using EfficientFormer-L1
    
    Why EfficientFormer-L1?
    - Attention mechanism captures global context (fur patterns, facial structure)
    - Much better than ResNet for fine-grained recognition
    - Efficient enough to run on consumer GPUs
    - Pre-trained on ImageNet for good initialization
    - Fine-tuned with Triplet Loss for same-dog recognition
    """
    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()
        
        # Load EfficientFormer-L1 backbone (Vision Transformer)
        # num_classes=0 removes the classification head, gives us raw features
        print("Loading EfficientFormer-L1 backbone...")
        self.backbone = timm.create_model(
            'efficientformer_l1', 
            pretrained=pretrained, 
            num_classes=0,  # Remove classification head
            global_pool=''  # We'll do our own pooling
        )
        
        # Get the feature dimension from the model
        # EfficientFormer-L1 outputs 448-dimensional features
        self.feature_dim = self.backbone.num_features
        print(f"Backbone feature dimension: {self.feature_dim}")
        
        # Embedding head: Convert ViT features to metric learning embeddings
        # This is crucial for clustering - we want L2-normalized embeddings
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Small dropout for robustness
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x):
        """
        Forward pass with proper feature extraction and normalization
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            L2-normalized embeddings (B, embedding_dim)
        """
        # Extract features from ViT backbone
        features = self.backbone.forward_features(x)
        
        # Global average pooling over spatial dimensions
        # ViT outputs (B, num_patches, feature_dim)
        # We average across patches to get (B, feature_dim)
        if len(features.shape) == 3:
            features = features.mean(dim=1)
        elif len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # Generate embeddings
        embeddings = self.embedding_head(features)
        
        # L2 normalization - CRITICAL for cosine similarity / metric learning
        # This puts all embeddings on the unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


def load_model(device, embedding_dim=512, use_finetuned=True):
    """Load the EfficientFormer embedding model"""
    print("=" * 60)
    print("Loading EfficientFormer-L1 Embedding Model")
    print("=" * 60)
    
    model = EfficientFormerEmbedder(embedding_dim=embedding_dim, pretrained=True)
    
    # Load fine-tuned weights if available
    if use_finetuned:
        finetuned_path = 'efficientformer_triplet_best.pth'
        try:
            print(f"Loading fine-tuned weights from {finetuned_path}...")
            state_dict = torch.load(finetuned_path, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Fine-tuned model loaded successfully!")
            print("   (Triplet Loss trained on Max dog dataset)")
        except FileNotFoundError:
            print(f"⚠️  Fine-tuned weights not found at {finetuned_path}")
            print("   Using ImageNet pretrained weights only")
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    print(f"Embedding dimension: {embedding_dim}")
    return model


# Enhanced preprocessing for Vision Transformers
# ViTs work best with standard ImageNet preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientFormer uses 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    )
])

# Test-time augmentation transforms for robustness
transform_flip = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_crop = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_embedding(model, image_path, device, use_tta=True):
    """
    Extract embedding from an image with optional test-time augmentation
    
    Args:
        model: EfficientFormer model
        image_path: Path to image
        device: CPU or CUDA
        use_tta: Whether to use test-time augmentation (slower but more robust)
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if use_tta:
            # Test-time augmentation: Extract embeddings from multiple views
            img_base = transform(image).unsqueeze(0).to(device)
            img_flip = transform_flip(image).unsqueeze(0).to(device)
            img_crop = transform_crop(image).unsqueeze(0).to(device)
            
            # Extract embeddings
            with torch.no_grad():
                emb_base = model(img_base)
                emb_flip = model(img_flip)
                emb_crop = model(img_crop)
            
            # Average and re-normalize
            # Weighted average: give more weight to base view
            embedding = (emb_base * 2.0 + emb_flip * 1.0 + emb_crop * 1.0) / 4.0
            embedding = F.normalize(embedding, p=2, dim=1)
        else:
            # Single pass (faster)
            img = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(img)
        
        return embedding.cpu().numpy().flatten()
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def extract_embeddings_batch(model, image_paths, device):
    """
    Extract embeddings in batches for speed
    
    Args:
        model: EfficientFormer model
        image_paths: List of image paths
        device: CPU or CUDA
    """
    embeddings = []
    
    # Load and transform images
    images = []
    valid_indices = []
    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            embeddings.append(None)
    
    if not images:
        return embeddings
    
    # Stack into batch
    batch = torch.stack(images).to(device)
    
    # Extract embeddings
    with torch.no_grad():
        batch_embeddings = model(batch)
    
    # Place results in correct positions
    result = [None] * len(image_paths)
    for idx, emb in zip(valid_indices, batch_embeddings.cpu().numpy()):
        result[idx] = emb
    
    return result


def main():
    print("=" * 60)
    print("EfficientFormer-L1 Vision Transformer Embedding Extraction")
    print("State-of-the-art for Dog Re-Identification")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model = load_model(device, embedding_dim=EMBEDDING_SIZE)
    
    # Database setup
    DATABASE_URL = f'sqlite:///{Config.DB_PATH}'
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get all dog embeddings
        dogs = session.query(PetEmbedding).filter(
            PetEmbedding.species == 'dog'
        ).all()
        
        print(f"\nFound {len(dogs)} dogs to process")
        print(f"Processing in batches of {BATCH_SIZE}")
        
        successful = 0
        failed = 0
        
        # Process in batches
        for i in tqdm(range(0, len(dogs), BATCH_SIZE), desc="Extracting embeddings"):
            batch_dogs = dogs[i:i+BATCH_SIZE]
            
            # Get image paths
            batch_paths = []
            batch_dog_objs = []
            for dog in batch_dogs:
                media = session.query(MediaFile).filter_by(id=dog.media_id).first()
                if media:
                    batch_paths.append(media.path)
                    batch_dog_objs.append(dog)
                else:
                    failed += 1
            
            # Extract embeddings for batch
            if batch_paths:
                batch_embeddings = extract_embeddings_batch(model, batch_paths, device)
                
                # Update database
                for dog, embedding in zip(batch_dog_objs, batch_embeddings):
                    if embedding is not None:
                        dog.embedding = embedding.astype(np.float32).tobytes()
                        successful += 1
                    else:
                        failed += 1
            
            # Commit every batch
            session.commit()
        
        print("\n" + "=" * 60)
        print("Embedding Extraction Complete!")
        print("=" * 60)
        print(f"Successful: {successful}/{len(dogs)}")
        print(f"Failed: {failed}/{len(dogs)}")
        print(f"Embedding dimension: {EMBEDDING_SIZE}")
        print(f"\nModel: EfficientFormer-L1 (Vision Transformer)")
        print(f"Features: Global attention, fine-grained recognition")
        
        if successful > 0:
            print("\n✅ Next step: Run clustering with strict agglomerative method")
            print("  python cluster_agglomerative_strict.py")
            print("\n💡 With ViT embeddings, you should see MUCH better separation!")
            print("  - Fewer large mixed-breed clusters")
            print("  - Better individual dog identification")
            print("  - More meaningful groupings based on facial features")
        
    finally:
        session.close()


if __name__ == '__main__':
    main()
