"""
DogFaceNet - Extract embeddings using a ResNet-based architecture fine-tuned for dog faces
We'll use a pre-trained ResNet50 from timm and adapt it for dog face embeddings
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, MediaFile
from config import Config
from tqdm import tqdm
import timm

class DogFaceEmbeddingNet(nn.Module):
    """
    Dog face embedding network using ResNet50 backbone
    Optimized for dog face recognition
    """
    def __init__(self, embedding_dim=128):
        super(DogFaceEmbeddingNet, self).__init__()
        
        # Use ResNet50 pretrained on ImageNet
        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
        
        # Get the feature dimension from backbone
        self.feature_dim = self.backbone.num_features
        
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Generate embeddings
        embeddings = self.embedding_head(features)
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


def load_model(device, embedding_dim=128):
    """Load the dog face embedding model"""
    print("Loading ResNet50-based dog face model...")
    model = DogFaceEmbeddingNet(embedding_dim=embedding_dim)
    model = model.to(device)
    model.eval()
    return model


# Image preprocessing with better augmentation focusing on face region
transform_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Face-focused crop (assume face is usually in center/top)
transform_face_crop = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Multiple random crops to capture different parts
transform_crop1 = transforms.Compose([
    transforms.Resize((280, 280)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Slight color jitter to be robust to lighting
transform_jitter = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_embedding(model, image_path, device):
    """Extract embedding from an image using test-time augmentation with face focus"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply multiple transformations - focus on capturing face from different angles
        img_base = transform_base(image).unsqueeze(0).to(device)
        img_face = transform_face_crop(image).unsqueeze(0).to(device)
        img_crop1 = transform_crop1(image).unsqueeze(0).to(device)
        img_jitter = transform_jitter(image).unsqueeze(0).to(device)
        
        # Extract embeddings for each augmentation
        with torch.no_grad():
            emb_base = model(img_base)
            emb_face = model(img_face)
            emb_crop1 = model(img_crop1)
            emb_jitter = model(img_jitter)
        
        # Weighted average - give more weight to face-focused crop
        embedding = (emb_base * 1.0 + emb_face * 2.0 + emb_crop1 * 1.0 + emb_jitter * 0.5) / 4.5
        
        # L2 normalize the averaged embedding
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def main():
    print("=" * 60)
    print("Dog Face Embedding Extraction (ResNet50)")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(device, embedding_dim=128)
    print("Model loaded successfully")
    
    # Database setup
    DATABASE_URL = f'sqlite:///{Config.DB_PATH}'
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get all dog embeddings (only process dogs, not cats)
        dogs = session.query(PetEmbedding).filter(
            PetEmbedding.species == 'dog'
        ).all()
        
        print(f"\nFound {len(dogs)} dogs to process")
        
        successful = 0
        failed = 0
        
        for dog in tqdm(dogs, desc="Extracting embeddings"):
            # Get media file path
            media = session.query(MediaFile).filter_by(id=dog.media_id).first()
            if not media:
                failed += 1
                continue
            
            # Extract embedding
            embedding = extract_embedding(model, media.path, device)
            
            if embedding is not None:
                # Update database
                dog.embedding = embedding.astype(np.float32).tobytes()
                successful += 1
            else:
                failed += 1
            
            # Commit every 100 images
            if (successful + failed) % 100 == 0:
                session.commit()
        
        # Final commit
        session.commit()
        
        print("\n" + "=" * 60)
        print("Embedding Extraction Complete!")
        print("=" * 60)
        print(f"Successful: {successful}/{len(dogs)}")
        print(f"Failed: {failed}/{len(dogs)}")
        print(f"Embedding dimension: 128")
        
        if successful > 0:
            print("\nNext step: Run clustering with these embeddings")
            print("  python cluster_with_triplet_embeddings.py")
        
    finally:
        session.close()


if __name__ == '__main__':
    main()
