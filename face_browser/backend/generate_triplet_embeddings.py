"""
Generate Triplet Loss Embeddings for All Pets
Uses the trained triplet loss model to create new embeddings for all pet detections
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, MediaFile
from config import Config
from tqdm import tqdm

# Configuration
MODEL_PATH = 'triplet_model_final.pth'
EMBEDDING_DIM = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# Database setup - use the same database as the rest of the app
DATABASE_URL = f'sqlite:///{Config.DB_PATH}'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


class EmbeddingNet(nn.Module):
    """ResNet-based embedding network with L2 normalization"""
    def __init__(self, embedding_dim=512):
        super(EmbeddingNet, self).__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Add custom embedding head
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2 normalize embeddings
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def load_model(model_path):
    """Load the trained triplet loss model"""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    model = EmbeddingNet(embedding_dim=checkpoint.get('embedding_dim', EMBEDDING_DIM))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def get_transform():
    """Get image transform for inference"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def extract_embedding(model, image_path, transform):
    """Extract embedding for a single image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            embedding = model(img_tensor)
        
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def generate_embeddings():
    """Generate embeddings for all pet detections"""
    session = Session()
    
    try:
        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Please train the model first.")
        
        model = load_model(MODEL_PATH)
        transform = get_transform()
        
        # Get all pet embeddings
        all_pets = session.query(PetEmbedding).all()
        print(f"\nGenerating triplet loss embeddings for {len(all_pets)} pets...")
        
        updated_count = 0
        failed_count = 0
        
        for pet_emb in tqdm(all_pets, desc="Processing pets"):
            if not pet_emb.thumbnail_path:
                failed_count += 1
                continue
            
            if not os.path.exists(pet_emb.thumbnail_path):
                failed_count += 1
                continue
            
            # Generate new embedding
            embedding = extract_embedding(model, pet_emb.thumbnail_path, transform)
            
            if embedding is not None:
                # Update embedding in database
                pet_emb.embedding = embedding.tobytes()
                pet_emb.embedding_model = 'triplet_resnet50'
                updated_count += 1
            else:
                failed_count += 1
        
        # Commit changes
        session.commit()
        
        print(f"\n{'='*60}")
        print(f"Embedding Generation Complete!")
        print(f"{'='*60}")
        print(f"Successfully updated: {updated_count}")
        print(f"Failed: {failed_count}")
        print(f"Total: {len(all_pets)}")
        
        # Show species breakdown
        dogs = session.query(PetEmbedding).filter(PetEmbedding.species == 'dog').count()
        cats = session.query(PetEmbedding).filter(PetEmbedding.species == 'cat').count()
        print(f"\nBreakdown:")
        print(f"  Dogs: {dogs}")
        print(f"  Cats: {cats}")
        
    finally:
        session.close()


if __name__ == '__main__':
    print("="*60)
    print("Triplet Loss Embedding Generation")
    print("="*60)
    
    generate_embeddings()
    
    print("\nNext step: Run cluster_with_triplet_embeddings.py to re-cluster pets")
