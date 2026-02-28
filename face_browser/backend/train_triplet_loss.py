"""
Triplet Loss Training for Pet Re-identification
Trains a deep metric learning model to create embeddings that separate individual dogs/cats
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding, MediaFile
from config import Config
import random
from tqdm import tqdm

# Configuration
MAX_FOLDER = r"C:\Users\david\Downloads\Max"
BATCH_SIZE = 32
EMBEDDING_DIM = 512
MARGIN = 0.5
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
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


class TripletLoss(nn.Module):
    """Triplet loss with margin"""
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class TripletDataset(Dataset):
    """Dataset that generates triplets: (anchor, positive, negative)"""
    def __init__(self, max_images_paths, other_dog_paths, transform=None):
        self.max_images = max_images_paths
        self.other_dogs = other_dog_paths
        self.transform = transform
        
        # Each epoch will sample from max images
        self.num_samples = len(max_images_paths) * 10  # 10x oversampling
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Anchor: random Max image
        anchor_path = random.choice(self.max_images)
        anchor_img = Image.open(anchor_path).convert('RGB')
        
        # Positive: different Max image
        positive_path = random.choice(self.max_images)
        while positive_path == anchor_path and len(self.max_images) > 1:
            positive_path = random.choice(self.max_images)
        positive_img = Image.open(positive_path).convert('RGB')
        
        # Negative: random other dog
        negative_path = random.choice(self.other_dogs)
        negative_img = Image.open(negative_path).convert('RGB')
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img


def get_max_images():
    """Get all Max image paths"""
    max_folder = Path(MAX_FOLDER)
    images = list(max_folder.glob('*.jpg')) + list(max_folder.glob('*.JPG'))
    print(f"Found {len(images)} Max images in {MAX_FOLDER}")
    return [str(p) for p in images]


def get_other_dog_thumbnails():
    """Get thumbnail paths for all other dogs from database"""
    session = Session()
    try:
        # Get all dog embeddings (species='dog')
        dog_embeddings = session.query(PetEmbedding).filter(
            PetEmbedding.species == 'dog'
        ).all()
        
        thumbnail_paths = []
        for emb in dog_embeddings:
            if emb.thumbnail_path and os.path.exists(emb.thumbnail_path):
                thumbnail_paths.append(emb.thumbnail_path)
        
        print(f"Found {len(thumbnail_paths)} other dog thumbnails from database")
        return thumbnail_paths
    finally:
        session.close()


def train():
    """Train the triplet loss model"""
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get data
    max_images = get_max_images()
    other_dogs = get_other_dog_thumbnails()
    
    if len(max_images) < 2:
        raise ValueError("Need at least 2 Max images for training")
    if len(other_dogs) < 1:
        raise ValueError("Need at least 1 other dog image for negatives")
    
    # Create dataset and dataloader
    dataset = TripletDataset(max_images, other_dogs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Create model, loss, optimizer
    model = EmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print(f"\nTraining Configuration:")
    print(f"  Max images: {len(max_images)}")
    print(f"  Other dogs: {len(other_dogs)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  Margin: {MARGIN}")
    print(f"  Device: {DEVICE}\n")
    
    # Training loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for batch_idx, (anchor, positive, negative) in enumerate(progress_bar):
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}')
        
        scheduler.step()
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'triplet_model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    final_model_path = 'triplet_model_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': EMBEDDING_DIM,
    }, final_model_path)
    print(f'\nFinal model saved: {final_model_path}')
    
    return model


if __name__ == '__main__':
    print("="*60)
    print("Triplet Loss Training for Pet Re-identification")
    print("="*60)
    
    model = train()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run generate_triplet_embeddings.py to create new embeddings")
    print("2. Run cluster_with_triplet_embeddings.py to re-cluster pets")
