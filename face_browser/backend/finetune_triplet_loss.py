"""
Fine-tune EfficientFormer-L1 with Triplet Loss for dog re-identification
Uses Max dog photos (39 images) as positive samples and other dogs as negatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
from pathlib import Path
import random
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import PetEmbedding
from config import Config

# Configuration
MAX_DOG_FOLDER = Path(r"C:\Users\USERNAME\Downloads\Max")
DATABASE_URL = f'sqlite:///{Config.DB_PATH}'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-5
MARGIN = 0.5  # Triplet loss margin
EMBEDDING_DIM = 512


class EfficientFormerEmbedder(nn.Module):
    """EfficientFormer-L1 with custom embedding head"""
    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()
        # Load pretrained model
        self.backbone = timm.create_model('efficientformer_l1', pretrained=pretrained, num_classes=0)
        backbone_dim = 448  # EfficientFormer-L1 output dimension
        
        # Custom embedding head for metric learning
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        # L2 normalize for cosine similarity
        return F.normalize(embeddings, p=2, dim=1)


class TripletDataset(Dataset):
    """Dataset that generates triplets (anchor, positive, negative)"""
    def __init__(self, max_dog_paths, other_dog_embeddings, transform=None):
        self.max_dog_paths = list(max_dog_paths)
        self.other_dogs = other_dog_embeddings  # List of (path, cluster_id)
        self.transform = transform
        
        # Group other dogs by cluster for hard negative mining
        self.dog_clusters = {}
        for path, cluster_id in other_dog_embeddings:
            if cluster_id not in self.dog_clusters:
                self.dog_clusters[cluster_id] = []
            self.dog_clusters[cluster_id].append(path)
    
    def __len__(self):
        # Generate many triplets from Max photos
        return len(self.max_dog_paths) * 10
    
    def __getitem__(self, idx):
        # Anchor: Max dog photo
        anchor_path = random.choice(self.max_dog_paths)
        
        # Positive: Different Max dog photo
        positive_path = random.choice([p for p in self.max_dog_paths if p != anchor_path])
        
        # Negative: Random dog from different cluster
        negative_cluster = random.choice(list(self.dog_clusters.keys()))
        negative_path = random.choice(self.dog_clusters[negative_cluster])
        
        # Load images
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative


def triplet_loss(anchor, positive, negative, margin=0.5):
    """Triplet loss: d(anchor, positive) + margin < d(anchor, negative)"""
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def get_other_dogs_paths(session, limit=500):
    """Get paths of other dogs (not Max) for negative samples"""
    dogs = session.query(PetEmbedding).filter(
        PetEmbedding.species == 'dog',
        PetEmbedding.cluster_id.isnot(None)
    ).limit(limit).all()
    
    paths = []
    for dog in dogs:
        media_path = dog.media.path if dog.media else None
        if media_path and Path(media_path).exists():
            paths.append((media_path, dog.cluster_id))
    
    return paths


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for anchor, positive, negative in pbar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        # Forward pass
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        # Compute triplet loss
        loss = triplet_loss(anchor_emb, positive_emb, negative_emb, margin=MARGIN)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, max_dog_paths, other_dog_paths, transform, device):
    """Validate by checking if Max photos cluster together"""
    model.eval()
    
    with torch.no_grad():
        # Get embeddings for all Max photos
        max_embeddings = []
        for path in max_dog_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            emb = model(img_tensor)
            max_embeddings.append(emb.cpu().numpy())
        
        max_embeddings = np.vstack(max_embeddings)
        
        # Get embeddings for sample of other dogs
        other_embeddings = []
        for path, _ in random.sample(other_dog_paths, min(50, len(other_dog_paths))):
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            emb = model(img_tensor)
            other_embeddings.append(emb.cpu().numpy())
        
        other_embeddings = np.vstack(other_embeddings)
        
        # Compute average intra-Max similarity vs Max-to-others similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        intra_max_sim = cosine_similarity(max_embeddings, max_embeddings)
        # Exclude diagonal
        intra_max_sim = intra_max_sim[~np.eye(intra_max_sim.shape[0], dtype=bool)]
        
        max_to_others_sim = cosine_similarity(max_embeddings, other_embeddings)
        
        return {
            'intra_max_mean': intra_max_sim.mean(),
            'intra_max_min': intra_max_sim.min(),
            'max_to_others_mean': max_to_others_sim.mean(),
            'max_to_others_max': max_to_others_sim.max(),
            'separation': intra_max_sim.mean() - max_to_others_sim.mean()
        }


def main():
    print("="*70)
    print("TRIPLET LOSS FINE-TUNING FOR DOG RE-IDENTIFICATION")
    print("="*70)
    
    # Get Max dog paths
    max_dog_paths = list(MAX_DOG_FOLDER.glob("*.jpg")) + list(MAX_DOG_FOLDER.glob("*.JPG"))
    print(f"\nMax dog photos found: {len(max_dog_paths)}")
    
    if len(max_dog_paths) < 10:
        print("ERROR: Need at least 10 photos of Max for training")
        return
    
    # Get other dogs for negatives
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    print("Loading other dogs for negative samples...")
    other_dog_paths = get_other_dogs_paths(session, limit=500)
    print(f"Other dogs loaded: {len(other_dog_paths)} from {len(set(c for _, c in other_dog_paths))} clusters")
    
    session.close()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = TripletDataset(max_dog_paths, other_dog_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"\nDataset size: {len(dataset)} triplets")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Load pretrained model
    print(f"\nLoading EfficientFormer-L1 on {DEVICE}...")
    model = EfficientFormerEmbedder(embedding_dim=EMBEDDING_DIM, pretrained=True).to(DEVICE)
    
    # Optimizer (fine-tuning, so small learning rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Margin: {MARGIN}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    
    # Baseline validation
    print("\n" + "="*70)
    print("BASELINE (before training)")
    print("="*70)
    baseline = validate(model, max_dog_paths, other_dog_paths, val_transform, DEVICE)
    print(f"Intra-Max similarity: {baseline['intra_max_mean']:.4f} (min: {baseline['intra_max_min']:.4f})")
    print(f"Max-to-others similarity: {baseline['max_to_others_mean']:.4f} (max: {baseline['max_to_others_max']:.4f})")
    print(f"Separation: {baseline['separation']:.4f}")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_separation = baseline['separation']
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, DEVICE, epoch)
        scheduler.step()
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            metrics = validate(model, max_dog_paths, other_dog_paths, val_transform, DEVICE)
            print(f"\nEpoch {epoch} - Loss: {avg_loss:.4f}")
            print(f"  Intra-Max: {metrics['intra_max_mean']:.4f}, Max-to-others: {metrics['max_to_others_mean']:.4f}")
            print(f"  Separation: {metrics['separation']:.4f} (best: {best_separation:.4f})")
            
            # Early stopping
            if metrics['separation'] > best_separation:
                best_separation = metrics['separation']
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'efficientformer_triplet_best.pth')
                print("  ✅ Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping (patience {patience})")
                    break
    
    # Final validation
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    # Load best model
    model.load_state_dict(torch.load('efficientformer_triplet_best.pth'))
    final = validate(model, max_dog_paths, other_dog_paths, val_transform, DEVICE)
    
    print(f"Baseline separation: {baseline['separation']:.4f}")
    print(f"Final separation: {final['separation']:.4f}")
    print(f"Improvement: {final['separation'] - baseline['separation']:.4f}")
    
    print(f"\nBefore training:")
    print(f"  Intra-Max: {baseline['intra_max_mean']:.4f}, Max-to-others: {baseline['max_to_others_mean']:.4f}")
    print(f"After training:")
    print(f"  Intra-Max: {final['intra_max_mean']:.4f}, Max-to-others: {final['max_to_others_mean']:.4f}")
    
    print("\n✅ Training complete!")
    print("✅ Best model saved to: efficientformer_triplet_best.pth")
    print("\nNext steps:")
    print("1. Run extract_efficientformer_embeddings.py with fine-tuned model")
    print("2. Re-cluster with updated embeddings")
    print("3. Compare clustering quality")


if __name__ == '__main__':
    main()
