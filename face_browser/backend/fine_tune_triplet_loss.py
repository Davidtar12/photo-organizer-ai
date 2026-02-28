#!/usr/bin/env python3
"""
Triplet Loss Fine-tuning for Pet Clustering

Fine-tunes the EfficientFormer-L1 Vision Transformer using Triplet Loss
to improve embedding quality for better pet clustering.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import sys
sys.path.append('.')

from database import get_session, PetEmbedding, PetCluster, MediaFile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PetTripletDataset(Dataset):
    """Dataset for Triplet Loss training using current cluster assignments as pseudo-labels"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_to_indices = self._create_label_indices()

    def _create_label_indices(self):
        """Create mapping from labels to indices for efficient triplet mining"""
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load image and return with its label"""
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            logger.warning(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a dummy image if loading fails
            return torch.zeros(3, 224, 224), self.labels[idx]

class TripletLoss(nn.Module):
    """Triplet Loss for metric learning"""

    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)

class EfficientFormerEmbedder(nn.Module):
    """EfficientFormer-L1 with custom embedding head for fine-tuning"""

    def __init__(self, embedding_dim=512, pretrained=True):
        super(EfficientFormerEmbedder, self).__init__()

        # Load EfficientFormer-L1 backbone
        self.backbone = timm.create_model('efficientformer_l1', pretrained=pretrained, num_classes=0)

        # Custom embedding head
        backbone_features = 448  # EfficientFormer-L1 feature dimension
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)

        # Generate embedding
        embedding = self.embedding_head(features)

        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)

        return embedding

def create_triplets_batch(embeddings, labels, batch_size=32):
    """Create triplets for training from embeddings and labels"""

    triplets = []
    unique_labels = np.unique(labels)

    for _ in range(batch_size):
        # Randomly select anchor label
        anchor_label = np.random.choice(unique_labels)

        # Get indices for this label
        anchor_indices = np.where(labels == anchor_label)[0]

        if len(anchor_indices) < 2:
            continue  # Need at least 2 samples of same class

        # Select anchor and positive
        anchor_idx, positive_idx = np.random.choice(anchor_indices, 2, replace=False)
        anchor_emb = embeddings[anchor_idx]
        positive_emb = embeddings[positive_idx]

        # Select negative (different class)
        negative_labels = unique_labels[unique_labels != anchor_label]
        if len(negative_labels) == 0:
            continue

        negative_label = np.random.choice(negative_labels)
        negative_indices = np.where(labels == negative_label)[0]

        if len(negative_indices) == 0:
            continue

        negative_idx = np.random.choice(negative_indices)
        negative_emb = embeddings[negative_idx]

        triplets.append((anchor_emb, positive_emb, negative_emb))

    return triplets

def prepare_data_for_finetuning(session):
    """Prepare dataset from current clustering results"""

    logger.info("Preparing data for Triplet Loss fine-tuning...")

    # Get all dog clusters and their pets
    dog_clusters = session.query(PetCluster).filter(PetCluster.species == 'dog').all()

    image_paths = []
    labels = []
    cluster_id_map = {}

    for cluster_idx, cluster in enumerate(dog_clusters):
        cluster_id_map[cluster.id] = cluster_idx

        # Get all pets in this cluster
        pets = session.query(PetEmbedding).filter(PetEmbedding.cluster_id == cluster.id).all()

        for pet in pets:
            # Get media file path
            media = session.query(MediaFile).filter(MediaFile.id == pet.media_id).first()
            if media and os.path.exists(media.file_path):
                image_paths.append(media.file_path)
                labels.append(cluster_idx)

    logger.info(f"Prepared {len(image_paths)} dog images across {len(dog_clusters)} clusters")

    return image_paths, labels

def fine_tune_with_triplet_loss(image_paths, labels, num_epochs=10, batch_size=32, learning_rate=1e-4):
    """Fine-tune the model using Triplet Loss"""

    logger.info("Starting Triplet Loss fine-tuning...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    model = EfficientFormerEmbedder(embedding_dim=512, pretrained=True)
    model.to(device)

    # Freeze backbone initially, only train embedding head
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Only train embedding head first
    optimizer = optim.Adam(model.embedding_head.parameters(), lr=learning_rate)
    criterion = TripletLoss(margin=0.5)

    # Data transforms
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = PetTripletDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Training loop
    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_images, batch_labels in progress_bar:
            batch_images = batch_images.to(device)

            # Get embeddings for the batch
            embeddings = model(batch_images)

            # Convert to numpy for triplet creation
            embeddings_np = embeddings.detach().cpu().numpy()
            labels_np = batch_labels.numpy()

            # Create triplets
            triplets = create_triplets_batch(embeddings_np, labels_np, batch_size=len(batch_images))

            if len(triplets) == 0:
                continue

            # Convert triplets back to tensors
            anchors = torch.stack([torch.tensor(t[0]) for t in triplets]).to(device)
            positives = torch.stack([torch.tensor(t[1]) for t in triplets]).to(device)
            negatives = torch.stack([torch.tensor(t[2]) for t in triplets]).to(device)

            # Compute loss
            loss = criterion(anchors, positives, negatives)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'efficientformer_triplet_finetuned.pth')
            logger.info("Saved best model checkpoint")

    # Unfreeze backbone for final fine-tuning
    logger.info("Unfreezing backbone for final fine-tuning...")
    for param in model.backbone.parameters():
        param.requires_grad = True

    # Lower learning rate for backbone
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': learning_rate * 0.1},
        {'params': model.embedding_head.parameters(), 'lr': learning_rate}
    ])

    # Final epochs with full model
    for epoch in range(5):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Full FT Epoch {epoch+1}/5")

        for batch_images, batch_labels in progress_bar:
            batch_images = batch_images.to(device)

            embeddings = model(batch_images)
            embeddings_np = embeddings.detach().cpu().numpy()
            labels_np = batch_labels.numpy()

            triplets = create_triplets_batch(embeddings_np, labels_np, batch_size=len(batch_images))

            if len(triplets) == 0:
                continue

            anchors = torch.stack([torch.tensor(t[0]) for t in triplets]).to(device)
            positives = torch.stack([torch.tensor(t[1]) for t in triplets]).to(device)
            negatives = torch.stack([torch.tensor(t[2]) for t in triplets]).to(device)

            loss = criterion(anchors, positives, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Full FT Epoch {epoch+1}/5, Average Loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'efficientformer_triplet_finetuned.pth')
            logger.info("Saved best model checkpoint")

    logger.info("Triplet Loss fine-tuning completed!")
    return model

def main():
    """Main function to run Triplet Loss fine-tuning"""

    logger.info("Starting Triplet Loss Fine-tuning for Pet Clustering")

    # Get database session
    session = get_session()

    try:
        # Prepare data
        image_paths, labels = prepare_data_for_finetuning(session)

        if len(image_paths) < 100:
            logger.error("Not enough data for fine-tuning. Need at least 100 images.")
            return

        # Fine-tune model
        model = fine_tune_with_triplet_loss(image_paths, labels)

        logger.info("✅ Triplet Loss fine-tuning completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Run extract_efficientformer_embeddings.py with the fine-tuned model")
        logger.info("2. Re-run clustering with improved embeddings")
        logger.info("3. Compare results with previous clustering")

    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\dscodingpython\File organizers\face_browser\backend\fine_tune_triplet_loss.py