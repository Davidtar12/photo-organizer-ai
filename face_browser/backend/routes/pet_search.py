"""
CLIP-based search endpoint for text-to-image and image-to-image pet search.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import faiss
import numpy as np
import torch
from flask import Blueprint, jsonify, request
from PIL import Image
from sqlalchemy import select

try:
    import open_clip
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "open-clip-torch"])
    import open_clip

from database import session_scope
from models import PetEmbedding
from schemas import PetEmbeddingSchema

bp = Blueprint("pet_search", __name__, url_prefix="/api/pets/search")
pet_schema = PetEmbeddingSchema(many=True)

logger = logging.getLogger(__name__)

# Global CLIP model and FAISS index (loaded on first request)
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_FAISS_INDEX = None
_PET_IDS = None
_DEVICE = None


def _load_clip_model():
    """Load CLIP model once globally."""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _DEVICE
    
    if _CLIP_MODEL is None:
        logger.info("Loading CLIP model for search...")
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _CLIP_MODEL, _, _CLIP_PREPROCESS = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='openai'
        )
        _CLIP_MODEL.to(_DEVICE).eval()
        logger.info(f"✓ CLIP loaded on {_DEVICE}")


def _build_faiss_index():
    """Build FAISS index from all pet embeddings in database."""
    global _FAISS_INDEX, _PET_IDS
    
    logger.info("Building FAISS index from pet embeddings...")
    
    with session_scope() as session:
        # Get all pets with valid thumbnails
        stmt = select(PetEmbedding).where(
            PetEmbedding.thumbnail_path.isnot(None)
        ).order_by(PetEmbedding.id)
        pets = session.execute(stmt).scalars().all()
        
        if len(pets) == 0:
            logger.warning("No pets found for indexing")
            return
        
        # Extract CLIP embeddings for all pets
        embeddings = []
        pet_ids = []
        
        for i, pet in enumerate(pets):
            if i % 100 == 0:
                logger.info(f"Extracting embeddings for FAISS: {i}/{len(pets)}")
            
            if not Path(pet.thumbnail_path).exists():
                continue
            
            try:
                img = Image.open(pet.thumbnail_path).convert('RGB')
                img_tensor = _CLIP_PREPROCESS(img).unsqueeze(0).to(_DEVICE)
                
                with torch.no_grad():
                    emb = _CLIP_MODEL.encode_image(img_tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                
                embeddings.append(emb.cpu().numpy().flatten())
                pet_ids.append(pet.id)
            except Exception as e:
                logger.error(f"Failed to extract embedding for pet {pet.id}: {e}")
                continue
        
        if len(embeddings) == 0:
            logger.error("No valid embeddings for FAISS index")
            return
        
        # Build FAISS index (Inner Product = cosine similarity for normalized vectors)
        embeddings_array = np.vstack(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        _FAISS_INDEX = faiss.IndexFlatIP(dimension)  # Inner product (cosine for L2-normalized)
        _FAISS_INDEX.add(embeddings_array)
        _PET_IDS = np.array(pet_ids)
        
        logger.info(f"✓ FAISS index built with {len(embeddings)} pets")


@bp.get("/text")
def search_by_text():
    """
    Text-to-Image search: Find pets matching a natural language description.
    
    Query params:
        q: Text query (e.g., "a golden retriever sitting on couch")
        k: Number of results (default 20)
        species: Optional filter for 'dog' or 'cat'
    
    Returns:
        List of pets ranked by similarity to text query
    """
    query_text = request.args.get('q', type=str)
    k = request.args.get('k', 20, type=int)
    species_filter = request.args.get('species', type=str)
    
    if not query_text:
        return jsonify({"error": "Query text 'q' is required"}), 400
    
    # Load model and index if needed
    _load_clip_model()
    if _FAISS_INDEX is None:
        _build_faiss_index()
    
    if _FAISS_INDEX is None:
        return jsonify({"error": "FAISS index not available"}), 500
    
    try:
        # Encode text query
        with torch.no_grad():
            text_tokens = open_clip.tokenize([query_text]).to(_DEVICE)
            text_emb = _CLIP_MODEL.encode_text(text_tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            query_vector = text_emb.cpu().numpy().astype('float32')
        
        # Search FAISS index
        distances, indices = _FAISS_INDEX.search(query_vector, k * 3)  # Get more for filtering
        
        # Get pet IDs
        result_pet_ids = _PET_IDS[indices[0]].tolist()
        scores = distances[0].tolist()
        
        # Fetch pets from database
        with session_scope() as session:
            stmt = select(PetEmbedding).where(PetEmbedding.id.in_(result_pet_ids))
            pets = session.execute(stmt).scalars().all()
            
            # Create dict for fast lookup
            pets_dict = {pet.id: pet for pet in pets}
            
            # Order results by FAISS ranking and apply species filter
            results = []
            for pet_id, score in zip(result_pet_ids, scores):
                pet = pets_dict.get(pet_id)
                if pet is None:
                    continue
                
                # Apply species filter if specified
                if species_filter and pet.species != species_filter:
                    continue
                
                pet_data = {
                    "id": pet.id,
                    "species": pet.species,
                    "cluster_id": pet.cluster_id,
                    "detection_confidence": pet.detection_confidence,
                    "similarity_score": float(score),
                    "media": {
                        "id": pet.media.id,
                        "path": pet.media.path
                    } if pet.media else None
                }
                results.append(pet_data)
                
                if len(results) >= k:
                    break
        
        return jsonify({
            "query": query_text,
            "results": results,
            "count": len(results)
        })
    
    except Exception as e:
        logger.error(f"Text search error: {e}")
        return jsonify({"error": str(e)}), 500


@bp.post("/image")
def search_by_image():
    """
    Image-to-Image search: Find similar pets by uploading a reference photo.
    
    Request body (JSON):
        image: Base64-encoded image data
        k: Number of results (default 20)
        species: Optional filter for 'dog' or 'cat'
    
    Returns:
        List of pets most similar to the uploaded image
    """
    data = request.get_json()
    
    if not data or 'image' not in data:
        return jsonify({"error": "Image data is required"}), 400
    
    k = data.get('k', 20)
    species_filter = data.get('species')
    
    # Load model and index if needed
    _load_clip_model()
    if _FAISS_INDEX is None:
        _build_faiss_index()
    
    if _FAISS_INDEX is None:
        return jsonify({"error": "FAISS index not available"}), 500
    
    try:
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:  # Remove data URL prefix if present
            image_data = image_data.split(',', 1)[1]
        
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Extract CLIP embedding from query image
        img_tensor = _CLIP_PREPROCESS(img).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            img_emb = _CLIP_MODEL.encode_image(img_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            query_vector = img_emb.cpu().numpy().astype('float32')
        
        # Search FAISS index
        distances, indices = _FAISS_INDEX.search(query_vector, k * 3)
        
        # Get pet IDs
        result_pet_ids = _PET_IDS[indices[0]].tolist()
        scores = distances[0].tolist()
        
        # Fetch pets from database
        with session_scope() as session:
            stmt = select(PetEmbedding).where(PetEmbedding.id.in_(result_pet_ids))
            pets = session.execute(stmt).scalars().all()
            
            pets_dict = {pet.id: pet for pet in pets}
            
            results = []
            for pet_id, score in zip(result_pet_ids, scores):
                pet = pets_dict.get(pet_id)
                if pet is None:
                    continue
                
                if species_filter and pet.species != species_filter:
                    continue
                
                pet_data = {
                    "id": pet.id,
                    "species": pet.species,
                    "cluster_id": pet.cluster_id,
                    "detection_confidence": pet.detection_confidence,
                    "similarity_score": float(score),
                    "media": {
                        "id": pet.media.id,
                        "path": pet.media.path
                    } if pet.media else None
                }
                results.append(pet_data)
                
                if len(results) >= k:
                    break
        
        return jsonify({
            "results": results,
            "count": len(results)
        })
    
    except Exception as e:
        logger.error(f"Image search error: {e}")
        return jsonify({"error": str(e)}), 500


@bp.post("/rebuild-index")
def rebuild_index():
    """Rebuild the FAISS index (useful after adding new pets)."""
    try:
        _build_faiss_index()
        return jsonify({"status": "success", "message": "FAISS index rebuilt"})
    except Exception as e:
        logger.error(f"Failed to rebuild index: {e}")
        return jsonify({"error": str(e)}), 500
