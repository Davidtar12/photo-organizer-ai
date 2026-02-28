"""
Run the complete face indexing pipeline:
1. Extract faces and embeddings from all photos
2. Build Faiss index for fast similarity search
3. Cluster faces into person groups
4. Generate active learning suggestions

This is a one-shot script that runs all steps in sequence with detailed progress.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add backend directory to Python path
script_dir = Path(__file__).resolve().parent
backend_dir = script_dir / "backend"
sys.path.insert(0, str(backend_dir))

# Now import using relative paths from backend
from services.face_indexer import FaceIndexer
from services.faiss_index import FaissIndex
from services.face_clusterer import FaceClusterer
from services.suggestion_generator import SuggestionGenerator
from database import engine
from models import Base
from logging_config import setup_logging, get_logger

# Setup centralized logging
setup_logging(use_colors=True)
logger = get_logger(__name__)


def log_stage(stage: int, total_stages: int, name: str, start: bool = True):
    """Log pipeline stage with progress percentage."""
    pct = ((stage - 1) / total_stages * 100) if not start else (stage / total_stages * 100)
    symbol = "▶" if start else "✓"
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"{symbol} STAGE {stage}/{total_stages} ({pct:.0f}%): {name}")
    logger.info("=" * 80)
    logger.info("")


def main():
    overall_start = time.time()
    total_stages = 4  # Updated to include suggestion generation

    logger.info("=" * 80)
    logger.info("🎯 FACE BROWSER - FULL INDEXING PIPELINE")
    logger.info("=" * 80)
    logger.info("")

    # Initialize database tables
    logger.info("🔧 Creating database tables...")
    Base.metadata.create_all(engine)
    logger.info("✓ Database ready")
    logger.info("")

    # Step 1: Extract faces and embeddings
    log_stage(1, total_stages, "Extracting faces and embeddings from photos")
    stage_start = time.time()
    indexer = FaceIndexer()
    # Run the indexer with optimized batch processing
    indexer.run(full_scan=False, batch_size=128)  # Increased from 16 to 32 for max GPU utilization
    stage_time = time.time() - stage_start
    logger.info(f"⏱️  Stage 1 completed in {stage_time/60:.1f} minutes")

    # Step 2: Build Faiss index
    log_stage(2, total_stages, "Building Faiss index for similarity search")
    stage_start = time.time()
    faiss_builder = FaissIndex()
    faiss_builder.build()
    stage_time = time.time() - stage_start
    logger.info(f"⏱️  Stage 2 completed in {stage_time:.1f} seconds")

    # Step 3: Cluster faces into people (Pass 1 - Seed Clusters)
    log_stage(3, total_stages, "Clustering faces into person groups (Pass 1 - Strict/High-Purity)")
    stage_start = time.time()
    clusterer = FaceClusterer(min_cluster_size=2)
    
    # Use STRICT threshold for Pass 1 (creates pure seed clusters)
    # 0.315 L2 distance on normalized vectors ≈ high confidence same person
    clusterer.cluster_all(distance_threshold=0.315)
    
    stage_time = time.time() - stage_start
    logger.info(f"⏱️  Stage 3 (Pass 1) completed in {stage_time:.1f} seconds")

    # Step 3b: Assign noise to centroids (Pass 2 - Capture the Long Tail)
    logger.info("")
    logger.info("=" * 80)
    logger.info("▶ STAGE 3b/4: Assigning 'noise' faces to clusters (Pass 2 - Centroid Assignment)")
    logger.info("=" * 80)
    logger.info("")
    stage_start = time.time()
    
    # Use a lenient threshold for this pass (0.4-0.5 recommended)
    # 0.4 L2 distance on normalized vectors ≈ 0.6 cosine similarity
    assigned_count = clusterer.assign_noise_to_centroids(distance_threshold=0.5)
    
    stage_time = time.time() - stage_start
    logger.info(f"⏱️  Stage 3b (Pass 2) completed in {stage_time:.1f} seconds ({assigned_count} faces assigned)")

    # Step 3c: Merge similar clusters (Pass 3 - Fix Split Errors)
    logger.info("")
    logger.info("=" * 80)
    logger.info("▶ STAGE 3c/4: Merging similar clusters (Pass 3 - Auto-Merge)")
    logger.info("=" * 80)
    logger.info("")
    stage_start = time.time()
    
    # Use a stricter threshold for merging (0.3-0.35 recommended)
    # Only merge if centroids are very close (high confidence same person)
    merged_count = clusterer.merge_similar_clusters(distance_threshold=0.35)
    
    stage_time = time.time() - stage_start
    logger.info(f"⏱️  Stage 3c (Pass 3) completed in {stage_time:.1f} seconds ({merged_count} clusters merged)")

    # Step 4: Generate active learning suggestions
    log_stage(4, total_stages, "Generating active learning suggestions")
    stage_start = time.time()
    suggestion_gen = SuggestionGenerator()
    num_suggestions = suggestion_gen.generate_suggestions(max_suggestions=100)
    stage_time = time.time() - stage_start
    logger.info(f"⏱️  Stage 4 completed in {stage_time:.1f} seconds ({num_suggestions} suggestions)")

    total_time = time.time() - overall_start
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ PIPELINE 100% COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start the backend: python run_backend.py")
    logger.info("  2. View clusters at: http://localhost:5052/api/persons/")
    logger.info("  3. Review suggestions at: http://localhost:5052/api/suggestions/pending")
    logger.info("  4. Check health: http://localhost:5052/api/health")
    logger.info("")


if __name__ == "__main__":
    main()
