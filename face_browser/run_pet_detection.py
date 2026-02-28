"""
Standalone script to detect and classify pets (dogs/cats) in the Organized folder.
This script:
1. Scans all photos in the Organized folder
2. Detects dogs and cats using YOLO
3. Classifies them as "Max" (or other pets) using the trained pet recognizer
4. Stores results in the pet_embeddings table

This script does NOT process human faces (no buffalo_l/InsightFace).

Usage:
    python run_pet_detection.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from backend.config import Config
from backend.database import session_scope, engine
from backend.models import Base, PetEmbedding, MediaFile
from backend.services.pet_detector import PetDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Run pet detection on all photos in the Organized folder."""
    
    logger.info("=" * 80)
    logger.info("🐕 STARTING PET DETECTION (YOLO + pets.pkl)")
    logger.info("=" * 80)
    
    # Ensure database tables exist
    Base.metadata.create_all(bind=engine)
    
    # Initialize the pet detector (YOLO + pets.pkl only, no buffalo_l)
    detector = PetDetector()
    
    # List all media files
    logger.info("📂 Scanning Organized folder for photos...")
    files = detector.list_media_files()
    logger.info(f"Found {len(files)} candidate media files")
    
    # Process files in batches
    logger.info("🔍 Starting pet detection (this may take a while)...")
    detector.run(full_scan=True, batch_size=16)
    
    # Count detected pets
    with session_scope() as session:
        pet_count = session.query(PetEmbedding).count()
        media_with_pets = session.query(MediaFile).filter(MediaFile.pet_count > 0).count()
    
    logger.info("=" * 80)
    logger.info("✅ PET DETECTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total pets detected: {pet_count}")
    logger.info(f"Media files with pets: {media_with_pets}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start the backend server: cd backend && python app.py")
    logger.info("2. Open the pets app: http://127.0.0.1:5052/pets.html")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Pet detection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
