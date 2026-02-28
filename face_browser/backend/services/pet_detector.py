"""
Pet-only detector for scanning photos and identifying dogs/cats.
This is separate from the FaceIndexer (which handles human faces).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from sqlalchemy import delete

from config import Config
from database import session_scope
from models import MediaFile, PetEmbedding, TaskProgress

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    path: Path
    size: int
    mtime: float


class PetDetector:
    """Detect and classify pets (dogs/cats) in photos using YOLO and fastai."""

    def __init__(self, cfg: Config = Config()) -> None:
        self.cfg = cfg
        
        # Load YOLO for object detection (dogs/cats)
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            logger.info("✓ YOLOv8 loaded for pet detection (dogs/cats)")
        except ImportError:
            logger.error("❌ Could not import ultralytics. Install it: pip install ultralytics")
            raise SystemExit("Fatal Error: ultralytics not installed")
        except Exception as e:
            logger.error(f"❌ YOLOv8 failed to load: {e}")
            raise SystemExit(f"Fatal Error: YOLOv8 could not be loaded. {e}")
        
        # Load fastai pet recognizer model (pets.pkl)
        self.pet_recognizer = None
        pets_model_path = Path(__file__).parent.parent / 'models' / 'pets.pkl'
        try:
            if pets_model_path.exists():
                from fastai.vision.all import load_learner
                self.pet_recognizer = load_learner(pets_model_path)
                logger.info("✓ Pet recognizer loaded from pets.pkl")
            else:
                logger.warning("⚠️  pets.pkl not found - will use YOLO detection only (no specific pet classification)")
        except ImportError:
            logger.warning("⚠️  fastai not available - will use YOLO detection only")
        except Exception as e:
            logger.warning(f"⚠️  Could not load pets.pkl: {e}")

    def list_media_files(self) -> List[FileInfo]:
        """List all media files in the Organized directory."""
        organized = self.cfg.ORGANIZED_DIR
        candidates: List[FileInfo] = []

        for path in organized.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.cfg.SUPPORTED_EXTENSIONS:
                continue
            stat = path.stat()
            candidates.append(FileInfo(path=path, size=stat.st_size, mtime=stat.st_mtime))

        logger.info("Discovered %s candidate media files", len(candidates))
        return candidates

    def run(self, full_scan: bool = False, batch_size: int = 16) -> None:
        """Run pet detection on all media files.
        
        Args:
            full_scan: If True, reprocess all files. If False, skip already processed files.
            batch_size: Number of images to process in parallel
        """
        logger.info("🔍 Starting pet detection scan...")
        files = self.list_media_files()
        total = len(files)
        processed = 0

        logger.info(
            "Starting pet detection on %s files (full_scan=%s, batch_size=%s)",
            total, full_scan, batch_size
        )
        self._update_progress("pet_detection", total_items=total, processed=0, message="Starting")

        # Filter files that need processing
        files_to_process = []
        for file_info in files:
            should_process = full_scan or self._should_process(file_info)
            if should_process:
                files_to_process.append(file_info)
        
        logger.info("📊 %s/%s files need processing (%.1f%% up-to-date)", 
                   len(files_to_process), total, 
                   ((total - len(files_to_process)) / total * 100) if total else 0)

        # Process files in batches
        batch = []
        for idx, file_info in enumerate(files):
            should_process = full_scan or self._should_process(file_info)
            
            if should_process:
                batch.append(file_info)
                
                if len(batch) >= batch_size or idx == len(files) - 1:
                    logger.info("Processing batch [%s-%s/%s] (%.1f%%)", 
                               processed + 1, processed + len(batch), total, 
                               ((processed + len(batch)) / total * 100))
                    self._process_batch(batch)
                    processed += len(batch)
                    batch = []
                    
                    if processed % 100 == 0 or processed == len(files_to_process):
                        logger.info(
                            "🎯 MILESTONE: %s/%s files completed (%.1f%%)",
                            processed, len(files_to_process),
                            (processed / len(files_to_process) * 100) if len(files_to_process) else 0,
                        )
                        self._update_progress(
                            "pet_detection",
                            total_items=total,
                            processed=processed,
                            message=f"Processed {processed}/{len(files_to_process)}",
                        )
            else:
                processed += 1

        self._update_progress("pet_detection", total_items=total, processed=total, message="Completed")
        logger.info("✅ Pet detection complete: processed %s files", len(files_to_process))

    def _should_process(self, info: FileInfo) -> bool:
        """Check if a file needs to be processed."""
        with session_scope() as session:
            media = session.query(MediaFile).filter_by(path=str(info.path)).one_or_none()
            if not media:
                return True
            if not media.last_scanned_at:
                return True
            if abs(media.last_scanned_at.timestamp() - info.mtime) > 1:
                return True
            return False

    def _process_batch(self, batch: List[FileInfo]) -> None:
        """Process a batch of files."""
        if not batch:
            return
        
        batch_results = []
        for file_info in batch:
            pets = self._detect_pets(file_info.path)
            batch_results.append((file_info, pets))
        
        # Commit all results to database
        with session_scope() as session:
            for file_info, pets in batch_results:
                try:
                    self._store_results(session, file_info, pets)
                except Exception as e:
                    logger.error("Failed to store results for %s: %s", file_info.path, e)
                    continue

    def _detect_pets(self, image_path: Path, conf_threshold: float = 0.4) -> List[dict]:
        """Detect pets using YOLO and classify with pets.pkl."""
        if not self.yolo_model:
            return []
        
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning("Could not read image: %s", image_path)
                return []
            
            results = self.yolo_model(img, conf=conf_threshold, verbose=False)
            detected_pets = []
            
            if results:
                names = self.yolo_model.names
                for r in results:
                    for i, c in enumerate(r.boxes.cls):
                        label = names[int(c)]
                        confidence = float(r.boxes.conf[i])
                        bbox_xyxy = r.boxes.xyxy[i].tolist()
                        
                        # Only process dogs and cats
                        if label not in ['dog', 'cat']:
                            continue
                        
                        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
                        bbox = {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}
                        
                        # Get embedding from fastai model if available
                        embedding = None
                        if self.pet_recognizer:
                            crop = img[y1:y2, x1:x2]
                            if crop.size > 0:
                                embedding = self._get_pet_embedding(crop)
                        
                        detected_pets.append({
                            'species': label,
                            'confidence': confidence,
                            'bbox': bbox,
                            'embedding': embedding
                        })
            
            return detected_pets
        except Exception as e:
            logger.error(f"Pet detection failed for {image_path}: {e}")
            return []

    def _get_pet_embedding(self, crop: np.ndarray) -> Optional[List[float]]:
        """Extract embedding from pet crop using fastai model."""
        if not self.pet_recognizer:
            return None
        
        try:
            from fastai.vision.all import PILImage
            import torch
            
            # Convert OpenCV BGR to RGB PIL Image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.create(crop_rgb)
            
            # Get embedding from model
            with torch.no_grad():
                # Get prediction and use it as embedding
                pred, pred_idx, probs = self.pet_recognizer.predict(pil_img)
                # Use probability distribution as embedding
                embedding = probs.cpu().numpy()
            
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Pet embedding extraction failed: {e}")
            return None

    def _store_results(self, session, file_info: FileInfo, pets: List[dict]) -> None:
        """Store pet detection results for a single file."""
        media = session.query(MediaFile).filter_by(path=str(file_info.path)).one_or_none()
        if not media:
            media = MediaFile(path=str(file_info.path))
        media.size_bytes = file_info.size
        media.last_scanned_at = datetime.fromtimestamp(file_info.mtime)
        media.sha256 = self._sha256(file_info.path)

        width, height = self._get_dimensions(file_info.path)
        media.width = width
        media.height = height

        session.add(media)
        session.flush()

        # Remove previous pet detections for re-indexing
        session.execute(delete(PetEmbedding).where(PetEmbedding.media_id == media.id))

        pets_created = 0
        for idx, pet_data in enumerate(pets):
            embedding = pet_data.get("embedding")
            if embedding is None or len(embedding) == 0:
                continue
            
            embedding_array = np.array(embedding, dtype=np.float32)
            bbox = pet_data.get("bbox", {})
            confidence = pet_data.get("confidence", 0.0)
            species = pet_data.get("species", "unknown")
            thumb_path = self._create_thumbnail(media.path, bbox, media.sha256, idx)
            
            pet = PetEmbedding(
                media_id=media.id,
                embedding=embedding_array.tobytes(),
                embedding_dim=embedding_array.size,
                embedding_index=idx,
                bbox=bbox,
                detection_confidence=confidence,
                species=species,
                model_name="pets.pkl" if self.pet_recognizer else "yolo_only",
                detector="yolov8n",
                thumbnail_path=str(thumb_path) if thumb_path else None,
            )
            session.add(pet)
            pets_created += 1
        
        media.pet_count = pets_created
        session.add(media)

        logger.debug("Indexed %s pets in %s", pets_created, file_info.path)

    def _sha256(self, path: Path) -> str:
        """Calculate SHA256 hash of file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _get_dimensions(self, path: Path) -> tuple[int, int]:
        """Get image dimensions."""
        try:
            with Image.open(path) as img:
                return img.width, img.height
        except Exception:
            return 0, 0

    def _create_thumbnail(self, image_path: str, bbox: dict, sha256: Optional[str], idx: int) -> Optional[Path]:
        """Create a thumbnail of the cropped pet region."""
        if not bbox:
            return None

        image_file = Path(image_path)
        cache_dir = Path(self.cfg.THUMB_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        thumb_name = f"pet_{sha256 or image_file.stem}_{idx}.jpg"
        thumb_path = cache_dir / thumb_name

        if thumb_path.exists():
            return thumb_path

        try:
            with Image.open(image_file) as img:
                left = bbox.get("x", 0)
                top = bbox.get("y", 0)
                width = bbox.get("w", 0)
                height = bbox.get("h", 0)

                if width <= 0 or height <= 0:
                    return None

                crop = img.crop((left, top, left + width, top + height))
                crop.thumbnail((256, 256))
                crop.save(thumb_path, "JPEG", quality=85)
                return thumb_path
        except Exception as exc:
            logger.error("Failed to create thumbnail for %s: %s", image_path, exc)
            return None

    def _update_progress(self, task_name: str, *, total_items: Optional[int], processed: int, message: str) -> None:
        """Update task progress in database."""
        with session_scope() as session:
            task = session.query(TaskProgress).filter_by(task_name=task_name).one_or_none()
            if not task:
                task = TaskProgress(task_name=task_name)

            task.total_items = total_items
            task.processed_items = processed
            task.last_message = message
            session.add(task)
