from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
from PIL import Image
from sqlalchemy import delete

from config import Config
from database import session_scope
from models import FaceEmbedding, MediaFile, TaskProgress

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    path: Path
    size: int
    mtime: float


class FaceIndexer:
    """Index faces within the Organized folder."""

    def __init__(self, cfg: Config = Config()) -> None:
        self.cfg = cfg
        self.model_name = cfg.FACE_MODEL_NAME
        self.detector_backend = cfg.DETECTOR_BACKEND
        
        # Try InsightFace with AuraFace for superior embeddings
        self.use_insightface = False
        self.insightface_app = None
        
        try:
            import insightface
            
            logger.info("Loading buffalo_l with TensorRT optimization...")
            
            # TensorRT cache directory
            trt_cache_dir = Path.home() / '.insightface' / 'trt_cache'
            trt_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure TensorRT and CUDA providers
            providers = [
                ('TensorrtExecutionProvider', {  # Note: case-sensitive, lowercase 'rt'
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': str(trt_cache_dir),
                    'trt_max_workspace_size': 2147483648,
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
            
            # Load buffalo_l (InsightFace's best pretrained model)
            self.insightface_app = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=providers
            )
            self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Log which provider was actually selected
            rec_model = self.insightface_app.models.get('recognition')
            if rec_model:
                active_provider = rec_model.session.get_providers()[0]
                if active_provider == 'TensorrtExecutionProvider':
                    logger.info("✓ buffalo_l using TensorRT (FP16) - first run will build engines (~60-90s)")
                elif active_provider == 'CUDAExecutionProvider':
                    logger.info("✓ buffalo_l using CUDA (TensorRT unavailable or failed)")
                else:
                    logger.warning("⚠ buffalo_l using CPU (GPU acceleration unavailable)")
            
            self.use_insightface = True
            self.model_name = "buffalo_l"
            logger.info("buffalo_l ready (SCRFD detector + ArcFace-R50 embeddings)")
            
            # Plan C (AuraFace - manual integration, commented due to complexity):
            # from huggingface_hub import snapshot_download
            # auraface_dir = Path.home() / '.insightface' / 'models' / 'auraface'
            # if not (auraface_dir / 'scrfd_10g_bnkps.onnx').exists():
            #     logger.info("Downloading AuraFace-v1 model package from HuggingFace (one-time, ~400MB)...")
            #     snapshot_download("fal/AuraFace-v1", local_dir=str(auraface_dir))
            #     logger.info(f"✓ AuraFace-v1 downloaded to {auraface_dir}")
            # from insightface.model_zoo import model_zoo
            # det_model = model_zoo.get_model(str(auraface_dir / 'scrfd_10g_bnkps.onnx'), providers=providers)
            # det_model.prepare(ctx_id=0, input_size=(640, 640))
            # rec_model = model_zoo.get_model(str(auraface_dir / 'glintr100.onnx'), providers=providers)
            # rec_model.prepare(ctx_id=0)
            # self.insightface_app = insightface.app.FaceAnalysis(providers=providers)
            # self.insightface_app.models = {'detection': det_model, 'recognition': rec_model}
            # self.insightface_app.det_model = det_model

            
        except Exception as e:
            logger.critical(f"InsightFace/AuraFace failed to load. This is a fatal error. The program will exit. Error: {e}")
            raise SystemExit(f"Fatal Error: InsightFace/AuraFace could not be loaded. {e}")
            # logger.warning(f"InsightFace not available, falling back to DeepFace: {e}")
            # from deepface import DeepFace
            # logger.info("Loading DeepFace model (%s)...", self.model_name)
            # self.model = DeepFace.build_model(self.model_name)
            # logger.info("Model ready. Detector: %s", self.detector_backend)
        
        # Try loading YOLO for object detection (dogs/cats)
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            
            self.yolo_model = YOLO('yolov8n.pt')
            logger.info("YOLOv8 loaded for pet detection (dogs/cats)")
        except ImportError:
            logger.warning("Could not import ultralytics. Skipping pet detection.")
        except Exception as e:
            logger.warning(f"YOLOv8 not available, skipping pet detection: {e}")
        
        # Try loading fastai pet recognizer model (pets.pkl)
        self.pet_recognizer = None
        pets_model_path = Path(__file__).parent.parent / 'models' / 'pets.pkl'
        try:
            if pets_model_path.exists():
                from fastai.vision.all import load_learner
                self.pet_recognizer = load_learner(pets_model_path)
                logger.info("Pet recognizer loaded from pets.pkl")
            else:
                logger.info("pets.pkl not found - pet recognition disabled (YOLO detection only)")
        except ImportError:
            logger.warning("fastai not available - pet recognition disabled (YOLO detection only)")
        except Exception as e:
            logger.warning(f"Could not load pets.pkl: {e}")

    def list_media_files(self) -> List[FileInfo]:
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
        """Run face indexing with batch processing for GPU efficiency.
        
        Args:
            full_scan: If True, reprocess all files. If False, skip already processed files.
            batch_size: Number of images to process in parallel (default 16 for optimal GPU utilization)
        """
        logger.info("🔍 Starting run() method, calling list_media_files()...")
        files = self.list_media_files()
        total = len(files)
        processed = 0

        logger.info(
            "Starting face indexing on %s files (full_scan=%s, batch_size=%s, progress every %s files)",
            total,
            full_scan,
            batch_size,
            self.cfg.PROGRESS_EVERY,
        )
        self._update_progress("face_index", total_items=total, processed=0, message="Starting")

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
                
                # Process batch when it reaches batch_size or is the last file
                if len(batch) >= batch_size or idx == len(files) - 1:
                    logger.info("Processing batch [%s-%s/%s] (%.1f%%)", 
                               processed + 1, processed + len(batch), total, 
                               ((processed + len(batch)) / total * 100))
                    self._process_batch(batch)
                    processed += len(batch)
                    batch = []
                    
                    if processed % self.cfg.PROGRESS_EVERY == 0 or processed == len(files_to_process):
                        logger.info(
                            "🎯 MILESTONE: %s/%s files completed (%.1f%%)",
                            processed,
                            len(files_to_process),
                            (processed / len(files_to_process) * 100) if len(files_to_process) else 0,
                        )
                        self._update_progress(
                            "face_index",
                            total_items=total,
                            processed=processed,
                            message=f"Processed {processed}/{len(files_to_process)}",
                        )
            else:
                processed += 1

        self._update_progress("face_index", total_items=total, processed=total, message="Completed")
        logger.info("✅ Face indexing complete: processed %s files in batches of %s", len(files_to_process), batch_size)

    def _should_process(self, info: FileInfo) -> bool:
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
        """Process a batch of files for optimal GPU utilization.
        
        This method:
        1. Loads all images in the batch into memory
        2. Performs batch face detection with InsightFace (single GPU call)
        3. Commits all results to database in a single transaction
        """
        if not batch:
            return
        
        logger.debug("Processing batch of %s files", len(batch))
        
        # Detect faces for entire batch (single GPU call)
        if self.use_insightface:
            batch_results = self._detect_faces_insightface_batch(batch)
        else:
            logger.warning("InsightFace is not available. Skipping batch.")
            return
        
        # Commit all results to database in single transaction
        with session_scope() as session:
            for file_info, (faces, pets) in zip(batch, batch_results):
                try:
                    self._store_file_results(session, file_info, faces, pets)
                except Exception as e:
                    logger.error("Failed to store results for %s: %s", file_info.path, e)
                    continue
    
    def _detect_faces_insightface_batch(self, batch: List[FileInfo]) -> List[tuple]:
        """Detect faces and pets in a batch of images for optimal GPU utilization.
        
        Returns:
            List of (face_representations, pet_detections) for each image in the batch
        """
        batch_results = []
        
        # Load all images in batch
        images = []
        valid_indices = []
        for idx, file_info in enumerate(batch):
            try:
                img = cv2.imread(str(file_info.path))
                if img is not None:
                    images.append(img)
                    valid_indices.append(idx)
                else:
                    logger.warning("Could not read image: %s", file_info.path)
                    batch_results.append(([], []))
            except Exception as e:
                logger.error("Error loading %s: %s", file_info.path, e)
                batch_results.append(([], []))
        
        if not images:
            return [([], [])] * len(batch)
        
        # Process all images in single batch (GPU optimization)
        try:
            for idx in range(len(batch)):
                if idx in valid_indices:
                    img_idx = valid_indices.index(idx)
                    img = images[img_idx]
                    
                    # Detect faces
                    detected_faces = self.insightface_app.get(img)
                    faces = []
                    if detected_faces:
                        for face in detected_faces:
                            bbox_xyxy = face.bbox.astype(int)
                            x, y = bbox_xyxy[0], bbox_xyxy[1]
                            w = bbox_xyxy[2] - bbox_xyxy[0]
                            h = bbox_xyxy[3] - bbox_xyxy[1]
                            
                            embedding = face.normed_embedding
                            
                            faces.append({
                                'embedding': embedding.tolist(),
                                'facial_area': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                                'confidence': float(face.det_score)
                            })
                    
                    # Detect pets
                    pets = self._detect_pets_from_img(img, batch[idx].path)
                    
                    batch_results.append((faces, pets))
                else:
                    batch_results.append(([], []))
        except Exception as e:
            logger.error(f"Batch face detection failed: {e}")
            return [([], [])] * len(batch)
        
        return batch_results
    
    def _store_file_results(self, session, file_info: FileInfo, faces: List[dict], pets: List[dict]) -> None:
        """Store face and pet detection results for a single file (called within batch transaction)."""
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

        # Remove previous faces and pets for re-indexing
        session.execute(delete(FaceEmbedding).where(FaceEmbedding.media_id == media.id))
        from models import PetEmbedding
        session.execute(delete(PetEmbedding).where(PetEmbedding.media_id == media.id))

        faces_created = 0
        for idx, rep in enumerate(faces):
            embedding = np.array(rep.get("embedding", []), dtype=np.float32)
            if embedding.size == 0:
                continue

            bbox = rep.get("facial_area") or {}
            confidence = rep.get("face_confidence") or rep.get("confidence")
            thumb_path = self._create_thumbnail(media.path, bbox, media.sha256, idx)

            face = FaceEmbedding(
                media_id=media.id,
                embedding=embedding.tobytes(),
                embedding_dim=embedding.size,
                embedding_index=idx,
                bbox=bbox,
                detection_confidence=confidence,
                model_name=self.model_name,
                detector=self.detector_backend if not self.use_insightface else "SCRFD",
                thumbnail_path=str(thumb_path) if thumb_path else None,
            )
            session.add(face)
            faces_created += 1

        pets_created = 0
        for idx, pet_data in enumerate(pets):
            embedding = pet_data.get("embedding")
            if embedding is None or len(embedding) == 0:
                continue
            
            embedding_array = np.array(embedding, dtype=np.float32)
            bbox = pet_data.get("bbox", {})
            confidence = pet_data.get("confidence", 0.0)
            species = pet_data.get("species", "unknown")
            thumb_path = self._create_thumbnail(media.path, bbox, media.sha256, idx, prefix="pet")
            
            pet = PetEmbedding(
                media_id=media.id,
                embedding=embedding_array.tobytes(),
                embedding_dim=embedding_array.size,
                embedding_index=idx,
                bbox=bbox,
                detection_confidence=confidence,
                species=species,
                model_name="pets.pkl" if self.pet_recognizer else "none",
                detector="yolov8n",
                thumbnail_path=str(thumb_path) if thumb_path else None,
            )
            session.add(pet)
            pets_created += 1
        
        media.face_count = faces_created
        media.pet_count = pets_created
        session.add(media)

        logger.debug("Indexed %s faces, %s pets in %s", faces_created, pets_created, file_info.path)

    def _process_file(self, info: FileInfo) -> None:
        logger.debug("Processing %s", info.path)
        
        # Detect faces using InsightFace or DeepFace
        if self.use_insightface:
            representations = self._detect_faces_insightface(info.path)
        else:
            # representations = self._detect_faces_deepface(info.path)
            logger.warning("InsightFace is not available. Skipping face detection for %s", info.path)
            return
        
        if not isinstance(representations, list):
            logger.warning("Unexpected representation format for %s", info.path)
            return

        # Detect objects (dogs, cats, etc.) if YOLO is available
        detected_pets = []
        if self.yolo_model:
            detected_pets = self._detect_pets(info.path)

        with session_scope() as session:
            media = session.query(MediaFile).filter_by(path=str(info.path)).one_or_none()
            if not media:
                media = MediaFile(path=str(info.path))
            media.size_bytes = info.size
            media.last_scanned_at = datetime.fromtimestamp(info.mtime)
            media.sha256 = self._sha256(info.path)

            width, height = self._get_dimensions(info.path)
            media.width = width
            media.height = height

            session.add(media)
            session.flush()

            # Remove previous faces and pets for re-indexing
            session.execute(delete(FaceEmbedding).where(FaceEmbedding.media_id == media.id))
            from models import PetEmbedding
            session.execute(delete(PetEmbedding).where(PetEmbedding.media_id == media.id))

            faces_created = 0
            for idx, rep in enumerate(representations):
                embedding = np.array(rep.get("embedding", []), dtype=np.float32)
                if embedding.size == 0:
                    continue

                bbox = rep.get("facial_area") or {}
                confidence = rep.get("face_confidence") or rep.get("confidence")
                thumb_path = self._create_thumbnail(media.path, bbox, media.sha256, idx)

                face = FaceEmbedding(
                    media_id=media.id,
                    embedding=embedding.tobytes(),
                    embedding_dim=embedding.size,
                    embedding_index=idx,
                    bbox=bbox,
                    detection_confidence=confidence,
                    model_name=self.model_name,
                    detector=self.detector_backend if not self.use_insightface else "SCRFD",
                    thumbnail_path=str(thumb_path) if thumb_path else None,
                )
                session.add(face)
                faces_created += 1

            media.face_count = faces_created
            
            # Process pet embeddings
            pets_created = 0
            for idx, pet_data in enumerate(detected_pets):
                embedding = pet_data.get("embedding")
                if embedding is None or len(embedding) == 0:
                    continue
                
                embedding_array = np.array(embedding, dtype=np.float32)
                bbox = pet_data.get("bbox", {})
                confidence = pet_data.get("confidence", 0.0)
                species = pet_data.get("species", "unknown")
                thumb_path = self._create_thumbnail(media.path, bbox, media.sha256, idx, prefix="pet")
                
                pet = PetEmbedding(
                    media_id=media.id,
                    embedding=embedding_array.tobytes(),
                    embedding_dim=embedding_array.size,
                    embedding_index=idx,
                    bbox=bbox,
                    detection_confidence=confidence,
                    species=species,
                    model_name="pets.pkl" if self.pet_recognizer else "none",
                    detector="yolov8n",
                    thumbnail_path=str(thumb_path) if thumb_path else None,
                )
                session.add(pet)
                pets_created += 1
            
            media.pet_count = pets_created
            session.add(media)

            logger.debug("Indexed %s faces, %s pets in %s", faces_created, pets_created, info.path)

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _get_dimensions(self, path: Path) -> tuple[int, int]:
        try:
            with Image.open(path) as img:
                return img.width, img.height
        except Exception:
            return 0, 0

    def _create_thumbnail(self, image_path: str, bbox: dict, sha256: Optional[str], idx: int, prefix: str = "face") -> Optional[Path]:
        """Create a thumbnail of the cropped region. prefix can be 'face' or 'pet'."""
        if not bbox:
            return None

        image_file = Path(image_path)
        cache_dir = Path(self.cfg.THUMB_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        thumb_name = f"{prefix}_{sha256 or image_file.stem}_{idx}.jpg"
        thumb_path = cache_dir / thumb_name

        if thumb_path.exists():
            return thumb_path

        try:
            with Image.open(image_file) as img:
                left = bbox.get("x", bbox.get("left", 0))
                top = bbox.get("y", bbox.get("top", 0))
                width = bbox.get("w", bbox.get("width", 0))
                height = bbox.get("h", bbox.get("height", 0))

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
        with session_scope() as session:
            task = session.query(TaskProgress).filter_by(task_name=task_name).one_or_none()
            if not task:
                task = TaskProgress(task_name=task_name)

            task.total_items = total_items
            task.processed_items = processed
            task.last_message = message
            session.add(task)

    def _detect_faces_insightface(self, image_path: Path) -> List[dict]:
        """Detect faces using AuraFace (InsightFace-compatible)."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning("Could not read image: %s", image_path)
                return []
            
            # AuraFace returns faces with normed_embedding already included
            detected_faces = self.insightface_app.get(img)
            if not detected_faces:
                return []

            # Format representations for database storage
            representations = []
            for face in detected_faces:
                bbox_xyxy = face.bbox.astype(int)
                x, y = bbox_xyxy[0], bbox_xyxy[1]
                w = bbox_xyxy[2] - bbox_xyxy[0]
                h = bbox_xyxy[3] - bbox_xyxy[1]
                
                # Use the normed_embedding directly (already L2-normalized)
                embedding = face.normed_embedding
                
                representations.append({
                    'embedding': embedding.tolist(),
                    'facial_area': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'confidence': float(face.det_score)
                })
            
            return representations
        except Exception as e:
            logger.error(f"AuraFace detection failed for {image_path}: {e}")
            return []

    def _detect_faces_deepface(self, image_path: Path) -> List[dict]:
        """Fallback: Detect faces using DeepFace."""
        # from deepface import DeepFace
        # try:
        #     return DeepFace.represent(
        #         img_path=str(image_path),
        #         model_name=self.model_name,
        #         detector_backend=self.detector_backend,
        #         enforce_detection=False,
        #         align=True,
        #     )
        # except Exception as exc:
        #     logger.error("Failed to detect faces in %s: %s", image_path, exc)
        #     return []
        logger.warning("DeepFace fallback is disabled. Skipping face detection.")
        return []

    def _detect_pets(self, image_path: Path, conf_threshold: float = 0.4) -> List[dict]:
        """Detect pets (dogs, cats) using YOLOv12 and extract embeddings."""
        if not self.yolo_model:
            return []
        
        try:
            results = self.yolo_model(str(image_path), conf=conf_threshold, verbose=False)
            detected_pets = []
            
            if results:
                names = self.yolo_model.names
                img = cv2.imread(str(image_path))
                if img is None:
                    return []
                
                for r in results:
                    for i, c in enumerate(r.boxes.cls):
                        label = names[int(c)]
                        confidence = float(r.boxes.conf[i])
                        bbox_xyxy = r.boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                        
                        # Only process dogs and cats
                        if label not in ['dog', 'cat']:
                            continue
                        
                        # Convert YOLO bbox to our format
                        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
                        bbox = {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}
                        
                        # Extract pet crop for embedding
                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        
                        # Get embedding from fastai model if available
                        embedding = None
                        if self.pet_recognizer:
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
    
    def _detect_pets_from_img(self, img: np.ndarray, image_path: Path, conf_threshold: float = 0.4) -> List[dict]:
        """Detect pets from an already loaded image."""
        if not self.yolo_model:
            return []
        
        try:
            results = self.yolo_model(img, conf=conf_threshold, verbose=False)
            detected_pets = []
            
            if results:
                names = self.yolo_model.names
                for r in results:
                    for i, c in enumerate(r.boxes.cls):
                        label = names[int(c)]
                        confidence = float(r.boxes.conf[i])
                        bbox_xyxy = r.boxes.xyxy[i].tolist()
                        
                        if label not in ['dog', 'cat']:
                            continue
                        
                        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
                        bbox = {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}
                        
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
            
            # Get embedding from model (before final classification layer)
            with torch.no_grad():
                # This assumes pets.pkl was trained with fastai and has .model attribute
                # The embedding is typically from the penultimate layer
                # You'll need to adjust this based on your actual fastai architecture
                embedding = self.pet_recognizer.model[:-1](pil_img.unsqueeze(0))
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Pet embedding extraction failed: {e}")
            return None

    def _detect_objects(self, image_path: Path, conf_threshold: float = 0.4) -> List[dict]:
        """Detect objects (dogs, cats, etc.) using YOLO - DEPRECATED, use _detect_pets instead."""
        if not self.yolo_model:
            return []
        
        try:
            results = self.yolo_model(str(image_path), conf=conf_threshold, verbose=False)
            detected_objects = []
            
            if results:
                names = self.yolo_model.names
                for r in results:
                    for i, c in enumerate(r.boxes.cls):
                        label = names[int(c)]
                        confidence = float(r.boxes.conf[i])
                        bbox = r.boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                        # Skip 'person' to avoid overlap with face detector (InsightFace)
                        if label == 'person':
                            continue
                        detected_objects.append({
                            'label': label,
                            'confidence': confidence,
                            'bbox': bbox
                        })
            
            return detected_objects
        except Exception as e:
            logger.error(f"Object detection failed for {image_path}: {e}")
            return []


if __name__ == "__main__":  # pragma: no cover
    indexer = FaceIndexer()
    indexer.run(full_scan=False)

    # Update test image path to user-provided folder
    test_images = list(Path('C:/Users/USERNAME/Downloads/Max').glob('*.jpg')) if Path('C:/Users/USERNAME/Downloads/Max').exists() else []
