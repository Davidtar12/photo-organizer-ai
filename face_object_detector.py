"""Face and Object Detection Module

Detects and identifies:
1. Human faces using DeepFace (ArcFace model)
2. Dogs and cats using YOLOv8 (object detection / bounding boxes)
3. Dog identity using DogFaceNet (embedding + nearest-neighbor) when available,
   with automatic fallback to a custom fastai classifier if DogFaceNet isn't loaded

Recommended combined flow for pet identification:
- YOLO detects and crops dog faces (or dog regions)
- DogFaceNet identifies which specific dog by comparing embeddings to a gallery
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
import sqlite3
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Global model instances (loaded once at startup)
DEEPFACE_AVAILABLE = False
INSIGHTFACE_AVAILABLE = False
YOLO_AVAILABLE = False
DOG_CLASSIFIER_AVAILABLE = False
DOGFACENET_AVAILABLE = False

deepface = None
insightface_app = None
yolo_model = None
dog_classifier = None
dogfacenet_session = None  # onnxruntime.InferenceSession, if loaded
dog_gallery = None         # Optional identification gallery for DogFaceNet


def initialize_models(
    dog_classifier_path: Optional[Path] = None,
    dogfacenet_onnx_path: Optional[Path] = None,
    dog_gallery_path: Optional[Path] = None,
):
    """
    Initialize all detection models. Call this ONCE in main() before parallel processing.
    
    Args:
        dog_classifier_path: Path to custom fastai dog classifier (.pkl file)
        dogfacenet_onnx_path: Optional path to DogFaceNet ONNX model for dog identification
        dog_gallery_path: Optional path to a JSON file with known dog embeddings for identification
    """
    global DEEPFACE_AVAILABLE, INSIGHTFACE_AVAILABLE, YOLO_AVAILABLE, DOG_CLASSIFIER_AVAILABLE, DOGFACENET_AVAILABLE
    global deepface, insightface_app, yolo_model, dog_classifier, dogfacenet_session, dog_gallery
    
    # 1. Load InsightFace for human face recognition (preferred)
    try:
        import insightface
        insightface_app = insightface.app.FaceAnalysis(
            name='buffalo_l',  # Contains SCRFD-10GKP detector
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        insightface_app.prepare(ctx_id=0, det_size=(640, 640))
        INSIGHTFACE_AVAILABLE = True
        logger.info("InsightFace loaded successfully (using SCRFD-10G detector + ArcFace recognition)")
    except Exception as e:
        logger.warning(f"Could not load InsightFace: {e}")
        INSIGHTFACE_AVAILABLE = False
        
        # Fallback to DeepFace if InsightFace unavailable
        try:
            from deepface import DeepFace
            deepface = DeepFace
            DEEPFACE_AVAILABLE = True
            logger.info("DeepFace loaded as fallback (using ArcFace model)")
        except Exception as e2:
            logger.warning(f"Could not load DeepFace fallback: {e2}")
            DEEPFACE_AVAILABLE = False
    
    # 2. Load YOLOv8 for object detection
    try:
        from ultralytics import YOLO
        # Download yolov8n.pt (nano - smallest and fastest) on first run
        yolo_model = YOLO('yolov8n.pt')
        YOLO_AVAILABLE = True
        logger.info("YOLOv8 model loaded successfully (nano version)")
    except Exception as e:
        logger.warning(f"Could not load YOLOv8: {e}")
        YOLO_AVAILABLE = False
    
    # 3. Load custom dog classifier if provided
    if dog_classifier_path and dog_classifier_path.exists():
        try:
            from fastai.vision.all import load_learner
            dog_classifier = load_learner(dog_classifier_path)
            DOG_CLASSIFIER_AVAILABLE = True
            logger.info(f"Custom dog classifier loaded from {dog_classifier_path}")
        except Exception as e:
            logger.warning(f"Could not load custom dog classifier: {e}")
            DOG_CLASSIFIER_AVAILABLE = False

    # 4. Load DogFaceNet (ONNX) if provided
    if dogfacenet_onnx_path and dogfacenet_onnx_path.exists():
        try:
            import onnxruntime as ort
            providers = []
            try:
                # Prefer CUDA if available
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                _ = ort.get_device()
            except Exception:
                providers = ['CPUExecutionProvider']
            dogfacenet_session = ort.InferenceSession(str(dogfacenet_onnx_path), providers=providers)
            DOGFACENET_AVAILABLE = True
            logger.info(f"DogFaceNet ONNX model loaded from {dogfacenet_onnx_path}")

            # Optionally load a gallery of known dogs (JSON: list of {label, embedding})
            if dog_gallery_path and dog_gallery_path.exists():
                try:
                    with open(dog_gallery_path, 'r', encoding='utf-8') as f:
                        dog_gallery = json.load(f)
                    logger.info(f"DogFaceNet gallery loaded from {dog_gallery_path} (entries: {len(dog_gallery)})")
                except Exception as ge:
                    logger.warning(f"Could not load dog gallery from {dog_gallery_path}: {ge}")
        except Exception as e:
            logger.warning(f"Could not load DogFaceNet ONNX model: {e}")
            DOGFACENET_AVAILABLE = False
    
    return {
        'deepface': DEEPFACE_AVAILABLE,
        'insightface': INSIGHTFACE_AVAILABLE,
        'yolo': YOLO_AVAILABLE,
        'dog_classifier': DOG_CLASSIFIER_AVAILABLE,
        'dogfacenet': DOGFACENET_AVAILABLE
    }


def detect_faces(image_path: Path) -> List[Dict[str, Any]]:
    """
    Detect all human faces in an image using InsightFace (SCRFD-10G) or DeepFace fallback.
    
    Returns:
        List of face dictionaries containing:
            - embedding: 512-dimensional face embedding vector (for comparison)
            - bbox: (x, y, w, h) bounding box
            - confidence: detection confidence (0-1)
            - facial_area: dict with detailed bbox info
    """
    # Prefer InsightFace (faster and more accurate)
    if INSIGHTFACE_AVAILABLE:
        return _detect_faces_insightface(image_path)
    elif DEEPFACE_AVAILABLE:
        return _detect_faces_deepface(image_path)
    else:
        return []


def _detect_faces_insightface(image_path: Path) -> List[Dict[str, Any]]:
    """Detect faces using InsightFace SCRFD-10G detector."""
    import cv2
    import numpy as np
    
    faces = []
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return []
        
        # Get faces from InsightFace
        detected_faces = insightface_app.get(img)
        
        for i, face in enumerate(detected_faces):
            # Extract bounding box in (x, y, w, h) format
            bbox_xyxy = face.bbox.astype(int)  # [x1, y1, x2, y2]
            x, y = bbox_xyxy[0], bbox_xyxy[1]
            w = bbox_xyxy[2] - bbox_xyxy[0]
            h = bbox_xyxy[3] - bbox_xyxy[1]
            
            faces.append({
                'face_id': i,
                'embedding': face.embedding.tolist(),  # 512-dim ArcFace embedding
                'facial_area': {
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h)
                },
                'confidence': float(face.det_score)
            })
        
        if faces:
            logger.debug(f"Detected {len(faces)} face(s) in {image_path.name} with InsightFace")
    
    except Exception as e:
        logger.debug(f"InsightFace detection failed for {image_path}: {e}")
    
    return faces


def _detect_faces_deepface(image_path: Path) -> List[Dict[str, Any]]:
    """Fallback: Detect faces using DeepFace."""
    faces = []
    try:
        # Use ArcFace model - state-of-the-art for face recognition
        # enforce_detection=False prevents crashes on photos without faces
        result = deepface.represent(
            img_path=str(image_path),
            model_name='ArcFace',
            enforce_detection=False,
            detector_backend='opencv'  # Faster than default
        )
        
        # DeepFace returns a list of detected faces
        for i, face_data in enumerate(result):
            faces.append({
                'face_id': i,
                'embedding': face_data['embedding'],  # 512-dim vector for comparison
                'facial_area': face_data.get('facial_area', {}),
                'confidence': face_data.get('face_confidence', 0.0)
            })
        
        if faces:
            logger.debug(f"Detected {len(faces)} face(s) in {image_path.name} with DeepFace")
    
    except Exception as e:
        logger.debug(f"DeepFace detection failed for {image_path}: {e}")
    
    return faces


def detect_objects(image_path: Path, conf_threshold: float = 0.4) -> List[Dict[str, Any]]:
    """
    Detect objects (including dogs and cats) using YOLOv8.
    
    Args:
        image_path: Path to image file
        conf_threshold: Minimum confidence threshold (0-1)
    
    Returns:
        List of detected objects with:
            - label: object class name (e.g., 'dog', 'cat', 'person')
            - confidence: detection confidence
            - bbox: (x1, y1, x2, y2) bounding box coordinates
    """
    if not YOLO_AVAILABLE:
        return []
    
    detected_objects = []
    try:
        # Run YOLOv8 detection
        results = yolo_model(str(image_path), conf=conf_threshold, verbose=False)
        
        if results:
            names = yolo_model.names
            for r in results:
                for i, c in enumerate(r.boxes.cls):
                    label = names[int(c)]
                    if label == 'person':
                        continue  # Skip 'person' label, as it's handled by face detection
                    
                    confidence = float(r.boxes.conf[i])
                    bbox = r.boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                    
                    detected_objects.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': bbox
                    })
        
        if detected_objects:
            logger.debug(f"Detected {len(detected_objects)} objects in {image_path.name}: "
                        f"{', '.join(set(obj['label'] for obj in detected_objects))}")
    
    except Exception as e:
        logger.debug(f"Object detection failed for {image_path}: {e}")
    
    return detected_objects


def identify_dogs(image_path: Path, dog_bboxes: List[List[float]]) -> List[str]:
    """
    Identify specific dogs. Tries DogFaceNet + gallery (if available),
    falls back to custom fastai classifier otherwise.
    
    Args:
        image_path: Path to image file
        dog_bboxes: List of bounding boxes [[x1, y1, x2, y2], ...] for detected dogs
    
    Returns:
        List of dog identities (e.g., ['my_dog', 'dads_dog'])
    """
    if not dog_bboxes:
        return []
    
    # Prefer DogFaceNet if available and gallery present
    if DOGFACENET_AVAILABLE:
        try:
            return identify_dogs_dogfacenet(image_path, dog_bboxes)
        except Exception as e:
            logger.warning(f"DogFaceNet identification failed, falling back if possible: {e}")

    # Fallback to fastai classifier if available
    if DOG_CLASSIFIER_AVAILABLE:
        dog_identities = []
        try:
            img = Image.open(image_path)
            for bbox in dog_bboxes:
                x1, y1, x2, y2 = bbox
                cropped_dog = img.crop((x1, y1, x2, y2))
                pred, pred_idx, probs = dog_classifier.predict(cropped_dog)
                dog_identities.append(str(pred))
                logger.debug(
                    f"Identified dog in {image_path.name} as '{pred}' (confidence: {float(max(probs)):.2f})"
                )
            return dog_identities
        except Exception as e:
            logger.warning(f"Dog identification (fastai) failed for {image_path}: {e}")

    return []


def _preprocess_dogfacenet(crop: Image.Image, size: int = 224) -> "np.ndarray":
    """Preprocess a cropped dog image for DogFaceNet ONNX model.

    Note: Input normalization may vary per model. This uses a generic 0-1 scaling.
    Adjust mean/std if your specific DogFaceNet requires it.
    """
    import numpy as np

    img = crop.convert('RGB').resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # CHW layout expected by most ONNX vision models
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)  # NCHW
    return arr


def _cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    import numpy as np
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def identify_dogs_dogfacenet(image_path: Path, dog_bboxes: List[List[float]],
                             top_k: int = 1, similarity_threshold: float = 0.60) -> List[str]:
    """Identify dogs using DogFaceNet embeddings compared against a gallery.

    Expects a gallery loaded in `dog_gallery` with entries like:
    [ { "label": "my_dog", "embedding": [..] }, ... ]

    Args:
        image_path: Path to the source image
        dog_bboxes: list of [x1, y1, x2, y2] boxes
        top_k: how many top labels to return per box (default 1)
        similarity_threshold: minimum cosine similarity to accept a match
    Returns:
        List of predicted labels (length equals number of bboxes). If no match, returns "unknown".
    """
    import numpy as np

    if not DOGFACENET_AVAILABLE or dogfacenet_session is None:
        raise RuntimeError("DogFaceNet session not available")

    # Build gallery arrays
    gallery_labels: List[str] = []
    gallery_embs: List[np.ndarray] = []
    if isinstance(dog_gallery, list):
        for entry in dog_gallery:
            if 'label' in entry and 'embedding' in entry:
                gallery_labels.append(entry['label'])
                gallery_embs.append(np.array(entry['embedding'], dtype=np.float32))
    elif isinstance(dog_gallery, dict) and 'labels' in dog_gallery and 'embeddings' in dog_gallery:
        gallery_labels = list(dog_gallery['labels'])
        gallery_embs = [np.array(e, dtype=np.float32) for e in dog_gallery['embeddings']]

    if not gallery_labels or not gallery_embs:
        logger.warning("DogFaceNet gallery is empty; returning unknown for all dogs")
        return ["unknown" for _ in dog_bboxes]

    img = Image.open(image_path)
    input_name = dogfacenet_session.get_inputs()[0].name
    output_name = dogfacenet_session.get_outputs()[0].name

    predictions: List[str] = []
    for bbox in dog_bboxes:
        x1, y1, x2, y2 = bbox
        crop = img.crop((x1, y1, x2, y2))
        inp = _preprocess_dogfacenet(crop)
        # Run inference to get embedding
        out = dogfacenet_session.run([output_name], {input_name: inp})[0]
        emb = out[0].astype(np.float32)

        # Compare with gallery by cosine similarity
        best_label = "unknown"
        best_score = -1.0
        for label, g_emb in zip(gallery_labels, gallery_embs):
            score = _cosine_similarity(emb, g_emb)
            if score > best_score:
                best_score = score
                best_label = label

        if best_score >= similarity_threshold:
            predictions.append(best_label)
        else:
            predictions.append("unknown")

    return predictions


def process_image_for_detection(image_path: Path) -> Dict[str, Any]:
    """
    Run all detection tasks on a single image.
    
    Returns:
        Dictionary containing:
            - faces: List of detected faces with embeddings
            - objects: List of detected objects
            - dogs: List of identified dogs
            - has_person: bool
            - has_dog: bool
            - has_cat: bool
    """
    result = {
        'faces': [],
        'objects': [],
        'dogs': [],
        'has_person': False,
        'has_dog': False,
        'has_cat': False
    }
    
    # 1. Detect human faces
    result['faces'] = detect_faces(image_path)
    
    # 2. Detect objects (dogs, cats, etc.)
    result['objects'] = detect_objects(image_path)
    
    # 3. Extract dog bounding boxes for custom identification
    dog_bboxes = [obj['bbox'] for obj in result['objects'] if obj['label'] == 'dog']
    if dog_bboxes and (DOG_CLASSIFIER_AVAILABLE or DOGFACENET_AVAILABLE):
        result['dogs'] = identify_dogs(image_path, dog_bboxes)
    
    # 4. Set convenience flags
    object_labels = {obj['label'] for obj in result['objects']}
    result['has_person'] = len(result['faces']) > 0
    result['has_dog'] = 'dog' in object_labels
    result['has_cat'] = 'cat' in object_labels
    
    return result


# ============================================================================
# (Legacy database management code removed. The main web application's
# SQLAlchemy models and services now handle all database operations.)
# ============================================================================
