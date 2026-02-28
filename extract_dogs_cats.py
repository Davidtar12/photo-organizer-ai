"""Extract all photos containing dogs and cats using YOLOv8

This script scans your photo library and copies all photos containing dogs or cats
to separate folders, making it easy to then organize them for training.
"""

import argparse
import logging
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Image extensions to scan
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.raw', '.cr2', 
                   '.nef', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.jfif'}


def scan_images(root_dir: Path, max_files: int = None):
    """Find all image files in the directory."""
    logger.info(f"Scanning for images in {root_dir}...")
    
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(root_dir.rglob(f'*{ext}'))
        images.extend(root_dir.rglob(f'*{ext.upper()}'))
    
    if max_files:
        images = images[:max_files]
    
    logger.info(f"Found {len(images)} images to process")
    return images


def detect_animals(image_path: Path, model, conf_threshold: float = 0.5):
    """
    Detect dogs and cats in an image using YOLOv8.
    
    Returns:
        dict with keys 'has_dog', 'has_cat', 'dog_count', 'cat_count', 'dog_boxes', 'cat_boxes'
    """
    result = {
        'has_dog': False,
        'has_cat': False,
        'dog_count': 0,
        'cat_count': 0,
        'dog_boxes': [],
        'cat_boxes': []
    }
    
    try:
        # Run YOLOv8 detection
        results = model(str(image_path), conf=conf_threshold, verbose=False)
        
        if results:
            names = model.names
            for r in results:
                for i, c in enumerate(r.boxes.cls):
                    label = names[int(c)]
                    confidence = float(r.boxes.conf[i])
                    bbox = r.boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                    
                    if label == 'dog':
                        result['has_dog'] = True
                        result['dog_count'] += 1
                        result['dog_boxes'].append({'bbox': bbox, 'confidence': confidence})
                    elif label == 'cat':
                        result['has_cat'] = True
                        result['cat_count'] += 1
                        result['cat_boxes'].append({'bbox': bbox, 'confidence': confidence})
    
    except Exception as e:
        logger.debug(f"Detection failed for {image_path}: {e}")
    
    return result


def crop_and_save_animal(image_path: Path, bbox: list, output_path: Path, animal_type: str, index: int):
    """
    Crop the detected animal from the image and save it.
    
    Args:
        image_path: Original image path
        bbox: [x1, y1, x2, y2] bounding box
        output_path: Directory to save cropped image
        animal_type: 'dog' or 'cat'
        index: Index for multiple animals in same image
    """
    try:
        img = Image.open(image_path)
        x1, y1, x2, y2 = bbox
        
        # Crop the animal
        cropped = img.crop((x1, y1, x2, y2))
        
        # Create filename: original_name_dog_1.jpg
        stem = image_path.stem
        ext = image_path.suffix if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png'] else '.jpg'
        output_filename = f"{stem}_{animal_type}_{index}{ext}"
        
        # Save
        output_file = output_path / output_filename
        cropped.save(output_file, quality=95)
        
        return output_file
    
    except Exception as e:
        logger.warning(f"Failed to crop {animal_type} from {image_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract all photos containing dogs and cats using YOLOv8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Extract from photo library
  python extract_dogs_cats.py --path "C:\\Users\\david\\OneDrive\\Documents\\Pictures" --output dog_cat_extraction

  # Quick test on first 1000 photos
  python extract_dogs_cats.py --path "C:\\Users\\david\\OneDrive\\Documents\\Pictures" --output dog_cat_extraction --max-files 1000

  # Also save cropped versions
  python extract_dogs_cats.py --path "C:\\Users\\david\\OneDrive\\Documents\\Pictures" --output dog_cat_extraction --crop

After extraction:
  1. Review dog_cat_extraction/dogs/ folder
  2. Move your dog photos to: dog_classifier/my_dog/
  3. Move other dogs to: dog_classifier/other_dogs/
  4. Train classifier: python train_dog_classifier.py --data-dir dog_classifier
        '''
    )
    
    parser.add_argument('--path', type=Path, required=True,
                       help='Root directory to scan for photos')
    parser.add_argument('--output', type=Path, default=Path('dog_cat_extraction'),
                       help='Output directory (default: dog_cat_extraction)')
    parser.add_argument('--max-files', type=int,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for detection (0-1, default: 0.5)')
    parser.add_argument('--crop', action='store_true',
                       help='Also save cropped versions of detected animals')
    parser.add_argument('--copy-originals', action='store_true', default=True,
                       help='Copy original photos to output folders (default: True)')
    
    args = parser.parse_args()
    
    # Create output directories
    dogs_dir = args.output / 'dogs'
    cats_dir = args.output / 'cats'
    both_dir = args.output / 'dogs_and_cats'
    dogs_cropped_dir = args.output / 'dogs_cropped'
    cats_cropped_dir = args.output / 'cats_cropped'
    
    dogs_dir.mkdir(parents=True, exist_ok=True)
    cats_dir.mkdir(parents=True, exist_ok=True)
    both_dir.mkdir(parents=True, exist_ok=True)
    
    if args.crop:
        dogs_cropped_dir.mkdir(parents=True, exist_ok=True)
        cats_cropped_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLOv8 model
    logger.info("Loading YOLOv8 model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Nano version for speed
        logger.info("✓ YOLOv8 loaded successfully")
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return 1
    except Exception as e:
        logger.error(f"Failed to load YOLOv8: {e}")
        return 1
    
    # Scan for images
    images = scan_images(args.path, args.max_files)
    
    if not images:
        logger.error("No images found!")
        return 1
    
    # Process images
    stats = defaultdict(int)
    
    logger.info(f"Processing {len(images)} images...")
    logger.info("=" * 60)
    
    for image_path in tqdm(images, desc="Detecting animals"):
        result = detect_animals(image_path, model, args.conf_threshold)
        
        # Skip if no dogs or cats
        if not result['has_dog'] and not result['has_cat']:
            continue
        
        # Determine destination
        if result['has_dog'] and result['has_cat']:
            dest_dir = both_dir
            stats['both'] += 1
        elif result['has_dog']:
            dest_dir = dogs_dir
            stats['dogs_only'] += 1
        else:
            dest_dir = cats_dir
            stats['cats_only'] += 1
        
        # Copy original photo
        if args.copy_originals:
            try:
                dest_file = dest_dir / image_path.name
                # Handle name conflicts
                counter = 1
                while dest_file.exists():
                    dest_file = dest_dir / f"{image_path.stem}_{counter}{image_path.suffix}"
                    counter += 1
                
                shutil.copy2(image_path, dest_file)
                stats['copied'] += 1
            except Exception as e:
                logger.warning(f"Failed to copy {image_path}: {e}")
        
        # Crop and save individual animals
        if args.crop:
            for idx, dog_box in enumerate(result['dog_boxes'], 1):
                crop_and_save_animal(image_path, dog_box['bbox'], dogs_cropped_dir, 'dog', idx)
                stats['dogs_cropped'] += 1
            
            for idx, cat_box in enumerate(result['cat_boxes'], 1):
                crop_and_save_animal(image_path, cat_box['bbox'], cats_cropped_dir, 'cat', idx)
                stats['cats_cropped'] += 1
        
        stats['total_dogs'] += result['dog_count']
        stats['total_cats'] += result['cat_count']
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Extraction Complete!")
    logger.info("=" * 60)
    logger.info(f"Total images processed: {len(images)}")
    logger.info(f"Photos with dogs only: {stats['dogs_only']}")
    logger.info(f"Photos with cats only: {stats['cats_only']}")
    logger.info(f"Photos with both: {stats['both']}")
    logger.info(f"Total photos copied: {stats['copied']}")
    logger.info(f"Total dogs detected: {stats['total_dogs']}")
    logger.info(f"Total cats detected: {stats['total_cats']}")
    
    if args.crop:
        logger.info(f"Dogs cropped: {stats['dogs_cropped']}")
        logger.info(f"Cats cropped: {stats['cats_cropped']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Next Steps:")
    logger.info("=" * 60)
    logger.info(f"1. Review extracted photos in: {args.output}")
    logger.info(f"   - Dogs: {dogs_dir}")
    logger.info(f"   - Cats: {cats_dir}")
    logger.info(f"   - Both: {both_dir}")
    
    if args.crop:
        logger.info(f"   - Cropped dogs: {dogs_cropped_dir}")
        logger.info(f"   - Cropped cats: {cats_cropped_dir}")
    
    logger.info("\n2. Organize for training:")
    logger.info("   - Create: dog_classifier/my_dog/")
    logger.info("   - Create: dog_classifier/other_dogs/")
    logger.info("   - Move your dog photos to my_dog/")
    logger.info("   - Move other dogs to other_dogs/")
    
    logger.info("\n3. Train the classifier:")
    logger.info("   python train_dog_classifier.py --data-dir dog_classifier")


if __name__ == '__main__':
    main()
