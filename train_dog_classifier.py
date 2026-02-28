"""Custom Dog Classifier Training Module

Train a fastai model to distinguish between individual dogs.

Setup Instructions:
1. Create training folders:
   dog_classifier/
       my_dog/          (30-50 clear photos of your dog)
       dads_dog/        (30-50 clear photos of dad's dog)
       other_dog/       (optional - other dogs for better generalization)

2. Run: python train_dog_classifier.py --data-dir dog_classifier --export-path dog_model.pkl

3. The trained model will be saved as dog_model.pkl
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_directory(data_dir: Path) -> bool:
    """
    Validate that the data directory has the correct structure.
    
    Expected:
        data_dir/
            class1/
                img1.jpg
                img2.jpg
                ...
            class2/
                img1.jpg
                ...
    """
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if len(subdirs) < 2:
        logger.error(f"Need at least 2 dog classes. Found {len(subdirs)} in {data_dir}")
        logger.info("Create folders like: my_dog/, dads_dog/")
        return False
    
    # Check each class has images
    total_images = 0
    for subdir in subdirs:
        image_files = list(subdir.glob('*.jpg')) + list(subdir.glob('*.jpeg')) + \
                     list(subdir.glob('*.png')) + list(subdir.glob('*.JPEG')) + \
                     list(subdir.glob('*.JPG'))
        
        num_images = len(image_files)
        total_images += num_images
        
        logger.info(f"Class '{subdir.name}': {num_images} images")
        
        if num_images < 10:
            logger.warning(f"Class '{subdir.name}' has only {num_images} images. "
                          f"Recommend at least 30-50 for good accuracy.")
    
    if total_images < 30:
        logger.error(f"Only {total_images} total images. Recommend at least 60+ (30 per dog)")
        return False
    
    logger.info(f"✓ Data directory valid: {len(subdirs)} classes, {total_images} total images")
    return True


def train_dog_classifier(data_dir: Path,
                        export_path: Path,
                        epochs: int = 10,
                        img_size: int = 224,
                        batch_size: int = 16,
                        valid_pct: float = 0.2) -> Optional[Path]:
    """
    Train a fastai ResNet34 model to identify different dogs.
    
    Args:
        data_dir: Path to data directory with dog class folders
        export_path: Where to save the trained model (.pkl)
        epochs: Number of training epochs
        img_size: Image size (224 is standard for ResNet)
        batch_size: Batch size for training
        valid_pct: Percentage of data for validation (0.2 = 20%)
    
    Returns:
        Path to exported model if successful
    """
    try:
        from fastai.vision.all import (
            ImageDataLoaders, vision_learner, resnet34, 
            accuracy, error_rate, Resize, RandomResizedCrop,
            aug_transforms
        )
    except ImportError:
        logger.error("fastai not installed. Run: pip install fastai")
        return None
    
    if not check_data_directory(data_dir):
        return None
    
    logger.info("=" * 60)
    logger.info("Training Custom Dog Classifier")
    logger.info("=" * 60)
    
    try:
        # 1. Create DataLoaders with augmentation for better generalization
        logger.info("Loading and augmenting images...")
        dls = ImageDataLoaders.from_folder(
            data_dir,
            valid_pct=valid_pct,
            seed=42,
            item_tfms=Resize(img_size + 32),  # Slightly larger for cropping
            batch_tfms=aug_transforms(
                size=img_size,
                max_rotate=10.0,  # Small rotations
                max_lighting=0.2,  # Lighting changes
                max_warp=0.0,     # No perspective warp (keeps dog recognizable)
                p_affine=0.75,
                p_lighting=0.75
            ),
            bs=batch_size
        )
        
        # Show sample
        logger.info(f"Classes found: {dls.vocab}")
        logger.info(f"Training samples: {len(dls.train_ds)}")
        logger.info(f"Validation samples: {len(dls.valid_ds)}")
        
        # 2. Create learner with ResNet34 (good balance of accuracy and speed)
        logger.info("Creating model (ResNet34)...")
        learn = vision_learner(dls, resnet34, metrics=[error_rate, accuracy])
        
        # 3. Find optimal learning rate
        logger.info("Finding optimal learning rate...")
        lr_min, lr_steep = learn.lr_find()
        logger.info(f"Suggested LR: {lr_steep:.2e}")
        
        # 4. Fine-tune the model
        logger.info(f"Training for {epochs} epochs...")
        logger.info("Phase 1: Train head (last layers)")
        learn.fine_tune(epochs, base_lr=lr_steep)
        
        # 5. Show results
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        
        # Get final metrics
        final_error = learn.validate()[1]  # error_rate
        final_accuracy = 1.0 - final_error
        
        logger.info(f"Final Accuracy: {final_accuracy*100:.2f}%")
        logger.info(f"Final Error Rate: {final_error*100:.2f}%")
        
        # Show confusion matrix (which dogs are being confused)
        logger.info("\nConfusion Matrix:")
        interp = learn.interpret()
        interp.plot_confusion_matrix(figsize=(6, 6))
        
        # 6. Export the model
        logger.info(f"\nExporting model to {export_path}...")
        learn.export(export_path)
        
        logger.info("✓ Model saved successfully!")
        logger.info(f"Use this model in photo-organizer.py with: --dog-classifier {export_path}")
        
        return export_path
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_classifier(model_path: Path, test_image: Path):
    """
    Test the trained classifier on a single image.
    
    Args:
        model_path: Path to exported .pkl model
        test_image: Path to test image
    """
    try:
        from fastai.vision.all import load_learner
        from PIL import Image
    except ImportError:
        logger.error("fastai not installed")
        return
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    if not test_image.exists():
        logger.error(f"Test image not found: {test_image}")
        return
    
    logger.info(f"Loading model from {model_path}...")
    learn = load_learner(model_path)
    
    logger.info(f"Testing on {test_image}...")
    pred, pred_idx, probs = learn.predict(str(test_image))
    
    logger.info("\n" + "=" * 60)
    logger.info("Prediction Results")
    logger.info("=" * 60)
    logger.info(f"Predicted: {pred}")
    logger.info(f"Confidence: {max(probs)*100:.2f}%")
    logger.info("\nAll probabilities:")
    for i, class_name in enumerate(learn.dls.vocab):
        logger.info(f"  {class_name}: {probs[i]*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Train a custom dog classifier using fastai',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train with default settings
  python train_dog_classifier.py --data-dir dog_classifier

  # Train with more epochs for better accuracy
  python train_dog_classifier.py --data-dir dog_classifier --epochs 20

  # Test the trained model
  python train_dog_classifier.py --test --model dog_model.pkl --image test_dog.jpg

Setup:
  1. Create folders: dog_classifier/my_dog/, dog_classifier/dads_dog/
  2. Add 30-50 clear photos of each dog
  3. Run training command above
  4. Use trained model with: --dog-classifier dog_model.pkl
        '''
    )
    
    parser.add_argument('--data-dir', type=Path, default=Path('dog_classifier'),
                       help='Directory containing dog class folders')
    parser.add_argument('--export-path', type=Path, default=Path('dog_model.pkl'),
                       help='Where to save the trained model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size for training (default: 224)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16, reduce if out of memory)')
    parser.add_argument('--valid-pct', type=float, default=0.2,
                       help='Validation percentage (default: 0.2)')
    
    # Testing mode
    parser.add_argument('--test', action='store_true',
                       help='Test a trained model on a single image')
    parser.add_argument('--model', type=Path,
                       help='Path to trained model (.pkl) for testing')
    parser.add_argument('--image', type=Path,
                       help='Path to test image')
    
    args = parser.parse_args()
    
    if args.test:
        if not args.model or not args.image:
            parser.error("--test requires --model and --image")
        test_classifier(args.model, args.image)
    else:
        result = train_dog_classifier(
            data_dir=args.data_dir,
            export_path=args.export_path,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            valid_pct=args.valid_pct
        )
        
        if result:
            logger.info("\n" + "=" * 60)
            logger.info("Next Steps:")
            logger.info("=" * 60)
            logger.info("1. Test the model:")
            logger.info(f"   python train_dog_classifier.py --test --model {result} --image your_dog_photo.jpg")
            logger.info("\n2. Use in photo organizer:")
            logger.info(f"   python photo-organizer.py --dog-classifier {result} --path ...")


if __name__ == '__main__':
    main()
