"""
ULTRA-SAFE Duplicate Cleanup Script
Removes duplicate files created by the old copy behavior, keeping BEST QUALITY version.

SAFETY FEATURES:
1. ONLY deletes exact SHA-256 duplicates (not similar, EXACT copies)
2. ALWAYS keeps the highest resolution/largest file
3. Creates deletion_log.txt with full list of what will be deleted
4. Requires manual 'YES' confirmation after reviewing the log
5. Shows detailed comparison for every duplicate pair
6. NEVER deletes unique files
"""
import os
import hashlib
from pathlib import Path
from collections import defaultdict
import logging
from PIL import Image
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.raw', '.cr2', '.nef', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.jfif'}

def get_file_quality_score(filepath):
    """
    Calculate quality score for a file.
    Higher score = better quality
    Score based on: resolution (if image) + file size
    """
    try:
        stat = filepath.stat()
        file_size = stat.st_size
        resolution = 0
        
        # Try to get image resolution
        if filepath.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                with Image.open(filepath) as img:
                    resolution = img.width * img.height
            except Exception:
                pass
        
        # Quality score = resolution * 1000 + file_size
        # This prioritizes resolution but uses file size as tiebreaker
        return (resolution * 1000) + file_size
    except Exception:
        return 0

def sha256sum(filepath):
    """Calculate SHA-256 hash of a file"""
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash {filepath}: {e}")
        return None

def find_all_files(root_dir, exclude_dirs=None):
    """Find all files in directory, excluding specified subdirectories"""
    if exclude_dirs is None:
        exclude_dirs = set()
    
    files = []
    root_path = Path(root_dir)
    
    for item in root_path.rglob('*'):
        if item.is_file():
            # Check if file is in excluded directory
            if not any(excluded in item.parts for excluded in exclude_dirs):
                files.append(item)
    
    return files

def main():
    # Root directory - scan all Pictures folders
    ROOT = Path(r'C:\Users\david\OneDrive\Documents\Pictures')
    ORGANIZED_DIR = ROOT / 'Fotos' / 'Organized'
    
    if not ORGANIZED_DIR.exists():
        logger.info("No 'Organized' folder found - nothing to clean up!")
        return
    
    logger.info("=" * 70)
    logger.info("SAFE DUPLICATE CLEANUP - DRY RUN MODE")
    logger.info("=" * 70)
    logger.info("This will identify duplicates between original and Organized folders")
    logger.info("")
    
    # Step 1: Build hash map of organized files
    logger.info("Step 1: Scanning Organized folder...")
    organized_files = find_all_files(ORGANIZED_DIR)
    logger.info(f"Found {len(organized_files)} files in Organized folder")
    
    logger.info("Calculating SHA-256 hashes for organized files...")
    organized_hashes = {}
    for idx, filepath in enumerate(organized_files, 1):
        if idx % 500 == 0:
            progress_pct = (idx / len(organized_files)) * 100
            logger.info(f"Progress: {idx}/{len(organized_files)} ({progress_pct:.1f}%) - {len(organized_hashes)} hashed so far")
        
        file_hash = sha256sum(filepath)
        if file_hash:
            organized_hashes[file_hash] = filepath
    
    logger.info(f"✓ Calculated hashes for {len(organized_hashes)} organized files")
    
    # Step 2: Scan original folders (excluding Organized, duplicates, photo_reports)
    logger.info("")
    logger.info("Step 2: Scanning original folders...")
    exclude_dirs = {'Organized', 'duplicates', 'photo_reports'}
    original_files = find_all_files(ROOT, exclude_dirs=exclude_dirs)
    logger.info(f"Found {len(original_files)} files in original folders")
    
    # Step 3: Find duplicates and determine which to keep
    logger.info("")
    logger.info("Step 3: Finding duplicates and comparing quality...")
    files_to_delete = []
    files_to_keep_better = []  # Cases where original is better than organized
    unique_files = []
    
    for idx, filepath in enumerate(original_files, 1):
        if idx % 500 == 0:
            progress_pct = (idx / len(original_files)) * 100
            logger.info(f"Progress: {idx}/{len(original_files)} ({progress_pct:.1f}%) - {len(files_to_delete)} duplicates, {len(unique_files)} unique")
        
        file_hash = sha256sum(filepath)
        if not file_hash:
            continue
        
        if file_hash in organized_hashes:
            # Found duplicate - compare quality
            original_path = filepath
            organized_path = organized_hashes[file_hash]
            
            original_quality = get_file_quality_score(original_path)
            organized_quality = get_file_quality_score(organized_path)
            
            if original_quality > organized_quality:
                # Original is BETTER quality - keep original, delete organized copy
                files_to_keep_better.append({
                    'keep': original_path,
                    'delete': organized_path,
                    'reason': f'Original has better quality (score: {original_quality} vs {organized_quality})'
                })
            else:
                # Organized is better or equal - delete original
                files_to_delete.append({
                    'delete': original_path,
                    'keep': organized_path,
                    'reason': f'Organized copy has equal/better quality (score: {organized_quality} vs {original_quality})'
                })
        else:
            # This file is NOT in Organized - keep it!
            unique_files.append(filepath)
    
    logger.info(f"✓ Analysis complete: {len(files_to_delete)} duplicates, {len(files_to_keep_better)} better originals, {len(unique_files)} unique")
    
    # Step 4: Create detailed deletion log
    logger.info("")
    logger.info("Step 4: Creating deletion log...")
    log_file = Path(__file__).parent / f"deletion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("DUPLICATE CLEANUP LOG\n")
        f.write(f"Created: {datetime.now()}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"  Files safely organized: {len(organized_hashes)}\n")
        f.write(f"  Lower quality duplicates (to delete): {len(files_to_delete)}\n")
        f.write(f"  Better quality originals (keep, delete organized copy): {len(files_to_keep_better)}\n")
        f.write(f"  Unique files (will be kept): {len(unique_files)}\n\n")
        
        if files_to_delete:
            f.write("=" * 100 + "\n")
            f.write(f"LOWER QUALITY ORIGINALS TO DELETE ({len(files_to_delete)} files):\n")
            f.write("=" * 100 + "\n\n")
            for i, item in enumerate(files_to_delete, 1):
                f.write(f"{i}. WILL DELETE (lower quality):\n")
                f.write(f"   {item['delete']}\n")
                f.write(f"   WILL KEEP (better quality):\n")
                f.write(f"   {item['keep']}\n")
                f.write(f"   Reason: {item['reason']}\n\n")
        
        if files_to_keep_better:
            f.write("=" * 100 + "\n")
            f.write(f"LOWER QUALITY ORGANIZED COPIES TO DELETE ({len(files_to_keep_better)} files):\n")
            f.write("=" * 100 + "\n\n")
            for i, item in enumerate(files_to_keep_better, 1):
                f.write(f"{i}. WILL DELETE (lower quality organized copy):\n")
                f.write(f"   {item['delete']}\n")
                f.write(f"   WILL KEEP (better quality original):\n")
                f.write(f"   {item['keep']}\n")
                f.write(f"   Reason: {item['reason']}\n\n")
        
        if unique_files:
            f.write("=" * 100 + "\n")
            f.write(f"UNIQUE FILES (WILL NOT BE DELETED - {len(unique_files)} files):\n")
            f.write("=" * 100 + "\n\n")
            for f_path in unique_files:
                f.write(f"  ✓ {f_path}\n")
    
    logger.info(f"✓ Deletion log created: {log_file}")
    logger.info("")
    
    # Step 5: Report findings
    logger.info("=" * 70)
    logger.info("RESULTS:")
    logger.info("=" * 70)
    logger.info(f"✓ Files safely organized: {len(organized_hashes)}")
    logger.info(f"⚠ Lower quality duplicates (will delete): {len(files_to_delete)}")
    logger.info(f"💎 Better quality originals (keep, delete organized copy): {len(files_to_keep_better)}")
    logger.info(f"💎 Unique files (will be kept): {len(unique_files)}")
    logger.info("")
    
    if files_to_keep_better:
        logger.info("⭐ Found originals with BETTER quality than organized copies:")
        for item in files_to_keep_better[:5]:
            logger.info(f"  KEEP: {item['keep']}")
            logger.info(f"  DELETE: {item['delete']}")
            logger.info(f"  Reason: {item['reason']}")
            logger.info("")
        if len(files_to_keep_better) > 5:
            logger.info(f"  ... and {len(files_to_keep_better) - 5} more")
        logger.info("")
    
    if unique_files:
        logger.warning("⚠ WARNING: Found unique files not in Organized folder:")
        for f in unique_files[:10]:
            logger.warning(f"  - {f}")
        if len(unique_files) > 10:
            logger.warning(f"  ... and {len(unique_files) - 10} more")
        logger.warning("")
        logger.warning("These files will NOT be deleted. You may want to re-run the organizer.")
    
    if not files_to_delete and not files_to_keep_better:
        logger.info("✓ No duplicates found - everything is clean!")
        return
    
    # Step 6: Show sample duplicates to delete
    if files_to_delete:
        logger.info("Sample lower-quality duplicates that will be removed:")
        for item in files_to_delete[:5]:
            logger.info(f"  ✗ DELETE: {item['delete']}")
            logger.info(f"    KEEP: {item['keep']}")
            logger.info(f"    {item['reason']}")
            logger.info("")
        
        if len(files_to_delete) > 5:
            logger.info(f"  ... and {len(files_to_delete) - 5} more")
    
    # Step 7: Ask for confirmation
    logger.info("")
    logger.info("=" * 70)
    logger.info("⚠️  IMPORTANT: PLEASE REVIEW THE DELETION LOG!")
    logger.info("=" * 70)
    logger.info(f"📄 Full deletion log saved to: {log_file}")
    logger.info("")
    logger.info("Please OPEN and REVIEW the log file to see exactly what will be deleted.")
    logger.info("")
    total_to_delete = len(files_to_delete) + len(files_to_keep_better)
    logger.info(f"Total files to be deleted: {total_to_delete}")
    logger.info(f"  - {len(files_to_delete)} lower quality originals")
    logger.info(f"  - {len(files_to_keep_better)} lower quality organized copies")
    logger.info("")
    logger.info("SAFETY GUARANTEE:")
    logger.info("  ✓ Only EXACT duplicates (SHA-256 match)")
    logger.info("  ✓ Always keeps HIGHEST quality version")
    logger.info("  ✓ NEVER deletes unique files")
    logger.info("")
    response = input("Have you reviewed the log? Type 'YES' to DELETE lower quality duplicates: ")
    
    if response != 'YES':
        logger.info("Cancelled - no files were deleted.")
        return
    
    # Step 8: Delete lower quality duplicates
    logger.info("")
    logger.info("Deleting lower quality duplicate files...")
    deleted_count = 0
    failed_count = 0
    
    # Delete lower quality originals
    for item in files_to_delete:
        try:
            item['delete'].unlink()
            deleted_count += 1
            if deleted_count % 100 == 0:
                logger.info(f"Deleted {deleted_count} files...")
        except Exception as e:
            logger.error(f"Failed to delete {item['delete']}: {e}")
            failed_count += 1
    
    # Delete lower quality organized copies (keep better originals)
    for item in files_to_keep_better:
        try:
            item['delete'].unlink()
            deleted_count += 1
            if deleted_count % 100 == 0:
                logger.info(f"Deleted {deleted_count} files...")
        except Exception as e:
            logger.error(f"Failed to delete {item['delete']}: {e}")
            failed_count += 1
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("CLEANUP COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"✓ Deleted: {deleted_count} lower quality duplicates")
    logger.info(f"✓ Kept: {len(organized_hashes) - len(files_to_keep_better) + len(files_to_keep_better)} best quality files")
    logger.info(f"✓ Preserved: {len(unique_files)} unique files")
    if failed_count:
        logger.warning(f"✗ Failed: {failed_count} files")
    logger.info("")
    logger.info("Your photo library now contains only the BEST QUALITY versions!")
    
    if files_to_keep_better:
        logger.info("")
        logger.info(f"⭐ Note: {len(files_to_keep_better)} better-quality originals were kept instead of organized copies")
        logger.info("   You may want to re-run photo-organizer.py to organize these properly")

if __name__ == '__main__':
    main()
