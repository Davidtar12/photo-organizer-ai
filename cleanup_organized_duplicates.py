"""
ORGANIZED FOLDER DEDUPLICATION
Removes ~50,000 duplicate files WITHIN the Organized folder itself.

This script finds files with identical SHA-256 hashes in the Organized folder
and keeps only ONE copy of each (the best quality version).

SAFETY FEATURES:
1. ONLY deletes exact SHA-256 duplicates
2. ALWAYS keeps the highest resolution/largest file
3. Creates detailed deletion log
4. Requires 'YES' confirmation
5. Progress tracking every 500 files
"""
import os
import hashlib
import argparse
import importlib
from pathlib import Path
import shutil
from collections import defaultdict
import logging
# Pillow optional; fall back to size-only quality scoring if not available
Image = None
try:
    Image = importlib.import_module("PIL.Image")
except Exception:
    Image = None
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.raw', '.cr2', '.nef', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.jfif'}

# SAFETY: Move duplicates to global duplicates folder instead of deleting
# Set to False to permanently delete duplicates
MOVE_DUPLICATES_TO_FOLDER = True
DUPLICATES_DIR = Path(r'C:\Users\david\OneDrive\Documents\Pictures\duplicates')

def get_file_quality_score(filepath):
    """Calculate quality score (resolution * 1000 + file_size)"""
    try:
        stat = filepath.stat()
        file_size = stat.st_size
        resolution = 0
        
        if Image is not None and filepath.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                with Image.open(filepath) as img:
                    resolution = img.width * img.height
            except Exception:
                pass
        
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

def safe_unique_path(dst_dir: Path, filename: str) -> Path:
    """Return a unique path inside dst_dir to avoid overwriting existing files."""
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dst_dir / (base + ext)
    counter = 1
    while candidate.exists():
        candidate = dst_dir / f"{base} ({counter}){ext}"
        counter += 1
    return candidate


def main():
    parser = argparse.ArgumentParser(description="Deduplicate exact duplicates inside Organized folder")
    parser.add_argument("--yes", action="store_true", help="Skip interactive prompt and proceed")
    parser.add_argument("--delete", action="store_true", help="Permanently delete duplicates instead of moving to duplicates folder")
    parser.add_argument("--organized-dir", type=str, default=r'C:\\Users\\david\\OneDrive\\Documents\\Pictures\\Organized', help="Path to Organized folder")
    parser.add_argument("--duplicates-dir", type=str, default=str(Path(r'C:\\Users\\david\\OneDrive\\Documents\\Pictures\\duplicates')), help="Path to global duplicates folder (when moving)")
    args = parser.parse_args()

    # FIX: Use the actual Organized path (without 'Fotos')
    ORGANIZED_DIR = Path(args.organized_dir)
    
    if not ORGANIZED_DIR.exists():
        logger.error("Organized folder not found!")
        return
    
    logger.info("=" * 70)
    logger.info("ORGANIZED FOLDER DEDUPLICATION")
    logger.info("=" * 70)
    logger.info("Finding duplicate files within the Organized folder...")
    logger.info("")
    
    # Step 1: Scan all files in Organized
    logger.info("Step 1: Scanning Organized folder...")
    all_files = list(ORGANIZED_DIR.rglob('*'))
    all_files = [f for f in all_files if f.is_file()]
    logger.info(f"Found {len(all_files)} files in Organized folder")
    
    # Step 2: Hash all files and group by hash
    logger.info("")
    logger.info("Step 2: Calculating SHA-256 hashes...")
    hash_groups = defaultdict(list)
    
    import time
    start_time = time.time()
    for idx, filepath in enumerate(all_files, 1):
        if idx % 500 == 0:
            elapsed = time.time() - start_time
            avg = elapsed / idx
            remaining = (len(all_files) - idx) * avg
            progress_pct = (idx / len(all_files)) * 100
            logger.info(f"Progress: {idx}/{len(all_files)} ({progress_pct:.1f}%) | ETA: {int(remaining)}s")
        
        file_hash = sha256sum(filepath)
        if file_hash:
            hash_groups[file_hash].append(filepath)
    
    logger.info(f"✓ Found {len(hash_groups)} unique files")
    
    # Step 3: Find duplicates (groups with more than 1 file)
    logger.info("")
    logger.info("Step 3: Identifying duplicates...")
    duplicate_groups = {h: files for h, files in hash_groups.items() if len(files) > 1}
    
    total_duplicates = sum(len(files) - 1 for files in duplicate_groups.values())
    logger.info(f"✓ Found {len(duplicate_groups)} duplicate sets")
    logger.info(f"✓ Total duplicate files to remove: {total_duplicates}")
    logger.info(f"✓ Will save approximately: {(total_duplicates / len(all_files)) * 100:.1f}% of space")
    
    # Step 4: Determine which files to keep/delete
    logger.info("")
    logger.info("Step 4: Determining best quality versions...")
    files_to_delete = []
    
    for file_hash, files in duplicate_groups.items():
        # Score each file by quality
        scored_files = [(f, get_file_quality_score(f)) for f in files]
        # Sort by quality (highest first)
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        # Keep the best, delete the rest
        best_file = scored_files[0][0]
        for filepath, score in scored_files[1:]:
            files_to_delete.append({
                'delete': filepath,
                'keep': best_file,
                'hash': file_hash,
                'delete_score': score,
                'keep_score': scored_files[0][1]
            })
    
    logger.info(f"✓ Will delete {len(files_to_delete)} duplicate files")
    logger.info(f"✓ Will keep {len(hash_groups)} unique files")
    
    # Step 5: Create deletion log
    logger.info("")
    logger.info("Step 5: Creating deletion log...")
    log_file = Path(__file__).parent / f"organized_dedup_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("ORGANIZED FOLDER DEDUPLICATION LOG\n")
        f.write(f"Created: {datetime.now()}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"  Total files scanned: {len(all_files)}\n")
        f.write(f"  Unique files (will keep): {len(hash_groups)}\n")
        f.write(f"  Duplicate files (will delete): {len(files_to_delete)}\n")
        f.write(f"  Space savings: {(len(files_to_delete) / len(all_files)) * 100:.1f}%\n\n")
        
        f.write("=" * 100 + "\n")
        f.write(f"FILES TO DELETE ({len(files_to_delete)} duplicates):\n")
        f.write("=" * 100 + "\n\n")
        
        for i, item in enumerate(files_to_delete, 1):
            f.write(f"{i}. DELETE (duplicate):\n")
            f.write(f"   {item['delete']}\n")
            f.write(f"   Quality score: {item['delete_score']}\n")
            f.write(f"   KEEP (best quality):\n")
            f.write(f"   {item['keep']}\n")
            f.write(f"   Quality score: {item['keep_score']}\n")
            f.write(f"   SHA-256: {item['hash']}\n\n")
    
    logger.info(f"✓ Deletion log created: {log_file}")
    
    # Step 6: Show summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("DEDUPLICATION SUMMARY:")
    logger.info("=" * 70)
    logger.info(f"📁 Organized folder: {ORGANIZED_DIR}")
    logger.info(f"📊 Total files: {len(all_files)}")
    logger.info(f"✅ Unique files: {len(hash_groups)}")
    logger.info(f"🗑️  Duplicates: {len(files_to_delete)}")
    logger.info(f"💾 Space savings: ~{(len(files_to_delete) / len(all_files)) * 100:.1f}%")
    logger.info("")
    
    # Show sample duplicates
    logger.info("Sample duplicates that will be removed:")
    for item in files_to_delete[:5]:
        logger.info(f"  ✗ DELETE: {item['delete'].name}")
        logger.info(f"    Location: {item['delete'].parent}")
        logger.info(f"    KEEP: {item['keep'].name}")
        logger.info(f"    Location: {item['keep'].parent}")
        logger.info("")
    
    if len(files_to_delete) > 5:
        logger.info(f"  ... and {len(files_to_delete) - 5} more")
    
    # Step 7: Confirmation
    logger.info("")
    logger.info("=" * 70)
    logger.info("⚠️  FINAL CONFIRMATION")
    logger.info("=" * 70)
    logger.info(f"📄 Review the log: {log_file}")
    logger.info("")
    logger.info(f"This will DELETE {len(files_to_delete)} duplicate files")
    logger.info(f"and KEEP {len(hash_groups)} unique files")
    logger.info("")
    logger.info("SAFETY GUARANTEE:")
    logger.info("  ✓ Only EXACT duplicates (SHA-256 match)")
    logger.info("  ✓ Always keeps HIGHEST quality version")
    logger.info("")

    proceed = args.yes
    if not proceed:
        response = input("Type 'YES' to DELETE duplicates and free up space: ")
        proceed = (response == 'YES')
    if not proceed:
        logger.info("Cancelled - no files were deleted.")
        return
    
    # Step 8: Remove duplicates (move to duplicates folder by default)
    logger.info("")
    move_to_folder = MOVE_DUPLICATES_TO_FOLDER and (not args.delete)
    duplicates_dir = Path(args.duplicates_dir)
    if move_to_folder:
        logger.info("Moving duplicate files to duplicates folder for safety...")
        duplicates_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Deleting duplicate files (permanent)...")
    deleted_count = 0
    failed_count = 0
    
    move_start = time.time()
    for idx, item in enumerate(files_to_delete, 1):
        try:
            if move_to_folder:
                dst = safe_unique_path(duplicates_dir, item['delete'].name)
                shutil.move(str(item['delete']), str(dst))
                deleted_count += 1
            else:
                item['delete'].unlink()
                deleted_count += 1
            if deleted_count % 500 == 0:
                elapsed = time.time() - move_start
                avg = elapsed / deleted_count
                remaining = (len(files_to_delete) - deleted_count) * avg
                logger.info(f"Progress: {deleted_count}/{len(files_to_delete)} | ETA: {int(remaining)}s")
        except Exception as e:
            logger.error(f"Failed to delete {item['delete']}: {e}")
            failed_count += 1
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ DEDUPLICATION COMPLETE!")
    logger.info("=" * 70)
    if move_to_folder:
        logger.info(f"✓ Moved: {deleted_count} duplicate files to {duplicates_dir}")
    else:
        logger.info(f"✓ Deleted: {deleted_count} duplicate files")
    logger.info(f"✓ Kept: {len(hash_groups)} unique files")
    if failed_count:
        logger.warning(f"✗ Failed: {failed_count} files")
    logger.info("")
    logger.info(f"Your Organized folder now has {len(hash_groups)} files instead of {len(all_files)}!")
    logger.info(f"Space freed: ~{(deleted_count / len(all_files)) * 100:.1f}%")

if __name__ == '__main__':
    main()
