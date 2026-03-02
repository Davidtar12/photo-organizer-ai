"""
UNKNOWN FOLDERS DEDUPLICATION
Removes duplicate files within Unknown folders, keeping best quality version.

Scans all folders with "Unknown" in the name and removes exact duplicates.
"""
import hashlib
from pathlib import Path
from collections import defaultdict
import logging
from PIL import Image
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.raw', '.cr2', '.nef', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.jfif'}

def get_file_quality_score(filepath):
    """Calculate quality score (resolution * 1000 + file_size)"""
    try:
        stat = filepath.stat()
        file_size = stat.st_size
        resolution = 0
        
        if filepath.suffix.lower() in IMAGE_EXTENSIONS:
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

def find_unknown_folders(root_dir):
    """Find all folders with 'Unknown' in the name"""
    root_path = Path(root_dir)
    unknown_folders = []
    
    for item in root_path.rglob('*'):
        if item.is_dir() and 'unknown' in item.name.lower():
            unknown_folders.append(item)
    
    return unknown_folders

def main():
    ROOT = Path(r'C:\Users\USERNAME\OneDrive\Documents\Pictures')
    
    logger.info("=" * 70)
    logger.info("UNKNOWN FOLDERS DEDUPLICATION")
    logger.info("=" * 70)
    logger.info("Finding folders with 'Unknown' in the name...")
    logger.info("")
    
    # Step 1: Find all Unknown folders
    unknown_folders = find_unknown_folders(ROOT)
    
    if not unknown_folders:
        logger.info("✓ No 'Unknown' folders found - nothing to clean up!")
        return
    
    logger.info(f"Found {len(unknown_folders)} Unknown folders:")
    for folder in unknown_folders:
        logger.info(f"  - {folder}")
    logger.info("")
    
    # Step 2: Process each Unknown folder
    total_files = 0
    total_duplicates = 0
    all_files_to_delete = []
    folder_logs = []
    
    for folder_idx, folder in enumerate(unknown_folders, 1):
        logger.info("=" * 70)
        logger.info(f"Scanning folder {folder_idx}/{len(unknown_folders)}: {folder.name}")
        logger.info("=" * 70)
        
        # Get all files in this folder
        all_files = list(folder.rglob('*'))
        all_files = [f for f in all_files if f.is_file()]
        
        if not all_files:
            logger.info("  Empty folder - skipping")
            continue
        
        logger.info(f"  Found {len(all_files)} files")
        total_files += len(all_files)
        
        # Hash all files
        hash_groups = defaultdict(list)
        
        for idx, filepath in enumerate(all_files, 1):
            if idx % 100 == 0:
                logger.info(f"  Hashing: {idx}/{len(all_files)}")
            
            file_hash = sha256sum(filepath)
            if file_hash:
                hash_groups[file_hash].append(filepath)
        
        # Find duplicates
        duplicate_groups = {h: files for h, files in hash_groups.items() if len(files) > 1}
        folder_duplicates = sum(len(files) - 1 for files in duplicate_groups.values())
        
        logger.info(f"  ✓ Found {len(hash_groups)} unique files")
        logger.info(f"  ✓ Found {folder_duplicates} duplicates to remove")
        
        if folder_duplicates == 0:
            logger.info("")
            continue
        
        total_duplicates += folder_duplicates
        
        # Determine which to keep/delete
        files_to_delete = []
        
        for file_hash, files in duplicate_groups.items():
            # Score each file
            scored_files = [(f, get_file_quality_score(f)) for f in files]
            scored_files.sort(key=lambda x: x[1], reverse=True)
            
            # Keep best, delete rest
            best_file = scored_files[0][0]
            for filepath, score in scored_files[1:]:
                files_to_delete.append({
                    'delete': filepath,
                    'keep': best_file,
                    'hash': file_hash,
                    'delete_score': score,
                    'keep_score': scored_files[0][1],
                    'folder': folder.name
                })
        
        all_files_to_delete.extend(files_to_delete)
        
        # Create log
        log_file = Path(__file__).parent / f"unknown_dedup_{folder.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"UNKNOWN FOLDER: {folder}\n")
            f.write(f"Total files: {len(all_files)}\n")
            f.write(f"Unique files: {len(hash_groups)}\n")
            f.write(f"Duplicates: {folder_duplicates}\n\n")
            
            for i, item in enumerate(files_to_delete, 1):
                f.write(f"{i}. DELETE: {item['delete']}\n")
                f.write(f"   KEEP: {item['keep']}\n\n")
        
        folder_logs.append(log_file)
        logger.info(f"  📄 Log: {log_file}")
        logger.info("")
    
    # Final summary
    logger.info("=" * 70)
    logger.info("SCAN COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Scanned: {len(unknown_folders)} Unknown folders")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Total duplicates found: {total_duplicates}")
    logger.info("")
    
    if not all_files_to_delete:
        logger.info("✅ No duplicates found - Unknown folders are clean!")
        return
    
    # Show summary by folder
    logger.info("DUPLICATES BY FOLDER:")
    folder_summary = defaultdict(int)
    for item in all_files_to_delete:
        folder_summary[item['folder']] += 1
    
    for folder_name, count in sorted(folder_summary.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {folder_name}: {count} duplicates")
    if len(folder_summary) > 10:
        logger.info(f"  ... and {len(folder_summary) - 10} more folders")
    
    logger.info("")
    logger.info("📄 Individual logs created for each folder")
    logger.info("")
    
    # Single confirmation for ALL deletions
    logger.info("=" * 70)
    logger.info("⚠️  READY TO DELETE")
    logger.info("=" * 70)
    logger.info(f"Total duplicates to delete: {len(all_files_to_delete)}")
    logger.info(f"Logs created: {len(folder_logs)}")
    logger.info("")
    logger.info("SAFETY:")
    logger.info("  ✓ Only exact SHA-256 duplicates")
    logger.info("  ✓ Keeps highest quality version")
    logger.info("  ✓ Individual logs for each folder")
    logger.info("")
    response = input(f"Delete ALL {len(all_files_to_delete)} duplicates from {len(folder_summary)} Unknown folders? Type 'YES': ")
    
    if response != 'YES':
        logger.info("Cancelled - no files deleted")
        return
    
    # Delete all duplicates
    logger.info("")
    logger.info("Deleting duplicates...")
    deleted = 0
    failed = 0
    
    for idx, item in enumerate(all_files_to_delete, 1):
        try:
            item['delete'].unlink()
            deleted += 1
            if deleted % 100 == 0:
                logger.info(f"  Progress: {deleted}/{len(all_files_to_delete)}")
        except Exception as e:
            logger.error(f"  Failed: {item['delete']}: {e}")
            failed += 1
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ CLEANUP COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed: {len(unknown_folders)} Unknown folders")
    logger.info(f"Total files scanned: {total_files}")
    logger.info(f"Duplicates deleted: {deleted}")
    if failed:
        logger.info(f"Failed deletions: {failed}")
    logger.info("")
    logger.info(f"📄 {len(folder_logs)} log files created")

if __name__ == '__main__':
    main()
