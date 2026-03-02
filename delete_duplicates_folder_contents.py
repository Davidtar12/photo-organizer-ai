"""
Permanently delete all files inside the duplicates folder.
This will free up storage space by removing all duplicate files.
The Organized folder remains untouched.
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DUPLICATES_DIR = Path(r'C:\Users\USERNAME\OneDrive\Documents\Pictures\duplicates')

def main():
    if not DUPLICATES_DIR.exists():
        logger.error(f"Duplicates folder not found: {DUPLICATES_DIR}")
        return
    
    all_files = [f for f in DUPLICATES_DIR.rglob('*') if f.is_file()]
    total_size = sum(f.stat().st_size for f in all_files)
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size / (1024 * 1024 * 1024)
    
    logger.info("=" * 70)
    logger.info("DELETE DUPLICATES FOLDER CONTENTS")
    logger.info("=" * 70)
    logger.info(f"Location: {DUPLICATES_DIR}")
    logger.info(f"Files to delete: {len(all_files):,}")
    logger.info(f"Space to free: {total_size_gb:.2f} GB ({total_size_mb:.1f} MB)")
    logger.info("")
    logger.info("⚠️  WARNING: This will PERMANENTLY delete all duplicate files!")
    logger.info("")
    
    response = input("Type 'DELETE DUPLICATES' to confirm permanent deletion: ")
    
    if response != 'DELETE DUPLICATES':
        logger.info("Cancelled - no files were deleted.")
        return
    
    logger.info("")
    logger.info("Deleting files...")
    deleted = 0
    failed = 0
    
    for idx, file in enumerate(all_files, 1):
        try:
            file.unlink()
            deleted += 1
            if deleted % 1000 == 0:
                progress = (deleted / len(all_files)) * 100
                logger.info(f"Progress: {deleted:,}/{len(all_files):,} ({progress:.1f}%)")
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.error(f"Failed to delete {file.name}: {e}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ DELETION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Deleted: {deleted:,} files")
    logger.info(f"Freed: {total_size_gb:.2f} GB")
    if failed:
        logger.info(f"Failed: {failed} files")
    logger.info("")
    logger.info("Your Organized folder (54,544 unique photos) is safe and untouched!")

if __name__ == '__main__':
    main()
