"""
Remove all empty Unknown folders from Organized directory.
Run this after fix_unknown_subfolders.py to clean up empty directories.
"""
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

ORGANIZED_DIR = Path(r'C:\Users\david\OneDrive\Documents\Pictures\Organized')

def main():
    unknown_folders = [p for p in ORGANIZED_DIR.rglob('Unknown') if p.is_dir()]
    
    logger.info(f"Found {len(unknown_folders)} Unknown folders")
    
    deleted = 0
    not_empty = 0
    failed = 0
    
    for folder in unknown_folders:
        try:
            contents = list(folder.iterdir())
            if not contents:
                shutil.rmtree(str(folder))
                deleted += 1
                if deleted % 50 == 0:
                    logger.info(f"Deleted {deleted} empty Unknown folders...")
            else:
                not_empty += 1
                logger.warning(f"Not empty ({len(contents)} files): {folder}")
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.error(f"Failed to delete {folder}: {e}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"✅ Deleted: {deleted} empty Unknown folders")
    if not_empty:
        logger.info(f"⚠️  Skipped: {not_empty} non-empty Unknown folders")
    if failed:
        logger.info(f"❌ Failed: {failed} folders (permission denied)")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()
