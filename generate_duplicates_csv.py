"""
Generate duplicates.csv from existing duplicates folder
Creates CSV linking duplicate files to their originals in Organized folder
"""
from pathlib import Path
import hashlib
import csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def main():
    DUPLICATES_DIR = Path(r'C:\Users\USERNAME\OneDrive\Documents\Pictures\duplicates')
    ORGANIZED_DIR = Path(r'C:\Users\USERNAME\OneDrive\Documents\Pictures\Organized')
    OUTPUT_CSV = Path(r'C:\dscodingpython\File organizers\duplicates.csv')
    
    if not DUPLICATES_DIR.exists():
        logger.error(f"Duplicates folder not found: {DUPLICATES_DIR}")
        return
    
    if not ORGANIZED_DIR.exists():
        logger.error(f"Organized folder not found: {ORGANIZED_DIR}")
        return
    
    # Step 1: Hash all files in Organized folder
    logger.info("Step 1: Building hash index of Organized folder...")
    organized_hashes = {}
    organized_files = list(ORGANIZED_DIR.rglob('*'))
    organized_files = [f for f in organized_files if f.is_file()]
    
    logger.info(f"Found {len(organized_files)} files in Organized folder")
    
    for idx, filepath in enumerate(organized_files, 1):
        if idx % 1000 == 0:
            logger.info(f"Hashing: {idx}/{len(organized_files)} ({idx/len(organized_files)*100:.1f}%)")
        
        file_hash = sha256sum(filepath)
        if file_hash:
            organized_hashes[file_hash] = filepath
    
    logger.info(f"✓ Indexed {len(organized_hashes)} files from Organized folder")
    
    # Step 2: Hash duplicates and find their originals
    logger.info("")
    logger.info("Step 2: Finding originals for duplicate files...")
    duplicate_files = list(DUPLICATES_DIR.rglob('*'))
    duplicate_files = [f for f in duplicate_files if f.is_file()]
    
    logger.info(f"Found {len(duplicate_files)} files in duplicates folder")
    
    pairs = []
    matched = 0
    unmatched = 0
    
    for idx, dup_path in enumerate(duplicate_files, 1):
        if idx % 100 == 0:
            logger.info(f"Processing: {idx}/{len(duplicate_files)} ({matched} matched, {unmatched} unmatched)")
        
        dup_hash = sha256sum(dup_path)
        if not dup_hash:
            unmatched += 1
            continue
        
        if dup_hash in organized_hashes:
            original_path = organized_hashes[dup_hash]
            pairs.append((str(original_path), str(dup_path)))
            matched += 1
        else:
            unmatched += 1
            logger.warning(f"No match found for: {dup_path}")
    
    logger.info(f"✓ Matched {matched} duplicates to originals")
    if unmatched > 0:
        logger.warning(f"⚠ {unmatched} duplicates have no matching original")
    
    # Step 3: Write CSV
    if not pairs:
        logger.warning("No duplicate pairs found - nothing to write")
        return
    
    logger.info("")
    logger.info(f"Step 3: Writing duplicates.csv with {len(pairs)} pairs...")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original', 'duplicate'])
        for original, duplicate in pairs:
            writer.writerow([original, duplicate])
    
    logger.info(f"✓ CSV written to: {OUTPUT_CSV}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUCCESS!")
    logger.info("=" * 70)
    logger.info(f"Created duplicates.csv with {len(pairs)} duplicate pairs")
    logger.info(f"Webapp can now load this file from: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
