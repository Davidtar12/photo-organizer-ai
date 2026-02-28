"""
Filter duplicates.csv to only include pairs where both files still exist
"""
from pathlib import Path
import csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    INPUT_CSV = Path(r'C:\dscodingpython\File organizers\duplicates.csv')
    OUTPUT_CSV = Path(r'C:\dscodingpython\File organizers\duplicates_filtered.csv')
    
    if not INPUT_CSV.exists():
        logger.error(f"CSV not found: {INPUT_CSV}")
        return
    
    existing_pairs = []
    deleted_pairs = 0
    
    logger.info("Filtering duplicates.csv to only existing files...")
    
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_path = row.get('original', '').strip()
            duplicate_path = row.get('duplicate', '').strip()
            
            if not original_path or not duplicate_path:
                continue
            
            original_exists = Path(original_path).exists()
            duplicate_exists = Path(duplicate_path).exists()
            
            if original_exists and duplicate_exists:
                existing_pairs.append((original_path, duplicate_path))
            else:
                deleted_pairs += 1
    
    logger.info(f"Found {len(existing_pairs)} existing pairs, {deleted_pairs} deleted/missing")
    
    # Write filtered CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original', 'duplicate'])
        for original, duplicate in existing_pairs:
            writer.writerow([original, duplicate])
    
    logger.info(f"✓ Filtered CSV written to: {OUTPUT_CSV}")
    logger.info("")
    logger.info("To use this filtered list:")
    logger.info(f"1. Backup original: copy duplicates.csv duplicates_backup.csv")
    logger.info(f"2. Replace: move duplicates_filtered.csv duplicates.csv")
    logger.info(f"3. Refresh webapp")

if __name__ == '__main__':
    main()
