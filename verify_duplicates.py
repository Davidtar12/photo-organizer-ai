"""
Duplicate Verification Tool
Helps you verify duplicate files before deleting them permanently.
"""
import hashlib
import pandas as pd
from pathlib import Path
from PIL import Image
from datetime import datetime
import argparse

def sha256sum(path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def get_file_info(path: Path) -> dict:
    """Get detailed file information."""
    info = {
        'path': str(path),
        'exists': path.exists(),
        'size_mb': 0,
        'modified': None,
        'resolution': None,
        'sha256': None
    }
    
    if not path.exists():
        return info
    
    stat = path.stat()
    info['size_mb'] = round(stat.st_size / (1024 * 1024), 2)
    info['modified'] = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    info['sha256'] = sha256sum(path)
    
    try:
        with Image.open(path) as img:
            w, h = img.size
            info['resolution'] = f"{w}x{h} ({round(w*h/1000000, 1)}MP)"
    except Exception:
        pass
    
    return info

def verify_duplicates(csv_path: Path, sample_size: int = 10):
    """Verify duplicates from the CSV report."""
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"📊 Total duplicate pairs found: {len(df)}\n")
    
    # Check if duplicate files still exist
    missing_originals = 0
    missing_duplicates = 0
    
    for idx, row in df.iterrows():
        if not Path(row['original']).exists():
            missing_originals += 1
        if not Path(row['duplicate']).exists():
            missing_duplicates += 1
    
    if missing_originals > 0:
        print(f"⚠️  {missing_originals} original files missing/moved")
    if missing_duplicates > 0:
        print(f"✅ {missing_duplicates} duplicate files already removed/moved")
    
    print(f"\n🔍 Verifying first {sample_size} duplicate pairs:\n")
    print("=" * 100)
    
    for idx, row in df.head(sample_size).iterrows():
        original = Path(row['original'])
        duplicate = Path(row['duplicate'])
        
        print(f"\n📌 Pair #{idx + 1}:")
        print(f"   Original:  {original.name}")
        print(f"   Duplicate: {duplicate.name}")
        
        orig_info = get_file_info(original)
        dup_info = get_file_info(duplicate)
        
        print(f"\n   ORIGINAL FILE:")
        print(f"      Path:       {orig_info['path']}")
        print(f"      Exists:     {orig_info['exists']}")
        print(f"      Size:       {orig_info['size_mb']} MB")
        print(f"      Modified:   {orig_info['modified']}")
        print(f"      Resolution: {orig_info['resolution']}")
        print(f"      SHA-256:    {orig_info['sha256'][:16]}...")
        
        print(f"\n   DUPLICATE FILE:")
        print(f"      Path:       {dup_info['path']}")
        print(f"      Exists:     {dup_info['exists']}")
        print(f"      Size:       {dup_info['size_mb']} MB")
        print(f"      Modified:   {dup_info['modified']}")
        print(f"      Resolution: {dup_info['resolution']}")
        print(f"      SHA-256:    {dup_info['sha256'][:16]}...")
        
        # Verify hashes match
        if orig_info['sha256'] and dup_info['sha256']:
            if orig_info['sha256'] == dup_info['sha256']:
                print(f"\n   ✅ VERIFIED: SHA-256 hashes MATCH (identical files)")
            else:
                print(f"\n   ❌ WARNING: SHA-256 hashes DO NOT MATCH!")
        
        print("=" * 100)
    
    print(f"\n💡 To see all {len(df)} pairs, check: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Verify duplicate files')
    parser.add_argument('--csv', type=Path, 
                       default=Path('photo_reports/duplicates.csv'),
                       help='Path to duplicates.csv report')
    parser.add_argument('--sample', type=int, default=10,
                       help='Number of duplicate pairs to verify (default: 10)')
    args = parser.parse_args()
    
    verify_duplicates(args.csv, args.sample)

if __name__ == '__main__':
    main()
