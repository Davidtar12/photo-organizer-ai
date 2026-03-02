"""Simple script to show duplicate groups and allow deletion"""

import pandas as pd
from pathlib import Path
import os

# Load duplicates
csv_path = r"C:\Users\USERNAME\OneDrive\Documents\Pictures\photo_reports\duplicates.csv"
df = pd.read_csv(csv_path)

print(f"\n{'='*80}")
print(f"Found {len(df)} duplicate pairs")
print(f"{'='*80}\n")

# Group by original
groups = df.groupby('original')

for i, (original, group) in enumerate(groups, 1):
    duplicates = group['duplicate'].tolist()
    
    print(f"\n--- Group {i} of {len(groups)} ---")
    print(f"ORIGINAL: {original}")
    
    if Path(original).exists():
        size_mb = Path(original).stat().st_size / (1024*1024)
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Exists: YES")
    else:
        print(f"  Exists: NO")
    
    print(f"\nDUPLICATES ({len(duplicates)}):")
    for j, dup in enumerate(duplicates, 1):
        print(f"  {j}. {dup}")
        if Path(dup).exists():
            size_mb = Path(dup).stat().st_size / (1024*1024)
            print(f"     Size: {size_mb:.2f} MB - EXISTS")
        else:
            print(f"     DOES NOT EXIST")
    
    print(f"\nWasted space: {sum(Path(d).stat().st_size for d in duplicates if Path(d).exists()) / (1024*1024):.2f} MB")
    
    # Ask what to do
    choice = input(f"\nDelete duplicates? (y/n/q to quit): ").strip().lower()
    
    if choice == 'q':
        print("\nQuitting...")
        break
    elif choice == 'y':
        for dup in duplicates:
            if Path(dup).exists():
                try:
                    os.remove(dup)
                    print(f"  ✓ Deleted: {dup}")
                except Exception as e:
                    print(f"  ✗ Failed to delete: {e}")
    else:
        print("  Skipped")

print(f"\n{'='*80}")
print("Done!")
print(f"{'='*80}\n")
