import csv
import os

csv_file = r'C:\dscodingpython\File organizers\duplicates.csv'

total = 0
both_exist = 0
original_exists = 0
duplicate_exists = 0
neither_exists = 0

with open(csv_file, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        orig = row['original']
        dup = row['duplicate']
        
        orig_ex = os.path.exists(orig)
        dup_ex = os.path.exists(dup)
        
        if orig_ex and dup_ex:
            both_exist += 1
        elif orig_ex:
            original_exists += 1
        elif dup_ex:
            duplicate_exists += 1
        else:
            neither_exists += 1
        
        # Show first existing pair
        if both_exist == 1:
            print(f"First existing pair found at row {total}:")
            print(f"  Original: {orig}")
            print(f"  Duplicate: {dup}")
            print()

print(f"Total pairs: {total}")
print(f"Both exist: {both_exist}")
print(f"Only original exists: {original_exists}")
print(f"Only duplicate exists: {duplicate_exists}")
print(f"Neither exists: {neither_exists}")
print(f"\nPercentage with both files: {both_exist/total*100:.1f}%")
