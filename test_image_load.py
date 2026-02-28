import csv
import os
from pathlib import Path

# Read first duplicate pair
csv_file = r'C:\dscodingpython\File organizers\duplicates.csv'
with open(csv_file, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    first_row = next(reader)
    
print(f"First duplicate pair:")
print(f"Original: {first_row['original']}")
print(f"Duplicate: {first_row['duplicate']}")
print()

# Check if files exist
orig_path = first_row['original']
dup_path = first_row['duplicate']

print(f"Original exists: {os.path.exists(orig_path)}")
print(f"Duplicate exists: {os.path.exists(dup_path)}")

if os.path.exists(orig_path):
    print(f"Original size: {os.path.getsize(orig_path)} bytes")
if os.path.exists(dup_path):
    print(f"Duplicate size: {os.path.getsize(dup_path)} bytes")

# Test image endpoint
import requests
print("\nTesting image endpoint...")
try:
    response = requests.get(
        'http://127.0.0.1:5000/api/image',
        params={'path': orig_path},
        timeout=5
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Content-Length: {len(response.content)} bytes")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
