"""
Quick script to clear out old human face data from pet_clusters table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from backend.database import session_scope
from backend.models import PetCluster

with session_scope() as session:
    count = session.query(PetCluster).count()
    print(f"Found {count} entries in pet_clusters table")
    
    if count > 0:
        response = input(f"Delete all {count} entries? (yes/no): ")
        if response.lower() == 'yes':
            session.query(PetCluster).delete()
            session.commit()
            print(f"✅ Deleted {count} entries from pet_clusters")
        else:
            print("Cancelled")
    else:
        print("Table is already empty")
