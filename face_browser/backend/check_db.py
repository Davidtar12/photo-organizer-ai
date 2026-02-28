import sys
from pathlib import Path

# Add backend path to allow importing database and models
backend_path = Path(__file__).parent
sys.path.append(str(backend_path))

try:
    from database import session_scope
    from models import PetEmbedding, PersonCluster
    print("Successfully imported database and models.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure this script is in the 'backend' directory and the venv is active.")
    sys.exit(1)

def check_counts():
    """Connects to the database and prints table counts."""
    try:
        with session_scope() as session:
            pet_embedding_count = session.query(PetEmbedding).count()
            person_cluster_count = session.query(PersonCluster).count()
            
            print("-" * 30)
            print(f"Found {pet_embedding_count} rows in 'pet_embeddings' table.")
            print(f"Found {person_cluster_count} rows in 'person_clusters' table.")
            print("-" * 30)

            if pet_embedding_count == 0:
                print("\n[!] The 'pet_embeddings' table is empty.")
                print("This means the pet detection script either hasn't been run or didn't save any results.")
            else:
                print("\n[+] Data exists! The issue is likely in the API logic.")

    except Exception as e:
        print(f"\nAn error occurred while connecting to the database: {e}")

if __name__ == "__main__":
    check_counts()
