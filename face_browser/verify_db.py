import sys
from pathlib import Path
from sqlalchemy import func, distinct

# Add backend directory to path to access database config
project_root = Path(__file__).resolve().parent
backend_dir = project_root / 'backend'
sys.path.insert(0, str(backend_dir))

def verify_database_state():
    """
    Connects to the database and prints a summary of the embeddings model
    and the latest scan time to verify the last pipeline run.
    """
    try:
        from database import session_scope
        from models import FaceEmbedding, MediaFile
        print("--- Database Verification ---")

        with session_scope() as session:
            # 1. Check which embedding models are in the database
            model_counts = session.query(
                FaceEmbedding.model_name, 
                func.count(FaceEmbedding.model_name)
            ).group_by(FaceEmbedding.model_name).all()

            if not model_counts:
                print("\n[ERROR] No face embeddings found in the database.")
                return

            print("\n✅ 1. Embedding Model(s) Found:")
            for model_name, count in model_counts:
                print(f"   - Model: '{model_name}', Count: {count} embeddings")
            
            if len(model_counts) > 1:
                print("\n   [WARNING] Multiple embedding models detected! This can lead to poor clustering results.")
            else:
                print("   [SUCCESS] A single, consistent embedding model is used.")

            # 2. Check the timestamp of the most recently scanned file
            latest_scan = session.query(func.max(MediaFile.last_scanned_at)).scalar()

            print("\n✅ 2. Latest File Scan Time:")
            if latest_scan:
                print(f"   - Most recent file processed at: {latest_scan.strftime('%Y-%m-%d %H:%M:%S')}")
                print("   [SUCCESS] The database contains recently processed files.")
            else:
                print("   [WARNING] Could not determine the latest scan time.")

        print("\n--- Verification Complete ---")

    except Exception as e:
        print(f"\nAn error occurred during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_database_state()
