import sys
from pathlib import Path

# Add backend directory to path
project_root = Path(__file__).resolve().parent
backend_dir = project_root / 'backend'
sys.path.insert(0, str(backend_dir))

def clear_old_embeddings():
    """Clear all face embeddings and clusters to prepare for new model."""
    try:
        from database import session_scope
        from models import FaceEmbedding, PersonCluster, ClusterSuggestion
        
        print("Clearing old embeddings and clusters...")
        
        with session_scope() as session:
            # Count before deletion
            face_count = session.query(FaceEmbedding).count()
            cluster_count = session.query(PersonCluster).count()
            suggestion_count = session.query(ClusterSuggestion).count()
            
            print(f"\nFound:")
            print(f"  - {face_count} face embeddings")
            print(f"  - {cluster_count} person clusters")
            print(f"  - {suggestion_count} suggestions")
            
            # Delete all
            session.query(ClusterSuggestion).delete()
            session.query(FaceEmbedding).delete()
            session.query(PersonCluster).delete()
            
            print("\n✅ All cleared! Ready for new model run.")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clear_old_embeddings()
