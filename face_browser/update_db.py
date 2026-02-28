import sys
from pathlib import Path
from sqlalchemy import inspect, text

# Add backend directory to path
project_root = Path(__file__).resolve().parent
backend_dir = project_root / 'backend'
sys.path.insert(0, str(backend_dir))

def update_database():
    """
    Applies necessary schema changes to the database using the backend's SQLAlchemy engine.
    This function should be run from the 'face_browser' directory.
    It now includes a manual check and ALTER TABLE for the 'objects_json' column.
    """
    try:
        from models import Base
        from database import engine
        print("Creating/updating database tables with SQLAlchemy...")

        # --- Manual Migrations for 'media_files' table ---
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('media_files')]

        # 1) Add objects_json if missing
        if 'objects_json' not in columns:
            print("Column 'objects_json' not found in 'media_files' table. Adding it...")
            with engine.connect() as connection:
                connection.execute(text('ALTER TABLE media_files ADD COLUMN objects_json TEXT'))
                connection.commit()
            print("...column 'objects_json' added successfully.")
        else:
            print("Column 'objects_json' already exists in 'media_files' table.")

        # 2) Add pet_count if missing
        if 'pet_count' not in columns:
            print("Column 'pet_count' not found in 'media_files' table. Adding it...")
            with engine.connect() as connection:
                connection.execute(text('ALTER TABLE media_files ADD COLUMN pet_count INTEGER DEFAULT 0'))
                connection.commit()
            print("...column 'pet_count' added successfully.")
        else:
            print("Column 'pet_count' already exists in 'media_files' table.")
        # --- End Manual Migrations ---

        print("\nEnsuring all other tables are created (if they don't exist)...")
        Base.metadata.create_all(bind=engine)
    print("\nSUCCESS: Database schema updated successfully.")
    print("Tables ensured: media_files, face_embeddings, person_clusters, merge_events, delete_log, task_progress, cluster_suggestions, pet_clusters, pet_embeddings")
    except Exception as e:
        print(f"An error occurred during database update: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    update_database()
