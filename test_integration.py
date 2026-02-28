"""Quick integration test to verify all components work together.

This script performs basic smoke tests without processing the full dataset.
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "face_browser" / "backend"
sys.path.insert(0, str(backend_dir))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from models import Base, MediaFile, FaceEmbedding, PersonCluster, ClusterSuggestion, TaskProgress
        print("  ✓ Models imported successfully")
    except ImportError as e:
        print(f"  ✗ Model import failed: {e}")
        return False
    
    try:
        from database import engine, session_scope
        print("  ✓ Database imported successfully")
    except ImportError as e:
        print(f"  ✗ Database import failed: {e}")
        return False
    
    try:
        from services.face_indexer import FaceIndexer
        from services.face_clusterer import FaceClusterer
        from services.suggestion_generator import SuggestionGenerator
        print("  ✓ Services imported successfully")
    except ImportError as e:
        print(f"  ✗ Service import failed: {e}")
        return False
    
    try:
        from routes import persons, suggestions, system
        print("  ✓ Routes imported successfully")
    except ImportError as e:
        print(f"  ✗ Routes import failed: {e}")
        return False
    
    try:
        from logging_config import setup_logging, get_logger
        print("  ✓ Logging config imported successfully")
    except ImportError as e:
        print(f"  ✗ Logging config import failed: {e}")
        return False
    
    return True


def test_database():
    """Test database connectivity and table creation."""
    print("\nTesting database...")
    
    try:
        from models import Base
        from database import engine
        
        # Create tables
        Base.metadata.create_all(engine)
        print("  ✓ Database tables created/verified")
        
        # Check that all expected tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        required_tables = [
            'media_files', 
            'face_embeddings', 
            'person_clusters', 
            'cluster_suggestions',
            'task_progress'
        ]
        
        for table in required_tables:
            if table in tables:
                print(f"  ✓ Table '{table}' exists")
            else:
                print(f"  ✗ Table '{table}' missing")
                return False
        
        # Verify new column exists
        columns = [col['name'] for col in inspector.get_columns('media_files')]
        if 'objects_json' in columns:
            print("  ✓ Column 'objects_json' exists in media_files")
        else:
            print("  ✗ Column 'objects_json' missing from media_files")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Database test failed: {e}")
        return False


def test_logging():
    """Test logging configuration."""
    print("\nTesting logging...")
    
    try:
        from logging_config import setup_logging, get_logger
        
        setup_logging(use_colors=True)
        logger = get_logger(__name__)
        
        logger.info("Test INFO message")
        logger.debug("Test DEBUG message")
        
        print("  ✓ Logging configured successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Logging test failed: {e}")
        return False


def test_flask_app():
    """Test Flask app creation."""
    print("\nTesting Flask app...")
    
    try:
        from app import create_app
        
        app = create_app()
        
        # Check blueprints are registered
        blueprints = list(app.blueprints.keys())
        required_bps = ['system', 'persons', 'suggestions']
        
        for bp in required_bps:
            if bp in blueprints:
                print(f"  ✓ Blueprint '{bp}' registered")
            else:
                print(f"  ✗ Blueprint '{bp}' not registered")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Flask app test failed: {e}")
        return False


def main():
    print("=" * 70)
    print("FACE BROWSER - INTEGRATION TEST")
    print("=" * 70)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Database", test_database),
        ("Logging", test_logging),
        ("Flask App", test_flask_app),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("\nNext steps:")
        print("  1. Run: python update_db.py (if not done yet)")
        print("  2. Run: python face_browser/run_pipeline.py")
        print("  3. Run: python run_backend.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
