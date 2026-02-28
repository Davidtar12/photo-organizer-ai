# Face Browser Webapp

Advanced AI-powered photo organization system with face recognition, clustering, and active learning.

## Features

- **Face Detection**: InsightFace SCRFD-10G (primary) + DeepFace (fallback)
- **Object Detection**: YOLOv8 for dogs and cats
- **Smart Clustering**: FAISS k-NN + connected components algorithm
- **Active Learning**: Human-in-the-loop cluster refinement
- **Progress Tracking**: Real-time updates for all operations
- **GPU Accelerated**: CUDA support for 10-20x speedup

## Quick Start

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Run integration tests
cd ../..
python test_integration.py

# 3. Migrate database
python update_db.py

# 4. Run full pipeline
python face_browser/run_pipeline.py

# 5. Start server
python run_backend.py
```

**Access at**: http://localhost:5052

## Documentation

- **[QUICK_REFERENCE.md](../QUICK_REFERENCE.md)** - Command cheat sheet
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Detailed setup & API docs
- **[IMPROVEMENTS_SUMMARY.md](../IMPROVEMENTS_SUMMARY.md)** - Recent changes & architecture

## Tech Stack

**Backend:**
- Flask + SQLAlchemy (REST API)
- InsightFace (face detection & embeddings)
- FAISS GPU (similarity search)
- YOLOv8 (object detection)
- SQLite (database)

**Frontend:**
- React + Vite
- Thumbnail caching
- Safe trash folder for deletions

## Recent Improvements (Oct 2025)

✅ Upgraded to InsightFace SCRFD-10G (2-3x faster)  
✅ Added YOLO object detection (dogs/cats)  
✅ Active learning system for cluster refinement  
✅ Progress tracking in all services  
✅ Centralized logging with colors  
✅ Complete REST API for suggestions  

See [IMPROVEMENTS_SUMMARY.md](../IMPROVEMENTS_SUMMARY.md) for details.
