# Face Browser - Quick Command Reference

## Setup (First Time Only)

```powershell
# Navigate to project
cd "c:\dscodingpython\File organizers"

# Install dependencies
cd face_browser\backend
pip install -r requirements.txt
cd ..\..

# Run integration tests
python test_integration.py

# Migrate database (adds cluster_suggestions table and objects_json column)
python update_db.py
```

## Daily Operations

### Option 1: Full Pipeline (All Stages)
```powershell
# Runs: indexing → clustering → suggestions
python face_browser\run_pipeline.py
```

### Option 2: Individual Services

```powershell
# Face indexing only
cd face_browser\backend
python -m services.face_indexer

# Clustering only (requires indexed faces)
python -m services.face_clusterer

# Generate suggestions (requires clusters)
python -m services.suggestion_generator
```

### Start Web Server
```powershell
# From "File organizers" directory
python run_backend.py
```
**Access at**: http://localhost:5052

## API Quick Reference

```bash
# Get pending suggestions (20 max)
curl http://localhost:5052/api/suggestions/pending?limit=20

# Accept suggestion (assign face to cluster)
curl -X POST http://localhost:5052/api/suggestions/123/accept

# Reject suggestion (keep current assignment)
curl -X POST http://localhost:5052/api/suggestions/123/reject

# Reject & create new cluster
curl -X POST http://localhost:5052/api/suggestions/123/reject \
  -H "Content-Type: application/json" \
  -d '{"create_new_cluster": true}'

# Skip suggestion
curl -X POST http://localhost:5052/api/suggestions/123/skip

# Get review statistics
curl http://localhost:5052/api/suggestions/stats

# List all person clusters
curl http://localhost:5052/api/persons/

# Health check
curl http://localhost:5052/api/health
```

## Troubleshooting

### GPU Not Working
```powershell
# Check CUDA
nvidia-smi

# Pipeline will auto-fallback to CPU if GPU unavailable
```

### Database Locked
```powershell
# Stop web server before running pipeline
# Only one process can access SQLite database at a time
```

### Import Errors
```powershell
# Activate virtual environment first
photoenv\Scripts\activate

# Then run commands
python face_browser\run_pipeline.py
```

### Reset Everything
```powershell
# Delete database and start fresh
cd "c:\dscodingpython\File organizers\face_browser\data"
del database.db

# Re-run migration
cd ..\..\..\..
python update_db.py

# Re-run pipeline
python face_browser\run_pipeline.py
```

## File Locations

```
c:\dscodingpython\File organizers\
├── update_db.py                    # Database migration script
├── test_integration.py             # Integration tests
├── run_backend.py                  # Web server launcher
├── IMPROVEMENTS_SUMMARY.md         # What changed & why
└── face_browser\
    ├── run_pipeline.py             # Full pipeline runner
    ├── INTEGRATION_GUIDE.md        # Detailed documentation
    └── backend\
        ├── app.py                  # Flask app
        ├── models.py               # Database schema
        ├── logging_config.py       # Centralized logging
        ├── requirements.txt        # Dependencies
        ├── routes\
        │   └── suggestions.py      # Active learning API
        └── services\
            ├── face_indexer.py     # Photo scanning + detection
            ├── face_clusterer.py   # k-NN clustering
            └── suggestion_generator.py  # Active learning
```

## Performance Tips

1. **GPU**: Ensure CUDA is available for 10-20x speedup
2. **Batch Size**: Adjust `PROGRESS_EVERY` in config.py for memory vs speed tradeoff
3. **Parallel Processing**: Services are single-threaded (SQLite limitation)
4. **Incremental**: `face_indexer.run(full_scan=False)` only processes new/changed files

## Common Workflows

### Initial Setup
```powershell
python update_db.py
python face_browser\run_pipeline.py
python run_backend.py
```

### Add New Photos
```powershell
# Copy photos to Organized folder, then:
python face_browser\backend\services\face_indexer.py  # Index new faces
python face_browser\backend\services\face_clusterer.py  # Re-cluster
python face_browser\backend\services\suggestion_generator.py  # Update suggestions
```

### Review & Improve Clusters
```powershell
# 1. Start server
python run_backend.py

# 2. Use API or build frontend UI to review suggestions at:
#    http://localhost:5052/api/suggestions/pending

# 3. Accept/reject suggestions via POST requests
```

---
**Need Help?** See `INTEGRATION_GUIDE.md` for detailed documentation
