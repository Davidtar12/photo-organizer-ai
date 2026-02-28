# SOLUTION: Install Packages into .venv

## The Situation
- ✅ `.venv` exists and is functional
- ❌ `photoenv` exists but is incomplete (no Scripts folder)
- ❌ `.venv` is missing required packages

## Quick Fix: Install All Required Packages

```powershell
# 1. Activate .venv (you're already using it)
.venv\Scripts\Activate.ps1

# 2. Install all requirements
cd "C:\dscodingpython\File organizers\face_browser\backend"
pip install -r requirements.txt

# This will install:
# - Flask, SQLAlchemy (web framework)
# - opencv-python (cv2)
# - insightface (face detection)
# - ultralytics (YOLO)
# - faiss-gpu (clustering)
# - All other dependencies

# 3. Run database migration
cd C:\dscodingpython
python update_db.py

# 4. Run integration tests
cd "C:\dscodingpython\File organizers"
python test_integration.py

# 5. Run pipeline
python face_browser\run_pipeline.py

# 6. Start server
python run_backend.py
```

## One-Line Install Command

```powershell
# From "File organizers" directory with .venv activated:
pip install Flask==3.0.3 Flask-Cors==4.0.1 SQLAlchemy==2.0.23 alembic==1.13.2 marshmallow==3.21.1 marshmallow-sqlalchemy==0.29.0 DeepFace==0.0.92 insightface==0.7.3 onnxruntime-gpu==1.16.3 ultralytics==8.0.220 numpy==1.26.4 opencv-python==4.10.0.84 Pillow==10.4.0 scikit-learn==1.5.1 scipy==1.11.4 faiss-gpu==1.7.2 psutil==6.0.0 python-dotenv==1.0.1 "tensorflow[and-cuda]>=2.15.0"
```

## Or Use Requirements File (Recommended)

```powershell
(.venv) PS C:\dscodingpython\File organizers> cd face_browser\backend
(.venv) PS C:\dscodingpython\File organizers\face_browser\backend> pip install -r requirements.txt
```

## Expected Installation Time
- ~5-10 minutes depending on your internet speed
- TensorFlow + CUDA packages are large (~2-3 GB)

## After Installation, Verify:

```powershell
python -c "import cv2; print('✓ OpenCV')"
python -c "import flask; print('✓ Flask')"
python -c "import insightface; print('✓ InsightFace')"
python -c "import ultralytics; print('✓ YOLO')"
python -c "import faiss; print('✓ FAISS')"
```

All should print without errors!
