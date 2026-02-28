# SETUP ISSUE - READ THIS FIRST!

## ❌ Problem Detected

You're using the **wrong virtual environment**!

- **Current**: `.venv` (missing all required packages)
- **Required**: `photoenv` (has face detection packages)

## ✅ Solution

### Step 1: Activate the Correct Environment

```powershell
# Deactivate current environment
deactivate

# Activate photoenv
C:\dscodingpython\photoenv\Scripts\Activate.ps1
```

### Step 2: Install Missing Packages (if needed)

```powershell
# Navigate to backend
cd "C:\dscodingpython\File organizers\face_browser\backend"

# Install all dependencies
pip install -r requirements.txt

# Go back to File organizers
cd "C:\dscodingpython\File organizers"
```

### Step 3: Run Database Migration

```powershell
# Run from the dscodingpython directory (parent of "File organizers")
cd C:\dscodingpython
python update_db.py
```

### Step 4: Run Integration Tests

```powershell
cd "C:\dscodingpython\File organizers"
python test_integration.py
```

Expected output: **All 4 tests should pass**

### Step 5: Run Pipeline

```powershell
python face_browser\run_pipeline.py
```

### Step 6: Start Web Server

```powershell
python run_backend.py
```

## 📋 Quick Copy-Paste Commands

```powershell
# Complete setup sequence
deactivate
C:\dscodingpython\photoenv\Scripts\Activate.ps1
cd "C:\dscodingpython\File organizers\face_browser\backend"
pip install -r requirements.txt
cd C:\dscodingpython
python update_db.py
cd "C:\dscodingpython\File organizers"
python test_integration.py
python face_browser\run_pipeline.py
python run_backend.py
```

## 🔍 How to Verify You're in the Right Environment

```powershell
# Should show: (photoenv) at the start of your prompt
# NOT (.venv)

# Check Python location
python -c "import sys; print(sys.executable)"
# Should show: C:\dscodingpython\photoenv\Scripts\python.exe

# Check if packages are installed
python -c "import cv2; print('✓ OpenCV installed')"
python -c "import flask; print('✓ Flask installed')"
python -c "import insightface; print('✓ InsightFace installed')"
```

## ⚠️ Common Mistakes

1. **Using .venv instead of photoenv** ← YOUR CURRENT ISSUE
2. Running update_db.py from wrong directory
3. Not installing requirements.txt
4. Having multiple terminals with different environments

## 🎯 Next Steps

1. Close all PowerShell terminals
2. Open ONE new PowerShell terminal
3. Run the "Quick Copy-Paste Commands" above
4. Everything should work!

---
**Issue**: `.venv` is missing opencv-python, flask, insightface, ultralytics, etc.  
**Fix**: Use `photoenv` which has all packages installed.
