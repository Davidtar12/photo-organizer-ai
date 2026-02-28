# Pet Recognition System

## Overview

The Face Browser now supports **individual pet recognition** - not just detecting "a dog" but recognizing "Fido" vs "Sparky".

## Architecture

### 1. Detection: YOLOv12
- **Model**: `yolov12n.pt` (Nano - fast and accurate)
- **Purpose**: Find all dog/cat bounding boxes in photos
- **Classes**: `dog`, `cat` (from COCO dataset)
- **Auto-download**: First run downloads ~6MB weights

### 2. Recognition: Custom fastai Model
- **Model**: `pets.pkl` (trained by you)
- **Purpose**: Identify individual pets by name ("Fido", "Sparky", etc.)
- **Training**: 50-100 labeled photos per pet
- **Architecture**: ResNet34 fine-tuned on your pet photos

### 3. Database Schema

```sql
-- New tables for pets (separate from faces)
CREATE TABLE pet_clusters (
    id INTEGER PRIMARY KEY,
    display_name TEXT,           -- "Fido", "Sparky", etc.
    species TEXT,                -- "dog", "cat"
    primary_pet_id INTEGER,      -- Best photo of this pet
    is_hidden BOOLEAN DEFAULT 0
);

CREATE TABLE pet_embeddings (
    id INTEGER PRIMARY KEY,
    media_id INTEGER,
    cluster_id INTEGER,          -- Links to pet_clusters
    embedding BLOB,              -- 512-dim vector from fastai
    embedding_dim INTEGER,
    bbox_json TEXT,
    detection_confidence REAL,   -- From YOLO
    species TEXT,                -- "dog" or "cat"
    model_name TEXT,             -- "pets.pkl"
    detector TEXT                -- "yolov12n"
);

-- MediaFile updated
ALTER TABLE media_files ADD COLUMN pet_count INTEGER DEFAULT 0;
```

## Workflow

### Phase 1: Initial Setup (One-time)

1. **Run pipeline without pet recognizer** (YOLO detection only):
   ```bash
   cd "c:\dscodingpython\File organizers\face_browser"
   C:\dscodingpython\photoenv\Scripts\python.exe run_pipeline.py
   ```
   - YOLOv12 detects all dogs/cats
   - No embeddings yet (pets.pkl doesn't exist)
   - Creates unlabeled `PetEmbedding` records

2. **Label pets using Jupyter notebook**:
   ```bash
   jupyter notebook notebooks/train_pet_recognizer.ipynb
   ```
   - Extracts 500 pet crops
   - You manually sort into folders: `Fido/`, `Sparky/`, etc.
   - Needs 50-100 photos per pet

3. **Train fastai model** (in notebook):
   - Fine-tunes ResNet34 for 5 epochs (~5-10 minutes)
   - Exports `backend/models/pets.pkl`
   - Now the pipeline can recognize individual pets!

4. **Re-run pipeline** (with pet recognizer):
   ```bash
   python run_pipeline.py
   ```
   - YOLO detects pets
   - `pets.pkl` identifies each one
   - Stores 512-dim embeddings
   - Ready for clustering

### Phase 2: Automatic Clustering

The same 3-pass clustering system runs on pets:

**Pass 1 (Strict)**: High-confidence seeds
- `k=50` nearest neighbors
- Distance threshold: `0.315`
- Creates initial pet clusters

**Pass 2 (Centroid Assignment)**: Assign unclustered pets
- Threshold: `0.50`
- Assigns "noise" to nearest cluster

**Pass 3 (Auto-Merge)**: Merge similar clusters
- Threshold: `0.35`
- Merges duplicate pet clusters

## API Endpoints

### Pet Browsing
```http
GET /api/pets/                    # List all pet clusters
GET /api/pets/{id}                # Get specific pet cluster
GET /api/pets/{id}/photos         # All photos of this pet
```

### Pet Management
```http
PATCH /api/pets/{id}              # Rename pet, hide, set primary photo
POST /api/pets/{id}/merge/{target_id}  # Merge duplicate clusters
DELETE /api/pets/{id}             # Hide pet cluster
```

### Unclustered Pets
```http
GET /api/pets/unclustered         # Pets not assigned to any cluster
POST /api/pets/{pet_id}/assign    # Manually assign pet to cluster
```

## Frontend UI

### Separate `/pets` View

```html
<!-- Similar to person browser, but for pets -->
<div class="pet-browser">
  <div class="pet-grid">
    <!-- Each pet cluster shows:
         - Primary pet photo
         - Pet name (e.g., "Fido")
         - Species tag (🐕/🐱)
         - Photo count
    -->
  </div>
</div>
```

## File Structure

```
face_browser/
├── backend/
│   ├── models.py                    # ✅ PetCluster, PetEmbedding added
│   ├── services/
│   │   └── face_indexer.py          # ✅ YOLOv12, pet detection/embedding
│   ├── routes/
│   │   ├── persons.py               # Existing face routes
│   │   └── pets.py                  # 🆕 TODO: Pet routes
│   └── models/
│       └── pets.pkl                 # 🆕 Trained fastai model (created by notebook)
├── notebooks/
│   └── train_pet_recognizer.ipynb   # ✅ Training workflow
└── frontend/
    ├── index.html                   # Existing person browser
    └── pets.html                    # 🆕 TODO: Pet browser
```

## Status

### ✅ Completed
- [x] Database schema (`PetCluster`, `PetEmbedding`)
- [x] YOLOv12 integration in `face_indexer.py`
- [x] Pet detection pipeline
- [x] Pet embedding extraction (placeholder for `pets.pkl`)
- [x] Training notebook (`train_pet_recognizer.ipynb`)

### 🔄 In Progress
- [ ] Backend API routes (`/api/pets/`)
- [ ] Frontend pet browser UI
- [ ] Pet clustering service (reuse `face_clusterer.py` logic)

### 📋 TODO
1. **Install dependencies**:
   ```bash
   pip install fastai torch torchvision ultralytics
   ```

2. **Run pipeline** (initial YOLO detection only)

3. **Label pets** (notebook)

4. **Train model** (notebook exports `pets.pkl`)

5. **Re-run pipeline** (with pet recognition)

6. **Build API routes** and frontend UI

## Benefits of This Approach

1. **Separation of Concerns**: Pets and faces are completely independent
   - Different FAISS indexes (can't mix 512-dim vectors from different models)
   - Different clustering parameters (pets might need different thresholds)
   - Different UX (pet browser vs person browser)

2. **Scalability**: Each pet model is custom-trained for your specific pets
   - No need for massive pre-trained model
   - 50-100 photos per pet is sufficient
   - Can retrain anytime with more data

3. **Flexibility**: The notebook workflow is:
   - Visual (see what you're labeling)
   - Interactive (tune training parameters)
   - Reusable (add new pets anytime)

## Notes

- **Performance**: YOLOv12 + TensorRT will be 2-4x faster than YOLOv8
- **Accuracy**: fastai ResNet34 is very accurate with 50-100 samples per class
- **Storage**: Each pet embedding is 512 floats = 2KB (same as faces)
