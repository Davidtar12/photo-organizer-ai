# Three-Pass Clustering System - Tuning Guide

## Overview
This system solves the "68k unclustered faces" problem using **three automatic passes** with different strategies and thresholds. Each pass is a "knob" you can tune.

---

## 🎯 The Three Passes

### Pass 1: Seed Clusters (STRICT)
**Method:** `cluster_all(distance_threshold=0.315)`  
**File:** `backend/services/face_clusterer.py`

**Purpose:**  
Create **pure, high-confidence clusters**. This pass intentionally leaves many faces unclustered ("noise") to avoid merge errors.

**Strategy:**
- k-NN graph (k=50 neighbors per face)
- L2 distance threshold: **0.315** (strict)
- Connected components → clusters
- Drop clusters < 2 faces

**Threshold Tuning:**
- **Lower (0.25-0.30)**: More strict → purer clusters, more noise
- **Higher (0.35-0.40)**: More lenient → fewer noise faces, but risk of merge errors
- **Recommended:** 0.315 (balanced strict)

**Expected Results:**
- ~5-15k faces in seed clusters
- ~60-70k faces left as "noise" (cluster_id=NULL)
- Zero merge errors (all clusters are pure)

---

### Pass 2: Centroid Assignment (LENIENT)
**Method:** `assign_noise_to_centroids(distance_threshold=0.5)`  
**File:** `backend/services/face_clusterer.py`

**Purpose:**  
**Clean up the noise pile** by assigning hard-to-match faces to the seed clusters. This is the main knob for reducing unclustered faces.

**Strategy:**
- Compute centroids (mean embedding) for all seed clusters
- For each noise face, find nearest cluster centroid
- Assign if distance < threshold
- Batch updates every 5000 faces

**Threshold Tuning:**
- **Lower (0.4)**: More conservative → fewer assignments, more noise remains
- **Higher (0.5-0.6)**: More aggressive → more assignments, less noise
- **Recommended:** Start at 0.5, increase if too many unclustered remain

**Expected Results:**
- ~40-60k faces assigned to clusters (50-80% reduction in noise)
- ~10-20k faces still unclustered (the truly hard cases)
- Some "split errors" (same person in 2+ clusters) are expected

**This is YOUR main tuning knob!** If you still have 30k+ unclustered after Pass 2:
- Increase threshold to 0.55 or 0.6
- Re-run just Pass 2 (no need to re-run Pass 1)

---

### Pass 3: Cluster Merging (MODERATE)
**Method:** `merge_similar_clusters(distance_threshold=0.35)`  
**File:** `backend/services/face_clusterer.py`

**Purpose:**  
**Fix split errors automatically**. Merges clusters whose centroids are very close (likely the same person split due to pose/lighting).

**Strategy:**
- Compute centroids for all clusters (min 3 faces per cluster)
- Build k-NN graph between centroids (k=10)
- Find connected components (merge groups)
- Merge into largest cluster in each group
- Delete empty source clusters

**Threshold Tuning:**
- **Lower (0.3)**: More strict → fewer merges, more manual work
- **Higher (0.4)**: More aggressive → risk merging different people
- **Recommended:** 0.35 (moderate/balanced)

**Expected Results:**
- ~50-200 clusters merged (depends on Pass 1 strictness)
- Eliminates most obvious "Bob-Frontal" + "Bob-With-Hat" splits
- Final cluster count reduced by 10-30%

---

## 🔧 Tuning Workflow

### Step 1: Run with defaults
```bash
python run_pipeline.py
```

**Default thresholds:**
- Pass 1: 0.315 (strict)
- Pass 2: 0.5 (moderate lenient)
- Pass 3: 0.35 (moderate)

### Step 2: Check results
```bash
python verify_db.py
```

Look at:
- Total embeddings
- Total clusters created
- **Most important:** Unclustered face count (cluster_id=NULL)

### Step 3: Adjust if needed

**Problem: Too many unclustered faces (>20k)**
- **Solution:** Increase Pass 2 threshold
- Edit `run_pipeline.py`, line ~95:
  ```python
  assigned_count = clusterer.assign_noise_to_centroids(distance_threshold=0.55)  # Was 0.5
  ```
- Re-run pipeline

**Problem: Too many clusters for same person**
- **Solution:** Increase Pass 3 threshold
- Edit `run_pipeline.py`, line ~110:
  ```python
  merged_count = clusterer.merge_similar_clusters(distance_threshold=0.4)  # Was 0.35
  ```
- Re-run pipeline

**Problem: Clusters contain wrong people (merge errors)**
- **Solution:** Decrease Pass 1 threshold (more strict)
- Edit `run_pipeline.py`, line ~83:
  ```python
  clusterer.cluster_all(distance_threshold=0.28)  # Was 0.315
  ```
- **WARNING:** This will increase noise pile, so also increase Pass 2 threshold

---

## 📊 Understanding Distance Thresholds

### L2 Distance on Normalized Vectors
All embeddings are L2-normalized (unit vectors), so:
- L2 distance ≈ cosine distance
- Lower distance = more similar
- Higher distance = less similar

### Rough Guidelines (AdaFace IR100)
- **0.0-0.25:** Same person, same conditions (frontal, good light)
- **0.25-0.35:** Same person, different conditions (pose, lighting, age)
- **0.35-0.5:** Possibly same person (borderline cases)
- **0.5-0.7:** Probably different people (but could be twins/siblings)
- **0.7+:** Definitely different people

### Threshold Ranges by Pass
- **Pass 1:** 0.25-0.35 (strict, high purity)
- **Pass 2:** 0.4-0.6 (lenient, noise cleanup)
- **Pass 3:** 0.3-0.4 (moderate, merge similar)

---

## 🎯 Target Metrics

### Good Results
- **Pass 1:** 5-20k faces clustered, 60-70k noise
- **Pass 2:** 50-70% noise reduction → 10-20k unclustered remain
- **Pass 3:** 10-30% cluster count reduction
- **Final:** <15k unclustered faces (manual review is feasible)

### Red Flags
- **Pass 1:** >30k faces clustered → too lenient, will have merge errors
- **Pass 2:** <30% noise reduction → too strict, increase threshold
- **Pass 3:** >50% clusters merged → too aggressive, different people merging

---

## 🧪 Advanced Tuning

### Per-Pass Re-runs
You can re-run individual passes without re-indexing:

**Re-run Pass 2 only:**
```python
from backend.services.face_clusterer import FaceClusterer
clusterer = FaceClusterer(min_cluster_size=2)
clusterer.assign_noise_to_centroids(distance_threshold=0.6)  # Try higher
```

**Re-run Pass 3 only:**
```python
clusterer.merge_similar_clusters(distance_threshold=0.4)  # Try higher
```

### A/B Testing
Keep a backup database to compare:
```bash
copy "data\face_index.db" "data\face_index_backup.db"
# Try different thresholds
# Compare results
```

### Visualization (Advanced)
Use the frontend to spot-check:
1. Start backend: `python run_backend.py`
2. Serve frontend: `cd frontend; python -m http.server 5173`
3. Visit: http://localhost:5173
4. Check for:
   - Merge errors (different people in same cluster)
   - Split errors (same person in multiple clusters)

---

## 🚨 Common Mistakes

### ❌ DON'T: Use lenient threshold in Pass 1
```python
clusterer.cluster_all(distance_threshold=0.5)  # BAD! Creates merge errors
```

**Why:** Creates polluted clusters. Merge errors are "eternal nightmare" to fix.

### ❌ DON'T: Use strict threshold in Pass 2
```python
clusterer.assign_noise_to_centroids(distance_threshold=0.3)  # BAD! Assigns nothing
```

**Why:** Defeats the purpose. You'll still have 60k unclustered.

### ✅ DO: Use strict→lenient→moderate progression
```python
# Pass 1: Strict (0.315) → Pure seed clusters
# Pass 2: Lenient (0.5) → Assign most noise
# Pass 3: Moderate (0.35) → Merge obvious splits
```

---

## 📝 Tuning Log Template

Keep notes on what works:

```markdown
## Run 1 (Baseline)
- Pass 1: 0.315 → 8,421 faces in 1,234 clusters
- Pass 2: 0.5 → 42,103 assigned, 25,588 unclustered
- Pass 3: 0.35 → 143 clusters merged
- **Result:** Too many unclustered

## Run 2 (Increase Pass 2)
- Pass 1: 0.315 → 8,421 faces in 1,234 clusters
- Pass 2: 0.55 → 51,203 assigned, 16,488 unclustered
- Pass 3: 0.35 → 143 clusters merged
- **Result:** Good! <17k unclustered is manageable

## Run 3 (Final Tune)
- Pass 1: 0.315 → 8,421 faces in 1,234 clusters
- Pass 2: 0.55 → 51,203 assigned, 16,488 unclustered
- Pass 3: 0.38 → 187 clusters merged (increased merge)
- **Result:** Perfect! Ready for production
```

---

## 🎓 Philosophy Recap

### The "Split vs Merge" Tradeoff

**Split Errors** (same person in 2+ clusters):
- ✅ Easy to fix: Click "merge" in UI
- ✅ Reversible: Can always split back
- ✅ Safe: No data contamination

**Merge Errors** (different people in 1 cluster):
- ❌ Nightmare to fix: Manual inspection of hundreds of photos
- ❌ Irreversible: Hard to un-mix faces
- ❌ Contaminating: Ruins cluster purity

**Therefore:** Always prefer split errors over merge errors.

### The Three-Pass Strategy

1. **Pass 1:** Create perfect seed clusters (accept lots of noise)
2. **Pass 2:** Reduce noise pile aggressively (accept some splits)
3. **Pass 3:** Auto-merge obvious splits (be conservative)

This gives you the "best of all worlds":
- Pure clusters (no merge errors)
- Minimal unclustered faces (<10-20k)
- Automatic split fixing
- Fast manual cleanup if needed

---

## 🏁 Quick Reference

| Pass | Method | Threshold | Purpose | Tune When |
|------|--------|-----------|---------|-----------|
| **1** | `cluster_all` | 0.315 | Create pure seeds | Merge errors appear |
| **2** | `assign_noise_to_centroids` | 0.5 | Reduce unclustered | >20k unclustered |
| **3** | `merge_similar_clusters` | 0.35 | Fix split errors | Too many person duplicates |

**Main knob:** Pass 2 threshold (0.5 → 0.6 if needed)  
**Safety valve:** Pass 1 strictness (keep <0.35 to avoid merge errors)  
**Polish:** Pass 3 merging (0.35-0.4 range)

---

Happy clustering! 🎉
