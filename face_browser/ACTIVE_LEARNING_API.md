# Active Learning API - Documentation

## Overview

The Active Learning API is the "fourth pass" of the face clustering system - the human-in-the-loop component for handling the remaining hard cases after automatic three-pass clustering.

After the automatic clustering reduces 68k unclustered faces down to ~10-20k, these endpoints help you quickly review and assign the remaining faces.

---

## New Endpoints

### 1. GET `/api/persons/unclustered`

**Purpose:** List all faces that are still unclustered (cluster_id is NULL)

**Query Parameters:**
- `limit` (int, default=100): Number of faces per page
- `offset` (int, default=0): Pagination offset

**Response:**
```json
{
  "total": 15234,
  "limit": 100,
  "offset": 0,
  "faces": [
    {
      "id": 12345,
      "media_id": 789,
      "embedding_index": 0,
      "bbox": {"x": 100, "y": 150, "w": 200, "h": 200},
      "detection_confidence": 0.99,
      "model_name": "AdaFace",
      "detector": "SCRFD-10G",
      "thumbnail_path": "/path/to/thumb.jpg",
      "cluster_id": null
    },
    ...
  ]
}
```

**Usage:**
```javascript
// Fetch first page of unclustered faces
fetch('/api/persons/unclustered?limit=50&offset=0')
  .then(r => r.json())
  .then(data => {
    console.log(`Total unclustered: ${data.total}`);
    // Display faces in UI for review
  });
```

---

### 2. GET `/api/persons/face/<face_id>/suggestions`

**Purpose:** Get top 5 cluster suggestions for a face using FAISS similarity search

**Parameters:**
- `face_id` (int): ID of the face to find suggestions for

**Response:**
```json
{
  "face_id": 12345,
  "suggestions": [
    {
      "cluster_id": 42,
      "similarity": 0.8934,
      "distance": 0.1066,
      "person_name": "Bob Smith",
      "face_count": 127,
      "sample_face_id": 5678
    },
    {
      "cluster_id": 17,
      "similarity": 0.8512,
      "distance": 0.1488,
      "person_name": "Alice Johnson",
      "face_count": 89,
      "sample_face_id": 2341
    },
    {
      "cluster_id": 103,
      "similarity": 0.7834,
      "distance": 0.2166,
      "person_name": "Person 103",
      "face_count": 34,
      "sample_face_id": 9876
    },
    {
      "cluster_id": 55,
      "similarity": 0.7123,
      "distance": 0.2877,
      "person_name": "Charlie Davis",
      "face_count": 156,
      "sample_face_id": 4321
    },
    {
      "cluster_id": 88,
      "similarity": 0.6745,
      "distance": 0.3255,
      "person_name": "Person 88",
      "face_count": 12,
      "sample_face_id": 7654
    }
  ],
  "num_clusters_total": 1234,
  "threshold_recommendation": {
    "high_confidence": 0.85,
    "moderate": 0.75,
    "low": 0.65
  }
}
```

**Algorithm:**
1. Load face embedding (512-dim AdaFace vector)
2. Load all cluster centroids (mean embedding per cluster)
3. Filter clusters with <3 faces (too small/noisy)
4. Build temporary FAISS IndexFlatIP (Inner Product = cosine similarity)
5. Search for k=5 nearest centroids
6. Return sorted by similarity (highest first)

**Interpreting Similarity Scores:**
- **>0.85:** Very likely same person (high confidence)
- **0.75-0.85:** Probably same person (moderate confidence)
- **0.65-0.75:** Maybe same person (low confidence)
- **<0.65:** Probably different person (reject)

**Usage:**
```javascript
// Get suggestions for unclustered face
fetch(`/api/persons/face/12345/suggestions`)
  .then(r => r.json())
  .then(data => {
    if (data.suggestions.length === 0) {
      // No good matches - suggest creating new person
      showCreateNewPersonUI();
    } else {
      // Show top 5 suggestions with thumbnails
      data.suggestions.forEach(sug => {
        showSuggestion({
          name: sug.person_name,
          similarity: `${(sug.similarity * 100).toFixed(1)}%`,
          thumbnailUrl: `/api/persons/face/${sug.sample_face_id}/thumbnail`,
          onClick: () => assignFaceToCluster(12345, sug.cluster_id)
        });
      });
    }
  });
```

---

### 3. POST `/api/persons/face/<face_id>/assign`

**Purpose:** Assign a face to a cluster (or create new cluster)

**Parameters:**
- `face_id` (int): ID of the face to assign

**Request Body:**
```json
{
  "cluster_id": 42,        // Assign to this existing cluster
  "create_new": false      // Or set to true to create new cluster
}
```

**Response (assign to existing):**
```json
{
  "status": "assigned",
  "face_id": 12345,
  "cluster_id": 42,
  "message": "Assigned face to cluster 42"
}
```

**Response (create new):**
```json
{
  "status": "created_new",
  "face_id": 12345,
  "cluster_id": 1567,
  "message": "Created new cluster for this face"
}
```

**Usage:**
```javascript
// Assign to existing cluster
fetch(`/api/persons/face/12345/assign`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({cluster_id: 42, create_new: false})
})
.then(r => r.json())
.then(data => console.log(data.message));

// Create new cluster
fetch(`/api/persons/face/12345/assign`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({create_new: true})
})
.then(r => r.json())
.then(data => {
  console.log(`Created cluster ${data.cluster_id}`);
  // Redirect to cluster to name it
  window.location.href = `/person/${data.cluster_id}`;
});
```

---

## Complete Active Learning Workflow

### Step 1: Fetch Unclustered Faces
```javascript
const response = await fetch('/api/persons/unclustered?limit=1&offset=0');
const data = await response.json();

if (data.total === 0) {
  showMessage("All faces clustered! 🎉");
  return;
}

const face = data.faces[0];
showFaceImage(face.id, `/api/persons/face/${face.id}/thumbnail`);
```

### Step 2: Get Suggestions
```javascript
const sugResponse = await fetch(`/api/persons/face/${face.id}/suggestions`);
const suggestions = await sugResponse.json();

if (suggestions.suggestions.length === 0) {
  showOptions([
    {label: "Create New Person", action: () => createNewPerson(face.id)}
  ]);
} else {
  // Show top 5 suggestions
  suggestions.suggestions.forEach(sug => {
    showOption({
      label: `${sug.person_name} (${(sug.similarity * 100).toFixed(1)}%)`,
      thumbnail: `/api/persons/face/${sug.sample_face_id}/thumbnail`,
      confidence: sug.similarity > 0.85 ? 'high' : (sug.similarity > 0.75 ? 'med' : 'low'),
      onClick: () => assignToCluster(face.id, sug.cluster_id)
    });
  });
  
  // Always offer "None of these" option
  showOption({
    label: "None of these / New Person",
    onClick: () => createNewPerson(face.id)
  });
}
```

### Step 3: Assign Face
```javascript
async function assignToCluster(faceId, clusterId) {
  const response = await fetch(`/api/persons/face/${faceId}/assign`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cluster_id: clusterId, create_new: false})
  });
  
  const result = await response.json();
  showSuccess(result.message);
  
  // Move to next unclustered face
  loadNextUnclusteredFace();
}

async function createNewPerson(faceId) {
  const response = await fetch(`/api/persons/face/${faceId}/assign`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({create_new: true})
  });
  
  const result = await response.json();
  showSuccess(`Created new person (Cluster ${result.cluster_id})`);
  
  // Optionally prompt for name
  const name = prompt("Name this person:");
  if (name) {
    await fetch(`/api/persons/${result.cluster_id}`, {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({display_name: name})
    });
  }
  
  loadNextUnclusteredFace();
}
```

---

## Performance Characteristics

### Centroid Computation
- **Time:** O(n) where n = number of faces
- **Typical:** ~2-5 seconds for 77k faces
- **Optimization:** Centroids could be pre-computed and cached

### FAISS Search
- **Time:** O(log n) for k=5 search (very fast)
- **Typical:** <10ms for 1-2k clusters
- **Index Type:** IndexFlatIP (exact cosine similarity)

### Total Latency
- **First call:** ~2-5 seconds (computes centroids)
- **Subsequent calls:** <100ms (if centroids cached)
- **Recommendation:** Cache centroids in memory or Redis

---

## Optimization: Centroid Caching

For production, pre-compute centroids after Pass 3:

```python
# After run_pipeline.py completes Pass 3
from backend.services.face_clusterer import FaceClusterer
from pathlib import Path
import json
import numpy as np

clusterer = FaceClusterer()

# Compute all centroids
with session_scope() as session:
    results = session.query(FaceEmbedding.cluster_id, FaceEmbedding.embedding).filter(
        FaceEmbedding.cluster_id.isnot(None)
    ).all()
    
    cluster_embeddings = defaultdict(list)
    for cluster_id, emb_bytes in results:
        vec = np.frombuffer(emb_bytes, dtype=np.float32)
        cluster_embeddings[cluster_id].append(vec)
    
    centroids = {}
    for cluster_id, vecs in cluster_embeddings.items():
        if len(vecs) >= 3:
            centroid = np.mean(np.vstack(vecs), axis=0)
            centroid /= np.linalg.norm(centroid)
            centroids[cluster_id] = centroid.tolist()

# Save to JSON
cache_file = Path("data/cluster_centroids.json")
with open(cache_file, 'w') as f:
    json.dump(centroids, f)

print(f"Cached {len(centroids)} centroids to {cache_file}")
```

Then modify `get_face_suggestions()` to load from cache:
```python
# At top of persons.py
CENTROID_CACHE = None

def load_centroids():
    global CENTROID_CACHE
    if CENTROID_CACHE is None:
        cache_file = Path("data/cluster_centroids.json")
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
            CENTROID_CACHE = {int(k): np.array(v, dtype=np.float32) for k, v in data.items()}
    return CENTROID_CACHE

# In get_face_suggestions():
centroids_dict = load_centroids()
if not centroids_dict:
    # Fallback to computing on-the-fly
    centroids_dict = compute_centroids_from_db()
```

---

## Error Handling

### No Clusters Available
```json
{
  "face_id": 12345,
  "suggestions": [],
  "message": "No clusters available yet. Run clustering first.",
  "num_clusters_total": 0
}
```

### Face Not Found
```json
{
  "error": "Face not found"
}
```
HTTP 404

### Invalid Embedding
```json
{
  "error": "Invalid embedding dimension: 128"
}
```
HTTP 400

---

## UI Integration Example

### Simple Active Learning Page
```html
<!DOCTYPE html>
<html>
<head>
  <title>Review Unclustered Faces</title>
  <style>
    .face-preview { max-width: 300px; border: 2px solid #ccc; }
    .suggestion { display: inline-block; margin: 10px; cursor: pointer; }
    .suggestion img { width: 100px; height: 100px; object-fit: cover; }
    .high-confidence { border: 3px solid green; }
    .moderate-confidence { border: 3px solid orange; }
    .low-confidence { border: 3px solid red; }
  </style>
</head>
<body>
  <h1>Review Unclustered Faces</h1>
  <div id="progress"></div>
  <div id="face-container">
    <h2>Who is this?</h2>
    <img id="face-image" class="face-preview">
  </div>
  <div id="suggestions-container"></div>
  <button onclick="createNew()">None of these / New Person</button>
  <button onclick="skip()">Skip</button>

  <script>
    let currentFace = null;
    let offset = 0;

    async function loadNext() {
      const res = await fetch(`/api/persons/unclustered?limit=1&offset=${offset}`);
      const data = await res.json();
      
      document.getElementById('progress').textContent = 
        `Unclustered: ${data.total} remaining`;
      
      if (data.faces.length === 0) {
        alert('All done! 🎉');
        return;
      }
      
      currentFace = data.faces[0];
      document.getElementById('face-image').src = 
        `/api/persons/face/${currentFace.id}/thumbnail`;
      
      loadSuggestions(currentFace.id);
    }

    async function loadSuggestions(faceId) {
      const res = await fetch(`/api/persons/face/${faceId}/suggestions`);
      const data = await res.json();
      
      const container = document.getElementById('suggestions-container');
      container.innerHTML = '<h3>Suggestions:</h3>';
      
      data.suggestions.forEach(sug => {
        const div = document.createElement('div');
        div.className = 'suggestion';
        if (sug.similarity > 0.85) div.classList.add('high-confidence');
        else if (sug.similarity > 0.75) div.classList.add('moderate-confidence');
        else div.classList.add('low-confidence');
        
        div.innerHTML = `
          <img src="/api/persons/face/${sug.sample_face_id}/thumbnail">
          <p>${sug.person_name}</p>
          <p>${(sug.similarity * 100).toFixed(1)}% match</p>
        `;
        div.onclick = () => assign(currentFace.id, sug.cluster_id);
        container.appendChild(div);
      });
    }

    async function assign(faceId, clusterId) {
      await fetch(`/api/persons/face/${faceId}/assign`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cluster_id: clusterId})
      });
      loadNext();
    }

    async function createNew() {
      const res = await fetch(`/api/persons/face/${currentFace.id}/assign`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({create_new: true})
      });
      const data = await res.json();
      const name = prompt('Name this person:');
      if (name) {
        await fetch(`/api/persons/${data.cluster_id}`, {
          method: 'PATCH',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({display_name: name})
        });
      }
      loadNext();
    }

    function skip() {
      offset++;
      loadNext();
    }

    loadNext();
  </script>
</body>
</html>
```

---

## Testing

### Test Suggestions Endpoint
```bash
# Start backend
python run_backend.py

# Get unclustered faces
curl http://localhost:5052/api/persons/unclustered?limit=1

# Get suggestions for a face
curl http://localhost:5052/api/persons/face/12345/suggestions

# Assign face to cluster
curl -X POST http://localhost:5052/api/persons/face/12345/assign \
  -H "Content-Type: application/json" \
  -d '{"cluster_id": 42}'
```

---

## Summary

The Active Learning API completes the four-pass clustering system:

1. **Pass 1:** Strict clustering → pure seed clusters (automatic)
2. **Pass 2:** Centroid assignment → 50-70% noise reduction (automatic)
3. **Pass 3:** Cluster merging → fix split errors (automatic)
4. **Pass 4:** Active learning → human review of remaining ~10-20k faces (manual + AI-assisted)

**Time Estimate:**
- Automatic passes: 3-5 minutes
- Manual review: 1-2 hours at 10-20 faces/minute

**Result:** 77,000 faces organized into ~1-2k person clusters with minimal manual effort! 🎉
