# AdaFace Migration Summary

## Model Change
**From:** InsightFace buffalo_l (ArcFace R50, 512-dim)  
**To:** AdaFace IR100 WebFace12M (512-dim)

## Why AdaFace?
- **Superior accuracy**: AdaFace IR100 is trained on WebFace12M (12 million identities)
- **Better generalization**: Handles diverse demographics and challenging conditions
- **Same dimensionality**: 512-dim embeddings (compatible with existing clustering code)
- **L2 normalized**: Cosine similarity via dot product (FAISS IndexFlatIP ready)

## Implementation Details
- **Detector**: Still using SCRFD-10G from buffalo_l (excellent face detection)
- **Embeddings**: Switched to AdaFace IR100 ONNX model
- **Model source**: HuggingFace (ibai/adaface)
- **Direct URL**: https://huggingface.co/ibai/adaface/resolve/main/adaface_ir100_webface12m.onnx
- **Auto-download**: Downloads to `~/.insightface/models/adaface/` on first run
- **Preprocessing**: Resize to 112x112, normalize to [-1, 1], L2 normalize output

## Database Changes
- `model_name` field now saves as "AdaFace" instead of "ArcFace"
- `detector` field remains "SCRFD-10G"
- `embedding_dim` remains 512

## Fallback Plan (buffalo_l)
The original buffalo_l implementation is kept as commented code in `face_indexer.py`:
```python
# Plan B (buffalo_l - ArcFace R50 embeddings):
# self.insightface_app = insightface.app.FaceAnalysis(
#     name='buffalo_l',
#     providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
# )
# self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
# self.use_insightface = True
# logger.info("InsightFace ready (SCRFD-10G + ArcFace R50)")
```

## Migration Steps
1. **Clear old embeddings** (if needed):
   ```bash
   python clear_embeddings.py
   ```

2. **Run pipeline** with AdaFace:
   ```bash
   python run_pipeline.py
   ```
   - AdaFace model will auto-download on first run (~180MB)
   - All faces will be re-indexed with new embeddings
   - Three-pass clustering will run automatically

3. **Verify** the change:
   ```bash
   python verify_db.py
   ```
   - Should show "AdaFace" as the model name
   - Should show consistent single model usage

## Performance Notes
- AdaFace inference is slightly slower than ArcFace R50 (IR100 vs R50 backbone)
- But the accuracy improvement is worth it for face clustering
- GPU acceleration via ONNX Runtime CUDAExecutionProvider helps

## Clustering Impact
The three-pass clustering system remains unchanged:
- **Pass 1**: Strict high-confidence clustering (~0.315 threshold)
- **Pass 2**: Centroid assignment (0.4-0.5 threshold)
- **Pass 3**: Cluster merging (0.35 threshold)

AdaFace's better embeddings should result in:
- Fewer "split errors" (same person in multiple clusters)
- Better handling of challenging faces (different angles, lighting, age)
- More robust centroid computation in Pass 2 and Pass 3

## Rollback Procedure
If AdaFace causes issues, revert to buffalo_l:
1. Edit `face_indexer.py`:
   - Comment out AdaFace code block
   - Uncomment buffalo_l Plan B code
2. Run `clear_embeddings.py`
3. Run `run_pipeline.py`
