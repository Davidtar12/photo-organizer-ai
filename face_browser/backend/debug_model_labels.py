import sys
from pathlib import Path

# Add backend path
backend_path = Path(__file__).parent
sys.path.append(str(backend_path))

try:
    from fastai.vision.all import load_learner
    
    pets_model_path = Path(__file__).parent / 'models' / 'pets.pkl'
    
    if not pets_model_path.exists():
        print(f"ERROR: Model file not found at {pets_model_path}")
        sys.exit(1)
    
    print(f"Loading model from: {pets_model_path}")
    learner = load_learner(pets_model_path)
    
    vocab = list(getattr(getattr(learner, 'dls', None), 'vocab', []) or [])
    
    print(f"\n{'='*50}")
    print(f"Model has {len(vocab)} classes:")
    print(f"{'='*50}")
    for idx, label in enumerate(vocab):
        print(f"  Index {idx}: '{label}'")
    
    print(f"\n{'='*50}")
    if 'Max' in vocab:
        max_idx = vocab.index('Max')
        print(f"✓ 'Max' found at index {max_idx}")
    else:
        print(f"✗ 'Max' NOT FOUND in vocabulary!")
        print(f"  Available labels: {vocab}")
    print(f"{'='*50}\n")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
