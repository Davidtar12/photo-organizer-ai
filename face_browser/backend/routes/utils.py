from __future__ import annotations

from pathlib import Path
from fastai.vision.all import load_learner, PILImage

# Load pet model
pets_model = None
try:
    pets_model_path = Path(__file__).parent / 'models' / 'pets.pkl'
    if pets_model_path.exists():
        pets_model = load_learner(pets_model_path)
except Exception as e:
    print(f"Could not load pets model: {e}")

def is_pet_face(thumbnail_path):
    if not pets_model or not thumbnail_path:
        return False
    try:
        img = PILImage.create(thumbnail_path)
        pred, _, _ = pets_model.predict(img)
        return str(pred) == 'Max'  # assuming the class is 'Max'
    except Exception as e:
        print(f"Error predicting pet: {e}")
        return False
