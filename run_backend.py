#!/usr/bin/env python
"""Run the face browser backend server."""

import sys
from pathlib import Path

# Add backend to Python path
backend_dir = Path(__file__).parent / "face_browser" / "backend"
sys.path.insert(0, str(backend_dir))

# Now import and run
from app import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5052, debug=True)
