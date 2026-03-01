# photo-organizer-ai

AI-powered photo organizer: YOLOv8 pet/object detection, face clustering, duplicate removal, React-based photo viewer.

## Features

- **Face clustering** — groups photos by person using facial recognition
- **Pet detection** — identifies cats and dogs using YOLOv8 object detection
- **Duplicate removal** — finds near-duplicate photos using perceptual hashing
- **React viewer** — browse organised albums in a web interface
- **Batch processing** — handles large photo libraries

## Prerequisites

- Python 3.9+
- Node.js 18+ (for the React viewer frontend)
- GPU recommended for YOLOv8 (CPU works but is slower)

## Setup

```bash
# Backend
pip install -r requirements.txt

# Frontend (React viewer)
cd face_browser
npm install
```

Refer to `INSTALL_PACKAGES.md` for platform-specific installation notes.

## Usage

```bash
# Organise a photo folder (detection + clustering + deduplication)
python photo_organizer.py --input /path/to/photos --output /path/to/organised

# Launch the React viewer
cd face_browser && npm start
# Open http://localhost:3000
```

See `QUICK_REFERENCE.md` for a full command cheatsheet.

## Notes

- YOLOv8 model downloads automatically on first run (~6 MB).
- Face clustering uses DBSCAN — no pre-labelling required.
- Processing time: ~2–5 seconds per photo on CPU, ~0.5s on GPU.

## Built with

Python · YOLOv8 (Ultralytics) · face_recognition · React · OpenCV  
AI-assisted development (Claude, GitHub Copilot) — architecture, requirements, QA validation and debugging by me.
