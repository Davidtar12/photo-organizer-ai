# Photo Duplicate Viewer - Setup & Usage Guide

## 🚀 Complete Setup Instructions

### Step 1: Generate Duplicates

Run the photo organizer to find duplicates in your photo library:

```bash
cd "C:\dscodingpython\File organizers"
photoenv\Scripts\activate
python photo-organizer.py --root "C:\Users\david\OneDrive\Documents\Pictures\Fotos" --dry-run
```

This will create `duplicates.csv` in the `File organizers` folder.

### Step 2: Install Backend Dependencies

```bash
cd "C:\dscodingpython\File organizers\duplicate-viewer-app\backend"
pip install flask flask-cors pillow
```

### Step 3: Install Frontend Dependencies

You'll need Node.js installed. Then:

```bash
cd "C:\dscodingpython\File organizers\duplicate-viewer-app"
npm install
```

### Step 4: Run the Application

#### Option A: Easy Start (Windows)
Double-click `start.bat` in the duplicate-viewer-app folder.

#### Option B: Manual Start

**Terminal 1 - Start Flask Backend:**
```bash
cd "C:\dscodingpython\File organizers\duplicate-viewer-app\backend"
python server.py
```

**Terminal 2 - Start React Frontend:**
```bash
cd "C:\dscodingpython\File organizers\duplicate-viewer-app"
npm run dev
```

Then open your browser to: http://localhost:3000

---

## 🎯 How to Use

1. **Navigate**: Use arrow keys (← →) or click the Previous/Next buttons to move between duplicate sets

2. **Review Images**: Look at both images side-by-side. The app shows:
   - Image preview
   - File path
   - Resolution (width × height)
   - File size (MB)
   - Date taken
   - SHA-256 hash

3. **Make Decisions**:
   - Click **"Keep"** on the image you want to preserve
   - Click **"Delete"** on the image you want to remove
   - **⚠️ DELETE IS IMMEDIATE** - The file will be deleted from your filesystem right away!

4. **Keyboard Shortcuts**:
   - Press `1` to keep the original (left image)
   - Press `2` to keep the duplicate (right image)
   - Press `←` to go to previous duplicate set
   - Press `→` to go to next duplicate set

5. **Export Decisions**: Click "Export Decisions" to save a JSON log of all your decisions

---

## ⚠️ Important Notes

### DELETE IS PERMANENT
- When you click "Delete", the file is **immediately removed** from your filesystem
- This is NOT reversible - make sure you're deleting the right file!
- The app will show "Deleted" badge and gray out deleted files

### Safety Features
- Only files in specific directories can be deleted (configurable in backend/server.py)
- Visual confirmation when files are deleted
- Decisions log is exported for your records

### File Access
- The app needs to access files on your local drive
- Images use `file://` protocol which may have browser restrictions
- All file operations happen through the Flask backend for security

---

## 🔧 Configuration

Edit `backend/server.py` to customize:

```python
# Line 18-20: Configure where your photos are
PHOTO_ROOT = Path(r'C:\dscodingpython\File organizers')
DUPLICATES_CSV = PHOTO_ROOT / 'duplicates.csv'

# Line 76-79: Configure which directories can have files deleted (SAFETY!)
allowed_dirs = [
    Path(r'C:\Users\david\OneDrive\Documents\Pictures'),
    Path(r'C:\dscodingpython\File organizers')
]
```

---

## 🐛 Troubleshooting

### "Cannot load duplicates"
- Make sure `duplicates.csv` exists in `C:\dscodingpython\File organizers\`
- Check that Flask backend is running (Terminal 1)
- Visit http://localhost:5000/api/health to test backend

### "Failed to delete file"
- Check that the file path is within allowed directories
- Verify you have write permissions for that file/folder
- Make sure the file isn't open in another program

### Images not displaying
- Some browsers block `file://` protocol for security
- Try a different browser (Chrome/Edge usually work)
- Check that the file paths in duplicates.csv are correct

### Port already in use
- Flask backend uses port 5000
- React frontend uses port 3000
- Kill any processes using these ports or change in config files

---

## 📊 Understanding the Output

### Decisions Log (JSON)
When you export decisions, you get a JSON file like:

```json
{
  "timestamp": "2025-10-28T10:30:00",
  "totalSets": 851,
  "reviewed": 45,
  "decisions": [
    {
      "original": "C:\\path\\to\\IMG_001.jpg",
      "duplicate": "C:\\path\\to\\IMG_001_copy.jpg",
      "keepOriginal": true,
      "keepDuplicate": false
    }
  ]
}
```

This log is saved in: `C:\dscodingpython\File organizers\deletion_decisions\`

---

## 🎨 Features

✅ Real-time file deletion
✅ Side-by-side image comparison  
✅ Keyboard shortcuts for speed
✅ Metadata display (resolution, size, date)
✅ Visual decision indicators
✅ Progress tracking
✅ Decisions export log
✅ Responsive design

---

## 🔐 Security

- File deletion is restricted to configured directories only
- All operations go through Flask backend (no direct file access from browser)
- File paths are validated before deletion
- Decisions are logged for audit trail

---

## 💡 Tips

1. **Start with dry-run**: Run photo-organizer with `--dry-run` first to see what duplicates exist
2. **Review carefully**: Since deletion is immediate, take your time reviewing each set
3. **Use keyboard shortcuts**: Much faster than clicking for large batches (851 duplicates!)
4. **Export regularly**: Export your decisions every 50-100 sets in case of browser crash
5. **Check file sizes**: Often the larger file is better quality
6. **Compare dates**: Usually want to keep the original (earliest date)

---

## Need Help?

Check the logs:
- Flask backend: Shows in Terminal 1
- React frontend: Shows in Terminal 2
- Browser console: F12 → Console tab
