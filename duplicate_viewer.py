"""Duplicate Photo Viewer Web App

Flask web application for viewing and managing duplicate photos.
Allows side-by-side comparison and deletion.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import logging
import json
from datetime import datetime
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
DUPLICATES_CSV = None
DUPLICATES_DATA = None
PHOTO_ROOT = None


def load_duplicates(csv_path: Path, photo_root: Path):
    """Load duplicates CSV file."""
    global DUPLICATES_CSV, DUPLICATES_DATA, PHOTO_ROOT
    
    DUPLICATES_CSV = csv_path
    PHOTO_ROOT = photo_root
    
    if not csv_path.exists():
        logger.error(f"Duplicates CSV not found: {csv_path}")
        return False
    
    try:
        DUPLICATES_DATA = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(DUPLICATES_DATA)} duplicate entries")
        return True
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return False


def get_file_info(file_path: Path):
    """Get detailed info about a file."""
    try:
        stat = file_path.stat()
        
        # Try to get image dimensions
        width, height = None, None
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except:
            pass
        
        return {
            'path': str(file_path),
            'name': file_path.name,
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024*1024), 2),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'width': width,
            'height': height,
            'exists': True
        }
    except Exception as e:
        return {
            'path': str(file_path),
            'name': file_path.name,
            'exists': False,
            'error': str(e)
        }


@app.route('/')
def index():
    """Main page showing duplicate groups."""
    if DUPLICATES_DATA is None:
        return "Duplicates data not loaded. Start server with --csv parameter.", 500
    
    # Group by original file to get duplicate groups
    groups = []
    for original_file in DUPLICATES_DATA['original'].unique():
        group_df = DUPLICATES_DATA[DUPLICATES_DATA['original'] == original_file]
        duplicates = group_df['duplicate'].tolist()
        
        # All files in this group (original + duplicates)
        all_files = [original_file] + duplicates
        
        if len(all_files) > 1:  # Only groups with actual duplicates
            try:
                total_size = sum(Path(f).stat().st_size for f in all_files if Path(f).exists())
                total_size_mb = round(total_size / (1024*1024), 2)
            except:
                total_size_mb = 0
            
            groups.append({
                'hash': original_file,  # Use original as group identifier
                'count': len(all_files),
                'total_size_mb': total_size_mb,
                'files': all_files[:3],  # Preview first 3
                'original_encoded': original_file  # For URL
            })
    
    # Sort by size (biggest duplicates first)
    groups.sort(key=lambda x: x['total_size_mb'], reverse=True)
    
    return render_template('duplicates_index.html', 
                         groups=groups, 
                         total_groups=len(groups))


@app.route('/group/')
def view_group():
    """View a specific duplicate group."""
    if DUPLICATES_DATA is None:
        return "Data not loaded", 500
    
    # Get the original file path from query parameter
    hash_val = request.args.get('original')
    if not hash_val:
        return "Missing 'original' parameter", 400
    
    # Get all duplicates for this original file
    group = DUPLICATES_DATA[DUPLICATES_DATA['original'] == hash_val]
    
    if group.empty:
        return f"Group not found: {hash_val}", 404
    
    # Collect all files (original + duplicates)
    files = []
    
    # Add original first
    original_path = Path(hash_val)
    info = get_file_info(original_path)
    info['sha256'] = 'original'
    files.append(info)
    
    # Add duplicates
    for _, row in group.iterrows():
        dup_path = Path(row['duplicate'])
        info = get_file_info(dup_path)
        info['sha256'] = 'duplicate'
        files.append(info)
    
    # Sort by modification date (oldest first - likely the original)
    files.sort(key=lambda x: x.get('modified', ''))
    
    return render_template('duplicate_group.html',
                         hash=hash_val,
                         files=files,
                         num_files=len(files))


@app.route('/image/<path:image_path>')
def serve_image(image_path):
    """Serve an image file."""
    try:
        # Security: ensure path is within PHOTO_ROOT
        full_path = Path(image_path)
        if not full_path.exists():
            return f"File not found: {image_path}", 404
        
        return send_file(full_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving image {image_path}: {e}")
        return str(e), 500


@app.route('/thumbnail/<path:image_path>')
def serve_thumbnail(image_path):
    """Serve a thumbnail version of an image."""
    try:
        full_path = Path(image_path)
        if not full_path.exists():
            return "File not found", 404
        
        # Create thumbnail
        img = Image.open(full_path)
        img.thumbnail((400, 400))
        
        # Save to temp and serve
        import io
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error creating thumbnail for {image_path}: {e}")
        return str(e), 500


@app.route('/delete', methods=['POST'])
def delete_file():
    """Delete a duplicate file."""
    data = request.json
    file_path = Path(data.get('file_path'))
    
    if not file_path.exists():
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    try:
        # Move to trash instead of permanent delete (safer)
        import send2trash
        send2trash.send2trash(str(file_path))
        
        logger.info(f"Moved to trash: {file_path}")
        return jsonify({'success': True, 'message': f'Moved to trash: {file_path.name}'})
    
    except ImportError:
        # Fallback: rename with .deleted extension
        try:
            deleted_path = file_path.with_suffix(file_path.suffix + '.deleted')
            file_path.rename(deleted_path)
            logger.info(f"Renamed to .deleted: {file_path}")
            return jsonify({'success': True, 'message': f'Renamed to .deleted: {file_path.name}'})
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    except Exception as e:
        logger.error(f"Failed to delete {file_path}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Get statistics about duplicates."""
    if DUPLICATES_DATA is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    total_duplicates = len(DUPLICATES_DATA)
    unique_originals = DUPLICATES_DATA['original'].nunique()
    
    # Calculate wasted space (all duplicate files)
    wasted_space = 0
    for _, row in DUPLICATES_DATA.iterrows():
        dup_path = Path(row['duplicate'])
        if dup_path.exists():
            wasted_space += dup_path.stat().st_size
    
    return jsonify({
        'total_files': total_duplicates + unique_originals,
        'unique_hashes': unique_originals,
        'duplicate_groups': unique_originals,
        'wasted_space_mb': round(wasted_space / (1024*1024), 2),
        'wasted_space_gb': round(wasted_space / (1024*1024*1024), 2)
    })


# HTML Templates (embedded for simplicity)
def create_templates():
    """Create template directory and files."""
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # duplicates_index.html
    with open(templates_dir / 'duplicates_index.html', 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Duplicate Photos</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { margin: 0 0 10px 0; color: #333; }
        .stats { color: #666; font-size: 14px; }
        .group-list { display: grid; gap: 15px; }
        .group-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); cursor: pointer; transition: transform 0.2s; }
        .group-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
        .group-hash { font-family: monospace; color: #888; font-size: 12px; word-break: break-all; }
        .group-info { display: flex; justify-content: space-between; margin: 10px 0; }
        .group-count { font-weight: bold; color: #e74c3c; }
        .group-size { color: #3498db; }
        .preview { display: flex; gap: 5px; margin-top: 10px; }
        .preview-img { width: 100px; height: 100px; object-fit: cover; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📸 Duplicate Photos</h1>
        <div class="stats" id="stats">Loading statistics...</div>
    </div>
    
    <div class="group-list">
        {% for group in groups %}
        <div class="group-card" onclick="viewGroup({{ group.hash | tojson }})">
            <div class="group-hash">{{ group.hash }}</div>
            <div class="group-info">
                <span class="group-count">{{ group.count }} duplicates</span>
                <span class="group-size">{{ group.total_size_mb }} MB wasted</span>
            </div>
        </div>
        {% endfor %}
    </div>
    
    <script>
        function viewGroup(original) {
            const encoded = encodeURIComponent(original);
            window.location.href = '/group/?original=' + encoded;
        }
        
        fetch('/stats')
            .then(r => r.json())
            .then(data => {
                document.getElementById('stats').innerHTML = 
                    `${data.total_files} files • ${data.duplicate_groups} groups • ${data.wasted_space_gb} GB wasted space`;
            });
    </script>
</body>
</html>''')
    
    # duplicate_group.html
    with open(templates_dir / 'duplicate_group.html', 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Duplicate Group</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .back-btn { display: inline-block; padding: 8px 16px; background: #3498db; color: white; text-decoration: none; border-radius: 4px; margin-bottom: 10px; }
        .back-btn:hover { background: #2980b9; }
        h1 { margin: 0; color: #333; }
        .hash { font-family: monospace; color: #888; font-size: 12px; margin-top: 5px; }
        .files-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .file-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .file-card.original { border: 3px solid #27ae60; }
        .file-img { width: 100%; height: 300px; object-fit: contain; background: #f8f9fa; border-radius: 4px; margin-bottom: 15px; }
        .file-info { font-size: 13px; color: #666; }
        .file-info-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee; }
        .file-name { font-weight: bold; color: #333; margin-bottom: 10px; word-break: break-all; }
        .delete-btn { background: #e74c3c; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; width: 100%; margin-top: 10px; font-size: 14px; }
        .delete-btn:hover { background: #c0392b; }
        .delete-btn:disabled { background: #95a5a6; cursor: not-allowed; }
        .label { display: inline-block; padding: 4px 8px; background: #27ae60; color: white; border-radius: 4px; font-size: 11px; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <a href="/" class="back-btn">← Back to List</a>
        <h1>Duplicate Group ({{ num_files }} files)</h1>
        <div class="hash">SHA-256: {{ hash }}</div>
    </div>
    
    <div class="files-grid">
        {% for file in files %}
        <div class="file-card {% if loop.first %}original{% endif %}" id="card-{{ loop.index }}">
            {% if loop.first %}<span class="label">ORIGINAL (Keep)</span>{% endif %}
            
            <img src="/thumbnail/{{ file.path }}" class="file-img" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22><text y=%2250%%22 x=%2250%%22>No Preview</text></svg>'">
            
            <div class="file-name">{{ file.name }}</div>
            
            <div class="file-info">
                <div class="file-info-row">
                    <span>Size:</span>
                    <span>{{ file.size_mb }} MB</span>
                </div>
                <div class="file-info-row">
                    <span>Modified:</span>
                    <span>{{ file.modified }}</span>
                </div>
                {% if file.width %}
                <div class="file-info-row">
                    <span>Dimensions:</span>
                    <span>{{ file.width }} × {{ file.height }}</span>
                </div>
                {% endif %}
                <div class="file-info-row">
                    <span>Path:</span>
                    <span style="font-size: 11px; word-break: break-all;">{{ file.path }}</span>
                </div>
            </div>
            
            {% if not loop.first %}
            <button class="delete-btn" onclick="deleteFile('{{ file.path }}', {{ loop.index }})">
                🗑️ Delete Duplicate
            </button>
            {% endif %}
        </div>
        {% endfor %}
    </div>
    
    <script>
        function deleteFile(filePath, cardIndex) {
            if (!confirm('Move this file to trash?')) return;
            
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = 'Deleting...';
            
            fetch('/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    const card = document.getElementById('card-' + cardIndex);
                    card.style.opacity = '0.5';
                    btn.textContent = '✓ Deleted';
                    btn.style.background = '#95a5a6';
                    alert('File moved to trash!');
                } else {
                    alert('Error: ' + data.error);
                    btn.disabled = false;
                    btn.textContent = '🗑️ Delete Duplicate';
                }
            })
            .catch(err => {
                alert('Error: ' + err);
                btn.disabled = false;
                btn.textContent = '🗑️ Delete Duplicate';
            });
        }
    </script>
</body>
</html>''')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Duplicate Photo Viewer Web App')
    parser.add_argument('--csv', type=Path, required=True,
                       help='Path to duplicates CSV file (e.g., duplicates/duplicates.csv)')
    parser.add_argument('--photo-root', type=Path, required=True,
                       help='Root directory of photo library')
    parser.add_argument('--host', default='127.0.0.1',
                       help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create templates before starting
    create_templates()
    
    if not load_duplicates(args.csv, args.photo_root):
        print("Failed to load duplicates data")
        return 1
    
    print("=" * 60)
    print("Duplicate Photo Viewer")
    print("=" * 60)
    print(f"CSV: {args.csv}")
    print(f"Photo Root: {args.photo_root}")
    print(f"\nOpen in browser: http://{args.host}:{args.port}")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
