"""
Flask server for Photo Duplicate Viewer
Handles file operations: delete, get metadata, load duplicates CSV
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import csv
import json
from pathlib import Path
from datetime import datetime
import hashlib
from PIL import Image
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PHOTO_ROOT = Path(r'C:\dscodingpython\File organizers')
DUPLICATES_CSV = PHOTO_ROOT / 'duplicates.csv'
DECISIONS_DIR = PHOTO_ROOT / 'deletion_decisions'
DECISIONS_DIR.mkdir(exist_ok=True)

def get_file_metadata(filepath):
    """Extract metadata from image/video file - OPTIMIZED: no SHA-256 by default"""
    try:
        path = Path(filepath)
        if not path.exists():
            return None
            
        stat = path.stat()
        file_size = stat.st_size / (1024 * 1024)  # MB
        
        metadata = {
            'path': str(path),
            'fileSize': round(file_size, 2),
            'exists': True
        }
        
        # Try to get image resolution - FAST
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif']:
            try:
                with Image.open(path) as img:
                    metadata['resolution'] = img.width * img.height
                    metadata['width'] = img.width
                    metadata['height'] = img.height
            except Exception as e:
                logger.debug(f"Could not read image dimensions for {path}: {e}")
                metadata['resolution'] = 0
                metadata['width'] = 0
                metadata['height'] = 0
        else:
            # Video or other file - use file size as proxy
            metadata['resolution'] = int(file_size * 1000000)  # Approximate
            metadata['width'] = 0
            metadata['height'] = 0
            
        # Get file modification time
        try:
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            metadata['dateTaken'] = mod_time.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            metadata['dateTaken'] = None
            
        # SKIP SHA-256 calculation - too slow for large files
        metadata['sha256'] = None
            
        return metadata
    except Exception as e:
        logger.error(f"Error getting metadata for {filepath}: {e}")
        return None

@app.route('/api/duplicates', methods=['GET'])
def get_duplicates():
    """Load duplicates from CSV with pagination - loads 100 pairs at a time"""
    try:
        if not DUPLICATES_CSV.exists():
            return jsonify({'error': 'duplicates.csv not found'}), 404
        
        # Pagination parameters
        offset = int(request.args.get('offset', 0))
        limit = int(request.args.get('limit', 100))
        filter_deleted = request.args.get('filter', 'false').lower() == 'true'
            
        duplicate_sets = []
        skipped = 0
        current_offset = 0
        
        with open(DUPLICATES_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                original_path = row.get('original', '').strip()
                duplicate_path = row.get('duplicate', '').strip()
                
                if not original_path or not duplicate_path:
                    continue
                
                # Only check existence if filter_deleted is True
                if filter_deleted:
                    original_exists = Path(original_path).exists()
                    duplicate_exists = Path(duplicate_path).exists()
                    
                    if not original_exists or not duplicate_exists:
                        skipped += 1
                        continue
                
                # Skip until we reach the offset
                if current_offset < offset:
                    current_offset += 1
                    continue
                
                # Stop when we reach the limit
                if len(duplicate_sets) >= limit:
                    break
                
                duplicate_sets.append({
                    'id': f'set-{offset + len(duplicate_sets) + 1}',
                    'original': {
                        'path': original_path,
                        'exists': True,
                        'decision': 'undecided'
                    },
                    'duplicate': {
                        'path': duplicate_path,
                        'exists': True,
                        'decision': 'undecided'
                    }
                })
                current_offset += 1
        
        # Count total for pagination info
        total_count = current_offset + skipped
        has_more = current_offset < total_count
        
        result = {
            'duplicates': duplicate_sets,
            'pagination': {
                'offset': offset,
                'limit': limit,
                'count': len(duplicate_sets),
                'total': total_count,
                'hasMore': has_more
            }
        }
                    
        logger.info(f"Loaded {len(duplicate_sets)} duplicate sets (offset={offset}, limit={limit}, filter={filter_deleted})")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error loading duplicates: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metadata/<int:set_index>', methods=['GET'])
def get_metadata(set_index):
    """Get metadata for a specific duplicate set (lazy loading)"""
    try:
        if not DUPLICATES_CSV.exists():
            return jsonify({'error': 'duplicates.csv not found'}), 404
            
        with open(DUPLICATES_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if set_index >= len(rows):
                return jsonify({'error': 'Invalid set index'}), 404
                
            row = rows[set_index]
            original_path = row.get('original', '').strip()
            duplicate_path = row.get('duplicate', '').strip()
            
            original_meta = get_file_metadata(original_path)
            duplicate_meta = get_file_metadata(duplicate_path)
            
            return jsonify({
                'original': original_meta,
                'duplicate': duplicate_meta
            })
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete', methods=['POST'])
def delete_file():
    """Delete a file from the filesystem"""
    try:
        data = request.json
        filepath = data.get('path')
        
        if not filepath:
            return jsonify({'error': 'No file path provided'}), 400
            
        path = Path(filepath)
        
        if not path.exists():
            return jsonify({'error': 'File does not exist'}), 404
            
        # Safety check - ensure file is within allowed directories
        allowed_dirs = [
            Path(r'C:\Users\USERNAME\OneDrive\Documents\Pictures'),
            Path(r'C:\dscodingpython\File organizers')
        ]
        
        if not any(path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
            return jsonify({'error': 'File is outside allowed directories'}), 403
            
        # Delete the file
        path.unlink()
        logger.info(f"Deleted file: {filepath}")
        
        return jsonify({'success': True, 'message': f'Deleted {path.name}'})
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-both', methods=['POST'])
def delete_both():
    """Delete both original and duplicate files"""
    try:
        data = request.json
        original_path = data.get('original')
        duplicate_path = data.get('duplicate')
        
        if not original_path or not duplicate_path:
            return jsonify({'error': 'Both paths required'}), 400
            
        original = Path(original_path)
        duplicate = Path(duplicate_path)
        
        # Safety check - ensure files are within allowed directories
        allowed_dirs = [
            Path(r'C:\Users\USERNAME\OneDrive\Documents\Pictures'),
            Path(r'C:\dscodingpython\File organizers')
        ]
        
        for path in [original, duplicate]:
            if not any(path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
                return jsonify({'error': f'File outside allowed directories: {path}'}), 403
        
        deleted = []
        errors = []
        
        # Try to delete both files
        for path in [original, duplicate]:
            if path.exists():
                try:
                    path.unlink()
                    deleted.append(str(path))
                    logger.info(f"Deleted file: {path}")
                except Exception as e:
                    errors.append(f"{path.name}: {str(e)}")
                    logger.error(f"Failed to delete {path}: {e}")
            else:
                logger.warning(f"File already deleted or not found: {path}")
        
        if errors:
            return jsonify({
                'success': False,
                'deleted': deleted,
                'errors': errors
            }), 500
        
        return jsonify({
            'success': True,
            'message': f'Deleted both files',
            'deleted': deleted
        })
    except Exception as e:
        logger.error(f"Error deleting both files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk-delete-duplicates', methods=['POST'])
def bulk_delete_duplicates():
    """Delete duplicate files for multiple pairs at once"""
    try:
        data = request.json
        pairs = data.get('pairs', [])
        
        if not pairs:
            return jsonify({'error': 'No pairs provided'}), 400
        
        allowed_dirs = [
            Path(r'C:\Users\USERNAME\OneDrive\Documents\Pictures'),
            Path(r'C:\dscodingpython\File organizers')
        ]
        
        deleted = []
        errors = []
        
        for pair in pairs:
            duplicate_path = pair.get('duplicate')
            if not duplicate_path:
                continue
            
            path = Path(duplicate_path)
            
            # Validate path is in allowed directories
            if not any(path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
                errors.append(f'{path.name}: Outside allowed directories')
                continue
            
            if path.exists():
                try:
                    path.unlink()
                    deleted.append(str(path))
                    logger.info(f"Bulk deleted: {path}")
                except Exception as e:
                    errors.append(f'{path.name}: {str(e)}')
                    logger.error(f"Failed to delete {path}: {e}")
            else:
                logger.warning(f"File already deleted: {path}")
        
        return jsonify({
            'success': True,
            'deleted': deleted,
            'deleted_count': len(deleted),
            'errors': errors,
            'error_count': len(errors)
        })
    except Exception as e:
        logger.error(f"Error in bulk delete: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-decisions', methods=['POST'])
def export_decisions():
    """Export deletion decisions to JSON file"""
    try:
        data = request.json
        decisions = data.get('decisions', [])
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'deletion_decisions_{timestamp}.json'
        filepath = DECISIONS_DIR / filename
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'totalSets': data.get('totalSets', 0),
            'reviewed': data.get('reviewed', 0),
            'decisions': decisions
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Exported decisions to {filepath}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath)
        })
    except Exception as e:
        logger.error(f"Error exporting decisions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/file-exists', methods=['POST'])
def check_file_exists():
    """Check if a file still exists"""
    try:
        data = request.json
        filepath = data.get('path')
        
        if not filepath:
            return jsonify({'error': 'No file path provided'}), 400
            
        exists = Path(filepath).exists()
        return jsonify({'exists': exists})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/api/image', methods=['GET'])
def serve_image():
    """Serve image file from absolute path"""
    try:
        filepath = request.args.get('path')
        if not filepath:
            return jsonify({'error': 'No path provided'}), 400
        
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Image not found: {path}")
            return jsonify({'error': 'Image not found'}), 404
        
        if not path.is_file():
            return jsonify({'error': 'Not a file'}), 400
        
        # Serve the image file
        return send_file(str(path), mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Photo Duplicate Viewer server...")
    logger.info(f"Looking for duplicates.csv at: {DUPLICATES_CSV}")
    app.run(debug=True, port=5000, host='127.0.0.1')
