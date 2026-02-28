"""Cloud Sync Attribute Handler

Handles OneDrive, Google Drive, and other cloud sync attributes
to prevent hidden folders and sync conflicts.
"""

import os
import logging
from pathlib import Path
from typing import Set, Dict, Any
import subprocess

logger = logging.getLogger(__name__)

# OneDrive attribute flags (Windows-specific)
ONEDRIVE_ATTRIBUTES = {
    'FILE_ATTRIBUTE_PINNED': 0x00080000,
    'FILE_ATTRIBUTE_UNPINNED': 0x00100000,
    'FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS': 0x00400000,
}


def detect_cloud_sync_folder(folder_path: Path) -> Dict[str, Any]:
    """
    Detect if a folder is under cloud sync (OneDrive, Google Drive, Dropbox, etc.)
    
    Returns:
        Dictionary with:
            - is_synced: bool
            - provider: str ('onedrive', 'googledrive', 'dropbox', 'icloud', None)
            - sync_status: str ('synced', 'syncing', 'error', 'unknown')
    """
    result = {
        'is_synced': False,
        'provider': None,
        'sync_status': 'unknown'
    }
    
    path_str = str(folder_path.resolve()).lower()
    
    # OneDrive detection
    if 'onedrive' in path_str:
        result['is_synced'] = True
        result['provider'] = 'onedrive'
        result['sync_status'] = _check_onedrive_status(folder_path)
    
    # Google Drive detection
    elif 'google drive' in path_str or 'googledrive' in path_str:
        result['is_synced'] = True
        result['provider'] = 'googledrive'
        result['sync_status'] = 'synced'  # Simplified - Google Drive File Stream uses different API
    
    # Dropbox detection
    elif 'dropbox' in path_str:
        result['is_synced'] = True
        result['provider'] = 'dropbox'
        result['sync_status'] = 'synced'
    
    # iCloud detection
    elif 'icloud' in path_str or 'icloudrive' in path_str:
        result['is_synced'] = True
        result['provider'] = 'icloud'
        result['sync_status'] = 'synced'
    
    logger.debug(f"Cloud sync detection for {folder_path}: {result}")
    return result


def _check_onedrive_status(folder_path: Path) -> str:
    """
    Check OneDrive sync status for a folder (Windows-specific).
    
    Returns: 'synced', 'syncing', 'error', or 'unknown'
    """
    try:
        # On Windows, OneDrive status can be checked via file attributes
        # This is a simplified check - full implementation would use Windows API
        if folder_path.exists():
            return 'synced'
    except Exception as e:
        logger.debug(f"Error checking OneDrive status: {e}")
    
    return 'unknown'


def normalize_cloud_attributes(folder_path: Path) -> bool:
    """
    Remove hidden/system attributes from cloud-synced folders.
    
    This prevents folders from being invisible in Windows Explorer
    while maintaining cloud sync functionality.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Windows: use attrib command to remove hidden and system attributes
        if os.name == 'nt':
            result = subprocess.run(
                ['attrib', '-h', '-s', str(folder_path)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.debug(f"Normalized attributes for {folder_path}")
                return True
            else:
                logger.warning(f"Failed to normalize attributes for {folder_path}: {result.stderr}")
                return False
        else:
            # Unix-like systems: just ensure folder is readable
            folder_path.chmod(0o755)
            return True
            
    except Exception as e:
        logger.warning(f"Error normalizing cloud attributes for {folder_path}: {e}")
        return False


def normalize_all_cloud_folders(root_path: Path, skip_dirs: Set[Path] = None) -> int:
    """
    Recursively normalize attributes for all cloud-synced folders.
    
    Args:
        root_path: Root directory to start from
        skip_dirs: Set of directory paths to skip
        
    Returns:
        Count of folders normalized
    """
    if skip_dirs is None:
        skip_dirs = set()
    
    normalized_count = 0
    
    try:
        for dirpath, dirs, _ in os.walk(root_path):
            current = Path(dirpath).resolve()
            
            # Skip excluded directories
            if current in skip_dirs:
                dirs[:] = []  # Don't descend into this directory
                continue
            
            # Normalize attributes for this directory
            cloud_info = detect_cloud_sync_folder(current)
            if cloud_info['is_synced']:
                if normalize_cloud_attributes(current):
                    normalized_count += 1
        
        logger.info(f"Normalized {normalized_count} cloud-synced folders under {root_path}")
        
    except Exception as e:
        logger.error(f"Error normalizing cloud folders: {e}")
    
    return normalized_count


def create_cloud_safe_folder(folder_path: Path) -> bool:
    """
    Create a folder with cloud-sync-safe attributes.
    
    Args:
        folder_path: Path to the folder to create
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create the folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Normalize attributes to ensure it's visible
        normalize_cloud_attributes(folder_path)
        
        # Set appropriate permissions
        if os.name == 'nt':
            # Windows: Ensure folder is not marked as system or hidden
            pass  # Already handled by normalize_cloud_attributes
        else:
            # Unix: Set standard directory permissions
            folder_path.chmod(0o755)
        
        logger.debug(f"Created cloud-safe folder: {folder_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating cloud-safe folder {folder_path}: {e}")
        return False


def check_sync_conflicts(file_path: Path) -> bool:
    """
    Check if a file has sync conflicts (e.g., OneDrive conflict files).
    
    Conflict files typically have patterns like:
    - filename-DESKTOP-ABC-1.jpg (OneDrive)
    - filename (conflicted copy 2023-01-15).jpg (Dropbox)
    - filename (1).jpg (Google Drive)
    
    Returns:
        True if file appears to be a sync conflict, False otherwise
    """
    name = file_path.name
    stem = file_path.stem
    
    # OneDrive conflict pattern
    if '-' in stem and stem.count('-') >= 2:
        parts = stem.split('-')
        if len(parts) >= 3 and parts[-1].isdigit():
            logger.debug(f"Detected potential OneDrive conflict file: {name}")
            return True
    
    # Dropbox conflict pattern
    if 'conflicted copy' in name.lower():
        logger.debug(f"Detected Dropbox conflict file: {name}")
        return True
    
    # Google Drive duplicate pattern (less reliable)
    if name.endswith(')' + file_path.suffix) and '(' in name:
        parent_part = name.rsplit('(', 1)[0].strip()
        number_part = name.rsplit('(', 1)[1].replace(file_path.suffix, '').replace(')', '').strip()
        if number_part.isdigit():
            logger.debug(f"Detected potential Google Drive duplicate: {name}")
            return True
    
    return False


def get_cloud_sync_recommendations(root_path: Path) -> Dict[str, Any]:
    """
    Analyze a photo library and provide cloud sync recommendations.
    
    Returns:
        Dictionary with recommendations and statistics
    """
    recommendations = {
        'is_cloud_synced': False,
        'provider': None,
        'conflict_files_found': [],
        'hidden_folders_found': [],
        'total_conflicts': 0,
        'total_hidden': 0,
        'recommendations': []
    }
    
    # Check if root is under cloud sync
    cloud_info = detect_cloud_sync_folder(root_path)
    recommendations['is_cloud_synced'] = cloud_info['is_synced']
    recommendations['provider'] = cloud_info['provider']
    
    if cloud_info['is_synced']:
        recommendations['recommendations'].append(
            f"Library is synced with {cloud_info['provider']}. "
            "Consider excluding 'Organized', 'duplicates', and 'photo_reports' "
            "folders from sync to save cloud storage space."
        )
    
    # Scan for hidden folders and conflict files (sample only, not full scan)
    try:
        for dirpath, dirs, files in os.walk(root_path):
            current = Path(dirpath)
            
            # Check for hidden folders (sample first 100)
            for d in dirs[:100]:
                dir_path = current / d
                try:
                    if os.name == 'nt':
                        # Check Windows attributes
                        import stat
                        attrs = dir_path.stat().st_file_attributes if hasattr(dir_path.stat(), 'st_file_attributes') else 0
                        if attrs & stat.FILE_ATTRIBUTE_HIDDEN:
                            recommendations['hidden_folders_found'].append(str(dir_path))
                            recommendations['total_hidden'] += 1
                except Exception:
                    pass
            
            # Check for conflict files (sample first 100)
            for f in files[:100]:
                file_path = current / f
                if check_sync_conflicts(file_path):
                    recommendations['conflict_files_found'].append(str(file_path))
                    recommendations['total_conflicts'] += 1
            
            # Limit search depth to avoid long scans
            if len(recommendations['hidden_folders_found']) > 50:
                break
    
    except Exception as e:
        logger.warning(f"Error scanning for cloud sync issues: {e}")
    
    if recommendations['total_hidden'] > 0:
        recommendations['recommendations'].append(
            f"Found {recommendations['total_hidden']} hidden folders. "
            "Run normalize_all_cloud_folders() to make them visible."
        )
    
    if recommendations['total_conflicts'] > 0:
        recommendations['recommendations'].append(
            f"Found {recommendations['total_conflicts']} potential sync conflict files. "
            "Review and resolve these conflicts manually."
        )
    
    return recommendations
