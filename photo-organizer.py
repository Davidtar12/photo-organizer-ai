import os
import shutil
import hashlib
import json
import logging
from pathlib import Path
import random # Added for auto_phash sampling
import concurrent.futures # Added for parallel processing
"""Photo Organizer Script

Resilient to missing heavy dependencies: imagehash (and its numpy stack) is optional.
Falls back to a lightweight perceptual hash implementation if imagehash import fails.

Improved for performance via parallel processing and batched I/O.
"""

# Heavy / optional deps (pandas, folium, jinja2, imagehash) are imported lazily or wrapped
# to prevent environment issues (e.g., broken numpy) impacting core functionality.
if os.environ.get('PHOTO_ORG_SKIP_IMAGEHASH') == '1':
    IMAGEHASH_AVAILABLE = False
else:
    try:  # Attempt to import imagehash (depends on numpy). If it fails, mark unavailable.
        import imagehash  # type: ignore
        IMAGEHASH_AVAILABLE = True
    except Exception:  # pragma: no cover
        IMAGEHASH_AVAILABLE = False
from PIL import Image, ExifTags, ImageFile
from datetime import datetime
import math
try:
    from geopy.geocoders import Nominatim  # type: ignore
    GEOPY_AVAILABLE = True
except Exception:  # pragma: no cover
    GEOPY_AVAILABLE = False
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Iterable, Tuple, Set
import argparse
import time

# Try to import tqdm for progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Optional format handlers moved behind env flags to avoid pulling heavy transitive deps unexpectedly.
HEIF_SUPPORTED = False
RAW_SUPPORTED = False
if os.environ.get('PHOTO_ORG_ENABLE_HEIF') == '1':  # opt-in
    try:  # pragma: no cover
        import pillow_heif  # type: ignore
        pillow_heif.register_heif_opener()
        HEIF_SUPPORTED = True
    except Exception:
        HEIF_SUPPORTED = False
if os.environ.get('PHOTO_ORG_ENABLE_RAW') == '1':  # opt-in
    try:  # pragma: no cover
        import rawpy  # type: ignore
        RAW_SUPPORTED = True
    except Exception:
        RAW_SUPPORTED = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define fallback perceptual hash implementation unconditionally so it exists
# when imagehash is skipped OR import failed.
if not IMAGEHASH_AVAILABLE:
    class _FallbackHash:
        """Simple stand‑in for imagehash.ImageHash supporting subtraction (Hamming distance)."""
        def __init__(self, bits):
            self.hash = bits  # int
        def __sub__(self, other):
            x = (self.hash ^ other.hash) & ((1 << 64) - 1)
            count = 0
            while x:
                x &= x - 1
                count += 1
            return count
        def __str__(self):  # for JSON/reporting
            return format(self.hash, '016x')
            
    def _compute_fallback_phash(pil_img) -> '_FallbackHash':  # type: ignore
        img = pil_img.convert('L').resize((8, 8))
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = 0
        for v in pixels:
            bits = (bits << 1) | (1 if v >= avg else 0)
        return _FallbackHash(bits)
    logger.debug("imagehash not available (skipped or failed); using fallback perceptual hash implementation")

# Configuration & Constants (tunable thresholds centralized here)
CONFIG = {
    'phash_distance_threshold': 6,       # Hamming distance threshold for similar images
    'similar_time_seconds': 3,           # Seconds window for burst detection
    'similar_distance_km': 0.1,          # Location proximity for similar photos (100m)
    'event_time_gap_hours': 2,           # Gap threshold to start a new event
    'event_location_gap_km': 1,          # Location gap (km) threshold to start new event
    'geocode_rate_limit_seconds': 1.0,   # Sleep between geocode calls
    'max_geocode_requests': 250,         # Hard cap per run to avoid abuse
    'map_default_center': (40.0, -74.0), # Default map center
}

# Image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.raw', '.cr2', '.nef', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.jfif'}

# Video extensions (comprehensive list)
VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', 
    '.mpg', '.mpeg', '.3gp', '.3g2', '.mts', '.m2ts', '.vob', '.ogv',
    '.ts', '.divx', '.xvid', '.f4v', '.asf', '.rm', '.rmvb'
}

# Combined media extensions
PHOTO_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# Global collection of extraction issues for post-run diagnostics
# This is populated in main() from the parallel processing results
EXTRACTION_ISSUES: List[Dict[str, Any]] = []

@dataclass
class PhotoInfo:
    path: Path
    datetime: Optional[datetime]
    gps: Optional[tuple]
    sha256: Optional[str]
    phash: Optional[Any]
    resolution: int
    location_name: Optional[str] = None
    organized_path: Optional[str] = None
    duplicate_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert complex objects for JSON safety
        if 'phash' in d and d['phash'] is not None:
             d['phash'] = str(self.phash)
        if 'path' in d and isinstance(self.path, Path):
            d['path'] = str(self.path)
        return d

def parse_args():
    parser = argparse.ArgumentParser(description='Organize and analyze a photo library.')
    # Changed default to None and made it required
    parser.add_argument('--root', type=Path, default=None, required=True, help='Root directory of photos (Required)')
    parser.add_argument('--dry-run', action='store_true', help='Simulate actions without modifying files')
    parser.add_argument('--max-photos', type=int, default=None, help='Limit number of photos processed (for testing)')
    parser.add_argument('--no-map', action='store_true', help='Skip map generation step')
    parser.add_argument('--phash-threshold', type=int, default=CONFIG['phash_distance_threshold'], help='Perceptual hash distance threshold')
    parser.add_argument('--auto-phash', action='store_true', help='Automatically recommend & use a perceptual hash threshold based on distance distribution')
    parser.add_argument('--event-gap-hours', type=float, default=CONFIG['event_time_gap_hours'], help='Time gap in hours to split events')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--allow-truncated', action='store_true', help='Attempt to load truncated/corrupt JPEGs (may salvage metadata)')
    return parser.parse_args()

def scan_photos(root: Path, skip_dirs: Set[Path]) -> Iterable[Path]:
    """Recursively scan for photo and video files, skipping generated directories (pathlib only)."""
    logger.info(f"Scanning media files (photos + videos) in: {root}")
    count = 0
    for dirpath, dirs, files in os.walk(root):
        current = Path(dirpath).resolve()
        # Prune skip_dirs from traversal
        dirs[:] = [d for d in dirs if (current / d).resolve() not in skip_dirs]
        for file in files:
            p = current / file
            if p.suffix.lower() in PHOTO_EXTENSIONS:
                count += 1
                if count % 500 == 0:
                    logger.info(f"Found {count} media files so far...")
                yield p
    logger.info(f"Total media files found: {count}")

def _parse_exif_dict(raw_exif: Any) -> Dict[str, Any]:
    """Helper to convert raw PIL EXIF data into a tag-name keyed dict."""
    if not raw_exif:
        return {}
    exif: Dict[str, Any] = {}
    for tag, val in raw_exif.items():
        exif[ExifTags.TAGS.get(tag, str(tag))] = val
    return exif

def get_datetime(exif: Dict[str, Any], img_path: Path) -> Optional[datetime]:
    """Extract datetime from EXIF, falling back to file modification time."""
    for field in ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']:
        raw = exif.get(field)
        if raw:
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode('utf-8', 'ignore')
                except Exception:
                    continue
            try:
                return datetime.strptime(str(raw), '%Y:%m:%d %H:%M:%S')
            except Exception:
                continue
    # Fallback to file modification time
    try:
        return datetime.fromtimestamp(img_path.stat().st_mtime)
    except Exception:
        return None

def get_gps(exif: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Extract GPS (lat, lon) from EXIF data."""
    gps = exif.get('GPSInfo')
    if not gps:
        return None
        
    def rat(v):
        """Handle EXIF rational values."""
        try:
            if hasattr(v, 'numerator') and hasattr(v, 'denominator'):
                return v.numerator / v.denominator
            if isinstance(v, (tuple, list)) and len(v) == 2:
                return v[0] / v[1]
            return float(v)
        except Exception:
            return 0.0
            
    def dms(vals):
        """Convert Degrees/Minutes/Seconds to decimal degrees."""
        if not vals or len(vals) < 3:
            return 0.0
        d, m, s = (rat(x) for x in vals[:3])
        return d + m/60 + s/3600
        
    try:
        lat = dms(gps.get(2))
        lon = dms(gps.get(4))
        lat_ref = gps.get(1, 'N'); lon_ref = gps.get(3, 'E')
        if isinstance(lat_ref, bytes): lat_ref = lat_ref.decode('utf-8','ignore')
        if isinstance(lon_ref, bytes): lon_ref = lon_ref.decode('utf-8','ignore')
        if lat_ref == 'S': lat = -lat
        if lon_ref == 'W': lon = -lon
        
        if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0 or lon != 0):
            return (lat, lon)
        return None
    except Exception as e:
        logger.debug(f"GPS parse failed {exif.get('GPSInfo')}: {e}")
        return None

def sha256sum(path: Path) -> Optional[str]:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        # Try normal path first
        with open(str(path), 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        # If we get "Invalid argument" error, it's likely due to special characters
        # Skip these files for now - they won't be detected as duplicates
        if e.errno == 22:  # Invalid argument
            logger.debug(f"Skipping file with special characters: {path}")
            return None
        logger.warning(f"Hash failed for {path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Hash failed for {path}: {e}")
        return None

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calculate distance between two GPS coordinates."""
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(h))

def extract_photo(path: Path) -> Tuple[Optional[PhotoInfo], Optional[Dict]]:
    """
    Extract all metadata from a single photo or video.
    Designed to be run in a separate process: opens the file ONCE.
    Returns (PhotoInfo, None) on success, or (None, ErrorDict) on failure.
    """
    sha = sha256sum(path)
    exif = {}
    gps = None
    dt = None
    phash = None
    res = 0
    
    is_video = path.suffix.lower() in VIDEO_EXTENSIONS
    
    # For videos, try to extract metadata without opening as image
    if is_video:
        # Videos: extract creation date from file metadata, skip image processing
        try:
            # Get file size as "resolution" proxy for videos (in MB)
            file_size = path.stat().st_size
            res = file_size  # Store file size for videos
            
            # Try to get creation time from file system
            dt = get_datetime({}, path)  # Empty EXIF, will use file time
            
            # Videos typically don't have GPS in standard metadata
            # (would need ffmpeg/exiftool for advanced extraction)
            gps = None
            phash = None  # Skip perceptual hash for videos
            
        except Exception as e:
            error_msg = f"Failed to process video metadata: {e}"
            logger.debug(f"EXTRACT FAIL | {path} | {error_msg}")
            return None, {
                'path': str(path),
                'type': 'video_metadata_fail',
                'message': error_msg
            }
    else:
        # Images: full EXIF extraction
        try:
            with Image.open(path) as img:
                # 1. Get EXIF
                raw_exif = img.getexif()
                exif = _parse_exif_dict(raw_exif)
                
                # 2. Get Resolution
                w, h = img.size
                res = w * h
                
                # 3. Get pHash
                try:
                    if IMAGEHASH_AVAILABLE:
                        phash = imagehash.phash(img)  # type: ignore[attr-defined]
                    else:  # fallback pure-python hash
                        phash = _compute_fallback_phash(img)
                except Exception as e:
                     logger.debug(f"pHash fail {path}: {e}")
                     # Non-fatal: continue processing

        except Exception as e:
            # This is a fatal error for this file (e.g., corrupt, unreadable)
            error_msg = f"Failed to open/process image: {e}"
            logger.debug(f"EXTRACT FAIL | {path} | {error_msg}")
            return None, {
                'path': str(path),
                'type': 'image_open_fail',
                'message': error_msg
            }

        # 4. Get Datetime (uses EXIF, falls back to file stat)
        dt = get_datetime(exif, path)
        
        # 5. Get GPS (uses EXIF)
        gps = get_gps(exif)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "EXTRACT | file=%s dt=%s gps=%s sha=%s res=%s phash=%s source=%s", 
            path.name,
            dt.isoformat() if dt else None,
            f"{gps[0]:.5f},{gps[1]:.5f}" if gps else None,
            (sha[:12] + '...') if sha else None,
            res,
            str(phash) if phash is not None else None,
            'imagehash' if IMAGEHASH_AVAILABLE else 'fallback'
        )

    # Check for non-fatal issues
    issues: List[Dict] = []
    if res == 0:
        issues.append({'type': 'resolution_zero', 'message': 'Image size could not be read'})
    if phash is None:
        issues.append({'type': 'missing_phash', 'message': 'Perceptual hash not generated'})
    if not sha:
        issues.append({'type': 'missing_sha256', 'message': 'SHA-256 hash not generated'})

    info = PhotoInfo(path=path, datetime=dt, gps=gps, sha256=sha, phash=phash, resolution=res)
    
    if issues:
        # Return the info we have, but also the errors
        return info, {
            'path': str(path),
            'type': 'metadata_warning',
            'message': ", ".join(i['message'] for i in issues)
        }
            
    return info, None

def organize_photos(photo_infos: List[PhotoInfo], organized_dir: Path, dry_run: bool = False) -> None:
    """Organize photos into year/month/location folder structure.
    For photos with missing metadata (Unknown/Unknown), organizes by file size and modified date.
    Respects dry-run mode (no filesystem modifications)."""
    logger.info("Organizing photos into folders..." + (" (dry-run)" if dry_run else ""))
    organized_count = 0
    if not dry_run:
        organized_dir.mkdir(exist_ok=True)

    for info in photo_infos:
        dt = info.datetime
        loc = info.location_name
        
        if loc:
            loc = loc.split(',')[0].strip()
            loc = ''.join(c for c in loc if c.isalnum() or c in (' ', '-', '_')).strip() or 'Unknown'
        else:
            loc = 'Unknown'
            
        year = str(dt.year) if dt else 'Unknown'
        month = f'{dt.month:02d}-{dt.strftime("%B")}' if dt else 'Unknown'
        
        # Enhanced organization for Unknown/Unknown photos
        if year == 'Unknown' and month == 'Unknown':
            # Get file size in MB
            try:
                file_size_mb = info.path.stat().st_size / (1024 * 1024)
                if file_size_mb < 1:
                    size_category = '1-Small_Under1MB'
                elif file_size_mb < 5:
                    size_category = '2-Medium_1-5MB'
                elif file_size_mb < 15:
                    size_category = '3-Large_5-15MB'
                else:
                    size_category = '4-XL_Over15MB'
            except Exception:
                size_category = '0-Unknown_Size'

            # Use file modification time as fallback
            try:
                mod_time = datetime.fromtimestamp(info.path.stat().st_mtime)
                mod_year = str(mod_time.year)
                mod_month = f'{mod_time.month:02d}-{mod_time.strftime("%B")}'
            except Exception:
                mod_year = 'Unknown_Date'
                mod_month = 'Unknown_Date'

            # Structure: Organized/Unknown_Metadata/[SizeCategory]/[Year]/[Month]/[Location?]
            if loc and loc != 'Unknown':
                target_dir = organized_dir / 'Unknown_Metadata' / size_category / mod_year / mod_month / loc
            else:
                target_dir = organized_dir / 'Unknown_Metadata' / size_category / mod_year / mod_month
        else:
            # Normal photos: avoid creating an extra 'Unknown' location folder
            if loc and loc != 'Unknown':
                target_dir = organized_dir / year / month / loc
            else:
                target_dir = organized_dir / year / month
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            
        filename = info.path.name
        target_path = target_dir / filename
        
        counter = 1
        while not dry_run and target_path.exists():
            # Handle filename collision
            name, ext = os.path.splitext(filename)
            target_path = target_dir / f"{name}_{counter}{ext}"
            counter += 1
            
        info.organized_path = str(target_path) # Record planned path even in dry-run
        
        if dry_run:
            continue
            
        try:
            # MOVE the file (not copy/link) to avoid creating duplicates
            shutil.move(str(info.path), str(target_path))
            organized_count += 1
        except Exception as e:
            logger.error(f"Failed to move {info.path}: {e}")
            info.organized_path = str(info.path) # Reset path on failure
                
    logger.info(f"Organized {organized_count} photos")

def find_duplicates(photo_infos: List[PhotoInfo], duplicate_dir: Path, dry_run: bool = False) -> List[tuple]:
    """Find and handle duplicate images based on SHA-256 hash.
    In dry-run, only logs actions."""
    logger.info("Finding duplicate images..." + (" (dry-run)" if dry_run else ""))
    hash_map: Dict[str, List[PhotoInfo]] = defaultdict(list)
    for info in photo_infos:
        if info.sha256:
            hash_map[info.sha256].append(info)
            
    duplicates: List[tuple] = []
    duplicate_count = 0
    if not dry_run:
        duplicate_dir.mkdir(exist_ok=True)
        
    for files in hash_map.values():
        if len(files) > 1:
            # Sort to find the "best" one to keep:
            # 1. Highest resolution (descending)
            # 2. Oldest modification time (ascending)
            # 3. Shortest path length (ascending)
            files.sort(key=lambda x: (-x.resolution, x.path.stat().st_mtime, len(str(x.path))))
            keep = files[0]
            
            for dup in files[1:]:
                filename = dup.path.name
                target = duplicate_dir / filename
                
                counter = 1
                while not dry_run and target.exists():
                    name, ext = os.path.splitext(filename)
                    target = duplicate_dir / f"{name}_dup{counter}{ext}"
                    counter += 1
                    
                dup.duplicate_path = str(target) # Record planned path
                duplicates.append((str(keep.path), str(dup.path)))
                
                if dry_run:
                    logger.info(f"[DRY-RUN] Would move duplicate {dup.path} -> {target}")
                    continue
                    
                try:
                    shutil.move(str(dup.path), str(target))
                    duplicate_count += 1
                except Exception as e:
                    logger.error(f"Duplicate move failed {dup.path}: {e}")
                    
    logger.info(f"Duplicate sets handled: {duplicate_count}" + (" (simulated)" if dry_run else ""))
    return duplicates

def group_similar_photos(photo_infos: List[PhotoInfo], phash_threshold: int, time_window: int, distance_km: float) -> List[List[PhotoInfo]]:
    """Group similar photos using perceptual hash, timestamp, and location - OPTIMIZED."""
    logger.info("Grouping similar photos (burst shots, etc.)...")
    groups: List[List[PhotoInfo]] = []
    used: set = set()
    valid_photos = [p for p in photo_infos if p.phash is not None]
    
    deep_debug = os.environ.get('PHOTO_ORG_DEEP_DEBUG') == '1'
    total_photos = len(valid_photos)
    logger.info(f"GROUP | photos_with_phash={total_photos} threshold={phash_threshold} time_window={time_window}s distance_km={distance_km:.3f}")
    
    # OPTIMIZATION 1: Sort by timestamp to reduce comparisons
    # Photos taken close in time are more likely to be similar
    valid_photos_with_time = [(i, p) for i, p in enumerate(valid_photos) if p.datetime]
    valid_photos_without_time = [(i, p) for i, p in enumerate(valid_photos) if not p.datetime]
    
    # Sort by time
    valid_photos_with_time.sort(key=lambda x: x[1].datetime)
    sorted_valid = valid_photos_with_time + valid_photos_without_time
    
    logger.info(f"⚡ FAST MODE: Only comparing photos within {time_window}s time window")
    logger.info(f"This reduces comparisons from ~{total_photos * total_photos // 2:,} to ~{total_photos * 100:,}")
    
    # Wrap with tqdm if available for progress bar
    iter_obj = tqdm(sorted_valid, desc="Grouping similar photos", unit="photo") if TQDM_AVAILABLE else sorted_valid
    
    for idx, (i, info1) in enumerate(iter_obj):
        if i in used:
            continue
        group = [info1]
        used.add(i)
        
        # Log progress every 500 photos
        if not TQDM_AVAILABLE and (idx + 1) % 500 == 0:
            progress_pct = ((idx + 1) / total_photos) * 100
            logger.info(f"GROUP PROGRESS | {idx+1}/{total_photos} ({progress_pct:.1f}%) - Found {len(groups)} groups so far")
        
        # OPTIMIZATION 2: Only compare with photos within time window
        # Start from current position and look ahead
        for j_idx in range(idx + 1, len(sorted_valid)):
            j, info2 = sorted_valid[j_idx]
            
            if j in used:
                continue
            
            # EARLY EXIT: If photos have timestamps and are too far apart in time, stop
            if info1.datetime and info2.datetime:
                time_diff = abs((info2.datetime - info1.datetime).total_seconds())
                if time_diff > time_window:
                    # Since sorted by time, all following photos will be even further
                    break
            
            is_similar = False
            
            # 1. Check pHash
            if info1.phash and info2.phash:
                ph_dist = (info1.phash - info2.phash)
                if ph_dist < phash_threshold:
                    is_similar = True
                if deep_debug and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "GROUP COMP | i=%d j=%d phash_dist=%s < %d ? %s", i, j, ph_dist, phash_threshold, ph_dist < phash_threshold
                    )
                    
            # 2. Check time (if not similar by pHash)
            if (not is_similar and info1.datetime and info2.datetime):
                time_diff = abs((info1.datetime - info2.datetime).total_seconds())
                if time_diff <= time_window:
                    is_similar = True
                    if deep_debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "GROUP COMP | i=%d j=%d time_diff=%.3fs <= %ds -> similar", i, j, time_diff, time_window
                        )
                        
            if is_similar:
                # 3. Check location
                # If both have GPS, they must be close.
                # If one or both lack GPS, group them anyway (assume same location).
                if info1.gps and info2.gps:
                    dist = haversine_km(info1.gps, info2.gps)
                    if deep_debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "GROUP DIST | i=%d j=%d haversine=%.4fkm <= %.3f ? %s", i, j, dist, distance_km, dist <= distance_km
                        )
                    if dist <= distance_km:
                        group.append(info2)
                        used.add(j)
                else:
                    # One or both lack GPS, group them
                    group.append(info2)
                    used.add(j)
                    
        if len(group) > 1:
            groups.append(group)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("GROUP FORMED | size=%d members=%s", len(group), [p.path.name for p in group])
                
    logger.info(f"✓ Found {len(groups)} groups of similar photos in optimized mode")
    return groups

def cluster_events(photo_infos: List[PhotoInfo], gap_hours: float, gap_loc_km: float) -> List[List[PhotoInfo]]:
    """Cluster sorted photos into events based on time and location gaps."""
    logger.info("Clustering photos into events...")
    events: List[List[PhotoInfo]] = []
    valid = sorted([p for p in photo_infos if p.datetime], key=lambda x: x.datetime)
    
    if not valid:
        return []

    event: List[PhotoInfo] = [valid[0]]
    last_time = valid[0].datetime
    last_loc = valid[0].gps

    for info in valid[1:]:
        time_gap = (info.datetime - last_time).total_seconds() / 3600 if last_time else 0
        loc_gap = 0
        
        if last_loc and info.gps:
            loc_gap = haversine_km(last_loc, info.gps)
            
        # Start a new event if time gap OR location gap is too large
        if time_gap > gap_hours or loc_gap > gap_loc_km:
            if event:
                events.append(event)
            event = [info]
        else:
            event.append(info)
            
        last_time = info.datetime
        # Only update location if the new photo *has* one
        if info.gps:
            last_loc = info.gps
            
    if event:
        events.append(event)
        
    logger.info(f"Clustered into {len(events)} events")
    return events

def reverse_geocode(gps, cache: Dict, rate_limit: float, geocoder):
    """Get location name from GPS coordinates with caching (safe + rate limited)."""
    if not gps or not geocoder:
        return None
        
    lat, lon = gps
    # Cache keys are strings "lat,lon" rounded to 3 decimals
    cache_key = f"{round(lat, 3)},{round(lon, 3)}"
    if cache_key in cache:
        return cache[cache_key]
        
    try:
        if rate_limit:
            time.sleep(rate_limit)
        location = geocoder.reverse(gps, timeout=10, language='en')
        result = location.address if location else None
        cache[cache_key] = result # Cache success or failure (None)
        return result
    except Exception as e:  # pragma: no cover
        logger.debug(f"Geocoding failed for {gps}: {e}")
        cache[cache_key] = None
        return None

def create_map_visualization(events: List[List[PhotoInfo]], report_dir: Path):
    """Create an interactive map showing photo events."""
    logger.info("Creating map visualization...")
    try:
        import folium  # type: ignore
    except Exception as e:
        logger.warning(f"folium not available, skipping map generation: {e}")
        return None
        
    # Find a center point for the map (first event with GPS)
    center_lat, center_lon = CONFIG['map_default_center']
    for event in events:
        for photo in event:
            if photo.gps:
                center_lat, center_lon = photo.gps
                break
        if (center_lat, center_lon) != CONFIG['map_default_center']:
            break
            
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
    
    for i, event in enumerate(events):
        event_gps = None
        # Find first GPS coordinate in the event to mark it
        for photo in event:
            if photo.gps:
                event_gps = photo.gps
                break
                
        if event_gps:
            dates = [p.datetime for p in event if p.datetime]
            date_range = f"{min(dates).strftime('%Y-%m-%d')} - {max(dates).strftime('%Y-%m-%d')}" if dates else 'Unknown'
            popup_text = (f"<b>Event {i+1}</b><br>Date: {date_range}<br>Photos: {len(event)}<br>"
                          f"Location: {event[0].location_name or 'Unknown'}")
                          
            folium.CircleMarker(
                location=event_gps,
                radius=min(len(event), 20), # Radius based on photo count, capped
                popup=popup_text,
                color='red',
                fill=True
            ).add_to(m)
            
    map_path = report_dir / 'photo_events_map.html'
    m.save(str(map_path))
    logger.info(f"Map saved to {map_path}")
    return str(map_path)

def main():
    args = parse_args()
    
    # --- 1. Setup Paths & Config ---
    PHOTO_ROOT = args.root.resolve()
    if not PHOTO_ROOT.exists():
        logger.error(f"Photo directory not found: {PHOTO_ROOT}")
        return
        
    # Define paths locally, not as globals
    REPORT_DIR = PHOTO_ROOT / 'photo_reports'
    DUPLICATE_DIR = PHOTO_ROOT / 'duplicates'
    ORGANIZED_DIR = PHOTO_ROOT / 'Organized'
    GEOCODE_CACHE_FILE = REPORT_DIR / 'geocode_cache.json'
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.allow_truncated:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        logger.info("Truncated image loading enabled (Pillow will attempt to open partial files)")
        
    # Create directories
    for d in (REPORT_DIR, DUPLICATE_DIR, ORGANIZED_DIR):
        d.mkdir(exist_ok=True)
        
    # Load geocode cache
    geocode_cache: Dict = {}
    if GEOCODE_CACHE_FILE.exists():
        try:
            with open(GEOCODE_CACHE_FILE, 'r', encoding='utf-8') as f:
                geocode_cache = json.load(f)
            logger.info(f"Loaded {len(geocode_cache)} items from geocode cache")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load geocode cache: {e}")
            geocode_cache = {}
            
    logger.info(f"Starting photo organization process at {PHOTO_ROOT} (dry-run={args.dry_run})")
    start_time = time.time()
    
    # --- 2. Scan and Process Photos (in Parallel) ---
    skip_dirs = {DUPLICATE_DIR.resolve(), REPORT_DIR.resolve(), ORGANIZED_DIR.resolve()}
    paths_to_process = list(scan_photos(PHOTO_ROOT, skip_dirs))
    
    if args.max_photos:
        paths_to_process = paths_to_process[:args.max_photos]
        logger.info(f"Processing limited to first {len(paths_to_process)} photos (--max-photos)")

    if not paths_to_process:
        logger.warning("No photos found to process.")
        return

    photo_infos: List[PhotoInfo] = []
    local_extraction_issues: List[Dict] = []
    total_photos = len(paths_to_process)
    processing_start = time.time()

    logger.info(f"Processing {total_photos} photos using parallel workers...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map processes the list in parallel
        future_to_path = executor.map(extract_photo, paths_to_process)
        
        # Wrap with tqdm if available
        iter_obj = tqdm(future_to_path, total=total_photos, desc="Processing photos", unit="photo") if TQDM_AVAILABLE else future_to_path

        for i, (info, error) in enumerate(iter_obj):
            if error:
                local_extraction_issues.append(error)
            if info:
                photo_infos.append(info)
                
            # Periodic progress log (in case tqdm is not available or for richer stats)
            if (i + 1) == 1 or ((i + 1) % 1000 == 0):
                elapsed = time.time() - processing_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_photos - (i + 1)) / rate if rate > 0 else float('inf')
                logger.info(
                    f"PROCESS | processed={i+1}/{total_photos} ({(i+1)/total_photos:.1%}) "
                    f"rate={rate:.1f} photos/s ETA={eta/60:.1f} min")

    # Update global issues list for final report
    global EXTRACTION_ISSUES
    EXTRACTION_ISSUES.extend(local_extraction_issues)
    
    # Count images vs videos
    image_count = sum(1 for p in photo_infos if p.path.suffix.lower() in IMAGE_EXTENSIONS)
    video_count = len(photo_infos) - image_count
    
    logger.info(f"Processed {len(photo_infos)} media files: {image_count} images, {video_count} videos (HEIF={HEIF_SUPPORTED}, RAW={RAW_SUPPORTED})")
    if not photo_infos:
        logger.warning("No photos were successfully processed.")
        return

    # --- 3. Batch Geocode (Efficiently) ---
    logger.info("Batching geocode requests...")
    gps_to_geocode: Set[Tuple[float, float]] = set()
    for info in photo_infos:
        if info.gps:
            cache_key = f"{round(info.gps[0], 3)},{round(info.gps[1], 3)}"
            if cache_key not in geocode_cache:
                gps_to_geocode.add(info.gps)
    
    geocoder = None
    if gps_to_geocode and GEOPY_AVAILABLE:
        try:
            geocoder = Nominatim(user_agent=f'photo_organizer_v1.2_{random.randint(1,10000)}')
        except Exception as e:
            logger.warning(f"Failed to initialize geocoder: {e}")
            
    if geocoder:
        num_to_geocode = min(len(gps_to_geocode), CONFIG['max_geocode_requests'])
        logger.info(f"Found {len(gps_to_geocode)} unique GPS coordinates to geocode (limit {num_to_geocode})")
        
        geocode_iter = list(gps_to_geocode)[:num_to_geocode]
        iter_obj = tqdm(geocode_iter, desc="Geocoding", unit="coord") if TQDM_AVAILABLE else geocode_iter
        
        for gps_tuple in iter_obj:
            # This will call the API and update the cache dict in-place
            reverse_geocode(gps_tuple, 
                            cache=geocode_cache, 
                            rate_limit=CONFIG['geocode_rate_limit_seconds'], 
                            geocoder=geocoder)
    
    # Populate location_name back into photo_infos
    for info in photo_infos:
        if info.gps:
            cache_key = f"{round(info.gps[0], 3)},{round(info.gps[1], 3)}"
            info.location_name = geocode_cache.get(cache_key)

    # --- 4. Organize, Find Duplicates, and Cluster ---
    organize_photos(photo_infos, ORGANIZED_DIR, dry_run=args.dry_run)
    
    duplicates = find_duplicates(photo_infos, DUPLICATE_DIR, dry_run=args.dry_run)
    
    # Optional adaptive pHash threshold recommendation
    effective_phash_threshold = args.phash_threshold
    if args.auto_phash:
        logger.info("Calculating automatic pHash threshold...")
        valid = [p for p in photo_infos if p.phash is not None]
        max_sample = 500  # Increased sample size, now using random
        
        if len(valid) > 1:
            # Use a random sample, not just the first N photos
            sample = random.sample(valid, min(len(valid), max_sample))
            dists = []
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    try:
                        d = sample[i].phash - sample[j].phash  # type: ignore
                        dists.append(int(d))
                    except Exception:
                        continue
            
            stats = {}
            if dists:
                dists.sort()
                import statistics
                stats['count'] = len(dists)
                stats['min'] = dists[0]
                stats['max'] = dists[-1]
                stats['median'] = int(statistics.median(dists))
                stats['p10'] = dists[max(0, int(0.10 * len(dists)) - 1)]
                stats['p25'] = dists[max(0, int(0.25 * len(dists)) - 1)]
                stats['p75'] = dists[max(0, int(0.75 * len(dists)) - 1)]
                stats['p90'] = dists[max(0, int(0.90 * len(dists)) - 1)]
                # Heuristic: choose threshold
                recommended = max(4, min(stats['p25'] - 2, stats['median'] - 4))
                stats['recommended_threshold'] = recommended
                logger.info("Auto pHash stats: %s", {k: stats[k] for k in sorted(stats) if 'hist' not in k})
                effective_phash_threshold = recommended
                
                # Persist histogram
                hist = defaultdict(int)
                for d in dists:
                    bucket = f"{(d // 2) * 2}-{(d // 2) * 2 + 1}"
                    hist[bucket] += 1
                stats['histogram_2wide'] = dict(sorted(hist.items(), key=lambda x: int(x[0].split('-')[0])))
                try:
                    with open(REPORT_DIR / 'phash_stats.json', 'w', encoding='utf-8') as f:
                        json.dump(stats, f, indent=2)
                except Exception as e:
                    logger.warning(f"Could not write phash_stats.json: {e}")
            else:
                logger.info("Auto pHash: insufficient data for statistics")
        else:
            logger.info("Auto pHash: Not enough photos with pHash to calculate stats.")

    similar_groups = group_similar_photos(photo_infos, effective_phash_threshold, CONFIG['similar_time_seconds'], CONFIG['similar_distance_km'])
    
    events = cluster_events(photo_infos, gap_hours=args.event_gap_hours, gap_loc_km=CONFIG['event_location_gap_km'])

    # --- 5. Generate Reports ---
    logger.info("Generating reports...")
    
    # Duplicates CSV - Enhanced with metadata for web viewer
    if duplicates:
        try:
            import csv as csv_module
            # Write to both report dir and File organizers root for web app
            csv_paths = [
                REPORT_DIR / 'duplicates.csv',
                PHOTO_ROOT / 'duplicates.csv'  # For web app
            ]
            
            for csv_path in csv_paths:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv_module.writer(f)
                    writer.writerow(['original', 'duplicate'])
                    for orig, dup in duplicates:
                        writer.writerow([orig, dup])
            
            logger.info(f"Duplicates CSV written to {csv_paths[0]} and {csv_paths[1]}")
        except Exception as e:
            logger.warning(f"Could not write duplicates.csv: {e}")
            
    # Similar groups JSON
    similar_report = [{
        'group_id': i,
        'photo_count': len(group),
        'photos': [str(p.path) for p in group],
        'date_range': str(min([p.datetime for p in group if p.datetime])) if any(p.datetime for p in group) else None
    } for i, group in enumerate(similar_groups)]
    with open(REPORT_DIR / 'similar_groups.json', 'w', encoding='utf-8') as f:
        json.dump(similar_report, f, ensure_ascii=False, indent=2, default=str)
        
    # Events JSON
    event_report = []
    for i, group in enumerate(events):
        dates = [p.datetime for p in group if p.datetime]
        locs = [p.location_name for p in group if p.location_name]
        gps_coords = [p.gps for p in group if p.gps]
        event_report.append({
            'event_id': i,
            'date_range': f"{min(dates)} - {max(dates)}" if dates else 'Unknown',
            'location': locs[0] if locs else 'Unknown',
            'gps_center': gps_coords[0] if gps_coords else None,
            'num_photos': len(group),
            'sample_photos': [p.organized_path or str(p.path) for p in group[:5]]
        })
    with open(REPORT_DIR / 'events.json', 'w', encoding='utf-8') as f:
        json.dump(event_report, f, ensure_ascii=False, indent=2, default=str)

    # Persist geocode cache
    try:
        with open(GEOCODE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(geocode_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Could not save geocode cache: {e}")
        
    # Map Visualization
    if events and not args.no_map and not args.dry_run:
        try:
            create_map_visualization(events, REPORT_DIR)
        except Exception as e:
            logger.warning(f"Map generation failed: {e}")
            
    # HTML Summary Report
    try:
        from jinja2 import Template  # type: ignore
    except Exception as e:
        logger.warning(f"jinja2 not available, skipping HTML summary: {e}")
        Template = None  # type: ignore

    if Template:
        html_template_str = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Photo Organization Summary</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background-color: #fdfdfd; }
                h1, h2, h3 { color: #333; }
                .event { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
                .photos { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
                .photos img { width: 150px; height: 150px; object-fit: cover; border-radius: 5px; border: 1px solid #eee; }
                .stats { background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0; }
                .stats p { margin: 5px 0; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Photo Organization Summary</h1>
            
            <div class="stats">
                <h2>Statistics</h2>
                <p><strong>Total Photos Processed:</strong> {{ total_photos }}</p>
                <p><strong>Events Detected:</strong> {{ num_events }}</p>
                <p><strong>Exact Duplicates Found:</strong> {{ num_duplicates }}</p>
                <p><strong>Similar Groups (Bursts):</strong> {{ num_similar_groups }}</p>
                <p><strong>Processing Time:</strong> {{ processing_time }} seconds</p>
            </div>
            
            {% if events and not no_map %}
            <p><a href="photo_events_map.html" target="_blank">View Interactive Map</a></p>
            {% endif %}

            <h2>Photo Events</h2>
            {% for event in events %}
            <div class="event">
                <h3>Event {{ loop.index }}: {{ event.location or 'Unknown Location' }}</h3>
                <p><strong>Date:</strong> {{ event.date_range }}</p>
                <p><strong>Photos:</strong> {{ event.num_photos }}</p>
                {% if event.gps_center %}
                <p><strong>GPS:</strong> {{ "%.4f, %.4f"|format(event.gps_center[0], event.gps_center[1]) }}</p>
                {% endif %}
                <div class="photos">
                    {% for photo in event.sample_photos %}
                    <img src="file:///{{ photo | e }}" alt="Event photo" title="{{ photo | e }}">
                    {% endfor %}
                </div>
            </div>
            {% else %}
            <p>No events were clustered.</p>
            {% endfor %}
        </body>
        </html>
        '''
        html_template = Template(html_template_str)
        processing_time = round(time.time() - start_time, 2)
        try:
            html = html_template.render(
                events=event_report,
                total_photos=len(photo_infos),
                num_events=len(events),
                num_duplicates=len(duplicates),
                num_similar_groups=len(similar_groups),
                processing_time=processing_time,
                no_map=args.no_map or args.dry_run
            )
            with open(REPORT_DIR / 'summary.html', 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception as e:
            logger.warning(f"Failed to generate HTML summary: {e}")
            
    # --- 6. Final Summary ---
    processing_time = round(time.time() - start_time, 2)
    logger.info("=" * 50)
    logger.info("PHOTO ORGANIZATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total photos processed: {len(photo_infos)}")
    logger.info(f"Events detected: {len(events)}")
    logger.info(f"Duplicates found: {len(duplicates)}")
    logger.info(f"Similar groups: {len(similar_groups)} (pHash threshold used: {effective_phash_threshold})")
    logger.info(f"Processing time: {processing_time} seconds")
    logger.info(f"Reports saved to: {REPORT_DIR}")

    # Summarize extraction issues
    if EXTRACTION_ISSUES:
        counts = defaultdict(int)
        for issue in EXTRACTION_ISSUES:
            counts[issue['type']] += 1
        logger.info("Extraction issue summary: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        try:
            with open(REPORT_DIR / 'extraction_issues.json', 'w', encoding='utf-8') as f:
                json.dump(EXTRACTION_ISSUES, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed extraction issues written to {REPORT_DIR / 'extraction_issues.json'}")
        except Exception as e:
            logger.warning(f"Could not write extraction issues file: {e}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()