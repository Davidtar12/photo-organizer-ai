"""Video Metadata Extraction using FFmpeg

Extracts comprehensive metadata from video files including:
- Duration
- Creation date from metadata
- GPS coordinates (if embedded)
- Resolution
- Codec information
- Frame rate
"""

import ffmpeg
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import re

logger = logging.getLogger(__name__)


def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a video file using ffmpeg.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing:
            - duration: float (seconds)
            - creation_time: datetime or None
            - gps: tuple (lat, lon) or None
            - resolution: int (width * height)
            - width: int
            - height: int
            - codec: str
            - fps: float
            - bitrate: int
            - file_size: int
    """
    metadata = {
        'duration': None,
        'creation_time': None,
        'gps': None,
        'resolution': 0,
        'width': 0,
        'height': 0,
        'codec': None,
        'fps': None,
        'bitrate': None,
        'file_size': 0
    }
    
    try:
        # Get file size
        metadata['file_size'] = video_path.stat().st_size
        
        # Probe video file
        probe = ffmpeg.probe(str(video_path))
        
        # Extract format information
        if 'format' in probe:
            fmt = probe['format']
            
            # Duration
            if 'duration' in fmt:
                try:
                    metadata['duration'] = float(fmt['duration'])
                except (ValueError, TypeError):
                    pass
            
            # Bitrate
            if 'bit_rate' in fmt:
                try:
                    metadata['bitrate'] = int(fmt['bit_rate'])
                except (ValueError, TypeError):
                    pass
            
            # Format tags (contains creation_time, GPS, etc.)
            if 'tags' in fmt:
                tags = fmt['tags']
                
                # Creation time (various formats)
                creation_time = _extract_creation_time(tags)
                if creation_time:
                    metadata['creation_time'] = creation_time
                
                # GPS coordinates
                gps = _extract_gps_from_tags(tags)
                if gps:
                    metadata['gps'] = gps
        
        # Extract video stream information
        if 'streams' in probe:
            for stream in probe['streams']:
                if stream.get('codec_type') == 'video':
                    # Resolution
                    if 'width' in stream and 'height' in stream:
                        try:
                            width = int(stream['width'])
                            height = int(stream['height'])
                            metadata['width'] = width
                            metadata['height'] = height
                            metadata['resolution'] = width * height
                        except (ValueError, TypeError):
                            pass
                    
                    # Codec
                    if 'codec_name' in stream:
                        metadata['codec'] = stream['codec_name']
                    
                    # Frame rate
                    if 'r_frame_rate' in stream:
                        try:
                            fps_str = stream['r_frame_rate']
                            if '/' in fps_str:
                                num, den = fps_str.split('/')
                                metadata['fps'] = float(num) / float(den)
                            else:
                                metadata['fps'] = float(fps_str)
                        except (ValueError, TypeError, ZeroDivisionError):
                            pass
                    
                    # Stream tags (another place for metadata)
                    if 'tags' in stream:
                        stream_tags = stream['tags']
                        
                        # Try to get creation time from stream if not found in format
                        if not metadata['creation_time']:
                            creation_time = _extract_creation_time(stream_tags)
                            if creation_time:
                                metadata['creation_time'] = creation_time
                        
                        # Try to get GPS from stream if not found in format
                        if not metadata['gps']:
                            gps = _extract_gps_from_tags(stream_tags)
                            if gps:
                                metadata['gps'] = gps
                    
                    break  # Only process first video stream
        
        logger.debug(f"Extracted video metadata from {video_path.name}: duration={metadata['duration']}s, "
                    f"resolution={metadata['width']}x{metadata['height']}, fps={metadata['fps']}")
        
    except ffmpeg.Error as e:
        logger.warning(f"FFmpeg error extracting metadata from {video_path}: {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        logger.warning(f"Error extracting video metadata from {video_path}: {e}")
    
    return metadata


def _extract_creation_time(tags: Dict[str, str]) -> Optional[datetime]:
    """
    Extract creation time from video tags.
    
    Tries multiple tag names and date formats.
    """
    # Common tag names for creation time
    time_tags = [
        'creation_time',
        'date',
        'creation_date',
        'com.apple.quicktime.creationdate',
        'com.android.capture.time'
    ]
    
    for tag_name in time_tags:
        if tag_name in tags:
            time_str = tags[tag_name]
            
            # Try multiple datetime formats
            formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',      # 2023-01-15T14:30:45.123Z
                '%Y-%m-%dT%H:%M:%SZ',         # 2023-01-15T14:30:45Z
                '%Y-%m-%d %H:%M:%S',          # 2023-01-15 14:30:45
                '%Y-%m-%d',                   # 2023-01-15
                '%Y%m%d',                     # 20230115
                '%Y:%m:%d %H:%M:%S',          # 2023:01:15 14:30:45 (like EXIF)
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(time_str, fmt)
                except (ValueError, TypeError):
                    continue
    
    return None


def _extract_gps_from_tags(tags: Dict[str, str]) -> Optional[Tuple[float, float]]:
    """
    Extract GPS coordinates from video tags.
    
    Tries multiple tag formats used by different devices.
    """
    # ISO 6709 format: +40.7614-073.9776/ (latitude+longitude)
    if 'location' in tags:
        loc_str = tags['location']
        match = re.match(r'([+-]\d+\.?\d*)([+-]\d+\.?\d*)', loc_str)
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            except ValueError:
                pass
    
    # Separate latitude/longitude tags
    lat_tags = ['location-latitude', 'com.apple.quicktime.location.latitude', 'GPS:Latitude']
    lon_tags = ['location-longitude', 'com.apple.quicktime.location.longitude', 'GPS:Longitude']
    
    lat = None
    lon = None
    
    for tag in lat_tags:
        if tag in tags:
            try:
                lat = float(tags[tag])
                break
            except ValueError:
                pass
    
    for tag in lon_tags:
        if tag in tags:
            try:
                lon = float(tags[tag])
                break
            except ValueError:
                pass
    
    if lat is not None and lon is not None:
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
    
    return None


def extract_video_frame(video_path: Path, output_path: Path, time_seconds: float = 1.0) -> bool:
    """
    Extract a single frame from a video at a specific time.
    
    Useful for generating thumbnails or for face detection.
    
    Args:
        video_path: Path to video file
        output_path: Path to save the extracted frame (JPEG)
        time_seconds: Time in seconds to extract frame from (default: 1.0)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        (
            ffmpeg
            .input(str(video_path), ss=time_seconds)
            .filter('scale', 1280, -1)  # Scale to 1280px wide, maintain aspect ratio
            .output(str(output_path), vframes=1, format='image2', vcodec='mjpeg')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        logger.debug(f"Extracted frame from {video_path.name} at {time_seconds}s -> {output_path}")
        return True
    except ffmpeg.Error as e:
        logger.warning(f"FFmpeg error extracting frame from {video_path}: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except Exception as e:
        logger.warning(f"Error extracting frame from {video_path}: {e}")
        return False


def get_video_duration(video_path: Path) -> Optional[float]:
    """Quick extraction of just the video duration (lightweight)."""
    try:
        probe = ffmpeg.probe(str(video_path))
        if 'format' in probe and 'duration' in probe['format']:
            return float(probe['format']['duration'])
    except Exception:
        pass
    return None
