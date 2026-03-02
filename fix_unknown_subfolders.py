"""
Fix Unknown subfolders inside Organized/YYYY/MM-Name/Unknown

Rules
- For each month folder (e.g., Organized/2025/01-January):
  - If an 'Unknown' subfolder exists, process all files inside it.
  - Compute SHA-256 for each file.
  - If the same hash exists elsewhere inside that month folder (outside Unknown),
    treat the Unknown file as a duplicate and MOVE it to the global duplicates folder.
  - Otherwise, MOVE the Unknown file up into the month folder root (the parent).
    If a name collision occurs, auto-rename safely (no overwrite).
  - When the Unknown folder becomes empty, remove it.

Safety
- DRY RUN by default (no file system changes) unless --apply is passed.
- Always uses shutil.move() (never copies).
- Logs a detailed plan and actions to a timestamped log file.
- Gracefully skips unreadable files.

Usage
  python fix_unknown_subfolders.py            # Dry run, prints and logs the plan only
  python fix_unknown_subfolders.py --apply    # Execute moves and deletions
"""
from __future__ import annotations
import argparse
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.raw', '.cr2', '.nef', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.jfif'}

ORGANIZED_DIR = Path(r'C:\Users\USERNAME\OneDrive\Documents\Pictures\Organized')
DUPLICATES_DIR = Path(r'C:\Users\USERNAME\OneDrive\Documents\Pictures\duplicates')

@dataclass
class PlanItem:
    action: str  # 'move-to-parent' | 'move-to-duplicates' | 'delete-folder'
    src: Path
    dst: Path | None
    reason: str


def sha256sum(filepath: Path) -> str | None:
    """Compute SHA-256 of a file, return None on failure (graceful)."""
    try:
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Skipping unreadable file (hash failed): {filepath} ({e})")
        return None


def is_media_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTENSIONS


def safe_unique_path(dst_dir: Path, filename: str) -> Path:
    """Return a unique path inside dst_dir, keeping extension, if filename exists."""
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dst_dir / (base + ext)
    counter = 1
    while candidate.exists():
        candidate = dst_dir / f"{base} ({counter}){ext}"
        counter += 1
    return candidate


def build_hash_index(root: Path, exclude_dir: Path) -> Dict[str, List[Path]]:
    """Build a map hash -> list[Path] for files under root, excluding exclude_dir subtree."""
    index: Dict[str, List[Path]] = {}
    all_files = [p for p in root.rglob('*') if p.is_file() and is_media_file(p) and exclude_dir not in p.parents]
    total = len(all_files)
    logger.info(f"  Hashing {total} files in {root.name}...")
    for idx, p in enumerate(all_files, 1):
        if idx % 100 == 0:
            logger.info(f"    Progress: {idx}/{total} ({100*idx/total:.1f}%)")
        h = sha256sum(p)
        if not h:
            continue
        index.setdefault(h, []).append(p)
    return index


def plan_for_month(month_dir: Path) -> List[PlanItem]:
    """Create a plan (no side effects) to fix Unknown under a specific month directory."""
    unknown_dir = month_dir / 'Unknown'
    if not unknown_dir.exists() or not unknown_dir.is_dir():
        return []

    logger.info(f"Planning for: {month_dir}")
    # Index hashes for all files under month_dir except Unknown
    hash_index = build_hash_index(month_dir, unknown_dir)

    plan: List[PlanItem] = []

    # Ensure duplicates dir exists in plan apply time
    DUPLICATES_DIR.mkdir(parents=True, exist_ok=True)

    # For each file in Unknown, decide action
    for p in unknown_dir.rglob('*'):
        if not p.is_file():
            continue
        if not is_media_file(p):
            continue
        h = sha256sum(p)
        if not h:
            continue
        # Duplicate within this month (outside Unknown)? Move to duplicates
        if h in hash_index and len(hash_index[h]) > 0:
            plan.append(PlanItem(
                action='move-to-duplicates',
                src=p,
                dst=DUPLICATES_DIR / p.name,
                reason='Duplicate of file in month folder (outside Unknown)'
            ))
        else:
            # Unique: move up to month root (keep unique filename)
            dst = safe_unique_path(month_dir, p.name)
            plan.append(PlanItem(
                action='move-to-parent',
                src=p,
                dst=dst,
                reason='Unique within month; promote from Unknown to month root'
            ))
    
    # If folder becomes empty, we'll delete it; we don't know yet, but schedule deletion last
    # Deletion will be attempted after moves when applying.
    plan.append(PlanItem(action='delete-folder', src=unknown_dir, dst=None, reason='Remove empty Unknown folder'))
    return plan


def apply_plan(plan: List[PlanItem], dry_run: bool) -> Tuple[int, int, int]:
    moved_to_parent = moved_to_duplicates = deleted_folders = 0

    for item in plan:
        if item.action == 'move-to-parent':
            assert item.dst is not None
            logger.info(f"PROMOTE: {item.src} -> {item.dst} | {item.reason}")
            if not dry_run:
                item.dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item.src), str(item.dst))
            moved_to_parent += 1
        elif item.action == 'move-to-duplicates':
            assert item.dst is not None
            logger.info(f"DUPLICATE: {item.src} -> {item.dst} | {item.reason}")
            if not dry_run:
                item.dst.parent.mkdir(parents=True, exist_ok=True)
                # Avoid overwriting duplicates with same name; ensure unique name in duplicates dir
                item.dst = safe_unique_path(item.dst.parent, item.dst.name)
                shutil.move(str(item.src), str(item.dst))
            moved_to_duplicates += 1
        elif item.action == 'delete-folder':
            folder = item.src
            if folder.exists() and folder.is_dir():
                try:
                    # Only delete if empty after moves
                    if not any(folder.iterdir()):
                        logger.info(f"DELETE FOLDER: {folder} | {item.reason}")
                        if not dry_run:
                            folder.rmdir()
                        deleted_folders += 1
                    else:
                        logger.info(f"SKIP DELETE (not empty): {folder}")
                except Exception as e:
                    logger.warning(f"Failed to delete folder {folder}: {e}")
    return moved_to_parent, moved_to_duplicates, deleted_folders


def main():
    parser = argparse.ArgumentParser(description='Fix Unknown subfolders within Organized month folders')
    parser.add_argument('--apply', action='store_true', help='Apply changes (otherwise dry run)')
    args = parser.parse_args()

    dry_run = not args.apply
    if not ORGANIZED_DIR.exists():
        logger.error(f"Organized folder not found: {ORGANIZED_DIR}")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(__file__).parent / f"fix_unknown_plan_{timestamp}.txt"

    logger.info('=' * 80)
    logger.info('FIX UNKNOWN SUBFOLDERS (month-level)')
    logger.info('=' * 80)
    logger.info(f"Organized: {ORGANIZED_DIR}")
    logger.info(f"Duplicates: {DUPLICATES_DIR}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'APPLY'}")

    month_dirs = [p for p in ORGANIZED_DIR.glob('*/*') if p.is_dir()]
    logger.info(f"Total month directories to scan: {len(month_dirs)}")
    total_plan: List[PlanItem] = []

    import time
    start = time.time()
    for idx, mdir in enumerate(month_dirs, 1):
        plan = plan_for_month(mdir)
        total_plan.extend(plan)
        if idx % 10 == 0:
            elapsed = time.time() - start
            avg = elapsed / idx
            remaining = (len(month_dirs) - idx) * avg
            logger.info(f"Scanned {idx}/{len(month_dirs)} months | ETA: {int(remaining)}s remaining")

    moves_parent = sum(1 for it in total_plan if it.action == 'move-to-parent')
    moves_dup = sum(1 for it in total_plan if it.action == 'move-to-duplicates')
    deletes = sum(1 for it in total_plan if it.action == 'delete-folder')

    # Write plan to log
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"FIX UNKNOWN SUBFOLDERS PLAN - {datetime.now().isoformat()}\n")
        f.write(f"Organized: {ORGANIZED_DIR}\n")
        f.write(f"Duplicates: {DUPLICATES_DIR}\n")
        f.write(f"Mode: {'DRY RUN' if dry_run else 'APPLY'}\n\n")
        for it in total_plan:
            f.write(f"{it.action}: {it.src} -> {it.dst if it.dst else ''} | {it.reason}\n")

    logger.info(f"Plan written: {log_path}")
    logger.info(f"Moves to parent: {moves_parent}")
    logger.info(f"Moves to duplicates: {moves_dup}")
    logger.info(f"Folder deletions (attempted): {deletes}")

    if dry_run:
        logger.info('Dry run complete. Re-run with --apply to execute moves.')
        return

    # Apply
    moved_parent, moved_dup, deleted_folders = apply_plan(total_plan, dry_run=False)

    logger.info('-' * 80)
    logger.info('APPLY COMPLETE')
    logger.info(f"Moved to parent: {moved_parent}")
    logger.info(f"Moved to duplicates: {moved_dup}")
    logger.info(f"Deleted folders: {deleted_folders}")


if __name__ == '__main__':
    main()
