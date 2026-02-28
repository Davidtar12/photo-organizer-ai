from __future__ import annotations

import logging
import sys
from pathlib import Path

# This script needs to be run from the 'face_browser' directory
# Add the project root to sys.path to allow for absolute imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.database import session_scope
from backend.models import FaceEmbedding, PersonCluster
from sqlalchemy import update

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reset_all_clusters():
    """
    Detach all faces from their clusters and delete all existing clusters.
    This prepares the database for a full re-clustering.
    """
    logger.info("Starting cluster reset process...")
    with session_scope() as session:
        # 1. Detach all faces from any cluster
        logger.info("Detaching all faces from their current clusters...")
        detached_count = session.execute(
            update(FaceEmbedding).values(cluster_id=None)
        ).rowcount
        logger.info(f"Detached {detached_count} face embeddings.")

        # 2. Delete all existing PersonCluster records
        logger.info("Deleting all existing person clusters...")
        deleted_clusters = session.query(PersonCluster).delete()
        logger.info(f"Deleted {deleted_clusters} clusters.")

        session.commit()
        logger.info("Cluster reset complete. The database is ready for re-clustering.")


if __name__ == "__main__":
    reset_all_clusters()
