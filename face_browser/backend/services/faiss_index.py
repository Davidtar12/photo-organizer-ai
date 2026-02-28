from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sqlalchemy import select

from config import Config
from database import session_scope
from models import FaceEmbedding

logger = logging.getLogger(__name__)


class FaissIndex:
    """Build and query Faiss index for fast face similarity search."""

    def __init__(self, cfg: Config = Config()) -> None:
        self.cfg = cfg
        self.index_path = cfg.DATA_DIR / "faces.index"
        self.index: faiss.Index | None = None

    def build(self) -> None:
        """Build Faiss index from all embeddings in DB."""
        logger.info("Building Faiss index from database...")

        all_embeddings = []
        all_ids = []

        with session_scope() as session:
            query = session.query(FaceEmbedding.id, FaceEmbedding.embedding)
            total = session.query(FaceEmbedding).count()
            logger.info("Loading %s embeddings...", total)

            for idx, (face_id, emb_bytes) in enumerate(query, 1):
                embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                all_embeddings.append(embedding)
                all_ids.append(face_id)

                if idx % 5000 == 0:
                    logger.info("Loaded %s/%s embeddings (%.1f%%)", idx, total, (idx / total * 100))

        if not all_embeddings:
            logger.warning("No embeddings found. Index not built.")
            return

        embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
        dimension = embeddings_matrix.shape[1]

        logger.info("Normalizing %s vectors (dim=%s) for cosine similarity...", len(all_embeddings), dimension)
        faiss.normalize_L2(embeddings_matrix)

        # Use IndexFlatL2 for exact search (cosine via normalized L2)
        index = faiss.IndexFlatL2(dimension)
        index_with_ids = faiss.IndexIDMap(index)
        index_with_ids.add_with_ids(embeddings_matrix, np.array(all_ids, dtype=np.int64))

        self.cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index_with_ids, str(self.index_path))
        logger.info("Faiss index saved to %s", self.index_path)

        self.index = index_with_ids

    def load(self) -> bool:
        """Load existing Faiss index from disk."""
        if not self.index_path.exists():
            logger.warning("Faiss index not found at %s", self.index_path)
            return False

        try:
            self.index = faiss.read_index(str(self.index_path))
            logger.info("Loaded Faiss index from %s", self.index_path)
            return True
        except Exception as exc:
            logger.error("Failed to load Faiss index: %s", exc)
            return False

    def search(self, target_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Find k most similar faces using Faiss index.

        Args:
            target_embedding: 1D numpy array of face embedding
            k: Number of similar faces to return

        Returns:
            List of (face_id, similarity_score) tuples, sorted by similarity (highest first)
        """
        if self.index is None:
            if not self.load():
                logger.error("Faiss index not loaded. Cannot search.")
                return []

        target_vec = np.array([target_embedding], dtype=np.float32)
        faiss.normalize_L2(target_vec)

        distances, face_ids = self.index.search(target_vec, k)

        results = []
        for i in range(len(face_ids[0])):
            face_id = int(face_ids[0][i])
            distance = float(distances[0][i])

            # Convert L2 distance to cosine similarity: similarity = 1 - (distance^2 / 2)
            similarity = 1 - (distance ** 2 / 2)

            if face_id != -1:
                results.append((face_id, similarity))

        return results


if __name__ == "__main__":  # pragma: no cover
    index_builder = FaissIndex()
    index_builder.build()
