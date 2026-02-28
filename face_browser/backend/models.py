from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MediaFile(Base):
    __tablename__ = "media_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False, index=True)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    orientation: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    pet_count: Mapped[int] = mapped_column(Integer, default=0)
    last_scanned_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_missing: Mapped[bool] = mapped_column(Boolean, default=False)
    objects_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    faces: Mapped[List["FaceEmbedding"]] = relationship(back_populates="media", cascade="all, delete-orphan")
    pets: Mapped[List["PetEmbedding"]] = relationship(back_populates="media", cascade="all, delete-orphan")


class PersonCluster(Base):
    __tablename__ = "person_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    primary_face_id: Mapped[Optional[int]] = mapped_column(ForeignKey("face_embeddings.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_hidden: Mapped[bool] = mapped_column(Boolean, default=False)

    faces: Mapped[List["FaceEmbedding"]] = relationship(
        "FaceEmbedding",
        back_populates="cluster",
        foreign_keys="[FaceEmbedding.cluster_id]"
    )
    primary_face: Mapped[Optional["FaceEmbedding"]] = relationship(
        "FaceEmbedding",
        foreign_keys=[primary_face_id],
        post_update=True
    )
    merge_events: Mapped[List["MergeEvent"]] = relationship(back_populates="target_cluster", foreign_keys="MergeEvent.target_cluster_id")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    __table_args__ = (
        UniqueConstraint("media_id", "embedding_index", name="uq_face_media_idx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    media_id: Mapped[int] = mapped_column(ForeignKey("media_files.id", ondelete="CASCADE"), nullable=False, index=True)
    cluster_id: Mapped[Optional[int]] = mapped_column(ForeignKey("person_clusters.id"), nullable=True, index=True)
    embedding_index: Mapped[int] = mapped_column(Integer, default=0)  # nth face in the media file
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, default=512)
    bbox_json: Mapped[str] = mapped_column(Text, nullable=False)
    detection_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_name: Mapped[str] = mapped_column(String(50), default="ArcFace")
    detector: Mapped[str] = mapped_column(String(50), default="retinaface")
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    media: Mapped[MediaFile] = relationship(back_populates="faces")
    cluster: Mapped[Optional[PersonCluster]] = relationship(
        "PersonCluster",
        back_populates="faces",
        foreign_keys=[cluster_id]
    )

    @property
    def bbox(self) -> Dict[str, Any]:
        return json.loads(self.bbox_json)

    @bbox.setter
    def bbox(self, value: Dict[str, Any]) -> None:
        self.bbox_json = json.dumps(value)


class MergeEvent(Base):
    __tablename__ = "merge_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_cluster_id: Mapped[int] = mapped_column(ForeignKey("person_clusters.id"), nullable=False)
    target_cluster_id: Mapped[int] = mapped_column(ForeignKey("person_clusters.id"), nullable=False)
    merged_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    initiated_by: Mapped[str] = mapped_column(String(64), default="manual")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    target_cluster: Mapped[PersonCluster] = relationship(back_populates="merge_events", foreign_keys=[target_cluster_id])


class DeleteLog(Base):
    __tablename__ = "delete_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    media_id: Mapped[int] = mapped_column(Integer, nullable=False)
    face_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    original_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    trash_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    deleted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


class TaskProgress(Base):
    __tablename__ = "task_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    total_items: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processed_items: Mapped[int] = mapped_column(Integer, default=0)
    last_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ClusterSuggestion(Base):
    """Active learning: stores uncertain face clustering decisions for human review."""
    __tablename__ = "cluster_suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    face_id: Mapped[int] = mapped_column(ForeignKey("face_embeddings.id", ondelete="CASCADE"), nullable=False, index=True)
    suggested_cluster_id: Mapped[int] = mapped_column(ForeignKey("person_clusters.id", ondelete="CASCADE"), nullable=False, index=True)
    distance: Mapped[float] = mapped_column(Float, nullable=False)  # Cosine distance to cluster
    reason: Mapped[str] = mapped_column(String(128), nullable=False)  # 'boundary', 'high_variance', etc.
    status: Mapped[str] = mapped_column(String(32), default='pending', index=True)  # 'pending', 'accepted', 'rejected', 'skipped'
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    face: Mapped["FaceEmbedding"] = relationship("FaceEmbedding", foreign_keys=[face_id])
    suggested_cluster: Mapped["PersonCluster"] = relationship("PersonCluster", foreign_keys=[suggested_cluster_id])


class PetCluster(Base):
    """Pet clusters for individual pet recognition."""
    __tablename__ = "pet_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    species: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # 'dog', 'cat', etc.
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    primary_pet_id: Mapped[Optional[int]] = mapped_column(ForeignKey("pet_embeddings.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_hidden: Mapped[bool] = mapped_column(Boolean, default=False)

    pets: Mapped[List["PetEmbedding"]] = relationship(
        "PetEmbedding",
        back_populates="cluster",
        foreign_keys="[PetEmbedding.cluster_id]"
    )
    primary_pet: Mapped[Optional["PetEmbedding"]] = relationship(
        "PetEmbedding",
        foreign_keys=[primary_pet_id],
        post_update=True
    )


class PetEmbedding(Base):
    """Pet embeddings for individual pet recognition (separate from face embeddings)."""
    __tablename__ = "pet_embeddings"
    __table_args__ = (
        UniqueConstraint("media_id", "embedding_index", name="uq_pet_media_idx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    media_id: Mapped[int] = mapped_column(ForeignKey("media_files.id", ondelete="CASCADE"), nullable=False, index=True)
    cluster_id: Mapped[Optional[int]] = mapped_column(ForeignKey("pet_clusters.id"), nullable=True, index=True)
    embedding_index: Mapped[int] = mapped_column(Integer, default=0)  # nth pet in the media file
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, default=512)
    bbox_json: Mapped[str] = mapped_column(Text, nullable=False)
    detection_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    species: Mapped[str] = mapped_column(String(50), default="unknown")  # 'dog', 'cat', etc. from YOLO
    model_name: Mapped[str] = mapped_column(String(50), default="pets.pkl")  # fastai model name
    detector: Mapped[str] = mapped_column(String(50), default="yolov8n")
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    media: Mapped[MediaFile] = relationship(back_populates="pets")
    cluster: Mapped[Optional[PetCluster]] = relationship(
        "PetCluster",
        back_populates="pets",
        foreign_keys=[cluster_id]
    )

    @property
    def bbox(self) -> Dict[str, Any]:
        return json.loads(self.bbox_json)

    @bbox.setter
    def bbox(self, value: Dict[str, Any]) -> None:
        self.bbox_json = json.dumps(value)

