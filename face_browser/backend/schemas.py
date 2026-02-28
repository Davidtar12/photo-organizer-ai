from __future__ import annotations

from marshmallow import Schema, fields


class BoundingBoxField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):  # pragma: no cover - simple pass-through
        return value


class MediaFileSchema(Schema):
    id = fields.Int()
    path = fields.Str()
    sha256 = fields.Str(allow_none=True)
    face_count = fields.Int()
    is_missing = fields.Bool()
    updated_at = fields.DateTime()


class FaceEmbeddingSchema(Schema):
    id = fields.Int()
    media = fields.Nested(MediaFileSchema)
    cluster_id = fields.Int(allow_none=True)
    bbox = fields.Dict()
    detection_confidence = fields.Float(allow_none=True)
    thumbnail_path = fields.Str(allow_none=True)


class PersonClusterSchema(Schema):
    id = fields.Int()
    display_name = fields.Str(allow_none=True)
    description = fields.Str(allow_none=True)
    primary_face_id = fields.Int(allow_none=True)
    face_count = fields.Int()
    created_at = fields.DateTime()
    updated_at = fields.DateTime()


class PetEmbeddingSchema(Schema):
    id = fields.Int()
    media = fields.Nested(MediaFileSchema)
    cluster_id = fields.Int(allow_none=True)
    bbox = fields.Dict()
    detection_confidence = fields.Float(allow_none=True)
    species = fields.Str()
    thumbnail_path = fields.Str(allow_none=True)


class PetClusterSchema(Schema):
    id = fields.Int()
    display_name = fields.Str(allow_none=True)
    species = fields.Str(allow_none=True)
    description = fields.Str(allow_none=True)
    primary_pet_id = fields.Int(allow_none=True)
    pet_count = fields.Int()
    created_at = fields.DateTime()
    updated_at = fields.DateTime()


class TaskProgressSchema(Schema):
    task_name = fields.Str()
    total_items = fields.Int(allow_none=True)
    processed_items = fields.Int()
    last_message = fields.Str(allow_none=True)
    started_at = fields.DateTime()
    updated_at = fields.DateTime()
