"""Node metadata schema for graph nodes."""

from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel, Field
from uuid import uuid4


class NodeMetadata(BaseModel):
    """Metadata for graph nodes stored in both vector and graph databases."""

    node_id: str = Field(default_factory=lambda: str(uuid4()))
    node_type: str  # e.g., "job", "company", "person", "interaction"
    content: str  # The actual text content to be embedded
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    source: str  # session_id that created this node
    project_id: str  # which project/memory space this belongs to
    embedding_model: str  # e.g., "text-embedding-3-small"
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)  # User-defined fields
