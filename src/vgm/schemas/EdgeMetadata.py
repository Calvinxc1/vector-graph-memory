"""Edge metadata schema for graph relationships."""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from uuid import uuid4


class EdgeMetadata(BaseModel):
    """Metadata for graph edges/relationships."""

    edge_id: str = Field(default_factory=lambda: str(uuid4()))
    from_node_id: str
    to_node_id: str
    relationship_type: str  # e.g., "works_at", "located_in", "applied_to"
    description: str = ""  # Freeform context about the relationship
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source: str  # session_id that created this edge
    project_id: str
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )  # Agent confidence in relationship
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
