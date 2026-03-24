"""Similar node schema for duplicate detection."""

from typing import Any, Dict
from pydantic import BaseModel, Field


class SimilarNode(BaseModel):
    """Result from similarity search for duplicate detection."""

    node_id: str
    content: str
    node_type: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any]
