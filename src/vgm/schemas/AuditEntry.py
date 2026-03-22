"""Audit entry schema for memory operations."""

from datetime import datetime
from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """Log entry for memory operations."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    project_id: str
    operation_type: Literal["add_node", "update_node", "add_edge", "merge_nodes"]
    summary: str  # Human-readable description of what happened
    commands: List[str]  # Database commands executed (content sanitized)
    metadata: Dict[str, Any]  # Metadata that was set (no content)
    affected_entities: List[str]  # Node/edge IDs involved
