"""Audit configuration schema."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class AuditConfig(BaseModel):
    """Configuration for audit log backend."""

    backend: Literal["jsonl", "mongodb"] = "jsonl"

    # JSONL-specific options
    log_dir: Optional[str] = Field(default=None)  # Default: ~/.vgm/logs
    rotation_size_mb: int = Field(default=10, gt=0)
    rotation_period: Literal["daily", "weekly", "monthly"] = "monthly"

    # MongoDB-specific options
    connection_string: Optional[str] = Field(default=None)  # e.g., "mongodb://localhost:27017"
    database: str = "vgm_audit"
    collection: str = "logs"
    ttl_days: Optional[int] = Field(default=None, gt=0)  # Auto-expire old logs
