"""Memory configuration schema."""

from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """Configuration for memory behavior and thresholds."""

    use_case_description: str  # Natural language: "Track job search activities..."
    memory_threshold_description: str  # Natural language: "Store specific job postings..."
    project_id: str  # Unique identifier for this memory project
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)  # Duplicate detection threshold
