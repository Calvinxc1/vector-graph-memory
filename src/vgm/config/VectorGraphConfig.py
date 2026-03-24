"""Vector and graph database configuration schema."""

from pydantic import BaseModel


class VectorGraphConfig(BaseModel):
    """Configuration for vector and graph database connections."""

    qdrant_collection: str = "vgm_memory"  # Qdrant collection name
    # JanusGraph doesn't need collection name, uses single graph with filtered queries
