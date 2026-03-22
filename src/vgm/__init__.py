"""Vector Graph Memory - Hybrid vector-graph database backend for AI agents."""

__version__ = "0.1.0"

# Main agent
from .MemoryAgent import MemoryAgent

# Storage layer
from .VectorGraphStore import VectorGraphStore

# Schemas
from .schemas import (
    NodeMetadata,
    EdgeMetadata,
    AuditEntry,
    SimilarNode,
)

# Configuration
from .config import (
    MemoryConfig,
    MemoryTriggerConfig,
    AuditConfig,
    VectorGraphConfig,
)

# Audit backends
from .audit import (
    AuditBackend,
    JSONLAuditBackend,
    MongoAuditBackend,
)

__all__ = [
    # Version
    "__version__",
    # Main API
    "MemoryAgent",
    "VectorGraphStore",
    # Schemas
    "NodeMetadata",
    "EdgeMetadata",
    "AuditEntry",
    "SimilarNode",
    # Configuration
    "MemoryConfig",
    "MemoryTriggerConfig",
    "AuditConfig",
    "VectorGraphConfig",
    # Audit backends
    "AuditBackend",
    "JSONLAuditBackend",
    "MongoAuditBackend",
]
