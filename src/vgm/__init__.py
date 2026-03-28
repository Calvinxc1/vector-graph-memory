"""Vector Graph Memory - Hybrid vector-graph database backend for AI agents."""

__version__ = "0.1.0"

# Main agent
from .MemoryAgent import MemoryAgent

# Storage layer
from .VectorGraphStore import VectorGraphStore
from .rag import (
    ConversationTurn,
    DspyRagSynthesizer,
    GraphFact,
    RagContext,
    RagContextBuilder,
    RagSynthesisResult,
    RetrievedPassage,
    build_dspy_lm,
    normalize_dspy_model_name,
)

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
    "ConversationTurn",
    "DspyRagSynthesizer",
    "GraphFact",
    "RagContext",
    "RagContextBuilder",
    "RagSynthesisResult",
    "RetrievedPassage",
    "build_dspy_lm",
    "normalize_dspy_model_name",
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
