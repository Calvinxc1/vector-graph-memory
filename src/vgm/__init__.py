"""Vector Graph Memory - Hybrid vector-graph database backend for AI agents."""

__version__ = "0.1.0"

# Main agent
from .MemoryAgent import MemoryAgent

# Storage layer
from .VectorGraphStore import VectorGraphStore
from .rag import (
    ConversationTurn,
    DEFAULT_DSPY_RUN_LOG_DIR,
    DEFAULT_EVAL_SOURCE_DIR,
    DEFAULT_EVAL_SUITE_PATH,
    DEFAULT_RAG_ARTIFACT_DIR,
    DEFAULT_RAG_EVAL_WEIGHTS,
    DspyArtifactManifest,
    DspyArtifactStore,
    DspyCompileManager,
    DspyCompileOutcome,
    DspyModelIdentity,
    DspyRagSynthesizer,
    DspyRunLogger,
    DspyRunSummary,
    GraphFact,
    LocalEvalSourceResolver,
    RagEvalCase,
    RagEvalCaseScore,
    RagEvalComponentScores,
    RagEvalReport,
    RagEvalTraceEntry,
    RagEvalRetrievalRef,
    RagEvalRubric,
    RagEvalWeights,
    RagContext,
    RagContextBuilder,
    RagSynthesisResult,
    RubricRagEvaluator,
    RetrievedPassage,
    build_dspy_lm,
    compute_rag_eval_score,
    load_rag_eval_cases,
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
    "DEFAULT_DSPY_RUN_LOG_DIR",
    "DEFAULT_EVAL_SOURCE_DIR",
    "DEFAULT_EVAL_SUITE_PATH",
    "DEFAULT_RAG_ARTIFACT_DIR",
    "DEFAULT_RAG_EVAL_WEIGHTS",
    "DspyArtifactManifest",
    "DspyArtifactStore",
    "DspyCompileManager",
    "DspyCompileOutcome",
    "DspyModelIdentity",
    "DspyRagSynthesizer",
    "DspyRunLogger",
    "DspyRunSummary",
    "GraphFact",
    "LocalEvalSourceResolver",
    "RagEvalCase",
    "RagEvalCaseScore",
    "RagEvalComponentScores",
    "RagEvalReport",
    "RagEvalTraceEntry",
    "RagEvalRetrievalRef",
    "RagEvalRubric",
    "RagEvalWeights",
    "RagContext",
    "RagContextBuilder",
    "RagSynthesisResult",
    "RubricRagEvaluator",
    "RetrievedPassage",
    "build_dspy_lm",
    "compute_rag_eval_score",
    "load_rag_eval_cases",
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
