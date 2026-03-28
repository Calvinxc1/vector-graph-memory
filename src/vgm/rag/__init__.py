"""RAG context assembly exports."""

from .artifacts import (
    DEFAULT_RAG_ARTIFACT_DIR,
    DspyArtifactManifest,
    DspyArtifactStore,
    DspyCompileOutcome,
    DspyModelIdentity,
)
from .compile_manager import DspyCompileManager
from .context_builder import RagContextBuilder
from .eval_dataset import RagEvalCase, RagEvalRetrievalRef, RagEvalRubric, load_rag_eval_cases
from .evaluation import (
    DEFAULT_EVAL_SOURCE_DIR,
    DEFAULT_EVAL_SUITE_PATH,
    LocalEvalSourceResolver,
    RagEvalCaseScore,
    RagEvalReport,
    RubricRagEvaluator,
)
from .eval_scoring import (
    DEFAULT_RAG_EVAL_WEIGHTS,
    RagEvalComponentScores,
    RagEvalWeights,
    compute_rag_eval_score,
)
from .models import (
    ConversationTurn,
    GraphFact,
    RagContext,
    RagSynthesisResult,
    RetrievedPassage,
)
from .synthesizer import DspyRagSynthesizer, build_dspy_lm, normalize_dspy_model_name

__all__ = [
    "ConversationTurn",
    "DspyRagSynthesizer",
    "DspyArtifactManifest",
    "DspyArtifactStore",
    "DspyCompileManager",
    "DspyCompileOutcome",
    "DspyModelIdentity",
    "GraphFact",
    "DEFAULT_RAG_EVAL_WEIGHTS",
    "DEFAULT_EVAL_SOURCE_DIR",
    "DEFAULT_EVAL_SUITE_PATH",
    "DEFAULT_RAG_ARTIFACT_DIR",
    "LocalEvalSourceResolver",
    "RagEvalCase",
    "RagEvalCaseScore",
    "RagEvalComponentScores",
    "RagEvalReport",
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
]
