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
from .eval_judge import (
    DspyRagEvalJudge,
    RagEvalJudge,
    RagEvalJudgeResult,
    build_evaluation_policy_key,
)
from .evaluation import (
    DEFAULT_EVAL_SOURCE_DIR,
    DEFAULT_EVAL_SUITE_PATH,
    LocalEvalSourceResolver,
    RagEvalCaseScore,
    RagEvalReport,
    RagEvalTraceEntry,
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
from .run_logging import DEFAULT_DSPY_RUN_LOG_DIR, DspyRunLogger, DspyRunSummary
from .synthesizer import DspyRagSynthesizer, build_dspy_lm, normalize_dspy_model_name

__all__ = [
    "ConversationTurn",
    "DspyRagSynthesizer",
    "DspyRagEvalJudge",
    "DspyArtifactManifest",
    "DspyArtifactStore",
    "DspyCompileManager",
    "DspyCompileOutcome",
    "DspyModelIdentity",
    "GraphFact",
    "DEFAULT_RAG_EVAL_WEIGHTS",
    "DEFAULT_EVAL_SOURCE_DIR",
    "DEFAULT_EVAL_SUITE_PATH",
    "DEFAULT_DSPY_RUN_LOG_DIR",
    "DEFAULT_RAG_ARTIFACT_DIR",
    "LocalEvalSourceResolver",
    "DspyRunLogger",
    "DspyRunSummary",
    "RagEvalCase",
    "RagEvalCaseScore",
    "RagEvalComponentScores",
    "RagEvalJudge",
    "RagEvalJudgeResult",
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
    "build_evaluation_policy_key",
    "build_dspy_lm",
    "compute_rag_eval_score",
    "load_rag_eval_cases",
    "normalize_dspy_model_name",
]
