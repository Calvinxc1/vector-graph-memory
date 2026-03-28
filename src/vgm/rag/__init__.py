"""RAG context assembly exports."""

from .context_builder import RagContextBuilder
from .eval_dataset import RagEvalCase, RagEvalRetrievalRef, RagEvalRubric, load_rag_eval_cases
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
    "GraphFact",
    "DEFAULT_RAG_EVAL_WEIGHTS",
    "RagEvalCase",
    "RagEvalComponentScores",
    "RagEvalRetrievalRef",
    "RagEvalRubric",
    "RagEvalWeights",
    "RagContext",
    "RagContextBuilder",
    "RagSynthesisResult",
    "RetrievedPassage",
    "build_dspy_lm",
    "compute_rag_eval_score",
    "load_rag_eval_cases",
    "normalize_dspy_model_name",
]
