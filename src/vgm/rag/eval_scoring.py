"""Scoring primitives for offline RAG synthesis evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class RagEvalComponentScores(BaseModel):
    """Normalized component scores for one eval example."""

    groundedness: float = Field(ge=0.0, le=1.0)
    abstention: float = Field(ge=0.0, le=1.0)
    source_alignment: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)


class RagEvalWeights(BaseModel):
    """Weighting policy for the v1 single-score metric."""

    groundedness: float = 0.50
    abstention: float = 0.25
    source_alignment: float = 0.15
    completeness: float = 0.10

    @model_validator(mode="after")
    def validate_total_weight(self) -> "RagEvalWeights":
        """Keep the scoring policy normalized."""

        total = (
            self.groundedness
            + self.abstention
            + self.source_alignment
            + self.completeness
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("RagEvalWeights must sum to 1.0")
        return self


DEFAULT_RAG_EVAL_WEIGHTS = RagEvalWeights()


def compute_rag_eval_score(
    components: RagEvalComponentScores,
    weights: RagEvalWeights = DEFAULT_RAG_EVAL_WEIGHTS,
) -> float:
    """Collapse component scores into the v1 single-score metric."""

    return (
        components.groundedness * weights.groundedness
        + components.abstention * weights.abstention
        + components.source_alignment * weights.source_alignment
        + components.completeness * weights.completeness
    )
