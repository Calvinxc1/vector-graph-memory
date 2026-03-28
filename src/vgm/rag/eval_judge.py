"""LLM-judge support for offline RAG synthesis evaluation."""

from __future__ import annotations

import re
from typing import Any, Protocol

import dspy
from pydantic import BaseModel, Field

from .eval_dataset import RagEvalCase
from .models import RagContext, RagSynthesisResult


class RagEvalJudgeResult(BaseModel):
    """Soft scores produced by an LLM judge."""

    groundedness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    backend: str = "dspy-judge"


class RagEvalJudge(Protocol):
    """Judge protocol used by the hybrid evaluator."""

    def judge(
        self,
        *,
        case: RagEvalCase,
        context: RagContext,
        result: RagSynthesisResult,
    ) -> RagEvalJudgeResult:
        """Return groundedness and completeness scores for one eval case."""


class RagEvalJudgeSignature(dspy.Signature):
    """Score one RAG answer against evidence and rubric."""

    question: str = dspy.InputField(desc="The user question being answered.")
    conversation_history: list[str] = dspy.InputField(
        desc="Prior conversation turns. This may be empty."
    )
    passages: list[str] = dspy.InputField(
        desc="Retrieved authoritative passages that should ground the answer."
    )
    rubric: str = dspy.InputField(
        desc=(
            "Evaluation rubric including required facts, forbidden claims, abstention "
            "expectations, and preferred source."
        )
    )
    answer: str = dspy.InputField(desc="The model answer being evaluated.")
    cited_source_ids: list[str] = dspy.InputField(
        desc="Source IDs cited by the model answer."
    )
    abstain: bool = dspy.InputField(
        desc="Whether the model abstained instead of answering."
    )
    groundedness_score: int = dspy.OutputField(
        desc=(
            "Integer from 0 to 5. 5 means fully supported by the retrieved evidence. "
            "0 means unsupported or contradictory."
        )
    )
    completeness_score: int = dspy.OutputField(
        desc=(
            "Integer from 0 to 5. 5 means it covers the required rule details for the "
            "question. 0 means it misses the core answer."
        )
    )
    rationale: str = dspy.OutputField(
        desc=(
            "Short explanation for the scores. Mention missing required facts or "
            "unsupported claims if relevant."
        )
    )


class RagEvalJudgeProgram(dspy.Module):
    """DSPy module used for hybrid evaluation scoring."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(RagEvalJudgeSignature)

    def forward(
        self,
        *,
        question: str,
        conversation_history: list[str],
        passages: list[str],
        rubric: str,
        answer: str,
        cited_source_ids: list[str],
        abstain: bool,
    ) -> Any:
        return self.predictor(
            question=question,
            conversation_history=conversation_history,
            passages=passages,
            rubric=rubric,
            answer=answer,
            cited_source_ids=cited_source_ids,
            abstain=abstain,
        )


class DspyRagEvalJudge:
    """LLM-backed judge for softer groundedness and completeness scoring."""

    def __init__(self, predictor: Any, backend_name: str = "dspy-judge"):
        self.predictor = predictor
        self.backend_name = backend_name

    @classmethod
    def from_lm(cls, lm: Any) -> "DspyRagEvalJudge":
        program = RagEvalJudgeProgram()
        predictor = getattr(program, "predictor", None)
        if predictor is not None and hasattr(predictor, "set_lm"):
            predictor.set_lm(lm)
        return cls(predictor=program)

    def judge(
        self,
        *,
        case: RagEvalCase,
        context: RagContext,
        result: RagSynthesisResult,
    ) -> RagEvalJudgeResult:
        prediction = self.predictor(
            question=context.current_question,
            conversation_history=[
                f"{turn.role}: {turn.content.strip()}"
                for turn in context.conversation_history
                if turn.content.strip()
            ],
            passages=[
                (
                    f"[source_id={passage.node_id}] "
                    f"[node_type={passage.node_type}] "
                    f"{passage.content}"
                )
                for passage in context.retrieved_passages
            ],
            rubric=self._format_rubric(case),
            answer=result.answer,
            cited_source_ids=result.cited_source_ids,
            abstain=result.abstain,
        )

        return RagEvalJudgeResult(
            groundedness=_normalize_judge_score(
                getattr(prediction, "groundedness_score", 0)
            ),
            completeness=_normalize_judge_score(
                getattr(prediction, "completeness_score", 0)
            ),
            rationale=str(getattr(prediction, "rationale", "")).strip(),
            backend=self.backend_name,
        )

    @staticmethod
    def _format_rubric(case: RagEvalCase) -> str:
        lines = [
            f"expected_abstain: {case.rubric.expected_abstain}",
            f"preferred_source_id: {case.rubric.preferred_source_id or 'none'}",
            "must_include:",
        ]
        if case.rubric.must_include:
            lines.extend(f"- {statement}" for statement in case.rubric.must_include)
        else:
            lines.append("- none")

        lines.append("must_not_include:")
        if case.rubric.must_not_include:
            lines.extend(f"- {statement}" for statement in case.rubric.must_not_include)
        else:
            lines.append("- none")

        lines.append(
            f"abstention_reason: {case.rubric.abstention_reason or 'none provided'}"
        )
        lines.append(f"notes: {case.rubric.notes or 'none'}")
        return "\n".join(lines)


def _normalize_judge_score(score: Any) -> float:
    try:
        numeric_score = int(score)
    except (TypeError, ValueError):
        numeric_score = 0
    clamped = max(0, min(5, numeric_score))
    return clamped / 5.0


def build_evaluation_policy_key(
    scoring_mode: str,
    *,
    judge_model_name: str | None = None,
    judge_model_version: str | None = None,
) -> str:
    """Build a stable cache/eval identity key for the active scoring policy."""

    normalized_mode = scoring_mode.strip().lower() or "deterministic"
    if normalized_mode != "hybrid":
        return "deterministic"

    normalized_model = _policy_token(judge_model_name or "shared-answer-model")
    normalized_version = _policy_token(judge_model_version or "unversioned")
    return f"hybrid--judge-{normalized_model}--{normalized_version}"


def _policy_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
