"""Structured eval dataset models for RAG synthesis benchmarking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .models import ConversationTurn

DocumentType = Literal["core_rules", "faq", "player_aid", "alien_addendum"]
RetrievalPriority = Literal["primary", "supporting"]
EvalMode = Literal["rules_reference"]


class RagEvalRetrievalRef(BaseModel):
    """A frozen retrieval reference anchored to a local source document page."""

    source_id: str
    document_id: str
    document_type: DocumentType
    authority_scope: str
    page: int = Field(ge=1)
    locator: str = Field(min_length=1)
    priority: RetrievalPriority = "primary"


class RagEvalRubric(BaseModel):
    """Scoring expectations for one synthesis example."""

    expected_abstain: bool = False
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    preferred_source_id: str | None = None
    abstention_reason: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def validate_abstention_contract(self) -> "RagEvalRubric":
        """Keep abstention and source expectations internally consistent."""

        if self.expected_abstain:
            if self.preferred_source_id is not None:
                raise ValueError("Abstaining cases cannot require a preferred source")
            if not self.abstention_reason:
                raise ValueError("Abstaining cases must include an abstention_reason")
        else:
            if not self.preferred_source_id:
                raise ValueError("Non-abstaining cases must include a preferred_source_id")
            if self.abstention_reason:
                raise ValueError("Non-abstaining cases cannot set an abstention_reason")
        return self


class RagEvalCase(BaseModel):
    """One offline eval example for the DSPy synthesis layer."""

    case_id: str
    suite_id: str
    game_id: str
    mode: EvalMode = "rules_reference"
    tags: list[str] = Field(default_factory=list)
    conversation: list[ConversationTurn]
    retrieval_refs: list[RagEvalRetrievalRef]
    rubric: RagEvalRubric

    @model_validator(mode="after")
    def validate_case_contract(self) -> "RagEvalCase":
        """Ensure the case can be used deterministically by later eval runners."""

        if not self.conversation:
            raise ValueError("Eval cases require at least one conversation turn")
        if self.conversation[-1].role != "user":
            raise ValueError("The final conversation turn must be from the user")
        if not self.retrieval_refs:
            raise ValueError("Eval cases require at least one retrieval reference")

        retrieval_source_ids = {ref.source_id for ref in self.retrieval_refs}
        if len(retrieval_source_ids) != len(self.retrieval_refs):
            raise ValueError("Retrieval reference source_ids must be unique per case")

        preferred_source_id = self.rubric.preferred_source_id
        if preferred_source_id and preferred_source_id not in retrieval_source_ids:
            raise ValueError(
                "rubric.preferred_source_id must reference one of the case retrieval_refs"
            )

        return self


def load_rag_eval_cases(path: str | Path) -> list[RagEvalCase]:
    """Load newline-delimited JSON eval cases from disk."""

    fixture_path = Path(path)
    cases: list[RagEvalCase] = []

    for raw_line in fixture_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(RagEvalCase.model_validate(json.loads(line)))

    return cases
