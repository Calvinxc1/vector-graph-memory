"""LLM-backed rules chat over retrieved graph evidence."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from .rulings import (
    LivePilotRulingInspection,
    RetrievedEvidenceNode,
    RulesRulingRequest,
    RulesRulingResult,
)


class RulesAdjudicationPrecedenceDraft(BaseModel):
    """Disabled legacy adjudication precedence draft kept for compatibility."""

    summary: str
    precedence_kind: str
    rule_node_id: str | None = None
    source_node_id: str | None = None


class RulesAdjudicationDraft(BaseModel):
    """Disabled legacy adjudication draft kept for compatibility."""

    question_id: str = "llm_adjudicated_question"
    seed_id: str | None = None
    subsystem: str | None = None
    ruling: str = ""
    primary_rule_id: str | None = None
    primary_source_id: str | None = None
    modifying_rule_ids: list[str] = Field(default_factory=list)
    modifying_source_ids: list[str] = Field(default_factory=list)
    supporting_rule_ids: list[str] = Field(default_factory=list)
    supporting_source_ids: list[str] = Field(default_factory=list)
    precedence_order: list[RulesAdjudicationPrecedenceDraft] = Field(default_factory=list)
    uncertainty: str | None = None
    abstain: bool = False


class RulesAdjudicationOutcome(BaseModel):
    """LLM rules response plus optional model thinking."""

    result: RulesRulingResult
    llm_result: RulesRulingResult | None = None
    adjudication_enabled: bool = False
    model_thinking: list[str] = Field(default_factory=list)
    verification_errors: list[str] = Field(default_factory=list)


class LlmRulesAdjudicator:
    """Answer a rules question with a typed chat-agent ruling over retrieved evidence."""

    def __init__(
        self,
        model: str | Model,
        *,
        agent: Agent | None = None,
        name: str = "rules_chat_agent",
        enable_adjudication: bool = False,
    ) -> None:
        self.model = model
        self.enable_adjudication = enable_adjudication
        self.agent = agent or Agent(
            model,
            name=name,
            system_prompt=_RULES_CHAT_SYSTEM_PROMPT,
        )

    def answer(
        self,
        request: RulesRulingRequest,
        inspection: LivePilotRulingInspection,
    ) -> RulesAdjudicationOutcome:
        """Return a ruling-shaped chat-agent response."""

        prompt = _build_chat_prompt(request, inspection)
        if self.enable_adjudication:
            run_result = self.agent.run_sync(prompt, output_type=RulesAdjudicationDraft)
            draft = run_result.output
            result, verification_errors = verify_adjudication_draft(
                request=request,
                inspection=inspection,
                draft=draft,
            )
            return RulesAdjudicationOutcome(
                result=result,
                adjudication_enabled=True,
                model_thinking=_extract_thinking_parts(run_result.all_messages()),
                verification_errors=verification_errors,
            )

        run_result = self.agent.run_sync(prompt, output_type=RulesRulingResult)
        llm_result = _coerce_ruling_result(run_result.output)
        evidence = inspection.evidence
        result = llm_result.model_copy(
            update={
                "question": request.question,
                "question_id": llm_result.question_id
                or (
                    inspection.selected_case.question_id
                    if inspection.selected_case is not None
                    else "llm_schema_ruling"
                ),
                "seed_id": llm_result.seed_id
                or inspection.selected_seed_id
                or evidence.seed_id
                or request.seed_id
                or "unknown",
                "subsystem": llm_result.subsystem or evidence.subsystem or "unknown",
                "backend": "llm-schema-chat",
            },
            deep=True,
        )
        return RulesAdjudicationOutcome(
            result=result,
            llm_result=llm_result,
            model_thinking=_extract_thinking_parts(run_result.all_messages()),
        )


def verify_adjudication_draft(
    *,
    request: RulesRulingRequest,
    inspection: LivePilotRulingInspection,
    draft: RulesAdjudicationDraft,
) -> tuple[RulesRulingResult, list[str]]:
    """Disabled legacy adjudication normalization kept behind enable_adjudication."""

    evidence = inspection.evidence
    errors = []
    if not draft.abstain:
        if not draft.ruling.strip():
            errors.append("ruling is required when abstain is false")
        if draft.primary_rule_id is None:
            errors.append("primary_rule_id is required when abstain is false")
        if draft.primary_source_id is None:
            errors.append("primary_source_id is required when abstain is false")
    result = RulesRulingResult(
        question=request.question,
        question_id=draft.question_id,
        seed_id=draft.seed_id or inspection.selected_seed_id or evidence.seed_id or request.seed_id or "unknown",
        subsystem=draft.subsystem or evidence.subsystem or "unknown",
        ruling=draft.ruling.strip() or "I cannot assemble a sufficiently supported live ruling for that question.",
        uncertainty=draft.uncertainty,
        abstain=bool(draft.abstain or errors),
        backend="llm-live-pilot-disabled",
    )
    return result, errors


def _coerce_ruling_result(output: Any) -> RulesRulingResult:
    if isinstance(output, RulesRulingResult):
        return output
    if isinstance(output, dict):
        return RulesRulingResult.model_validate(output)
    if hasattr(output, "model_dump"):
        return RulesRulingResult.model_validate(output.model_dump(mode="json"))
    return RulesRulingResult.model_validate_json(str(output))


def _build_chat_prompt(
    request: RulesRulingRequest,
    inspection: LivePilotRulingInspection,
) -> str:
    payload = {
        "task": "Return one RulesRulingResult JSON object for the SETI board-game rules question using the supplied graph evidence.",
        "game_context": (
            "SETI is the board game. Interpret terms such as orbiter, landing, "
            "probe, scan, and free action only as board-game rules terms, not as astronomy or physics."
        ),
        "question": request.question,
        "normalized_question": inspection.normalized_question,
        "premise_screen": inspection.premise_screen.model_dump(mode="json"),
        "issue_inference": inspection.issue_inference.model_dump(mode="json"),
        "selected_seed_id": inspection.selected_seed_id,
        "retrieved_node_ids": inspection.evidence.retrieved_node_ids,
        "expanded_node_ids": inspection.evidence.expanded_node_ids,
        "evidence_nodes": [
            _evidence_node_payload(node)
            for node in inspection.evidence.nodes
        ],
        "evidence_edges": [
            edge.model_dump(mode="json")
            for edge in inspection.evidence.edges
        ],
        "required_result_defaults": {
            "question": request.question,
            "question_id": (
                inspection.selected_case.question_id
                if inspection.selected_case is not None
                else "llm_schema_ruling"
            ),
            "seed_id": inspection.selected_seed_id or inspection.evidence.seed_id or request.seed_id or "unknown",
            "subsystem": inspection.evidence.subsystem or "unknown",
            "backend": "llm-schema-chat",
        },
    }
    return (
        "Use only this SETI board-game evidence payload. "
        "Return the same structured schema as the deterministic pilot response: RulesRulingResult. "
        "Populate question, question_id, seed_id, subsystem, ruling, primary_rule, primary_citation, "
        "modifying_rules, modifying_citations, supporting_rules, supporting_citations, "
        "precedence_order, uncertainty, abstain, and backend. Use null for absent optional objects "
        "and empty arrays for absent lists. Copy node IDs and citation labels exactly from evidence_nodes. "
        "Return a direct ruling in the ruling field, not a rules analysis essay. "
        "Start the ruling with Yes, No, or a short conditional answer when possible. "
        "If the evidence is insufficient, set abstain=true and explain the limitation in uncertainty.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=True)
    )


def _evidence_node_payload(node: RetrievedEvidenceNode) -> dict[str, Any]:
    return {
        "logical_node_id": node.logical_node_id,
        "node_kind": node.node_kind,
        "title": node.title,
        "content": node.content,
        "rule_kind": node.rule_kind,
        "normalized_statement": node.normalized_statement,
        "citation_label": node.citation_label,
        "citation_short": node.citation_short,
        "locator": node.locator,
        "authority_scope": node.authority_scope,
        "seed_id": node.seed_id,
        "subsystem": node.subsystem,
        "similarity_score": node.similarity_score,
    }


def _extract_thinking_parts(messages: list[Any]) -> list[str]:
    thinking: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", []):
            if type(part).__name__ != "ThinkingPart":
                continue
            content = getattr(part, "content", "")
            if isinstance(content, str) and content.strip():
                thinking.append(content.strip())
    return thinking


_RULES_CHAT_SYSTEM_PROMPT = """You issue concise SETI board-game rules rulings using only the supplied evidence graph.

SETI is a board game. Do not answer from astronomy, physics, general knowledge,
or outside game assumptions. Terms such as orbiter, landing, probe, scan, and
free action are game terms.

Return the requested RulesRulingResult schema. The ruling field must be the
answer, not an analysis plan. Start with the practical answer whenever possible.
Keep citations tied to supplied evidence labels, but do not fabricate rules,
cards, game state, citations, or source IDs.
"""
