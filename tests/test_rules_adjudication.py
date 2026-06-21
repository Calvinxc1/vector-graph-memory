"""Tests for raw LLM rules chat."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ThinkingPart

from vgm.rules import (
    LlmRulesAdjudicator,
    LivePilotRulingInspection,
    PrecedenceEntry,
    RetrievedEvidenceNode,
    RetrievedRulingEvidence,
    RuleCitation,
    RuleReference,
    RulesAdjudicationDraft,
    RulesRulingRequest,
    RulesRulingResult,
    verify_adjudication_draft,
)


class _FakeStructuredOutput(BaseModel):
    question: str
    question_id: str
    seed_id: str
    subsystem: str
    ruling: str
    abstain: bool = False
    backend: str = "test"


class _FakeRunResult:
    def __init__(self, output, messages=None):
        self.output = output
        self.messages = messages or []

    def all_messages(self):
        return self.messages


class _FakeAgent:
    def __init__(self, output, messages=None):
        self.output = output
        self.messages = messages or []
        self.prompts = []

    def run_sync(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return _FakeRunResult(self.output, self.messages)


def _inspection() -> LivePilotRulingInspection:
    return LivePilotRulingInspection(
        question="Does an opponent's orbiter still reduce my landing cost?",
        normalized_question="opponent orbiter discount",
        evidence=RetrievedRulingEvidence(
            question="Does an opponent's orbiter still reduce my landing cost?",
            seed_id="seti_landing_orbiter_seed_v1",
            subsystem="landing_and_orbiter_interactions",
            retrieved_node_ids=[
                "rule_seti_landing_discount_if_orbiter_present",
                "src_seti_core_land_on_planet_or_moon",
            ],
            expanded_node_ids=[
                "rule_seti_landing_discount_if_orbiter_present",
                "src_seti_core_land_on_planet_or_moon",
            ],
            nodes=[
                RetrievedEvidenceNode(
                    storage_node_id="rule_seti_landing_discount_if_orbiter_present",
                    logical_node_id="rule_seti_landing_discount_if_orbiter_present",
                    node_kind="canonical_rule",
                    title="Landing discount with existing orbiter",
                    content="If an orbiter is already at the planet, landing there costs 1 less energy.",
                    rule_kind="core_rule",
                    normalized_statement="If an orbiter is already at the planet, landing there costs 1 less energy.",
                ),
                RetrievedEvidenceNode(
                    storage_node_id="src_seti_core_land_on_planet_or_moon",
                    logical_node_id="src_seti_core_land_on_planet_or_moon",
                    node_kind="source_passage",
                    title="Land on Planet or Moon",
                    content="Pay 3 energy to turn one of your probes into a lander. If an orbiter is already at the planet, landing there costs 1 less energy.",
                    citation_label="SETI Core Rules p.11",
                    citation_short="Core Rules p.11",
                    locator="p.11",
                    authority_scope="core",
                ),
            ],
        ),
        seed_inference={
            "normalized_question": "opponent orbiter discount",
            "selected_seed_id": "seti_landing_orbiter_seed_v1",
            "selected_score": 0.9,
            "candidates": [],
        },
        selected_seed_id="seti_landing_orbiter_seed_v1",
    )


def _llm_ruling_result() -> RulesRulingResult:
    return RulesRulingResult(
        question="model supplied question",
        question_id="llm-question-id",
        seed_id="model-seed",
        subsystem="model-subsystem",
        ruling="Yes. Apply the landing discount.",
        primary_rule=RuleReference(
            rule_node_id="rule_seti_landing_discount_if_orbiter_present",
            title="Landing discount with existing orbiter",
            rule_kind="core_rule",
            normalized_statement="If an orbiter is already at the planet, landing there costs 1 less energy.",
        ),
        primary_citation=RuleCitation(
            source_node_id="src_seti_core_land_on_planet_or_moon",
            citation_label="SETI Core Rules p.11",
            citation_short="Core Rules p.11",
            title="Land on Planet or Moon",
            locator="p.11",
            authority_scope="core",
            source_excerpt="Pay 3 energy to turn one of your probes into a lander.",
        ),
        precedence_order=[
            PrecedenceEntry(
                order=1,
                summary="Apply the core landing discount rule.",
                rule_node_id="rule_seti_landing_discount_if_orbiter_present",
                source_node_id="src_seti_core_land_on_planet_or_moon",
                precedence_kind="primary",
            )
        ],
        backend="model-backend",
    )


def test_adjudicator_returns_rules_ruling_result_schema():
    fake_agent = _FakeAgent(_llm_ruling_result())
    adjudicator = LlmRulesAdjudicator("test-model", agent=fake_agent)

    outcome = adjudicator.answer(
        RulesRulingRequest(question="Does an opponent's orbiter still reduce my landing cost?"),
        _inspection(),
    )

    assert outcome.result.ruling == "Yes. Apply the landing discount."
    assert outcome.result.abstain is False
    assert outcome.result.backend == "llm-schema-chat"
    assert outcome.llm_result is not None
    assert outcome.llm_result.backend == "model-backend"
    assert outcome.result.primary_rule is not None
    assert outcome.result.primary_rule.rule_node_id == "rule_seti_landing_discount_if_orbiter_present"
    assert outcome.result.primary_citation is not None
    assert outcome.result.primary_citation.source_node_id == "src_seti_core_land_on_planet_or_moon"
    assert outcome.result.precedence_order[0].precedence_kind == "primary"
    assert outcome.model_thinking == []
    assert "RulesRulingResult" in fake_agent.prompts[0]
    assert "rule_seti_landing_discount_if_orbiter_present" in fake_agent.prompts[0]


def test_adjudicator_preserves_model_thinking_for_schema_response():
    fake_agent = _FakeAgent(
        _llm_ruling_result(),
        messages=[
            ModelResponse(
                parts=[
                    ThinkingPart(content="Used the retrieved landing discount evidence."),
                ],
            ),
        ],
    )
    adjudicator = LlmRulesAdjudicator("test-model", agent=fake_agent)

    outcome = adjudicator.answer(
        RulesRulingRequest(question="Does an opponent's orbiter still reduce my landing cost?"),
        _inspection(),
    )

    assert outcome.model_thinking == ["Used the retrieved landing discount evidence."]


def test_adjudicator_accepts_model_dump_compatible_ruling_output():
    fake_agent = _FakeAgent(
        _FakeStructuredOutput(
            question="model question",
            question_id="model-question",
            seed_id="model-seed",
            subsystem="model-subsystem",
            ruling="Structured fallback.",
        )
    )
    adjudicator = LlmRulesAdjudicator("test-model", agent=fake_agent)

    outcome = adjudicator.answer(
        RulesRulingRequest(question="Does an opponent's orbiter still reduce my landing cost?"),
        _inspection(),
    )

    assert outcome.result.ruling == "Structured fallback."
    assert outcome.result.backend == "llm-schema-chat"


def test_legacy_adjudication_path_is_present_but_disabled_by_default():
    fake_agent = _FakeAgent(_llm_ruling_result())
    adjudicator = LlmRulesAdjudicator("test-model", agent=fake_agent)

    outcome = adjudicator.answer(
        RulesRulingRequest(question="Does an opponent's orbiter still reduce my landing cost?"),
        _inspection(),
    )

    assert outcome.adjudication_enabled is False
    assert outcome.result.backend == "llm-schema-chat"


def test_legacy_adjudication_verifier_remains_available():
    result, errors = verify_adjudication_draft(
        request=RulesRulingRequest(question="Does an opponent's orbiter still reduce my landing cost?"),
        inspection=_inspection(),
        draft=RulesAdjudicationDraft(ruling="Yes."),
    )

    assert result.abstain is True
    assert result.backend == "llm-live-pilot-disabled"
    assert errors == [
        "primary_rule_id is required when abstain is false",
        "primary_source_id is required when abstain is false",
    ]
