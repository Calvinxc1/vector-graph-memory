"""Tests for the rule extraction runner and comparison seam."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vgm.rules import (
    CandidateRuleExtractionBundle,
    PydanticAIRuleExtractionPredictor,
    RawPydanticAIRuleExtractionPredictor,
    RuleExtractionRequest,
    RuleExtractionRunner,
    ScopedSourceDocument,
    build_request_from_reference_bundle,
    compare_rule_extractions,
    load_bundle_from_seed_records,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "rag_eval"
MANIFEST_PATH = FIXTURE_DIR / "seti_landing_orbiter_seed_v1_manifest.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_reference_bundle():
    manifest = _load_json(MANIFEST_PATH)
    node_records = _load_jsonl(FIXTURE_DIR / "seti_landing_orbiter_seed_v1_nodes.jsonl")
    edge_records = _load_jsonl(FIXTURE_DIR / "seti_landing_orbiter_seed_v1_edges.jsonl")
    return load_bundle_from_seed_records(manifest, node_records, edge_records)


def _build_request_from_reference():
    reference = _load_reference_bundle()
    return build_request_from_reference_bundle(reference)


def test_rule_extraction_runner_builds_prompt_and_validates_bundle():
    reference = _load_reference_bundle()
    request = _build_request_from_reference()
    calls: list[tuple[str, str]] = []

    def predictor(system_prompt: str, user_prompt: str):
        calls.append((system_prompt, user_prompt))
        return reference

    runner = RuleExtractionRunner(predictor=predictor)

    bundle = runner.extract(request)

    assert bundle.seed_id == "seti_landing_orbiter_seed_v1"
    assert len(calls) == 1
    system_prompt, user_prompt = calls[0]
    assert "Return JSON only." in system_prompt
    assert "Only use these edge types" in system_prompt
    assert "Expected canonical rule IDs" in system_prompt
    assert "Q5. Can I land with an orbiter?" in user_prompt
    assert "Can an orbiter later land on the same planet?" in user_prompt
    assert "rule_seti_orbit_action_base" in user_prompt
    assert "edge_src_core_orbit_supports_rule_orbit_action_base" in user_prompt


def test_rule_extraction_runner_rejects_edges_to_unknown_nodes():
    reference = _load_reference_bundle()
    request = _build_request_from_reference()
    broken = reference.model_copy(deep=True)
    broken.edges[0].to_node_id = "missing-node"

    runner = RuleExtractionRunner(predictor=lambda _system, _user: broken)

    with pytest.raises(ValueError, match="unknown to_node_id=missing-node"):
        runner.extract(request)


def test_rule_extraction_comparison_detects_exact_match_and_field_drift():
    reference = _load_reference_bundle()

    exact = compare_rule_extractions(reference, reference)
    assert exact.is_exact_match is True

    drifted = reference.model_copy(deep=True)
    drifted.canonical_rules[0].normalized_statement = "changed statement"

    comparison = compare_rule_extractions(reference, drifted)

    assert comparison.is_exact_match is False
    assert comparison.field_mismatches[0].record_kind == "canonical_rule"
    assert comparison.field_mismatches[0].field_name == "normalized_statement"


def test_rule_extraction_comparison_ignores_trailing_whitespace_only_drift():
    reference = _load_reference_bundle()
    drifted = reference.model_copy(deep=True)
    drifted.source_passages[0].source_text = f"{drifted.source_passages[0].source_text}\n"

    comparison = compare_rule_extractions(reference, drifted)

    assert comparison.is_exact_match is True
    assert comparison.field_mismatches == []


def test_pydantic_ai_rule_extraction_predictor_reads_structured_output():
    reference = _load_reference_bundle()

    class FakeRunResult:
        def __init__(self, output):
            self.output = output

    class FakeAgent:
        def __init__(self):
            self.calls = []

        def run_sync(self, user_prompt, **kwargs):
            self.calls.append((user_prompt, kwargs))
            return FakeRunResult(reference)

    fake_agent = FakeAgent()
    predictor = PydanticAIRuleExtractionPredictor(
        model="test:model",
        agent=fake_agent,
    )

    result = predictor("system prompt", "user prompt")

    assert result == reference
    assert fake_agent.calls == [
        (
            "user prompt",
            {
                "instructions": "system prompt",
                "output_type": CandidateRuleExtractionBundle,
            },
        )
    ]


def test_build_request_from_reference_includes_expected_rule_and_edge_ids():
    request = _build_request_from_reference()

    assert "rule_seti_orbit_action_base" in request.expected_canonical_rule_ids
    assert "rule_seti_moon_landing_requires_access" in request.expected_canonical_rule_ids
    assert (
        "edge_src_core_orbit_supports_rule_orbit_action_base"
        in request.expected_edge_ids
    )
    assert "edge_rule_moon_access_applies_during_landing" in request.expected_edge_ids


def test_raw_pydantic_ai_rule_extraction_predictor_reads_dict_output():
    class FakeRunResult:
        def __init__(self, output):
            self.output = output

    class FakeAgent:
        def __init__(self):
            self.calls = []

        def run_sync(self, user_prompt, **kwargs):
            self.calls.append((user_prompt, kwargs))
            return FakeRunResult({"seed_id": "seti"})

    fake_agent = FakeAgent()
    predictor = RawPydanticAIRuleExtractionPredictor(
        model="test:model",
        agent=fake_agent,
    )

    result = predictor("system prompt", "user prompt")

    assert result == {"seed_id": "seti"}
    assert len(fake_agent.calls) == 1
    user_prompt, kwargs = fake_agent.calls[0]
    assert user_prompt == "user prompt"
    assert kwargs["instructions"] == "system prompt"
    assert kwargs["output_type"] == dict[str, Any]
