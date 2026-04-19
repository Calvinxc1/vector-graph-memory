"""Tests for DSPy-backed rules extraction scaffolding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from vgm.rules import (
    DspyRuleExtractor,
    RuleExtractionEvaluator,
    build_request_from_reference_bundle,
    load_bundle_from_seed_records,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "rag_eval"
MANIFEST_PATH = FIXTURE_DIR / "seti_landing_orbiter_seed_v1_manifest.json"
SECOND_MANIFEST_PATH = FIXTURE_DIR / "seti_free_action_authority_seed_v1_manifest.json"
SUITE_PATH = FIXTURE_DIR / "seti_rules_extraction_v1.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_reference_bundle():
    manifest = _load_json(MANIFEST_PATH)
    node_records = _load_jsonl(FIXTURE_DIR / "seti_landing_orbiter_seed_v1_nodes.jsonl")
    edge_records = _load_jsonl(FIXTURE_DIR / "seti_landing_orbiter_seed_v1_edges.jsonl")
    return load_bundle_from_seed_records(manifest, node_records, edge_records)


def test_rule_extraction_evaluator_scores_reference_bundle_as_exact_match():
    reference = _load_reference_bundle()
    score = RuleExtractionEvaluator.score_bundle(reference, reference)

    assert score.total_score == 1.0
    assert score.source_passage_recall == 1.0
    assert score.canonical_rule_recall == 1.0
    assert score.edge_recall == 1.0
    assert score.comparison.is_exact_match is True


def test_dspy_rule_extractor_parses_prediction_json_into_bundle():
    reference = _load_reference_bundle()
    request = build_request_from_reference_bundle(reference)

    class FakePredictor:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                source_passages_json=json.dumps(
                    [item.model_dump(mode="json") for item in reference.source_passages]
                ),
                canonical_rules_json=json.dumps(
                    [item.model_dump(mode="json") for item in reference.canonical_rules]
                ),
                edges_json=json.dumps(
                    [item.model_dump(mode="json") for item in reference.edges]
                ),
                uncertainties_json="[]",
            )

    predictor = FakePredictor()
    extractor = DspyRuleExtractor(predictor)
    bundle = extractor.extract(request)

    assert bundle.source_passages == reference.source_passages
    assert bundle.canonical_rules == reference.canonical_rules
    assert bundle.edges == reference.edges
    assert predictor.calls[0]["seed_id"] == "seti_landing_orbiter_seed_v1"
    assert "rule_seti_orbit_action_base" in predictor.calls[0]["expected_canonical_rule_ids"]
    assert (
        "edge_src_core_orbit_supports_rule_orbit_action_base"
        in predictor.calls[0]["expected_edge_ids"]
    )


def test_rule_extraction_evaluator_from_seed_manifest_builds_examples():
    evaluator = RuleExtractionEvaluator.from_seed_manifest(MANIFEST_PATH)

    assert len(evaluator.prepared_cases) == 1
    assert len(evaluator.trainset) == 1
    assert evaluator.trainset[0].seed_id == "seti_landing_orbiter_seed_v1"
    assert "rule_seti_orbit_action_base" in evaluator.trainset[0].expected_canonical_rule_ids
    assert (
        "edge_src_core_orbit_supports_rule_orbit_action_base"
        in evaluator.trainset[0].expected_edge_ids
    )


def test_rule_extraction_evaluator_from_manifest_paths_builds_multi_seed_examples():
    evaluator = RuleExtractionEvaluator.from_manifest_paths(
        [MANIFEST_PATH, SECOND_MANIFEST_PATH]
    )

    assert len(evaluator.prepared_cases) == 2
    assert {example.seed_id for example in evaluator.trainset} == {
        "seti_landing_orbiter_seed_v1",
        "seti_free_action_authority_seed_v1",
    }


def test_rule_extraction_evaluator_from_seed_suite_builds_multi_seed_examples():
    evaluator = RuleExtractionEvaluator.from_seed_suite(SUITE_PATH)

    assert len(evaluator.prepared_cases) == 2
    assert evaluator.prepared_cases[1].request.seed_id == "seti_free_action_authority_seed_v1"


def test_dspy_rule_extractor_coerces_input_shaped_source_documents():
    reference = _load_reference_bundle()
    request = build_request_from_reference_bundle(reference)

    class FakePredictor:
        def __call__(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                source_passages_json=json.dumps(
                    [document.model_dump(mode="json") for document in request.source_documents]
                ),
                canonical_rules_json="[]",
                edges_json="[]",
                uncertainties_json="[]",
            )

    extractor = DspyRuleExtractor(FakePredictor())
    bundle = extractor.extract(request)

    assert [record.node_id for record in bundle.source_passages] == [
        document.source_id for document in request.source_documents
    ]
    assert bundle.source_passages[0].source_text == request.source_documents[0].text
    assert bundle.source_passages[0].game_id == "seti"


def test_rule_extraction_metric_returns_zero_for_malformed_prediction():
    evaluator = RuleExtractionEvaluator.from_seed_manifest(MANIFEST_PATH)
    example = evaluator.trainset[0]

    prediction = SimpleNamespace(
        source_passages_json='[{"source_id":"broken"}]',
        canonical_rules_json="[]",
        edges_json="[]",
        uncertainties_json="[]",
    )

    assert evaluator.metric(example, prediction) == 0.0
