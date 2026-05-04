"""Tests for pre-import rule-load auditing."""

from __future__ import annotations

import json
from pathlib import Path

from vgm.rules import (
    audit_rule_extraction_bundle,
    build_request_from_reference_bundle,
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


def test_rule_load_audit_accepts_reference_bundle():
    bundle = _load_reference_bundle()
    request = build_request_from_reference_bundle(bundle)

    report = audit_rule_extraction_bundle(bundle, request=request)

    assert report.passes_required_gates is True
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.graph.source_grounded_rules == len(bundle.canonical_rules)
    assert report.source_coverage.requested_documents_represented == len(request.source_documents)


def test_rule_load_audit_rejects_rule_without_source_grounding_edge():
    bundle = _load_reference_bundle()
    broken = bundle.model_copy(deep=True)
    broken.edges = [
        edge
        for edge in broken.edges
        if edge.to_node_id != "rule_seti_landing_discount_if_orbiter_present"
        or edge.from_node_id not in {passage.node_id for passage in broken.source_passages}
    ]

    report = audit_rule_extraction_bundle(broken)

    assert report.passes_required_gates is False
    assert any(
        finding.code == "canonical_rule_missing_source_grounding"
        and "rule_seti_landing_discount_if_orbiter_present" in finding.related_ids
        for finding in report.findings
    )


def test_rule_load_audit_rejects_source_text_not_grounded_in_request():
    bundle = _load_reference_bundle()
    request = build_request_from_reference_bundle(bundle)
    broken = bundle.model_copy(deep=True)
    broken.source_passages[0].source_text = "This text is not in the requested source document."

    report = audit_rule_extraction_bundle(broken, request=request)

    assert report.passes_required_gates is False
    assert any(
        finding.code == "source_passage_text_not_in_requested_source"
        and broken.source_passages[0].node_id in finding.related_ids
        for finding in report.findings
    )
