"""Tests for rules-lawyer extraction contracts and seed adapters."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from vgm.rules import (
    build_seed_edge_records,
    build_seed_manifest,
    build_seed_node_records,
    load_bundle_from_seed_records,
    write_seed_fixture,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "rag_eval"
MANIFEST_PATH = FIXTURE_DIR / "seti_landing_orbiter_seed_v1_manifest.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_seti_manual_seed_fixture_loads_into_rule_extraction_bundle():
    manifest = _load_json(MANIFEST_PATH)
    node_records = _load_jsonl(FIXTURE_DIR / "seti_landing_orbiter_seed_v1_nodes.jsonl")
    edge_records = _load_jsonl(FIXTURE_DIR / "seti_landing_orbiter_seed_v1_edges.jsonl")

    bundle = load_bundle_from_seed_records(manifest, node_records, edge_records)

    assert bundle.seed_id == "seti_landing_orbiter_seed_v1"
    assert bundle.game_id == "seti"
    assert bundle.subsystem == "landing_and_orbiter_interactions"
    assert len(bundle.source_passages) == 5
    assert len(bundle.canonical_rules) == 9
    assert len(bundle.edges) == 15
    assert bundle.frozen_questions[0] == "Can an orbiter later land on the same planet?"

    q6_passage = next(
        passage
        for passage in bundle.source_passages
        if passage.node_id == "src_seti_faq_q6_opponent_orbiter_discount"
    )
    assert q6_passage.document_type == "faq"
    assert q6_passage.authority_scope == "rule_clarification"

    owner_rule = next(
        rule
        for rule in bundle.canonical_rules
        if rule.node_id == "rule_seti_landing_discount_not_owner_limited"
    )
    assert owner_rule.rule_kind == "clarification"
    assert "regardless of which player owns it" in owner_rule.normalized_statement


def test_seti_manual_seed_fixture_round_trips_through_rules_contract():
    manifest = _load_json(MANIFEST_PATH)
    node_records = _load_jsonl(FIXTURE_DIR / "seti_landing_orbiter_seed_v1_nodes.jsonl")
    edge_records = _load_jsonl(FIXTURE_DIR / "seti_landing_orbiter_seed_v1_edges.jsonl")

    bundle = load_bundle_from_seed_records(manifest, node_records, edge_records)

    rendered_nodes = build_seed_node_records(bundle)
    rendered_edges = build_seed_edge_records(bundle)
    rendered_manifest = build_seed_manifest(
        bundle,
        node_file=manifest["node_file"],
        edge_file=manifest["edge_file"],
    )

    assert rendered_nodes == node_records
    assert rendered_edges == edge_records
    assert rendered_manifest == manifest


def test_write_seed_fixture_round_trips_bundle_to_files():
    manifest = _load_json(MANIFEST_PATH)
    node_records = _load_jsonl(FIXTURE_DIR / "seti_landing_orbiter_seed_v1_nodes.jsonl")
    edge_records = _load_jsonl(FIXTURE_DIR / "seti_landing_orbiter_seed_v1_edges.jsonl")
    bundle = load_bundle_from_seed_records(manifest, node_records, edge_records)

    with TemporaryDirectory() as temp_dir:
        manifest_path, node_path, edge_path = write_seed_fixture(
            bundle,
            output_dir=temp_dir,
        )
        written_manifest = _load_json(manifest_path)
        written_nodes = _load_jsonl(node_path)
        written_edges = _load_jsonl(edge_path)
        restored = load_bundle_from_seed_records(
            written_manifest,
            written_nodes,
            written_edges,
        )

    assert manifest_path.name == "seti_landing_orbiter_seed_v1_manifest.json"
    assert node_path.name == "seti_landing_orbiter_seed_v1_nodes.jsonl"
    assert edge_path.name == "seti_landing_orbiter_seed_v1_edges.jsonl"
    assert restored == bundle
