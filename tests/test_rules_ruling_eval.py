"""Tests for the live pilot ruling evaluation layer."""

from __future__ import annotations

from pathlib import Path

from vgm.rules import (
    LivePilotRulingEngine,
    PilotRulingEvaluator,
    load_pilot_ruling_eval_cases,
    load_seti_pilot_bundles,
)
from vgm.schemas import SimilarNode


FIXTURE_SUITE = Path("tests/fixtures/rag_eval/seti_rules_ruling_eval_v1.jsonl")


class FakeRuleStore:
    def __init__(self):
        bundles = load_seti_pilot_bundles()
        self.payloads: dict[str, dict] = {}
        self.edges_by_node: dict[str, list[dict]] = {}
        for bundle in bundles.values():
            for passage in bundle.source_passages:
                self.payloads[passage.node_id] = {
                    "node_type": "source_passage",
                    "content": passage.rendered_content,
                    "project_id": bundle.project_id,
                    "seed_id": bundle.seed_id,
                    "game_id": bundle.game_id,
                    "node_kind": "source_passage",
                    "document_id": passage.document_id,
                    "document_type": passage.document_type,
                    "authority_scope": passage.authority_scope,
                    "title": passage.title,
                    "locator": passage.locator,
                    "page": passage.page,
                    "citation_label": passage.citation_label,
                    "citation_short": passage.citation_short,
                    "language": passage.language,
                    "subsystem": passage.subsystem,
                }
            for rule in bundle.canonical_rules:
                self.payloads[rule.node_id] = {
                    "node_type": "canonical_rule",
                    "content": rule.rendered_content,
                    "project_id": bundle.project_id,
                    "seed_id": bundle.seed_id,
                    "game_id": bundle.game_id,
                    "node_kind": "canonical_rule",
                    "rule_kind": rule.rule_kind,
                    "title": rule.title,
                    "normalized_statement": rule.normalized_statement,
                    "scope": rule.scope,
                    "subsystem": rule.subsystem,
                }
            for edge in bundle.edges:
                record = {
                    "edge_id": edge.edge_id,
                    "relationship_type": edge.edge_type,
                    "description": edge.rendered_description,
                    "from_node_id": edge.from_node_id,
                    "to_node_id": edge.to_node_id,
                }
                self.edges_by_node.setdefault(edge.from_node_id, []).append(record)
                self.edges_by_node.setdefault(edge.to_node_id, []).append(record)

    def search_similar_nodes(self, content: str, limit: int = 5, project_id: str | None = None):
        query_tokens = set(content.lower().replace('"', " ").replace("?", " ").split())
        scored: list[tuple[int, str, dict]] = []
        for node_id, payload in self.payloads.items():
            if project_id is not None and payload["project_id"] != project_id:
                continue
            text = " ".join(
                str(payload.get(key, ""))
                for key in ("title", "normalized_statement", "citation_label", "locator", "content")
            ).lower()
            score = sum(1 for token in query_tokens if token and token in text)
            if score > 0:
                scored.append((score, node_id, payload))
        scored.sort(reverse=True)
        return [
            SimilarNode(
                node_id=node_id,
                content=str(payload["content"]),
                node_type=str(payload["node_type"]),
                similarity_score=min(0.99, 0.5 + (score / max(len(query_tokens), 1))),
                metadata={k: v for k, v in payload.items() if k not in {"content", "node_type"}},
            )
            for score, node_id, payload in scored[:limit]
        ]

    def traverse_from_node(self, node_id: str, gremlin_steps: str):
        del gremlin_steps
        return list(self.edges_by_node.get(node_id, []))

    def get_nodes_batch(self, node_ids: list[str]):
        return {node_id: self.payloads[node_id] for node_id in node_ids if node_id in self.payloads}


def test_load_pilot_ruling_eval_cases_reads_tracked_suite():
    cases = load_pilot_ruling_eval_cases(FIXTURE_SUITE)

    assert len(cases) == 8
    assert cases[0].suite_id == "seti_rules_ruling_eval_v1"
    assert cases[-1].expected_abstain is True


def test_pilot_ruling_evaluator_scores_live_engine_by_component():
    evaluator = PilotRulingEvaluator.from_suite(FIXTURE_SUITE)
    engine = LivePilotRulingEngine(FakeRuleStore(), project_id="seti_rules_lawyer")

    report = evaluator.evaluate_engine(engine)

    assert report.suite_id == "seti_rules_ruling_eval_v1"
    assert report.total_cases == 8
    assert report.failed_cases == 0
    assert report.average_total_score == 1.0
    assert report.average_retrieval_nodes == 1.0
    assert report.average_expanded_evidence == 1.0
    assert report.average_seed_inference == 1.0
    assert report.average_case_selection == 1.0
    assert report.average_primary_citation == 1.0
    assert report.average_modifier_selection == 1.0
    assert report.average_precedence_assembly == 1.0
    assert all(case.total_score == 1.0 for case in report.cases)
