"""Tests for the deterministic pilot ruling path."""

from __future__ import annotations

from vgm.rules import (
    DeterministicPilotRulingEngine,
    LivePilotRulingEngine,
    RulesRulingRequest,
    load_seti_pilot_bundles,
)
from vgm.schemas import SimilarNode


class FakeRuleStore:
    def __init__(self):
        bundles = load_seti_pilot_bundles()
        self.payloads: dict[str, dict] = {}
        self.edges_by_node: dict[str, list[dict]] = {}
        for bundle in bundles.values():
            for passage in bundle.source_passages:
                payload = {
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
                self.payloads[passage.node_id] = payload
            for rule in bundle.canonical_rules:
                payload = {
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
                self.payloads[rule.node_id] = payload
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
        results: list[SimilarNode] = []
        for rank, (score, node_id, payload) in enumerate(scored[:limit], start=1):
            results.append(
                SimilarNode(
                    node_id=node_id,
                    content=str(payload["content"]),
                    node_type=str(payload["node_type"]),
                    similarity_score=min(0.99, 0.5 + (score / max(len(query_tokens), 1))),
                    metadata={k: v for k, v in payload.items() if k not in {"content", "node_type"}},
                )
            )
        return results

    def traverse_from_node(self, node_id: str, gremlin_steps: str):
        del gremlin_steps
        return list(self.edges_by_node.get(node_id, []))

    def get_nodes_batch(self, node_ids: list[str]):
        return {node_id: self.payloads[node_id] for node_id in node_ids if node_id in self.payloads}


def test_pilot_ruling_engine_answers_all_frozen_questions_without_abstaining():
    engine = DeterministicPilotRulingEngine.for_seti_pilot()

    cases = [
        (
            "Can an orbiter later land on the same planet?",
            "seti_landing_orbiter_seed_v1",
            "rule_seti_orbiter_is_permanent",
            "src_seti_faq_q5_land_with_orbiter",
        ),
        (
            "Does an opponent's orbiter still reduce my landing cost?",
            "seti_landing_orbiter_seed_v1",
            "rule_seti_landing_discount_if_orbiter_present",
            "src_seti_core_land_on_planet_or_moon",
        ),
        (
            "Does an existing orbiter also reduce the cost to land on that planet's moon?",
            "seti_landing_orbiter_seed_v1",
            "rule_seti_moon_landing_inherits_planet_discount_logic",
            "src_seti_faq_q7_moon_discount",
        ),
        (
            "If another player's orbiter is at Jupiter and I have the tech that lets me land on moons, do I get the discount when landing on one of Jupiter's moons?",
            "seti_landing_orbiter_seed_v1",
            "rule_seti_moon_landing_inherits_planet_discount_logic",
            "src_seti_faq_q7_moon_discount",
        ),
        (
            "Can I interrupt a main action with a free action, and can I interrupt one free action with another free action?",
            "seti_free_action_authority_seed_v1",
            "rule_seti_free_actions_can_interrupt_main_action",
            "src_seti_faq_q4_free_action_timing",
        ),
        (
            "During a Scan action, can I interrupt the Scan with a free action and then continue the Scan?",
            "seti_free_action_authority_seed_v1",
            "rule_seti_scan_action_can_resume_after_free_action_interrupt",
            "src_seti_faq_q4_scan_example",
        ),
        (
            "If placing data triggers the income increase effect, can I place more data first and then resolve that income effect?",
            "seti_free_action_authority_seed_v1",
            "rule_seti_income_increase_free_action_cannot_be_nested",
            "src_seti_faq_q4_income_example",
        ),
        (
            'The player aid says free actions are "NO LIMIT". Does that allow one free action to interrupt another?',
            "seti_free_action_authority_seed_v1",
            "rule_seti_free_actions_cannot_interrupt_free_action",
            "src_seti_faq_q4_free_action_timing",
        ),
    ]

    for question, seed_id, expected_rule_id, expected_source_id in cases:
        result = engine.answer(RulesRulingRequest(question=question))
        assert result.abstain is False
        assert result.seed_id == seed_id
        assert result.primary_rule is not None
        assert result.primary_rule.rule_node_id == expected_rule_id
        assert result.primary_citation is not None
        assert result.primary_citation.source_node_id == expected_source_id
        assert result.precedence_order


def test_pilot_ruling_engine_exposes_authority_conflict_in_player_aid_case():
    engine = DeterministicPilotRulingEngine.for_seti_pilot()

    result = engine.answer(
        RulesRulingRequest(
            question='The player aid says free actions are "NO LIMIT". Does that allow one free action to interrupt another?'
        )
    )

    assert result.abstain is False
    assert "does not authorize free-action nesting" in result.ruling
    assert [entry.precedence_kind for entry in result.precedence_order] == [
        "primary",
        "authority",
        "support",
    ]
    assert result.modifying_citations[0].source_node_id == "src_seti_aid_free_actions_summary"


def test_pilot_ruling_engine_abstains_for_untracked_question():
    engine = DeterministicPilotRulingEngine.for_seti_pilot()

    result = engine.answer(
        RulesRulingRequest(
            question="What is the best opening move in SETI?",
            seed_id="seti_landing_orbiter_seed_v1",
        )
    )

    assert result.abstain is True
    assert result.question_id == "unsupported_question"
    assert result.primary_rule is None
    assert "frozen SETI questions" in (result.uncertainty or "")


def test_live_pilot_ruling_engine_answers_frozen_question_from_retrieved_graph_evidence():
    engine = LivePilotRulingEngine(FakeRuleStore(), project_id="seti_rules_lawyer")

    result = engine.answer(
        RulesRulingRequest(
            question="If another player already has an orbiter there, do I still get the cheaper landing cost?",
        )
    )

    assert result.abstain is False
    assert result.backend == "live-pilot"
    assert result.seed_id == "seti_landing_orbiter_seed_v1"
    assert result.primary_rule is not None
    assert result.primary_rule.rule_node_id == "rule_seti_landing_discount_if_orbiter_present"
    assert result.primary_citation is not None
    assert result.primary_citation.source_node_id == "src_seti_core_land_on_planet_or_moon"
    assert result.modifying_rules[0].rule_node_id == "rule_seti_landing_discount_not_owner_limited"


def test_live_pilot_ruling_engine_handles_player_aid_authority_conflict():
    engine = LivePilotRulingEngine(FakeRuleStore(), project_id="seti_rules_lawyer")

    result = engine.answer(
        RulesRulingRequest(
            question='The player aid says free actions are "NO LIMIT". Does that allow one free action to interrupt another?',
        )
    )

    assert result.abstain is False
    assert result.seed_id == "seti_free_action_authority_seed_v1"
    assert result.primary_rule is not None
    assert result.primary_rule.rule_node_id == "rule_seti_free_actions_cannot_interrupt_free_action"
    assert result.modifying_citations[0].source_node_id == "src_seti_aid_free_actions_summary"
    assert [entry.precedence_kind for entry in result.precedence_order] == [
        "primary",
        "authority",
        "support",
    ]


def test_live_pilot_ruling_engine_handles_scan_paraphrase_with_same_subsystem():
    engine = LivePilotRulingEngine(FakeRuleStore(), project_id="seti_rules_lawyer")

    result = engine.answer(
        RulesRulingRequest(
            question="If I pause a Scan to do a free action, can I go back and finish the Scan afterward?",
        )
    )

    assert result.abstain is False
    assert result.question_id == "seti-rules-021-scan-interrupt-publicity"
    assert result.primary_rule is not None
    assert result.primary_rule.rule_node_id == "rule_seti_scan_action_can_resume_after_free_action_interrupt"


def test_live_pilot_ruling_inspection_exposes_retrieved_evidence_separately():
    engine = LivePilotRulingEngine(FakeRuleStore(), project_id="seti_rules_lawyer")

    inspection = engine.inspect_request(
        RulesRulingRequest(
            question="If another player already has an orbiter there, do I still get the cheaper landing cost?",
        )
    )

    assert inspection.evidence.seed_id == "seti_landing_orbiter_seed_v1"
    assert inspection.evidence.subsystem == "landing_and_orbiter_interactions"
    assert "rule_seti_landing_discount_if_orbiter_present" in inspection.evidence.nodes_by_logical_id
    assert "src_seti_core_land_on_planet_or_moon" in inspection.evidence.nodes_by_logical_id
    assert inspection.evidence.rule_nodes
    assert inspection.evidence.source_nodes


def test_live_pilot_ruling_inspection_separates_seed_inference_and_case_selection():
    engine = LivePilotRulingEngine(FakeRuleStore(), project_id="seti_rules_lawyer")

    inspection = engine.inspect_request(
        RulesRulingRequest(
            question="If I pause a Scan to do a free action, can I go back and finish the Scan afterward?",
        )
    )

    assert inspection.seed_inference.selected_seed_id == "seti_free_action_authority_seed_v1"
    assert inspection.seed_inference.selected_score >= 0.14
    assert inspection.selected_seed_id == "seti_free_action_authority_seed_v1"
    assert inspection.selected_case is not None
    assert inspection.selected_case.question_id == "seti-rules-021-scan-interrupt-publicity"
    assert inspection.selected_case.question_score >= 0.14
    assert inspection.selected_case.evidence_score >= 2
    assert inspection.candidate_cases


def test_live_pilot_ruling_inspection_leaves_case_unselected_for_unsupported_question():
    engine = LivePilotRulingEngine(FakeRuleStore(), project_id="seti_rules_lawyer")

    inspection = engine.inspect_request(
        RulesRulingRequest(
            question="What is the best opening move in SETI?",
        )
    )

    assert inspection.selected_case is None
    assert inspection.selected_seed_id == "seti_landing_orbiter_seed_v1"
    assert inspection.seed_inference.selected_seed_id is None
    assert inspection.candidate_cases == []


def test_live_pilot_ruling_engine_abstains_for_untracked_question():
    engine = LivePilotRulingEngine(FakeRuleStore(), project_id="seti_rules_lawyer")

    result = engine.answer(
        RulesRulingRequest(
            question="What is the best opening move in SETI?",
        )
    )

    assert result.abstain is True
    assert result.backend == "live-pilot"
