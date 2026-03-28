"""Tests for offline RAG evaluation helpers."""

import json

from vgm.rag import RagSynthesisResult, RubricRagEvaluator


class FakeSynthesizer:
    """Return a fixed synthesis result for evaluation tests."""

    def __init__(self, result: RagSynthesisResult):
        self.result = result

    def synthesize(self, context):
        return self.result


def write_eval_fixture(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "seti-rules-en.txt").write_text(
        "Launch rules locator. By default, each player is limited to having one probe in space. "
        "Figures on the planetary board are not probes in space."
    )
    fixture_path = tmp_path / "suite.jsonl"
    fixture_path.write_text(
        json.dumps(
            {
                "case_id": "seti-test-1",
                "suite_id": "seti_rules_reference_v1",
                "game_id": "seti",
                "mode": "rules_reference",
                "tags": ["single_turn"],
                "conversation": [
                    {
                        "role": "user",
                        "content": "How many probes can I have in space by default?",
                    }
                ],
                "retrieval_refs": [
                    {
                        "source_id": "source-1",
                        "document_id": "seti-rules-en",
                        "document_type": "core_rules",
                        "authority_scope": "general_rules",
                        "page": 1,
                        "locator": "Launch rules locator.",
                        "priority": "primary",
                    }
                ],
                "rubric": {
                    "expected_abstain": False,
                    "must_include": [
                        "By default, a player can have only one probe on the solar system board at a time.",
                        "Figures on planetary boards do not count as probes in space for that limit.",
                    ],
                    "must_not_include": [
                        "The default limit is two probes.",
                    ],
                    "preferred_source_id": "source-1",
                    "notes": "",
                },
            }
        )
        + "\n"
    )
    return fixture_path, source_dir


def test_rubric_evaluator_scores_fully_grounded_answer(tmp_path):
    fixture_path, source_dir = write_eval_fixture(tmp_path)
    evaluator = RubricRagEvaluator.from_suite(
        suite_path=fixture_path,
        source_dir=source_dir,
        use_case_description="Board game rules reference",
        project_id="seti-test",
    )
    synthesizer = FakeSynthesizer(
        RagSynthesisResult(
            answer=(
                "By default, a player can have only one probe on the solar system board at a time, "
                "and figures on planetary boards do not count as probes in space for that limit."
            ),
            cited_source_ids=["source-1"],
            abstain=False,
            backend="fake-baseline",
        )
    )

    report = evaluator.evaluate_synthesizer(synthesizer)

    assert report.total_score >= 0.95
    assert report.average_groundedness == 1.0
    assert report.average_source_alignment == 1.0
