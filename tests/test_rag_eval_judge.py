"""Tests for the DSPy-backed RAG eval judge wrapper."""

import json
from types import SimpleNamespace

from vgm.rag import (
    DspyRagEvalJudge,
    RagSynthesisResult,
    RubricRagEvaluator,
    build_evaluation_policy_key,
)


class FakePredictor:
    """Capture judge inputs and return fixed scores."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            groundedness_score="4",
            completeness_score=5,
            rationale="Supported and complete.",
        )


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


def test_dspy_rag_eval_judge_normalizes_scores_and_formats_inputs(tmp_path):
    fixture_path, source_dir = write_eval_fixture(tmp_path)
    evaluator = RubricRagEvaluator.from_suite(
        suite_path=fixture_path,
        source_dir=source_dir,
        use_case_description="Board game rules reference",
        project_id="seti-test",
    )
    prepared_case = evaluator.prepared_cases[0]
    predictor = FakePredictor()
    judge = DspyRagEvalJudge(predictor=predictor)

    result = judge.judge(
        case=prepared_case.case,
        context=prepared_case.context,
        result=RagSynthesisResult(
            answer="Only one probe can be in space by default.",
            cited_source_ids=["source-1"],
            abstain=False,
            backend="fake-baseline",
        ),
    )

    assert result.groundedness == 0.8
    assert result.completeness == 1.0
    assert result.rationale == "Supported and complete."
    assert predictor.calls[0]["question"] == "How many probes can I have in space by default?"
    assert "must_include:" in predictor.calls[0]["rubric"]
    assert "The default limit is two probes." in predictor.calls[0]["rubric"]


def test_build_evaluation_policy_key_includes_judge_identity():
    assert build_evaluation_policy_key("deterministic") == "deterministic"
    assert (
        build_evaluation_policy_key(
            "hybrid",
            judge_model_name="openai/gpt-5.4",
            judge_model_version="2026-03-01",
        )
        == "hybrid--judge-openai-gpt-5-4--2026-03-01"
    )
