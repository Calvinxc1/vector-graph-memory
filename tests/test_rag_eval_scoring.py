"""Tests for the v1 RAG eval scoring contract."""

from vgm.rag import (
    DEFAULT_RAG_EVAL_WEIGHTS,
    RagEvalComponentScores,
    RagEvalWeights,
    compute_rag_eval_score,
)


def test_compute_rag_eval_score_uses_default_weighting():
    components = RagEvalComponentScores(
        groundedness=1.0,
        abstention=0.5,
        source_alignment=0.5,
        completeness=0.0,
    )

    assert compute_rag_eval_score(components) == 0.7


def test_default_weights_prioritize_groundedness_over_completeness():
    grounded_answer = RagEvalComponentScores(
        groundedness=1.0,
        abstention=1.0,
        source_alignment=0.0,
        completeness=0.0,
    )
    complete_but_risky_answer = RagEvalComponentScores(
        groundedness=0.0,
        abstention=1.0,
        source_alignment=1.0,
        completeness=1.0,
    )

    assert compute_rag_eval_score(grounded_answer) > compute_rag_eval_score(
        complete_but_risky_answer
    )


def test_custom_weights_must_sum_to_one():
    try:
        RagEvalWeights(
            groundedness=0.4,
            abstention=0.2,
            source_alignment=0.2,
            completeness=0.3,
        )
    except ValueError as exc:
        assert "must sum to 1.0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid score weights")


def test_default_weights_are_normalized():
    total = (
        DEFAULT_RAG_EVAL_WEIGHTS.groundedness
        + DEFAULT_RAG_EVAL_WEIGHTS.abstention
        + DEFAULT_RAG_EVAL_WEIGHTS.source_alignment
        + DEFAULT_RAG_EVAL_WEIGHTS.completeness
    )

    assert total == 1.0
