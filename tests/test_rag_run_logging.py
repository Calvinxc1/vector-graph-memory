"""Tests for persistent DSPy proof-run logging."""

import sys

from vgm.rag import (
    DspyModelIdentity,
    DspyRunLogger,
    RagEvalBucketReport,
    RagEvalReport,
    RagEvalTraceEntry,
)
from vgm.rag.artifacts import DspyArtifactManifest, DspyCompileOutcome


def build_identity() -> DspyModelIdentity:
    return DspyModelIdentity(
        provider="openai",
        model_id="gpt-4o-mini",
        retrieval_schema_version="1",
        synthesis_program_version="1",
        eval_suite_id="seti_rules_reference_v1",
    )


def build_report(score: float, backend: str) -> RagEvalReport:
    return RagEvalReport(
        suite_id="seti_rules_reference_v1",
        backend=backend,
        case_results=[],
        bucket_reports={
            "standard": RagEvalBucketReport(
                bucket_name="standard",
                num_cases=1,
                average_groundedness=score,
                average_abstention=score,
                average_source_alignment=score,
                average_completeness=score,
                total_score=score,
            )
        },
        average_groundedness=score,
        average_abstention=score,
        average_source_alignment=score,
        average_completeness=score,
        total_score=score,
    )


def test_run_logger_persists_baseline_report(tmp_path):
    logger = DspyRunLogger(base_dir=tmp_path)
    trace_entries = [
        RagEvalTraceEntry(
            case_id="seti-case-1",
            conversation=[{"role": "user", "content": "Question"}],
            retrieval_refs=[{"source_id": "source-1"}],
            rag_context={"current_question": "Question"},
            rubric={"expected_abstain": False},
            result={
                "case_id": "seti-case-1",
                "backend": "dspy-baseline",
                "answer": "Answer",
                "cited_source_ids": ["source-1"],
                "abstain": False,
                "groundedness": 1.0,
                "abstention": 1.0,
                "source_alignment": 1.0,
                "completeness": 1.0,
                "total_score": 1.0,
            },
        )
    ]

    run_dir = logger.log_baseline_eval(
        identity=build_identity(),
        report=build_report(0.9, "dspy-baseline"),
        trace_entries=trace_entries,
        metadata={"proof": True},
    )

    assert (run_dir / "report.json").exists()
    assert (run_dir / "trace.json").exists()
    assert (run_dir / "summary.json").exists()
    assert '"proof": true' in (run_dir / "summary.json").read_text().lower()
    assert '"bucket_reports"' in (run_dir / "summary.json").read_text()


def test_run_logger_persists_compile_reports_and_transcript(tmp_path):
    logger = DspyRunLogger(base_dir=tmp_path)
    identity = build_identity()
    trace_entries = [
        RagEvalTraceEntry(
            case_id="seti-case-1",
            conversation=[{"role": "user", "content": "Question"}],
            retrieval_refs=[{"source_id": "source-1"}],
            rag_context={"current_question": "Question"},
            rubric={"expected_abstain": False},
            result={
                "case_id": "seti-case-1",
                "backend": "dspy-baseline",
                "answer": "Answer",
                "cited_source_ids": ["source-1"],
                "abstain": False,
                "groundedness": 1.0,
                "abstention": 1.0,
                "source_alignment": 1.0,
                "completeness": 1.0,
                "total_score": 1.0,
            },
        )
    ]
    outcome = DspyCompileOutcome(
        promoted=False,
        reason="candidate regressed groundedness",
        manifest=DspyArtifactManifest(
            identity=identity,
            promoted=False,
            baseline_total_score=0.9,
            compiled_total_score=0.8,
            baseline_groundedness=1.0,
            compiled_groundedness=0.9,
        ),
        baseline_report=build_report(0.9, "dspy-baseline"),
        compiled_report=build_report(0.8, "dspy-compiled"),
    )

    run_dir = logger.log_compile_outcome(
        identity=identity,
        outcome=outcome,
        transcript="compile transcript",
        baseline_traces=trace_entries,
        compiled_traces=trace_entries,
    )

    assert (run_dir / "baseline_report.json").exists()
    assert (run_dir / "compiled_report.json").exists()
    assert (run_dir / "baseline_trace.json").exists()
    assert (run_dir / "compiled_trace.json").exists()
    assert (run_dir / "compile.log").read_text() == "compile transcript"
    assert '"baseline_bucket_reports"' in (run_dir / "summary.json").read_text()


def test_capture_output_collects_stdout_and_stderr():
    def emit():
        print("stdout line")
        sys.stderr.write("stderr line\n")
        return "done"

    result, transcript = DspyRunLogger.capture_output(emit)

    assert result == "done"
    assert "stdout line" in transcript
    assert "stderr line" in transcript
