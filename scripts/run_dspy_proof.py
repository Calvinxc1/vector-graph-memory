"""Manual proof runner for baseline DSPy eval and compile comparison."""

from __future__ import annotations

import argparse
import json
import os

import dspy
from dotenv import load_dotenv

from vgm.rag import (
    DEFAULT_DSPY_RUN_LOG_DIR,
    DEFAULT_EVAL_SOURCE_DIR,
    DEFAULT_EVAL_SUITE_PATH,
    DEFAULT_RAG_ARTIFACT_DIR,
    DspyArtifactStore,
    DspyCompileManager,
    DspyRagEvalJudge,
    DspyModelIdentity,
    DspyRagSynthesizer,
    DspyRunLogger,
    RubricRagEvaluator,
    build_evaluation_policy_key,
    build_dspy_lm,
    normalize_dspy_model_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline DSPy eval and optional compile comparison."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "openai:gpt-4o-mini"),
        help="PydanticAI/DSPy model string. Defaults to LLM_MODEL.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Run DSPy compilation and compare baseline vs compiled.",
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("PROJECT_ID", "dspy-proof-run"),
        help="Project identifier used in eval contexts.",
    )
    parser.add_argument(
        "--use-case",
        default=os.getenv(
            "MEMORY_USE_CASE", "Board game rules reference and clarification"
        ),
        help="Use-case text supplied to the synthesizer.",
    )
    parser.add_argument(
        "--eval-suite",
        default=os.getenv("RAG_DSPY_EVAL_SUITE_PATH", str(DEFAULT_EVAL_SUITE_PATH)),
        help="Path to the tracked eval JSONL suite.",
    )
    parser.add_argument(
        "--eval-source-dir",
        default=os.getenv("RAG_DSPY_EVAL_SOURCE_DIR", str(DEFAULT_EVAL_SOURCE_DIR)),
        help="Path to extracted local source documents.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=os.getenv("RAG_DSPY_CACHE_DIR", str(DEFAULT_RAG_ARTIFACT_DIR)),
        help="Directory for promoted compiled artifacts.",
    )
    parser.add_argument(
        "--run-log-dir",
        default=os.getenv("RAG_DSPY_RUN_LOG_DIR", str(DEFAULT_DSPY_RUN_LOG_DIR)),
        help="Directory for baseline/compile proof logs.",
    )
    parser.add_argument(
        "--program-version",
        default=os.getenv("RAG_DSPY_PROGRAM_VERSION", "1"),
        help="Synthesis program version for cache identity.",
    )
    parser.add_argument(
        "--retrieval-schema-version",
        default=os.getenv("RAG_RETRIEVAL_SCHEMA_VERSION", "1"),
        help="Retrieval schema version for cache identity.",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=("deterministic", "hybrid"),
        default=os.getenv("RAG_DSPY_EVAL_SCORING_MODE", "hybrid"),
        help="Offline eval scoring mode. Hybrid adds an LLM judge for softer scoring.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("RAG_DSPY_JUDGE_MODEL"),
        help="Optional provider:model judge override, e.g. openai:gpt-5.4.",
    )
    return parser.parse_args()


def build_identity(
    *,
    model_name: str,
    evaluator: RubricRagEvaluator,
    program_version: str,
    retrieval_schema_version: str,
    evaluation_policy_key: str,
) -> DspyModelIdentity:
    return DspyModelIdentity.from_model_name(
        normalize_dspy_model_name(model_name),
        model_version=os.getenv("DSPY_MODEL_VERSION"),
        api_base=os.getenv("DSPY_API_BASE"),
        model_type=os.getenv("DSPY_MODEL_TYPE"),
        retrieval_schema_version=retrieval_schema_version,
        synthesis_program_version=program_version,
        eval_suite_id=evaluator.suite_id,
        evaluation_policy_key=evaluation_policy_key,
    )


def main() -> int:
    load_dotenv()
    args = parse_args()

    lm = build_dspy_lm(
        args.model,
        model_name_override=os.getenv("DSPY_MODEL_NAME"),
        api_key=os.getenv("DSPY_API_KEY"),
        api_base=os.getenv("DSPY_API_BASE"),
        model_type=os.getenv("DSPY_MODEL_TYPE"),
    )
    dspy.settings.configure(lm=lm)

    judge_model = args.judge_model or args.model
    judge_model_name = os.getenv("RAG_DSPY_JUDGE_MODEL_NAME") or normalize_dspy_model_name(
        judge_model
    )
    judge_model_version = os.getenv("RAG_DSPY_JUDGE_MODEL_VERSION")
    judge_lm = None
    if args.scoring_mode == "hybrid":
        judge_lm = build_dspy_lm(
            judge_model,
            model_name_override=os.getenv("RAG_DSPY_JUDGE_MODEL_NAME"),
            api_key=os.getenv("RAG_DSPY_JUDGE_API_KEY") or os.getenv("DSPY_API_KEY"),
            api_base=os.getenv("RAG_DSPY_JUDGE_API_BASE") or os.getenv("DSPY_API_BASE"),
            model_type=os.getenv("RAG_DSPY_JUDGE_MODEL_TYPE")
            or os.getenv("DSPY_MODEL_TYPE"),
        )

    evaluator = RubricRagEvaluator.from_suite(
        suite_path=args.eval_suite,
        source_dir=args.eval_source_dir,
        use_case_description=args.use_case,
        project_id=args.project_id,
        judge=DspyRagEvalJudge.from_lm(judge_lm) if judge_lm is not None else None,
        scoring_mode=args.scoring_mode,
    )
    identity = build_identity(
        model_name=args.model,
        evaluator=evaluator,
        program_version=args.program_version,
        retrieval_schema_version=args.retrieval_schema_version,
        evaluation_policy_key=build_evaluation_policy_key(
            args.scoring_mode,
            judge_model_name=judge_model_name,
            judge_model_version=judge_model_version,
        ),
    )
    run_logger = DspyRunLogger(base_dir=args.run_log_dir)

    baseline_report, baseline_traces = evaluator.evaluate_synthesizer_with_trace(
        DspyRagSynthesizer.from_lm(lm)
    )
    baseline_run_dir = run_logger.log_baseline_eval(
        identity=identity,
        report=baseline_report,
        trace_entries=baseline_traces,
        metadata={
            "judge_model": judge_model if args.scoring_mode == "hybrid" else None,
            "model": args.model,
            "project_id": args.project_id,
            "scoring_mode": args.scoring_mode,
            "use_case": args.use_case,
        },
    )

    payload: dict[str, object] = {
        "baseline": {
            "run_dir": str(baseline_run_dir),
            "suite_id": baseline_report.suite_id,
            "backend": baseline_report.backend,
            "judge_model": judge_model if args.scoring_mode == "hybrid" else None,
            "scoring_mode": args.scoring_mode,
            "total_score": baseline_report.total_score,
            "average_groundedness": baseline_report.average_groundedness,
            "average_abstention": baseline_report.average_abstention,
            "average_source_alignment": baseline_report.average_source_alignment,
            "average_completeness": baseline_report.average_completeness,
            "num_cases": len(baseline_report.case_results),
        }
    }

    if args.compile:
        manager = DspyCompileManager(
            lm=lm,
            identity=identity,
            evaluator=evaluator,
            artifact_store=DspyArtifactStore(args.artifact_dir),
            auto_compile=True,
            run_logger=run_logger,
        )
        outcome = manager.compile_and_promote()
        if run_logger.last_compile_run_dir is None:
            raise RuntimeError("Compile run did not produce a persisted log directory")
        payload["compile"] = {
            "run_dir": str(run_logger.last_compile_run_dir),
            "judge_model": judge_model if args.scoring_mode == "hybrid" else None,
            "scoring_mode": args.scoring_mode,
            "promoted": outcome.promoted,
            "reason": outcome.reason,
            "baseline_total_score": outcome.baseline_report.total_score,
            "compiled_total_score": outcome.compiled_report.total_score,
            "baseline_groundedness": outcome.baseline_report.average_groundedness,
            "compiled_groundedness": outcome.compiled_report.average_groundedness,
            "artifact_key": outcome.manifest.identity.cache_key(),
        }

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
