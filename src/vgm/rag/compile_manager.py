"""Compilation and runtime-selection support for DSPy RAG synthesis."""

from __future__ import annotations

from typing import Any, Callable

import dspy

from .artifacts import (
    DspyArtifactManifest,
    DspyArtifactStore,
    DspyCompileOutcome,
    DspyModelIdentity,
)
from .evaluation import RubricRagEvaluator
from .run_logging import DspyRunLogger
from .synthesizer import (
    DspyRagSynthesizer,
    bind_program_lm,
    make_rag_answer_program,
)


class DspyCompileManager:
    """Own compilation, promotion, and cache loading for one model identity."""

    def __init__(
        self,
        *,
        lm: Any,
        identity: DspyModelIdentity,
        evaluator: RubricRagEvaluator,
        artifact_store: DspyArtifactStore,
        auto_compile: bool = True,
        run_logger: DspyRunLogger | None = None,
        compiler_factory: Callable[[Callable[..., float]], Any] | None = None,
    ):
        self.lm = lm
        self.identity = identity
        self.evaluator = evaluator
        self.artifact_store = artifact_store
        self.auto_compile = auto_compile
        self.run_logger = run_logger
        self.compiler_factory = compiler_factory or self._build_default_compiler
        self.compile_attempted = False

    def load_cached_synthesizer(self) -> DspyRagSynthesizer | None:
        """Load a previously promoted compiled synthesizer if present."""

        program = self.artifact_store.load_program(
            self.identity,
            program_factory=make_rag_answer_program,
        )
        if program is None:
            return None
        bind_program_lm(program, self.lm)
        return DspyRagSynthesizer.from_program(program, backend_name="dspy-compiled")

    def begin_auto_compile(self) -> bool:
        """Claim the one automatic compile attempt for this runtime."""

        if not self.auto_compile or self.compile_attempted:
            return False
        if self.artifact_store.has_artifact(self.identity):
            return False
        self.compile_attempted = True
        return True

    def compile_and_promote(self) -> DspyCompileOutcome:
        """Compile a candidate program, compare it to baseline, and persist the winner."""

        if self.run_logger is not None:
            (outcome, baseline_traces, compiled_traces), transcript = self.run_logger.capture_output(
                self._compile_and_promote_inner
            )
            self.run_logger.log_compile_outcome(
                identity=self.identity,
                outcome=outcome,
                transcript=transcript,
                baseline_traces=baseline_traces,
                compiled_traces=compiled_traces,
            )
            return outcome

        outcome, _, _ = self._compile_and_promote_inner()
        return outcome

    def _compile_and_promote_inner(self) -> tuple[DspyCompileOutcome, list[Any], list[Any]]:
        """Internal compile/promotion path shared by runtime and manual proof runs."""

        baseline_synthesizer = DspyRagSynthesizer.from_lm(self.lm)
        baseline_report, baseline_traces = self._evaluate_with_optional_trace(
            baseline_synthesizer
        )

        compiler = self.compiler_factory(self.evaluator.metric)
        student_program = make_rag_answer_program()
        bind_program_lm(student_program, self.lm)
        with dspy.context(lm=self.lm):
            compiled_program = compiler.compile(
                student_program,
                trainset=self.evaluator.trainset,
                valset=self.evaluator.valset,
            )
        bind_program_lm(compiled_program, self.lm)

        compiled_synthesizer = DspyRagSynthesizer.from_program(
            compiled_program,
            backend_name="dspy-compiled",
        )
        compiled_report, compiled_traces = self._evaluate_with_optional_trace(
            compiled_synthesizer
        )

        groundedness_ok = (
            compiled_report.average_groundedness >= baseline_report.average_groundedness
        )
        improved_total = compiled_report.total_score > baseline_report.total_score
        promoted = improved_total and groundedness_ok
        reason = (
            "compiled artifact outperformed baseline"
            if promoted
            else "compiled artifact did not improve total score without groundedness regression"
        )

        manifest = DspyArtifactManifest(
            identity=self.identity,
            promoted=promoted,
            baseline_total_score=baseline_report.total_score,
            compiled_total_score=compiled_report.total_score,
            baseline_groundedness=baseline_report.average_groundedness,
            compiled_groundedness=compiled_report.average_groundedness,
        )

        if promoted:
            self.artifact_store.save_promoted_artifact(
                identity=self.identity,
                program=compiled_program,
                manifest=manifest,
                eval_report=compiled_report,
            )

        return (
            DspyCompileOutcome(
                promoted=promoted,
                reason=reason,
                manifest=manifest,
                baseline_report=baseline_report,
                compiled_report=compiled_report,
            ),
            baseline_traces,
            compiled_traces,
        )

    def _evaluate_with_optional_trace(self, synthesizer: Any) -> tuple[Any, list[Any]]:
        """Support both trace-aware and report-only evaluator implementations in tests."""

        if hasattr(self.evaluator, "evaluate_synthesizer_with_trace"):
            return self.evaluator.evaluate_synthesizer_with_trace(synthesizer)
        return self.evaluator.evaluate_synthesizer(synthesizer), []

    @staticmethod
    def _build_default_compiler(metric: Callable[..., float]) -> Any:
        return dspy.MIPROv2(
            metric=metric,
            auto="light",
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            num_threads=1,
            verbose=False,
        )
