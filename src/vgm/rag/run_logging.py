"""Persistent local logging for manual DSPy eval and compile runs."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from .artifacts import DspyCompileOutcome, DspyModelIdentity
from .evaluation import RagEvalReport, RagEvalTraceEntry

DEFAULT_DSPY_RUN_LOG_DIR = Path(".vgm/dspy_runs")


class DspyRunSummary(BaseModel):
    """Metadata persisted for a single manual eval or compile run."""

    run_id: str
    run_type: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    identity: DspyModelIdentity
    suite_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DspyRunLogger:
    """Write local JSON and transcript artifacts for proof runs."""

    def __init__(self, base_dir: str | Path = DEFAULT_DSPY_RUN_LOG_DIR):
        self.base_dir = Path(base_dir)
        self.last_baseline_run_dir: Path | None = None
        self.last_compile_run_dir: Path | None = None

    def log_baseline_eval(
        self,
        *,
        identity: DspyModelIdentity,
        report: RagEvalReport,
        trace_entries: list[RagEvalTraceEntry] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist a baseline eval report and summary."""

        run_dir = self._create_run_dir("baseline_eval", identity)
        self._write_json(run_dir / "report.json", report.model_dump(mode="json"))
        if trace_entries is not None:
            self._write_json(
                run_dir / "trace.json",
                [entry.model_dump(mode="json") for entry in trace_entries],
            )
        summary = DspyRunSummary(
            run_id=run_dir.name,
            run_type="baseline_eval",
            identity=identity,
            suite_id=report.suite_id,
            metadata={
                "backend": report.backend,
                "total_score": report.total_score,
                "average_groundedness": report.average_groundedness,
                "average_abstention": report.average_abstention,
                "average_source_alignment": report.average_source_alignment,
                "average_completeness": report.average_completeness,
                **(metadata or {}),
            },
        )
        self._write_json(run_dir / "summary.json", summary.model_dump(mode="json"))
        self.last_baseline_run_dir = run_dir
        return run_dir

    def log_compile_outcome(
        self,
        *,
        identity: DspyModelIdentity,
        outcome: DspyCompileOutcome,
        transcript: str,
        baseline_traces: list[RagEvalTraceEntry] | None = None,
        compiled_traces: list[RagEvalTraceEntry] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist compile comparison reports, summary, and raw transcript."""

        run_dir = self._create_run_dir("compile_compare", identity)
        self._write_json(
            run_dir / "baseline_report.json",
            outcome.baseline_report.model_dump(mode="json"),
        )
        self._write_json(
            run_dir / "compiled_report.json",
            outcome.compiled_report.model_dump(mode="json"),
        )
        if baseline_traces is not None:
            self._write_json(
                run_dir / "baseline_trace.json",
                [entry.model_dump(mode="json") for entry in baseline_traces],
            )
        if compiled_traces is not None:
            self._write_json(
                run_dir / "compiled_trace.json",
                [entry.model_dump(mode="json") for entry in compiled_traces],
            )
        (run_dir / "compile.log").write_text(transcript)
        summary = DspyRunSummary(
            run_id=run_dir.name,
            run_type="compile_compare",
            identity=identity,
            suite_id=outcome.baseline_report.suite_id,
            metadata={
                "promoted": outcome.promoted,
                "reason": outcome.reason,
                "baseline_total_score": outcome.baseline_report.total_score,
                "compiled_total_score": outcome.compiled_report.total_score,
                "baseline_groundedness": outcome.baseline_report.average_groundedness,
                "compiled_groundedness": outcome.compiled_report.average_groundedness,
                "artifact_key": outcome.manifest.identity.cache_key(),
                **(metadata or {}),
            },
        )
        self._write_json(run_dir / "summary.json", summary.model_dump(mode="json"))
        self.last_compile_run_dir = run_dir
        return run_dir

    @staticmethod
    def capture_output(fn: Callable[[], Any]) -> tuple[Any, str]:
        """Capture stdout/stderr produced by one callable."""

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = fn()
        transcript = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()
        if stderr_text:
            transcript = f"{transcript}\n{stderr_text}" if transcript else stderr_text
        return result, transcript

    def _create_run_dir(self, run_type: str, identity: DspyModelIdentity) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.base_dir / f"{timestamp}--{run_type}--{identity.cache_key()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
