"""Local storage primitives for compiled DSPy RAG artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from pydantic import BaseModel, Field

from .evaluation import RagEvalReport

DEFAULT_RAG_ARTIFACT_DIR = Path(".vgm/dspy_artifacts")


class DspyModelIdentity(BaseModel):
    """Exact cache identity for one DSPy synthesis target."""

    provider: str
    model_id: str
    model_version: str | None = None
    api_base: str | None = None
    model_type: str | None = None
    retrieval_schema_version: str = "1"
    synthesis_program_version: str = "1"
    eval_suite_id: str = "seti_rules_reference_v1"

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        *,
        model_version: str | None = None,
        api_base: str | None = None,
        model_type: str | None = None,
        retrieval_schema_version: str = "1",
        synthesis_program_version: str = "1",
        eval_suite_id: str = "seti_rules_reference_v1",
    ) -> "DspyModelIdentity":
        """Build an identity from a normalized provider/model string."""

        provider, _, model_id = model_name.partition("/")
        if not model_id:
            provider = "unknown"
            model_id = model_name
        return cls(
            provider=provider,
            model_id=model_id,
            model_version=model_version,
            api_base=api_base,
            model_type=model_type,
            retrieval_schema_version=retrieval_schema_version,
            synthesis_program_version=synthesis_program_version,
            eval_suite_id=eval_suite_id,
        )

    def cache_key(self) -> str:
        """Build a readable but collision-resistant cache key."""

        readable_parts = [
            _sanitize(self.provider),
            _sanitize(self.model_id),
            _sanitize(self.model_version or "unversioned"),
            f"schema-{_sanitize(self.retrieval_schema_version)}",
            f"program-{_sanitize(self.synthesis_program_version)}",
            f"suite-{_sanitize(self.eval_suite_id)}",
        ]
        raw_key = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
        return "--".join(readable_parts + [digest])


class DspyArtifactManifest(BaseModel):
    """Metadata recorded for one promoted compiled artifact."""

    artifact_version: int = 1
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    identity: DspyModelIdentity
    promoted: bool
    baseline_total_score: float
    compiled_total_score: float
    baseline_groundedness: float
    compiled_groundedness: float
    program_path: str = "compiled_program.json"
    eval_report_path: str = "eval_report.json"


class DspyCompileOutcome(BaseModel):
    """Result of compiling and comparing a candidate DSPy program."""

    promoted: bool
    reason: str
    manifest: DspyArtifactManifest
    baseline_report: RagEvalReport
    compiled_report: RagEvalReport


class DspyArtifactStore:
    """Store and load promoted compiled DSPy artifacts on local disk."""

    def __init__(self, base_dir: str | Path = DEFAULT_RAG_ARTIFACT_DIR):
        self.base_dir = Path(base_dir)

    def artifact_dir(self, identity: DspyModelIdentity) -> Path:
        return self.base_dir / identity.cache_key()

    def has_artifact(self, identity: DspyModelIdentity) -> bool:
        artifact_dir = self.artifact_dir(identity)
        return (
            (artifact_dir / "manifest.json").exists()
            and (artifact_dir / "compiled_program.json").exists()
            and (artifact_dir / "eval_report.json").exists()
        )

    def load_manifest(self, identity: DspyModelIdentity) -> DspyArtifactManifest | None:
        manifest_path = self.artifact_dir(identity) / "manifest.json"
        if not manifest_path.exists():
            return None
        return DspyArtifactManifest.model_validate_json(manifest_path.read_text())

    def load_program(
        self,
        identity: DspyModelIdentity,
        *,
        program_factory: Callable[[], Any],
    ) -> Any | None:
        program_path = self.artifact_dir(identity) / "compiled_program.json"
        if not program_path.exists():
            return None
        program = program_factory()
        program.load(str(program_path), allow_pickle=False)
        return program

    def save_promoted_artifact(
        self,
        *,
        identity: DspyModelIdentity,
        program: Any,
        manifest: DspyArtifactManifest,
        eval_report: RagEvalReport,
    ) -> None:
        artifact_dir = self.artifact_dir(identity)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        program_path = artifact_dir / manifest.program_path
        report_path = artifact_dir / manifest.eval_report_path
        manifest_path = artifact_dir / "manifest.json"

        program.save(str(program_path), save_program=True)
        report_path.write_text(eval_report.model_dump_json(indent=2))
        manifest_path.write_text(manifest.model_dump_json(indent=2))


def _sanitize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
