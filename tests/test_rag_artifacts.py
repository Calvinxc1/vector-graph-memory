"""Tests for local DSPy artifact storage."""

from pathlib import Path

from vgm.rag import (
    DspyArtifactManifest,
    DspyArtifactStore,
    DspyModelIdentity,
    RagEvalReport,
)


class FakeProgram:
    """Minimal serializable program double."""

    def __init__(self):
        self.loaded_from = None

    def save(self, path, save_program=False):
        Path(path).write_text(f"saved:{save_program}")

    def load(self, path, allow_pickle=False):
        self.loaded_from = (Path(path).read_text(), allow_pickle)


def build_identity() -> DspyModelIdentity:
    return DspyModelIdentity(
        provider="openai",
        model_id="gpt-4o-mini",
        retrieval_schema_version="1",
        synthesis_program_version="1",
        eval_suite_id="seti_rules_reference_v1",
    )


def build_report(score: float) -> RagEvalReport:
    return RagEvalReport(
        suite_id="seti_rules_reference_v1",
        backend="dspy-compiled",
        case_results=[],
        average_groundedness=score,
        average_abstention=score,
        average_source_alignment=score,
        average_completeness=score,
        total_score=score,
    )


def test_artifact_store_round_trips_manifest_and_program(tmp_path):
    store = DspyArtifactStore(base_dir=tmp_path)
    identity = build_identity()
    manifest = DspyArtifactManifest(
        identity=identity,
        promoted=True,
        baseline_total_score=0.6,
        compiled_total_score=0.8,
        baseline_groundedness=1.0,
        compiled_groundedness=1.0,
    )

    store.save_promoted_artifact(
        identity=identity,
        program=FakeProgram(),
        manifest=manifest,
        eval_report=build_report(0.8),
    )

    assert store.has_artifact(identity) is True
    loaded_manifest = store.load_manifest(identity)
    assert loaded_manifest is not None
    assert loaded_manifest.compiled_total_score == 0.8

    loaded_program = store.load_program(identity, program_factory=FakeProgram)
    assert isinstance(loaded_program, FakeProgram)
    assert loaded_program.loaded_from == ("saved:True", False)


def test_cache_key_changes_with_evaluation_policy():
    deterministic = DspyModelIdentity(
        provider="openai",
        model_id="gpt-4o-mini",
        retrieval_schema_version="1",
        synthesis_program_version="1",
        eval_suite_id="seti_rules_reference_v1",
        evaluation_policy_key="deterministic",
    )
    hybrid = DspyModelIdentity(
        provider="openai",
        model_id="gpt-4o-mini",
        retrieval_schema_version="1",
        synthesis_program_version="1",
        eval_suite_id="seti_rules_reference_v1",
        evaluation_policy_key="hybrid--openai-gpt-5-4",
    )

    assert deterministic.cache_key() != hybrid.cache_key()
