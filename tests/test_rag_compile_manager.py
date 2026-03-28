"""Tests for DSPy compile/promotion policy."""

from types import SimpleNamespace

from vgm.rag.artifacts import DspyCompileOutcome, DspyModelIdentity
from vgm.rag.compile_manager import DspyCompileManager
from vgm.rag.evaluation import RagEvalReport


class FakeArtifactStore:
    """Capture promoted artifacts without touching disk."""

    def __init__(self, has_artifact=False):
        self._has_artifact = has_artifact
        self.saved = []

    def has_artifact(self, identity):
        return self._has_artifact

    def save_promoted_artifact(self, **kwargs):
        self.saved.append(kwargs)

    def load_program(self, identity, *, program_factory):
        return None


class FakeCompiler:
    """Return a fixed compiled program."""

    def __init__(self, compiled_program):
        self.compiled_program = compiled_program
        self.calls = []

    def compile(self, student, *, trainset, valset):
        self.calls.append({"student": student, "trainset": trainset, "valset": valset})
        return self.compiled_program


class FakeEvaluator:
    """Return preconfigured reports in call order."""

    def __init__(self, baseline_report, compiled_report):
        self.baseline_report = baseline_report
        self.compiled_report = compiled_report
        self.trainset = ["train"]
        self.valset = ["val"]
        self.metric = lambda *args, **kwargs: 1.0
        self.calls = []

    def evaluate_synthesizer(self, synthesizer):
        self.calls.append(synthesizer)
        if len(self.calls) == 1:
            return self.baseline_report
        return self.compiled_report


def build_identity() -> DspyModelIdentity:
    return DspyModelIdentity(
        provider="openai",
        model_id="gpt-4o-mini",
        retrieval_schema_version="1",
        synthesis_program_version="1",
        eval_suite_id="seti_rules_reference_v1",
    )


def build_report(score: float, groundedness: float, backend: str) -> RagEvalReport:
    return RagEvalReport(
        suite_id="seti_rules_reference_v1",
        backend=backend,
        case_results=[],
        average_groundedness=groundedness,
        average_abstention=score,
        average_source_alignment=score,
        average_completeness=score,
        total_score=score,
    )


def test_compile_manager_promotes_improving_candidate(monkeypatch):
    fake_store = FakeArtifactStore()
    fake_evaluator = FakeEvaluator(
        baseline_report=build_report(0.60, 1.0, "dspy-baseline"),
        compiled_report=build_report(0.82, 1.0, "dspy-compiled"),
    )
    fake_compiler = FakeCompiler(compiled_program=object())

    class FakeSynthFactory:
        @staticmethod
        def from_lm(lm):
            return SimpleNamespace(kind="baseline", lm=lm)

        @staticmethod
        def from_program(program, backend_name="dspy-compiled"):
            return SimpleNamespace(kind=backend_name, program=program)

    monkeypatch.setattr(
        "vgm.rag.compile_manager.DspyRagSynthesizer",
        FakeSynthFactory,
    )
    monkeypatch.setattr(
        "vgm.rag.compile_manager.bind_program_lm",
        lambda program, lm: None,
    )
    monkeypatch.setattr(
        "vgm.rag.compile_manager.make_rag_answer_program",
        lambda: object(),
    )

    manager = DspyCompileManager(
        lm=object(),
        identity=build_identity(),
        evaluator=fake_evaluator,
        artifact_store=fake_store,
        auto_compile=True,
        compiler_factory=lambda metric: fake_compiler,
    )

    outcome = manager.compile_and_promote()

    assert outcome.promoted is True
    assert len(fake_store.saved) == 1
    assert fake_store.saved[0]["manifest"].compiled_total_score == 0.82


def test_compile_manager_rejects_groundedness_regression(monkeypatch):
    fake_store = FakeArtifactStore()
    fake_evaluator = FakeEvaluator(
        baseline_report=build_report(0.70, 1.0, "dspy-baseline"),
        compiled_report=build_report(0.85, 0.75, "dspy-compiled"),
    )
    fake_compiler = FakeCompiler(compiled_program=object())

    class FakeSynthFactory:
        @staticmethod
        def from_lm(lm):
            return SimpleNamespace(kind="baseline", lm=lm)

        @staticmethod
        def from_program(program, backend_name="dspy-compiled"):
            return SimpleNamespace(kind=backend_name, program=program)

    monkeypatch.setattr(
        "vgm.rag.compile_manager.DspyRagSynthesizer",
        FakeSynthFactory,
    )
    monkeypatch.setattr(
        "vgm.rag.compile_manager.bind_program_lm",
        lambda program, lm: None,
    )
    monkeypatch.setattr(
        "vgm.rag.compile_manager.make_rag_answer_program",
        lambda: object(),
    )

    manager = DspyCompileManager(
        lm=object(),
        identity=build_identity(),
        evaluator=fake_evaluator,
        artifact_store=fake_store,
        auto_compile=True,
        compiler_factory=lambda metric: fake_compiler,
    )

    outcome = manager.compile_and_promote()

    assert outcome.promoted is False
    assert fake_store.saved == []


def test_begin_auto_compile_is_single_use():
    manager = DspyCompileManager(
        lm=object(),
        identity=build_identity(),
        evaluator=FakeEvaluator(
            baseline_report=build_report(0.6, 1.0, "dspy-baseline"),
            compiled_report=build_report(0.8, 1.0, "dspy-compiled"),
        ),
        artifact_store=FakeArtifactStore(),
        auto_compile=True,
        compiler_factory=lambda metric: FakeCompiler(compiled_program=object()),
    )

    assert manager.begin_auto_compile() is True
    assert manager.begin_auto_compile() is False
