"""DSPy program and evaluator for rules extraction against manual seeds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy
from pydantic import BaseModel, Field, ValidationError

from ..rag.synthesizer import bind_program_lm, build_dspy_lm
from .contracts import (
    RuleExtractionBundle,
    load_bundle_from_seed_records,
)
from .extraction import (
    CandidateCanonicalRule,
    CandidateRuleEdge,
    CandidateRuleExtractionBundle,
    CandidateSourcePassage,
    CandidateUncertainty,
    RuleExtractionComparison,
    RuleExtractionRequest,
    build_request_from_reference_bundle,
    compare_rule_extractions,
    normalize_candidate_bundle,
)


class ExtractionScoreReport(BaseModel):
    """Aggregate score report for one extraction bundle."""

    total_score: float
    source_passage_recall: float
    canonical_rule_recall: float
    edge_recall: float
    extra_record_penalty: float
    field_accuracy: float
    comparison: RuleExtractionComparison


@dataclass
class PreparedRuleExtractionCase:
    """Prepared DSPy-facing extraction case."""

    request: RuleExtractionRequest
    reference: RuleExtractionBundle
    example: dspy.Example


class RuleExtractionSignature(dspy.Signature):
    """Extract a bounded rules graph slice from scoped source passages."""

    seed_id: str = dspy.InputField(desc="Stable seed identifier for this pilot slice.")
    game_id: str = dspy.InputField(desc="Game identifier, for example 'seti'.")
    subsystem: str = dspy.InputField(desc="Bounded subsystem name for the extraction task.")
    expected_canonical_rule_ids: list[str] = dspy.InputField(
        desc="Canonical rule IDs that should be reused for this frozen pilot when the evidence supports them."
    )
    expected_edge_ids: list[str] = dspy.InputField(
        desc="Edge IDs that should be reused for this frozen pilot when the supported relationship matches."
    )
    allowed_edge_types: list[str] = dspy.InputField(desc="The only permitted edge types.")
    frozen_questions: list[str] = dspy.InputField(
        desc="Pilot questions the extracted rules must support."
    )
    source_documents_json: str = dspy.InputField(
        desc="JSON list of scoped source documents with citation metadata and text."
    )
    source_passages_json: str = dspy.OutputField(
        desc="JSON list of SourcePassage records. Preserve source IDs from the input."
    )
    canonical_rules_json: str = dspy.OutputField(
        desc=(
            "JSON list of CanonicalRule records. Keep rule units fine-grained and separate "
            "action base, prerequisite, clarification, and modifier rules."
        )
    )
    edges_json: str = dspy.OutputField(
        desc="JSON list of RuleEdge records using only the allowed edge types."
    )
    uncertainties_json: str = dspy.OutputField(
        desc="JSON list of ExtractionUncertainty records. Use this instead of guessing."
    )


class RuleExtractionProgram(dspy.Module):
    """DSPy module for rule extraction."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(RuleExtractionSignature)

    def forward(
        self,
        *,
        seed_id: str,
        game_id: str,
        subsystem: str,
        expected_canonical_rule_ids: list[str],
        expected_edge_ids: list[str],
        allowed_edge_types: list[str],
        frozen_questions: list[str],
        source_documents_json: str,
    ) -> Any:
        return self.predictor(
            seed_id=seed_id,
            game_id=game_id,
            subsystem=subsystem,
            expected_canonical_rule_ids=expected_canonical_rule_ids,
            expected_edge_ids=expected_edge_ids,
            allowed_edge_types=allowed_edge_types,
            frozen_questions=frozen_questions,
            source_documents_json=source_documents_json,
        )


def make_rule_extraction_program() -> RuleExtractionProgram:
    """Create a fresh DSPy rule-extraction module."""

    return RuleExtractionProgram()


class DspyRuleExtractor:
    """Wrap a DSPy program in an extraction-friendly interface."""

    def __init__(self, predictor: Any, *, backend_name: str = "dspy-baseline"):
        self.predictor = predictor
        self.backend_name = backend_name

    @classmethod
    def from_lm(cls, lm: Any) -> "DspyRuleExtractor":
        program = make_rule_extraction_program()
        bind_program_lm(program, lm)
        return cls(program)

    @classmethod
    def from_program(
        cls,
        program: Any,
        *,
        backend_name: str = "dspy-compiled",
    ) -> "DspyRuleExtractor":
        return cls(program, backend_name=backend_name)

    def extract(self, request: RuleExtractionRequest) -> RuleExtractionBundle:
        prediction = self.predictor(
            seed_id=request.seed_id,
            game_id=request.game_id,
            subsystem=request.subsystem,
            expected_canonical_rule_ids=request.expected_canonical_rule_ids,
            expected_edge_ids=request.expected_edge_ids,
            allowed_edge_types=request.allowed_edge_types,
            frozen_questions=request.frozen_questions,
            source_documents_json=json.dumps(
                [document.model_dump(mode="json") for document in request.source_documents],
                ensure_ascii=True,
            ),
        )
        return normalize_candidate_bundle(
            CandidateRuleExtractionBundle(
                seed_id=request.seed_id,
                game_id=request.game_id,
                scope=request.scope,
                subsystem=request.subsystem,
                project_id=request.project_id,
                source=request.source,
                source_passages=_load_json_records(
                    getattr(prediction, "source_passages_json", "[]"),
                    CandidateSourcePassage,
                ),
                canonical_rules=_load_json_records(
                    getattr(prediction, "canonical_rules_json", "[]"),
                    CandidateCanonicalRule,
                ),
                edges=_load_json_records(
                    getattr(prediction, "edges_json", "[]"),
                    CandidateRuleEdge,
                ),
                uncertainties=_load_candidate_uncertainties(
                    getattr(prediction, "uncertainties_json", "[]")
                ),
            ),
            request,
        )


class RuleExtractionEvaluator:
    """Evaluate extraction output against one or more reference bundles."""

    def __init__(self, prepared_cases: list[PreparedRuleExtractionCase]):
        if not prepared_cases:
            raise ValueError("At least one prepared extraction case is required")
        self.prepared_cases = prepared_cases
        self.trainset, self.valset = self._build_dspy_splits(prepared_cases)

    @classmethod
    def from_seed_manifest(cls, manifest_path: str | Path) -> "RuleExtractionEvaluator":
        return cls.from_manifest_paths([manifest_path])

    @classmethod
    def from_manifest_paths(
        cls,
        manifest_paths: list[str | Path],
    ) -> "RuleExtractionEvaluator":
        return cls(
            [cls._prepare_case_from_manifest_path(Path(manifest_path)) for manifest_path in manifest_paths]
        )

    @classmethod
    def from_seed_suite(cls, suite_path: str | Path) -> "RuleExtractionEvaluator":
        suite_path = Path(suite_path)
        suite = json.loads(suite_path.read_text())
        seed_manifests = suite.get("seed_manifests", [])
        if not seed_manifests:
            raise ValueError(f"Seed suite {suite_path} does not define any seed_manifests")
        return cls.from_manifest_paths(seed_manifests)

    def metric(self, example: dspy.Example, prediction: Any, trace: Any = None) -> float:
        del trace
        prepared_case = self._prepared_case_from_example(example)
        try:
            actual = normalize_candidate_bundle(
                CandidateRuleExtractionBundle(
                    seed_id=prepared_case.request.seed_id,
                    game_id=prepared_case.request.game_id,
                    scope=prepared_case.request.scope,
                    subsystem=prepared_case.request.subsystem,
                    project_id=prepared_case.request.project_id,
                    source=prepared_case.request.source,
                    source_passages=_load_json_records(
                        getattr(prediction, "source_passages_json", "[]"),
                        CandidateSourcePassage,
                    ),
                    canonical_rules=_load_json_records(
                        getattr(prediction, "canonical_rules_json", "[]"),
                        CandidateCanonicalRule,
                    ),
                    edges=_load_json_records(
                        getattr(prediction, "edges_json", "[]"),
                        CandidateRuleEdge,
                    ),
                    uncertainties=_load_candidate_uncertainties(
                        getattr(prediction, "uncertainties_json", "[]")
                    ),
                ),
                prepared_case.request,
            )
        except (json.JSONDecodeError, TypeError, ValidationError, ValueError):
            return 0.0
        return self.score_bundle(prepared_case.reference, actual).total_score

    @staticmethod
    def score_bundle(
        expected: RuleExtractionBundle,
        actual: RuleExtractionBundle,
    ) -> ExtractionScoreReport:
        comparison = compare_rule_extractions(expected, actual)
        source_recall = _recall_score(
            expected_total=len(expected.source_passages),
            missing=len(comparison.missing_source_passage_ids),
        )
        rule_recall = _recall_score(
            expected_total=len(expected.canonical_rules),
            missing=len(comparison.missing_canonical_rule_ids),
        )
        edge_recall = _recall_score(
            expected_total=len(expected.edges),
            missing=len(comparison.missing_edge_ids),
        )
        extra_total = (
            len(comparison.extra_source_passage_ids)
            + len(comparison.extra_canonical_rule_ids)
            + len(comparison.extra_edge_ids)
        )
        expected_total = max(1, len(expected.nodes) + len(expected.edges))
        extra_penalty = max(0.0, 1.0 - (extra_total / expected_total))
        comparable_field_total = (
            (len(expected.source_passages) - len(comparison.missing_source_passage_ids)) * 9
            + (len(expected.canonical_rules) - len(comparison.missing_canonical_rule_ids)) * 4
            + (len(expected.edges) - len(comparison.missing_edge_ids)) * 4
        )
        field_accuracy = 1.0
        if comparable_field_total > 0:
            field_accuracy = max(
                0.0,
                1.0 - (len(comparison.field_mismatches) / comparable_field_total),
            )

        total_score = (
            0.20 * source_recall
            + 0.35 * rule_recall
            + 0.30 * edge_recall
            + 0.10 * extra_penalty
            + 0.05 * field_accuracy
        )
        return ExtractionScoreReport(
            total_score=total_score,
            source_passage_recall=source_recall,
            canonical_rule_recall=rule_recall,
            edge_recall=edge_recall,
            extra_record_penalty=extra_penalty,
            field_accuracy=field_accuracy,
            comparison=comparison,
        )

    @staticmethod
    def _build_dspy_splits(
        prepared_cases: list[PreparedRuleExtractionCase],
    ) -> tuple[list[dspy.Example], list[dspy.Example]]:
        examples = [case.example for case in prepared_cases]
        return examples, examples[:]

    def _prepared_case_from_example(self, example: dspy.Example) -> PreparedRuleExtractionCase:
        for prepared_case in self.prepared_cases:
            if prepared_case.request.seed_id == example.seed_id:
                return prepared_case
        raise KeyError(f"No prepared extraction case for seed_id={example.seed_id}")

    @staticmethod
    def _prepare_case_from_manifest_path(manifest_path: Path) -> PreparedRuleExtractionCase:
        manifest = json.loads(manifest_path.read_text())
        node_records = _load_jsonl(Path(manifest["node_file"]))
        edge_records = _load_jsonl(Path(manifest["edge_file"]))
        reference = load_bundle_from_seed_records(manifest, node_records, edge_records)
        request = build_request_from_reference_bundle(reference)
        example = dspy.Example(
            seed_id=request.seed_id,
            game_id=request.game_id,
            subsystem=request.subsystem,
            expected_canonical_rule_ids=request.expected_canonical_rule_ids,
            expected_edge_ids=request.expected_edge_ids,
            allowed_edge_types=request.allowed_edge_types,
            frozen_questions=request.frozen_questions,
            source_documents_json=json.dumps(
                [document.model_dump(mode="json") for document in request.source_documents],
                ensure_ascii=True,
            ),
        ).with_inputs(
            "seed_id",
            "game_id",
            "subsystem",
            "expected_canonical_rule_ids",
            "expected_edge_ids",
            "allowed_edge_types",
            "frozen_questions",
            "source_documents_json",
        )
        return PreparedRuleExtractionCase(request=request, reference=reference, example=example)


def compile_rule_extractor(
    *,
    llm_model: str,
    evaluator: RuleExtractionEvaluator,
    compiler_factory: Any | None = None,
) -> Any:
    """Compile a DSPy rule extractor against the prepared extraction evaluator."""

    lm = build_dspy_lm(llm_model)
    compiler = compiler_factory or dspy.MIPROv2(
        metric=evaluator.metric,
        prompt_model=lm,
        task_model=lm,
        auto="light",
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_threads=1,
        verbose=False,
    )
    program = make_rule_extraction_program()
    bind_program_lm(program, lm)
    with dspy.context(lm=lm):
        compiled_program = compiler.compile(
            program,
            trainset=evaluator.trainset,
            valset=evaluator.valset,
        )
    bind_program_lm(compiled_program, lm)
    return compiled_program


def _recall_score(*, expected_total: int, missing: int) -> float:
    if expected_total <= 0:
        return 1.0
    return max(0.0, (expected_total - missing) / expected_total)


def _load_json_records(raw_json: str, model_type: Any) -> list[Any]:
    data = json.loads(raw_json or "[]")
    return [model_type.model_validate(item) for item in data]


def _load_candidate_uncertainties(
    raw_json: str,
) -> list[CandidateUncertainty] | dict[str, Any]:
    data = json.loads(raw_json or "[]")
    if isinstance(data, dict):
        return data
    return [CandidateUncertainty.model_validate(item) for item in data]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
