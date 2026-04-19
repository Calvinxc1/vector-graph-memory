"""Typed evaluation helpers for the live SETI pilot ruling path."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from .rulings import LivePilotRulingEngine, RulesRulingRequest, RulesRulingResult


class PilotRulingEvalCase(BaseModel):
    """One frozen evaluation case for the live pilot ruling path."""

    case_id: str
    suite_id: str
    seed_id: str | None = None
    question: str
    expected_abstain: bool = False
    expected_inferred_seed_id: str | None = None
    expected_selected_seed_id: str | None = None
    expected_question_id: str | None = None
    expected_retrieved_node_ids: list[str] = Field(default_factory=list)
    expected_expanded_node_ids: list[str] = Field(default_factory=list)
    expected_primary_rule_id: str | None = None
    expected_primary_source_id: str | None = None
    expected_modifying_rule_ids: list[str] = Field(default_factory=list)
    expected_modifying_source_ids: list[str] = Field(default_factory=list)
    expected_precedence_kinds: list[str] = Field(default_factory=list)
    retrieval_limit: int = Field(default=6, ge=1, le=20)
    notes: str = ""


class PilotRulingEvalComponentScores(BaseModel):
    """Component scores for one evaluation case."""

    retrieval_nodes: float
    expanded_evidence: float
    seed_inference: float
    case_selection: float
    primary_citation: float
    modifier_selection: float
    precedence_assembly: float


class PilotRulingEvalCaseReport(BaseModel):
    """Detailed evaluation result for one frozen ruling case."""

    case_id: str
    question: str
    expected_abstain: bool
    actual_abstain: bool
    component_scores: PilotRulingEvalComponentScores
    total_score: float
    actual_retrieved_node_ids: list[str] = Field(default_factory=list)
    actual_expanded_node_ids: list[str] = Field(default_factory=list)
    actual_seed_inference_id: str | None = None
    actual_selected_seed_id: str | None = None
    actual_question_id: str | None = None
    actual_primary_rule_id: str | None = None
    actual_primary_source_id: str | None = None
    actual_modifying_rule_ids: list[str] = Field(default_factory=list)
    actual_modifying_source_ids: list[str] = Field(default_factory=list)
    actual_precedence_kinds: list[str] = Field(default_factory=list)


class PilotRulingEvalReport(BaseModel):
    """Aggregate report across one frozen live-ruling eval suite."""

    suite_id: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    average_total_score: float
    average_retrieval_nodes: float
    average_expanded_evidence: float
    average_seed_inference: float
    average_case_selection: float
    average_primary_citation: float
    average_modifier_selection: float
    average_precedence_assembly: float
    cases: list[PilotRulingEvalCaseReport]


def load_pilot_ruling_eval_cases(path: str | Path) -> list[PilotRulingEvalCase]:
    """Load newline-delimited JSON pilot ruling eval cases from disk."""

    suite_path = Path(path)
    cases = [
        PilotRulingEvalCase.model_validate_json(line)
        for line in suite_path.read_text().splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError(f"Eval suite {suite_path} did not contain any cases")
    return cases


class PilotRulingEvaluator:
    """Evaluate the live pilot ruling path against frozen expectations."""

    def __init__(self, cases: list[PilotRulingEvalCase]):
        if not cases:
            raise ValueError("At least one pilot ruling eval case is required")
        self.cases = cases
        self.suite_id = cases[0].suite_id

    @classmethod
    def from_suite(cls, path: str | Path) -> "PilotRulingEvaluator":
        return cls(load_pilot_ruling_eval_cases(path))

    def evaluate_engine(self, engine: LivePilotRulingEngine) -> PilotRulingEvalReport:
        case_reports = [self._evaluate_case(engine, case) for case in self.cases]
        total_cases = len(case_reports)
        passed_cases = sum(1 for report in case_reports if report.total_score >= 0.999)
        return PilotRulingEvalReport(
            suite_id=self.suite_id,
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=total_cases - passed_cases,
            average_total_score=_average(report.total_score for report in case_reports),
            average_retrieval_nodes=_average(
                report.component_scores.retrieval_nodes for report in case_reports
            ),
            average_expanded_evidence=_average(
                report.component_scores.expanded_evidence for report in case_reports
            ),
            average_seed_inference=_average(
                report.component_scores.seed_inference for report in case_reports
            ),
            average_case_selection=_average(
                report.component_scores.case_selection for report in case_reports
            ),
            average_primary_citation=_average(
                report.component_scores.primary_citation for report in case_reports
            ),
            average_modifier_selection=_average(
                report.component_scores.modifier_selection for report in case_reports
            ),
            average_precedence_assembly=_average(
                report.component_scores.precedence_assembly for report in case_reports
            ),
            cases=case_reports,
        )

    def _evaluate_case(
        self,
        engine: LivePilotRulingEngine,
        case: PilotRulingEvalCase,
    ) -> PilotRulingEvalCaseReport:
        request = RulesRulingRequest(
            question=case.question,
            seed_id=case.seed_id,
            retrieval_limit=case.retrieval_limit,
        )
        inspection = engine.inspect_request(request)
        result = engine.answer(request)

        component_scores = PilotRulingEvalComponentScores(
            retrieval_nodes=_required_subset_score(
                case.expected_retrieved_node_ids,
                inspection.evidence.retrieved_node_ids,
            ),
            expanded_evidence=_required_subset_score(
                case.expected_expanded_node_ids,
                inspection.evidence.expanded_node_ids,
            ),
            seed_inference=_binary_score(
                case.expected_inferred_seed_id,
                inspection.seed_inference.selected_seed_id,
            ),
            case_selection=_score_case_selection(case, inspection.selected_seed_id, result.question_id),
            primary_citation=_score_primary_citation(case, result),
            modifier_selection=_score_modifier_selection(
                case,
                actual_rule_ids=sorted(rule.rule_node_id for rule in result.modifying_rules),
                actual_source_ids=sorted(citation.source_node_id for citation in result.modifying_citations),
            ),
            precedence_assembly=_binary_score(
                case.expected_precedence_kinds,
                [entry.precedence_kind for entry in result.precedence_order],
            ),
        )
        total_score = (
            component_scores.retrieval_nodes
            + component_scores.expanded_evidence
            + component_scores.seed_inference
            + component_scores.case_selection
            + component_scores.primary_citation
            + component_scores.modifier_selection
            + component_scores.precedence_assembly
        ) / 7.0
        return PilotRulingEvalCaseReport(
            case_id=case.case_id,
            question=case.question,
            expected_abstain=case.expected_abstain,
            actual_abstain=result.abstain,
            component_scores=component_scores,
            total_score=total_score,
            actual_retrieved_node_ids=inspection.evidence.retrieved_node_ids,
            actual_expanded_node_ids=inspection.evidence.expanded_node_ids,
            actual_seed_inference_id=inspection.seed_inference.selected_seed_id,
            actual_selected_seed_id=inspection.selected_seed_id,
            actual_question_id=result.question_id,
            actual_primary_rule_id=(result.primary_rule.rule_node_id if result.primary_rule else None),
            actual_primary_source_id=(
                result.primary_citation.source_node_id if result.primary_citation else None
            ),
            actual_modifying_rule_ids=[rule.rule_node_id for rule in result.modifying_rules],
            actual_modifying_source_ids=[
                citation.source_node_id for citation in result.modifying_citations
            ],
            actual_precedence_kinds=[entry.precedence_kind for entry in result.precedence_order],
        )


def _score_case_selection(
    case: PilotRulingEvalCase,
    actual_selected_seed_id: str | None,
    actual_question_id: str,
) -> float:
    return float(
        actual_selected_seed_id == case.expected_selected_seed_id
        and actual_question_id == case.expected_question_id
    )


def _score_primary_citation(case: PilotRulingEvalCase, result: RulesRulingResult) -> float:
    primary_rule_id = result.primary_rule.rule_node_id if result.primary_rule else None
    primary_source_id = result.primary_citation.source_node_id if result.primary_citation else None
    return float(
        primary_rule_id == case.expected_primary_rule_id
        and primary_source_id == case.expected_primary_source_id
    )


def _score_modifier_selection(
    case: PilotRulingEvalCase,
    *,
    actual_rule_ids: list[str],
    actual_source_ids: list[str],
) -> float:
    return float(
        sorted(case.expected_modifying_rule_ids) == actual_rule_ids
        and sorted(case.expected_modifying_source_ids) == actual_source_ids
    )


def _binary_score(expected: object, actual: object) -> float:
    return 1.0 if expected == actual else 0.0


def _required_subset_score(expected: list[str], actual: list[str]) -> float:
    return 1.0 if set(expected).issubset(set(actual)) else 0.0


def _average(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)
