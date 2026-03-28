"""Offline evaluation helpers for DSPy-backed RAG synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal

import dspy
from pydantic import BaseModel, Field

from .eval_dataset import RagEvalCase, RagEvalRetrievalRef, load_rag_eval_cases
from .eval_judge import RagEvalJudge, RagEvalJudgeResult
from .eval_scoring import RagEvalComponentScores, compute_rag_eval_score
from .models import RagContext, RagSynthesisResult, RetrievedPassage

DEFAULT_EVAL_SUITE_PATH = Path("tests/fixtures/rag_eval/seti_rules_reference_v1.jsonl")
DEFAULT_EVAL_SOURCE_DIR = Path("tests/fixtures/rag_eval/source_documents/extracted")
DEFAULT_EVAL_SOURCE_FILES = {
    "seti-rules-en": "seti-rules-en.txt",
    "seti-faq": "seti-faq.txt",
    "seti-player-aid-en": "seti-player-aid-en.txt",
    "seti-alien-species-en": "seti-alien-species-en.txt",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "later",
    "may",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "this",
    "to",
    "up",
    "when",
    "with",
    "you",
    "your",
}
_NEGATION_TOKENS = {"no", "not", "never", "cannot", "cant", "without"}


@dataclass
class PreparedRagEvalCase:
    """Prepared eval case with both typed context and DSPy example views."""

    case: RagEvalCase
    context: RagContext
    example: dspy.Example


class RagEvalCaseScore(BaseModel):
    """Scored result for one eval case."""

    case_id: str
    backend: str
    answer: str
    cited_source_ids: list[str] = Field(default_factory=list)
    abstain: bool
    groundedness: float
    abstention: float
    source_alignment: float
    completeness: float
    total_score: float
    bucket_labels: list[str] = Field(default_factory=list)
    score_details: dict[str, Any] = Field(default_factory=dict)


class RagEvalBucketReport(BaseModel):
    """Aggregate report for one eval bucket."""

    bucket_name: str
    num_cases: int
    average_groundedness: float
    average_abstention: float
    average_source_alignment: float
    average_completeness: float
    total_score: float


class RagEvalReport(BaseModel):
    """Aggregate report across one eval suite."""

    suite_id: str
    backend: str
    case_results: list[RagEvalCaseScore] = Field(default_factory=list)
    bucket_reports: dict[str, RagEvalBucketReport] = Field(default_factory=dict)
    average_groundedness: float
    average_abstention: float
    average_source_alignment: float
    average_completeness: float
    total_score: float


class RagEvalTraceEntry(BaseModel):
    """Detailed per-case trace for a logged eval run."""

    case_id: str
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_refs: list[dict[str, Any]] = Field(default_factory=list)
    rag_context: dict[str, Any] = Field(default_factory=dict)
    rubric: dict[str, Any] = Field(default_factory=dict)
    result: RagEvalCaseScore


class LocalEvalSourceResolver:
    """Resolve frozen eval retrieval references into exact local passages."""

    def __init__(
        self,
        source_dir: str | Path = DEFAULT_EVAL_SOURCE_DIR,
        source_files: dict[str, str] | None = None,
        excerpt_char_window: int = 500,
    ):
        self.source_dir = Path(source_dir)
        self.source_files = source_files or DEFAULT_EVAL_SOURCE_FILES
        self.excerpt_char_window = excerpt_char_window
        self._pages_cache: dict[str, list[str]] = {}

    def resolve_passage(self, retrieval_ref: RagEvalRetrievalRef) -> RetrievedPassage:
        """Resolve one retrieval reference into an excerpt-backed passage."""

        page_text = self._load_page(retrieval_ref.document_id, retrieval_ref.page)
        excerpt = self._extract_excerpt(page_text, retrieval_ref.locator)
        return RetrievedPassage(
            node_id=retrieval_ref.source_id,
            node_type=retrieval_ref.document_type,
            content=excerpt,
            similarity_score=1.0 if retrieval_ref.priority == "primary" else 0.9,
            metadata={
                "document_id": retrieval_ref.document_id,
                "authority_scope": retrieval_ref.authority_scope,
                "page": retrieval_ref.page,
                "locator": retrieval_ref.locator,
                "priority": retrieval_ref.priority,
            },
        )

    def _load_page(self, document_id: str, page_number: int) -> str:
        if document_id not in self._pages_cache:
            filename = self.source_files.get(document_id)
            if filename is None:
                raise FileNotFoundError(f"No local source file mapping configured for {document_id}")
            document_path = self.source_dir / filename
            if not document_path.exists():
                raise FileNotFoundError(
                    f"Expected local eval source document at {document_path}"
                )
            self._pages_cache[document_id] = document_path.read_text(errors="ignore").split(
                "\f"
            )
        pages = self._pages_cache[document_id]
        if page_number < 1 or page_number > len(pages):
            raise ValueError(
                f"Document {document_id} does not have page {page_number}; page count is {len(pages)}"
            )
        return pages[page_number - 1]

    def _extract_excerpt(self, page_text: str, locator: str) -> str:
        normalized_page = " ".join(page_text.split())
        locator_index = normalized_page.find(locator)
        if locator_index == -1:
            return normalized_page[: self.excerpt_char_window * 2].strip()

        start = max(0, locator_index - self.excerpt_char_window)
        end = min(len(normalized_page), locator_index + len(locator) + self.excerpt_char_window)
        excerpt = normalized_page[start:end].strip()
        if start > 0:
            excerpt = f"... {excerpt}"
        if end < len(normalized_page):
            excerpt = f"{excerpt} ..."
        return excerpt


class RubricRagEvaluator:
    """Evaluate DSPy synthesizer outputs against rubric-driven cases."""

    def __init__(
        self,
        prepared_cases: list[PreparedRagEvalCase],
        *,
        judge: RagEvalJudge | None = None,
        scoring_mode: Literal["deterministic", "hybrid"] = "deterministic",
    ):
        if not prepared_cases:
            raise ValueError("At least one prepared eval case is required")
        if scoring_mode == "hybrid" and judge is None:
            raise ValueError("Hybrid scoring mode requires a judge")
        self.prepared_cases = prepared_cases
        self.prepared_cases_by_id = {
            prepared_case.case.case_id: prepared_case for prepared_case in prepared_cases
        }
        self.judge = judge
        self.scoring_mode = scoring_mode
        self.suite_id = prepared_cases[0].case.suite_id
        self.trainset, self.valset = self._build_dspy_splits(prepared_cases)

    @classmethod
    def from_suite(
        cls,
        *,
        suite_path: str | Path = DEFAULT_EVAL_SUITE_PATH,
        source_dir: str | Path = DEFAULT_EVAL_SOURCE_DIR,
        use_case_description: str,
        project_id: str,
        resolver: LocalEvalSourceResolver | None = None,
        judge: RagEvalJudge | None = None,
        scoring_mode: Literal["deterministic", "hybrid"] = "deterministic",
    ) -> "RubricRagEvaluator":
        """Load, resolve, and prepare a full eval suite from disk."""

        loaded_cases = load_rag_eval_cases(suite_path)
        source_resolver = resolver or LocalEvalSourceResolver(source_dir=source_dir)
        prepared_cases = [
            cls._prepare_case(case, source_resolver, use_case_description, project_id)
            for case in loaded_cases
        ]
        return cls(
            prepared_cases=prepared_cases,
            judge=judge,
            scoring_mode=scoring_mode,
        )

    def evaluate_synthesizer(self, synthesizer: Any) -> RagEvalReport:
        """Run the full eval suite against one synthesizer."""

        report, _ = self.evaluate_synthesizer_with_trace(synthesizer)
        return report

    def evaluate_synthesizer_with_trace(
        self,
        synthesizer: Any,
    ) -> tuple[RagEvalReport, list[RagEvalTraceEntry]]:
        """Run the full eval suite and return both the aggregate report and per-case traces."""

        case_results: list[RagEvalCaseScore] = []
        trace_entries: list[RagEvalTraceEntry] = []
        for prepared_case in self.prepared_cases:
            synthesis_result = synthesizer.synthesize(prepared_case.context)
            scored_result = self._score_result(prepared_case, synthesis_result)
            case_results.append(scored_result)
            trace_entries.append(
                RagEvalTraceEntry(
                    case_id=prepared_case.case.case_id,
                    conversation=[
                        turn.model_dump(mode="json") for turn in prepared_case.case.conversation
                    ],
                    retrieval_refs=[
                        retrieval_ref.model_dump(mode="json")
                        for retrieval_ref in prepared_case.case.retrieval_refs
                    ],
                    rag_context={
                        "session_id": prepared_case.context.session_id,
                        "project_id": prepared_case.context.project_id,
                        "use_case_description": prepared_case.context.use_case_description,
                        "current_question": prepared_case.context.current_question,
                        "retrieval_query": prepared_case.context.retrieval_query,
                        "conversation_history": [
                            turn.model_dump(mode="json")
                            for turn in prepared_case.context.conversation_history
                        ],
                        "retrieved_passages": [
                            passage.model_dump(mode="json")
                            for passage in prepared_case.context.retrieved_passages
                        ],
                        "graph_facts": [],
                    },
                    rubric=prepared_case.case.rubric.model_dump(mode="json"),
                    result=scored_result,
                )
            )

        report = self._build_report(
            backend=case_results[0].backend if case_results else "unknown",
            case_results=case_results,
        )
        return report, trace_entries

    def metric(self, example: dspy.Example, prediction: Any, trace: Any = None) -> float:
        """DSPy-compatible metric callback for compilation/evaluation."""
        del trace
        prepared_case = self._prepared_case_from_example(example)
        result = self._result_from_prediction(prediction)
        return self._score_result(prepared_case, result).total_score

    @staticmethod
    def _prepare_case(
        case: RagEvalCase,
        resolver: LocalEvalSourceResolver,
        use_case_description: str,
        project_id: str,
    ) -> PreparedRagEvalCase:
        history = case.conversation[:-1]
        current_question = case.conversation[-1].content.strip()
        passages = [resolver.resolve_passage(retrieval_ref) for retrieval_ref in case.retrieval_refs]
        context = RagContext(
            session_id=f"rag-eval:{case.case_id}",
            project_id=project_id,
            use_case_description=use_case_description,
            current_question=current_question,
            retrieval_query=current_question,
            conversation_history=history,
            retrieved_passages=passages,
            graph_facts=[],
        )
        example = dspy.Example(
            case_id=case.case_id,
            suite_id=case.suite_id,
            conversation_history=[
                f"{turn.role}: {turn.content.strip()}"
                for turn in context.conversation_history
                if turn.content.strip()
            ],
            question=context.current_question,
            passages=[
                (
                    f"[source_id={passage.node_id}] "
                    f"[node_type={passage.node_type}] "
                    f"{passage.content}"
                )
                for passage in context.retrieved_passages
            ],
            graph_facts=[],
            use_case=context.use_case_description,
            expected_abstain=case.rubric.expected_abstain,
            must_include=case.rubric.must_include,
            must_not_include=case.rubric.must_not_include,
            preferred_source_id=case.rubric.preferred_source_id,
            abstention_reason=case.rubric.abstention_reason,
        ).with_inputs(
            "conversation_history",
            "question",
            "passages",
            "graph_facts",
            "use_case",
        )
        return PreparedRagEvalCase(case=case, context=context, example=example)

    @staticmethod
    def _build_dspy_splits(
        prepared_cases: list[PreparedRagEvalCase],
    ) -> tuple[list[dspy.Example], list[dspy.Example]]:
        trainset: list[dspy.Example] = []
        valset: list[dspy.Example] = []
        for index, prepared_case in enumerate(prepared_cases):
            target = valset if index % 5 == 4 else trainset
            target.append(prepared_case.example)
        if not valset:
            valset = trainset[:]
        return trainset, valset

    def _build_report(
        self,
        *,
        backend: str,
        case_results: list[RagEvalCaseScore],
    ) -> RagEvalReport:
        count = len(case_results)
        return RagEvalReport(
            suite_id=self.suite_id,
            backend=backend,
            case_results=case_results,
            bucket_reports=self._build_bucket_reports(case_results),
            average_groundedness=sum(result.groundedness for result in case_results) / count,
            average_abstention=sum(result.abstention for result in case_results) / count,
            average_source_alignment=sum(result.source_alignment for result in case_results)
            / count,
            average_completeness=sum(result.completeness for result in case_results) / count,
            total_score=sum(result.total_score for result in case_results) / count,
        )

    def _build_bucket_reports(
        self,
        case_results: list[RagEvalCaseScore],
    ) -> dict[str, RagEvalBucketReport]:
        bucketed_results: dict[str, list[RagEvalCaseScore]] = {}
        for result in case_results:
            for bucket_name in result.bucket_labels:
                bucketed_results.setdefault(bucket_name, []).append(result)

        bucket_reports: dict[str, RagEvalBucketReport] = {}
        for bucket_name, bucket_case_results in bucketed_results.items():
            count = len(bucket_case_results)
            bucket_reports[bucket_name] = RagEvalBucketReport(
                bucket_name=bucket_name,
                num_cases=count,
                average_groundedness=sum(
                    result.groundedness for result in bucket_case_results
                )
                / count,
                average_abstention=sum(
                    result.abstention for result in bucket_case_results
                )
                / count,
                average_source_alignment=sum(
                    result.source_alignment for result in bucket_case_results
                )
                / count,
                average_completeness=sum(
                    result.completeness for result in bucket_case_results
                )
                / count,
                total_score=sum(result.total_score for result in bucket_case_results) / count,
            )
        return bucket_reports

    def _score_result(
        self,
        prepared_case: PreparedRagEvalCase,
        result: RagSynthesisResult,
    ) -> RagEvalCaseScore:
        case = prepared_case.case
        deterministic_groundedness = self._groundedness_score(case, result.answer)
        abstention = 1.0 if result.abstain == case.rubric.expected_abstain else 0.0
        source_alignment = self._source_alignment_score(case, result.cited_source_ids)
        deterministic_completeness = self._completeness_score(case, result)
        judge_result = self._judge_result(prepared_case, result)
        groundedness = deterministic_groundedness
        completeness = deterministic_completeness
        if judge_result is not None:
            groundedness = min(deterministic_groundedness, judge_result.groundedness)
            completeness = judge_result.completeness
        components = RagEvalComponentScores(
            groundedness=groundedness,
            abstention=abstention,
            source_alignment=source_alignment,
            completeness=completeness,
        )
        return RagEvalCaseScore(
            case_id=case.case_id,
            backend=result.backend,
            answer=result.answer,
            cited_source_ids=result.cited_source_ids,
            abstain=result.abstain,
            groundedness=components.groundedness,
            abstention=components.abstention,
            source_alignment=components.source_alignment,
            completeness=components.completeness,
            total_score=compute_rag_eval_score(components),
            bucket_labels=self._bucket_labels(case),
            score_details={
                "scoring_mode": self.scoring_mode,
                "deterministic_groundedness": deterministic_groundedness,
                "deterministic_completeness": deterministic_completeness,
                "judge_groundedness": (
                    None if judge_result is None else judge_result.groundedness
                ),
                "judge_completeness": (
                    None if judge_result is None else judge_result.completeness
                ),
                "judge_rationale": None if judge_result is None else judge_result.rationale,
                "judge_backend": None if judge_result is None else judge_result.backend,
            },
        )

    @staticmethod
    def _bucket_labels(case: RagEvalCase) -> list[str]:
        if "hard_mode" in case.tags:
            return ["hard_mode"]
        return ["standard"]

    @staticmethod
    def _groundedness_score(case: RagEvalCase, answer: str) -> float:
        normalized_answer = _normalize_text(answer)
        if not normalized_answer and not case.rubric.expected_abstain:
            return 0.0
        for forbidden_statement in case.rubric.must_not_include:
            if _statement_supported(normalized_answer, forbidden_statement, threshold=0.8):
                return 0.0
        return 1.0

    @staticmethod
    def _source_alignment_score(case: RagEvalCase, cited_source_ids: list[str]) -> float:
        if case.rubric.expected_abstain:
            return 1.0 if not cited_source_ids else 0.5
        if not cited_source_ids:
            return 0.0
        return 1.0 if case.rubric.preferred_source_id in cited_source_ids else 0.0

    @staticmethod
    def _completeness_score(case: RagEvalCase, result: RagSynthesisResult) -> float:
        if case.rubric.expected_abstain:
            return 1.0 if result.abstain else 0.0
        if not case.rubric.must_include:
            return 1.0
        supported = [
            _statement_supported(result.answer, required_statement, threshold=0.65)
            for required_statement in case.rubric.must_include
        ]
        return sum(1.0 for item in supported if item) / len(supported)

    def _judge_result(
        self,
        prepared_case: PreparedRagEvalCase,
        result: RagSynthesisResult,
    ) -> RagEvalJudgeResult | None:
        if self.judge is None or self.scoring_mode != "hybrid":
            return None
        if prepared_case.case.rubric.expected_abstain or result.abstain:
            return None
        if not result.answer.strip():
            return None
        return self.judge.judge(
            case=prepared_case.case,
            context=prepared_case.context,
            result=result,
        )

    def _prepared_case_from_example(self, example: dspy.Example) -> PreparedRagEvalCase:
        return self.prepared_cases_by_id[example.case_id]

    @staticmethod
    def _result_from_prediction(prediction: Any) -> RagSynthesisResult:
        cited_source_ids = getattr(prediction, "cited_source_ids", [])
        if isinstance(cited_source_ids, str):
            cited_source_ids = [cited_source_ids]
        elif cited_source_ids is None:
            cited_source_ids = []
        return RagSynthesisResult(
            answer=str(getattr(prediction, "answer", "")).strip(),
            cited_source_ids=[str(source_id) for source_id in cited_source_ids],
            abstain=bool(getattr(prediction, "abstain", False)),
            backend="dspy-compiled",
        )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _meaningful_tokens(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(_normalize_text(text))
        if token not in _STOP_WORDS
    ]


def _statement_supported(answer: str, statement: str, *, threshold: float) -> bool:
    normalized_answer = _normalize_text(answer)
    normalized_statement = _normalize_text(statement)
    if normalized_statement and normalized_statement in normalized_answer:
        return True

    answer_tokens = set(_meaningful_tokens(normalized_answer))
    statement_tokens = _meaningful_tokens(normalized_statement)
    if not statement_tokens:
        return True

    coverage = len(answer_tokens.intersection(statement_tokens)) / len(statement_tokens)
    if _has_negation(answer_tokens) != _has_negation(set(statement_tokens)):
        coverage *= 0.5
    return coverage >= threshold


def _has_negation(tokens: set[str]) -> bool:
    return bool(tokens.intersection(_NEGATION_TOKENS))
