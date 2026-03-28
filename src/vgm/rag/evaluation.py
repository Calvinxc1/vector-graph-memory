"""Offline evaluation helpers for DSPy-backed RAG synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import dspy
from pydantic import BaseModel, Field

from .eval_dataset import RagEvalCase, RagEvalRetrievalRef, load_rag_eval_cases
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


class RagEvalReport(BaseModel):
    """Aggregate report across one eval suite."""

    suite_id: str
    backend: str
    case_results: list[RagEvalCaseScore] = Field(default_factory=list)
    average_groundedness: float
    average_abstention: float
    average_source_alignment: float
    average_completeness: float
    total_score: float


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

    def __init__(self, prepared_cases: list[PreparedRagEvalCase]):
        if not prepared_cases:
            raise ValueError("At least one prepared eval case is required")
        self.prepared_cases = prepared_cases
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
    ) -> "RubricRagEvaluator":
        """Load, resolve, and prepare a full eval suite from disk."""

        loaded_cases = load_rag_eval_cases(suite_path)
        source_resolver = resolver or LocalEvalSourceResolver(source_dir=source_dir)
        prepared_cases = [
            cls._prepare_case(case, source_resolver, use_case_description, project_id)
            for case in loaded_cases
        ]
        return cls(prepared_cases=prepared_cases)

    def evaluate_synthesizer(self, synthesizer: Any) -> RagEvalReport:
        """Run the full eval suite against one synthesizer."""

        case_results = [
            self._score_result(prepared_case.case, synthesizer.synthesize(prepared_case.context))
            for prepared_case in self.prepared_cases
        ]
        return self._build_report(
            backend=case_results[0].backend if case_results else "unknown",
            case_results=case_results,
        )

    def metric(self, example: dspy.Example, prediction: Any, trace: Any = None) -> float:
        """DSPy-compatible metric callback for compilation/evaluation."""
        del trace
        case = self._case_from_example(example)
        result = self._result_from_prediction(prediction)
        return self._score_result(case, result).total_score

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
            average_groundedness=sum(result.groundedness for result in case_results) / count,
            average_abstention=sum(result.abstention for result in case_results) / count,
            average_source_alignment=sum(result.source_alignment for result in case_results)
            / count,
            average_completeness=sum(result.completeness for result in case_results) / count,
            total_score=sum(result.total_score for result in case_results) / count,
        )

    def _score_result(self, case: RagEvalCase, result: RagSynthesisResult) -> RagEvalCaseScore:
        groundedness = self._groundedness_score(case, result.answer)
        abstention = 1.0 if result.abstain == case.rubric.expected_abstain else 0.0
        source_alignment = self._source_alignment_score(case, result.cited_source_ids)
        completeness = self._completeness_score(case, result)
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
        )

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

    def _case_from_example(self, example: dspy.Example) -> RagEvalCase:
        return next(
            prepared_case.case
            for prepared_case in self.prepared_cases
            if prepared_case.case.case_id == example.case_id
        )

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
