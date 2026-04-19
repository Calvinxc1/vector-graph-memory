"""Prompt-building, normalization, and comparison helpers for rules extraction."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic import AliasChoices, BaseModel, Field

from .contracts import RuleExtractionBundle, load_bundle_from_seed_records


class ScopedSourceDocument(BaseModel):
    """A bounded source fragment provided to the extraction runner."""

    source_id: str
    document_id: str
    document_type: str
    authority_scope: str
    title: str
    locator: str
    page: int | None = None
    text: str
    citation_label: str
    citation_short: str


class RuleExtractionRequest(BaseModel):
    """The complete structured request passed to a rules parser."""

    seed_id: str
    game_id: str
    scope: str = "rules_only"
    subsystem: str
    project_id: str
    source: str
    frozen_questions: list[str] = Field(default_factory=list)
    related_docs: list[str] = Field(default_factory=list)
    expected_canonical_rules: list["ExpectedCanonicalRule"] = Field(default_factory=list)
    expected_edges: list["ExpectedRuleEdge"] = Field(default_factory=list)
    expected_canonical_rule_ids: list[str] = Field(default_factory=list)
    expected_edge_ids: list[str] = Field(default_factory=list)
    allowed_edge_types: list[str] = Field(
        default_factory=lambda: [
            "supports",
            "clarifies",
            "modifies",
            "requires",
            "contributes_to",
            "applies_during",
            "conflicts_with",
        ]
    )
    source_documents: list[ScopedSourceDocument]


class ExpectedCanonicalRule(BaseModel):
    """Reference metadata for a canonical rule in a frozen pilot slice."""

    node_id: str
    rule_kind: str
    title: str
    normalized_statement: str
    scope: str


class ExpectedRuleEdge(BaseModel):
    """Reference metadata for an expected edge in a frozen pilot slice."""

    edge_id: str
    edge_type: str
    from_node_id: str
    to_node_id: str
    rationale: str


class CandidateSourcePassage(BaseModel):
    """Lenient typed model for LLM-emitted source passage records."""

    node_id: str = Field(validation_alias=AliasChoices("node_id", "source_id"))
    source_text: str = Field(
        validation_alias=AliasChoices("source_text", "text", "content")
    )
    game_id: str | None = None
    document_id: str | None = None
    document_type: str | None = None
    authority_scope: str | None = None
    title: str | None = None
    locator: str | None = None
    page: int | None = None
    citation_label: str | None = None
    citation_short: str | None = None
    language: str | None = None
    subsystem: str | None = None
    content: str | None = None


class CandidateCanonicalRule(BaseModel):
    """Lenient typed model for LLM-emitted canonical rule records."""

    node_id: str = Field(validation_alias=AliasChoices("node_id", "rule_id"))
    normalized_statement: str = Field(
        validation_alias=AliasChoices("normalized_statement", "text", "content")
    )
    game_id: str | None = None
    rule_kind: str | None = None
    title: str | None = None
    scope: str | None = None
    subsystem: str | None = None
    content: str | None = None


class CandidateRuleEdge(BaseModel):
    """Lenient typed model for LLM-emitted edge records."""

    edge_id: str
    from_node_id: str = Field(
        validation_alias=AliasChoices("from_node_id", "source_id")
    )
    to_node_id: str = Field(validation_alias=AliasChoices("to_node_id", "target_id"))
    edge_type: str | None = Field(
        default=None, validation_alias=AliasChoices("edge_type", "relationship_type")
    )
    rationale: str | None = None
    description: str | None = None
    confidence: float | None = None


class CandidateUncertainty(BaseModel):
    """Lenient typed model for LLM-emitted uncertainty records."""

    kind: str
    note: str
    source_ref: str | None = None
    related_ids: list[str] = Field(default_factory=list)


class CandidateRuleExtractionBundle(BaseModel):
    """Lenient typed model for rules-extraction LLM output."""

    seed_id: str
    game_id: str
    scope: str = "rules_only"
    subsystem: str
    project_id: str
    source: str
    source_passages: list[CandidateSourcePassage] = Field(default_factory=list)
    canonical_rules: list[CandidateCanonicalRule] = Field(default_factory=list)
    edges: list[CandidateRuleEdge] = Field(default_factory=list)
    uncertainties: list[CandidateUncertainty] | dict[str, Any] = Field(default_factory=list)


class RuleExtractionFieldMismatch(BaseModel):
    """One keyed mismatch between extracted output and the manual seed."""

    record_kind: str
    record_id: str
    field_name: str
    expected: Any
    actual: Any


class RuleExtractionComparison(BaseModel):
    """Comparison report between extracted output and a reference seed."""

    missing_source_passage_ids: list[str] = Field(default_factory=list)
    extra_source_passage_ids: list[str] = Field(default_factory=list)
    missing_canonical_rule_ids: list[str] = Field(default_factory=list)
    extra_canonical_rule_ids: list[str] = Field(default_factory=list)
    missing_edge_ids: list[str] = Field(default_factory=list)
    extra_edge_ids: list[str] = Field(default_factory=list)
    field_mismatches: list[RuleExtractionFieldMismatch] = Field(default_factory=list)

    @property
    def is_exact_match(self) -> bool:
        return not (
            self.missing_source_passage_ids
            or self.extra_source_passage_ids
            or self.missing_canonical_rule_ids
            or self.extra_canonical_rule_ids
            or self.missing_edge_ids
            or self.extra_edge_ids
            or self.field_mismatches
        )


def _stringify_request_payload(request: RuleExtractionRequest) -> str:
    return json.dumps(request.model_dump(mode="json"), indent=2, ensure_ascii=True)


def _find_duplicate_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


class RuleExtractionRunner:
    """Build prompts, invoke a parser callback, and validate the returned bundle."""

    def __init__(
        self,
        predictor: Callable[[str, str], RuleExtractionBundle | dict[str, Any]],
    ) -> None:
        self.predictor = predictor

    def build_system_prompt(self, request: RuleExtractionRequest) -> str:
        edge_types = ", ".join(request.allowed_edge_types)
        canonical_rule_ids = ", ".join(request.expected_canonical_rule_ids) or "none provided"
        edge_ids = ", ".join(request.expected_edge_ids) or "none provided"
        return (
            "You extract structured tabletop rules into a bounded graph-ready contract.\n"
            "Return JSON only.\n"
            "Do not answer the rules questions directly.\n"
            "Do not invent citations, pages, document types, or rule IDs not justified by the provided text.\n"
            "Prefer explicit uncertainty over unsupported inference.\n"
            f"Only use these edge types: {edge_types}.\n"
            "For this frozen pilot slice, preserve the expected canonical rule IDs and edge IDs when the evidence supports those exact rule units.\n"
            "Do not merge distinct rule units when the pilot expects them to remain separate.\n"
            f"Expected canonical rule IDs: {canonical_rule_ids}.\n"
            f"Expected edge IDs: {edge_ids}.\n"
            "Produce one RuleExtractionBundle with source_passages, canonical_rules, edges, and optional uncertainties."
        )

    def build_user_prompt(self, request: RuleExtractionRequest) -> str:
        return (
            "Extract a bounded rules-lawyer graph slice for the following scoped source material.\n"
            "Requirements:\n"
            "- Preserve citation and authority metadata from the input.\n"
            "- Create canonical rules only when the text expresses a reusable normative claim, restriction, prerequisite, or clarification.\n"
            "- Reuse the provided expected canonical rule IDs when the source supports those exact rule units.\n"
            "- Keep action base, prerequisite, clarification, and modifier rules separate for this pilot.\n"
            "- Reuse the provided source_id values as source_passage node_id values.\n"
            "- Reuse the provided expected edge IDs when the supported relationship matches the pilot reference structure.\n"
            "- Do not invent conflicts unless the sources explicitly contradict each other.\n"
            "- Ensure every edge references existing node IDs.\n"
            "- Keep the extraction focused on the frozen pilot questions.\n\n"
            f"{_stringify_request_payload(request)}"
        )

    def extract(self, request: RuleExtractionRequest) -> RuleExtractionBundle:
        raw_result = self.predictor(
            self.build_system_prompt(request),
            self.build_user_prompt(request),
        )
        bundle = normalize_candidate_bundle(raw_result, request)
        self._validate_bundle(bundle, request)
        return bundle

    @staticmethod
    def _validate_bundle(bundle: RuleExtractionBundle, request: RuleExtractionRequest) -> None:
        if bundle.seed_id != request.seed_id:
            raise ValueError(
                f"Extracted seed_id={bundle.seed_id} does not match request seed_id={request.seed_id}"
            )
        if bundle.game_id != request.game_id:
            raise ValueError(
                f"Extracted game_id={bundle.game_id} does not match request game_id={request.game_id}"
            )
        if bundle.subsystem != request.subsystem:
            raise ValueError(
                "Extracted subsystem does not match request subsystem: "
                f"{bundle.subsystem} != {request.subsystem}"
            )

        node_ids = [node.node_id for node in bundle.nodes]
        duplicate_node_ids = _find_duplicate_ids(node_ids)
        if duplicate_node_ids:
            raise ValueError(f"Duplicate node IDs in extraction output: {duplicate_node_ids}")

        edge_ids = [edge.edge_id for edge in bundle.edges]
        duplicate_edge_ids = _find_duplicate_ids(edge_ids)
        if duplicate_edge_ids:
            raise ValueError(f"Duplicate edge IDs in extraction output: {duplicate_edge_ids}")

        node_id_set = set(node_ids)
        for edge in bundle.edges:
            if edge.edge_type not in request.allowed_edge_types:
                raise ValueError(
                    f"Unsupported edge_type {edge.edge_type!r}; allowed={request.allowed_edge_types}"
                )
            if edge.from_node_id not in node_id_set:
                raise ValueError(
                    f"Edge {edge.edge_id} references unknown from_node_id={edge.from_node_id}"
                )
            if edge.to_node_id not in node_id_set:
                raise ValueError(
                    f"Edge {edge.edge_id} references unknown to_node_id={edge.to_node_id}"
                )


class PydanticAIRuleExtractionPredictor:
    """PydanticAI-backed predictor for structured rules extraction."""

    def __init__(
        self,
        model: str | Model,
        *,
        agent: Agent | None = None,
        name: str = "rules_extractor",
    ) -> None:
        self.model = model
        self.agent = agent or Agent(
            model,
            output_type=CandidateRuleExtractionBundle,
            name=name,
        )

    def __call__(self, system_prompt: str, user_prompt: str) -> CandidateRuleExtractionBundle:
        result = self.agent.run_sync(
            user_prompt,
            instructions=system_prompt,
            output_type=CandidateRuleExtractionBundle,
        )
        return result.output


class RawPydanticAIRuleExtractionPredictor:
    """PydanticAI-backed predictor that returns raw dict output for debugging."""

    def __init__(
        self,
        model: str | Model,
        *,
        agent: Agent | None = None,
        name: str = "rules_extractor_raw",
    ) -> None:
        self.model = model
        self.agent = agent or Agent(
            model,
            output_type=dict[str, Any],
            name=name,
        )

    def __call__(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        result = self.agent.run_sync(
            user_prompt,
            instructions=system_prompt,
            output_type=dict[str, Any],
        )
        return result.output


def compare_rule_extractions(
    expected: RuleExtractionBundle,
    actual: RuleExtractionBundle,
) -> RuleExtractionComparison:
    """Compare parser output against a hand-built reference bundle."""

    comparison = RuleExtractionComparison()

    expected_source = {record.node_id: record for record in expected.source_passages}
    actual_source = {record.node_id: record for record in actual.source_passages}
    expected_rules = {record.node_id: record for record in expected.canonical_rules}
    actual_rules = {record.node_id: record for record in actual.canonical_rules}
    expected_edges = {record.edge_id: record for record in expected.edges}
    actual_edges = {record.edge_id: record for record in actual.edges}

    comparison.missing_source_passage_ids = sorted(expected_source.keys() - actual_source.keys())
    comparison.extra_source_passage_ids = sorted(actual_source.keys() - expected_source.keys())
    comparison.missing_canonical_rule_ids = sorted(expected_rules.keys() - actual_rules.keys())
    comparison.extra_canonical_rule_ids = sorted(actual_rules.keys() - expected_rules.keys())
    comparison.missing_edge_ids = sorted(expected_edges.keys() - actual_edges.keys())
    comparison.extra_edge_ids = sorted(actual_edges.keys() - expected_edges.keys())

    for record_id in sorted(expected_source.keys() & actual_source.keys()):
        expected_record = expected_source[record_id]
        actual_record = actual_source[record_id]
        for field_name in (
            "document_id",
            "document_type",
            "authority_scope",
            "title",
            "locator",
            "page",
            "source_text",
            "citation_label",
            "citation_short",
        ):
            _record_field_mismatch(
                comparison,
                "source_passage",
                record_id,
                field_name,
                _normalize_comparison_value(getattr(expected_record, field_name)),
                _normalize_comparison_value(getattr(actual_record, field_name)),
            )

    for record_id in sorted(expected_rules.keys() & actual_rules.keys()):
        expected_record = expected_rules[record_id]
        actual_record = actual_rules[record_id]
        for field_name in ("rule_kind", "title", "normalized_statement", "scope"):
            _record_field_mismatch(
                comparison,
                "canonical_rule",
                record_id,
                field_name,
                _normalize_comparison_value(getattr(expected_record, field_name)),
                _normalize_comparison_value(getattr(actual_record, field_name)),
            )

    for record_id in sorted(expected_edges.keys() & actual_edges.keys()):
        expected_record = expected_edges[record_id]
        actual_record = actual_edges[record_id]
        for field_name in ("edge_type", "from_node_id", "to_node_id", "rationale"):
            _record_field_mismatch(
                comparison,
                "edge",
                record_id,
                field_name,
                _normalize_comparison_value(getattr(expected_record, field_name)),
                _normalize_comparison_value(getattr(actual_record, field_name)),
            )

    return comparison


def _record_field_mismatch(
    comparison: RuleExtractionComparison,
    record_kind: str,
    record_id: str,
    field_name: str,
    expected: Any,
    actual: Any,
) -> None:
    if expected != actual:
        comparison.field_mismatches.append(
            RuleExtractionFieldMismatch(
                record_kind=record_kind,
                record_id=record_id,
                field_name=field_name,
                expected=expected,
                actual=actual,
            )
        )


def _normalize_comparison_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def build_request_from_reference_bundle(
    bundle: RuleExtractionBundle,
) -> RuleExtractionRequest:
    """Build an extraction request from a reference bundle's scoped source passages."""

    return RuleExtractionRequest(
        seed_id=bundle.seed_id,
        game_id=bundle.game_id,
        scope=bundle.scope,
        subsystem=bundle.subsystem,
        project_id=bundle.project_id,
        source=bundle.source,
        frozen_questions=bundle.frozen_questions,
        related_docs=bundle.related_docs,
        expected_canonical_rules=[
            ExpectedCanonicalRule(
                node_id=record.node_id,
                rule_kind=record.rule_kind,
                title=record.title,
                normalized_statement=record.normalized_statement,
                scope=record.scope,
            )
            for record in bundle.canonical_rules
        ],
        expected_edges=[
            ExpectedRuleEdge(
                edge_id=record.edge_id,
                edge_type=record.edge_type,
                from_node_id=record.from_node_id,
                to_node_id=record.to_node_id,
                rationale=record.rationale,
            )
            for record in bundle.edges
        ],
        expected_canonical_rule_ids=[
            record.node_id for record in bundle.canonical_rules
        ],
        expected_edge_ids=[record.edge_id for record in bundle.edges],
        source_documents=[
            ScopedSourceDocument(
                source_id=passage.node_id,
                document_id=passage.document_id,
                document_type=passage.document_type,
                authority_scope=passage.authority_scope,
                title=passage.title,
                locator=passage.locator,
                page=passage.page,
                text=passage.source_text,
                citation_label=passage.citation_label,
                citation_short=passage.citation_short,
            )
            for passage in bundle.source_passages
        ],
    )


def load_reference_bundle_from_seed_fixture(
    manifest: dict[str, Any],
    node_records: list[dict[str, Any]],
    edge_records: list[dict[str, Any]],
) -> RuleExtractionBundle:
    """Load the current tracked seed fixture as a typed reference bundle."""

    return load_bundle_from_seed_records(manifest, node_records, edge_records)


def normalize_candidate_bundle(
    candidate: CandidateRuleExtractionBundle | RuleExtractionBundle | dict[str, Any],
    request: RuleExtractionRequest,
) -> RuleExtractionBundle:
    """Normalize lenient LLM output into the strict extraction bundle contract."""

    if isinstance(candidate, RuleExtractionBundle):
        return candidate

    parsed = (
        candidate
        if isinstance(candidate, CandidateRuleExtractionBundle)
        else CandidateRuleExtractionBundle.model_validate(candidate)
    )

    source_docs_by_id = {document.source_id: document for document in request.source_documents}
    expected_rules_by_id = {
        record.node_id: record for record in request.expected_canonical_rules
    }
    expected_edges_by_id = {record.edge_id: record for record in request.expected_edges}

    uncertainties = (
        [] if isinstance(parsed.uncertainties, dict) else parsed.uncertainties
    )

    return RuleExtractionBundle(
        seed_id=parsed.seed_id,
        game_id=parsed.game_id,
        scope=parsed.scope,
        subsystem=parsed.subsystem,
        project_id=parsed.project_id,
        source=parsed.source,
        created_at=request_created_at(),
        updated_at=request_created_at(),
        embedding_model="manual-seed-unembedded",
        related_docs=request.related_docs,
        frozen_questions=request.frozen_questions,
        source_passages=[
            _normalize_source_passage(record, request, source_docs_by_id)
            for record in parsed.source_passages
        ],
        canonical_rules=[
            _normalize_canonical_rule(record, request, expected_rules_by_id)
            for record in parsed.canonical_rules
        ],
        edges=[
            _normalize_edge(record, expected_edges_by_id)
            for record in parsed.edges
        ],
        uncertainties=[
            {
                "kind": record.kind,
                "note": record.note,
                "source_ref": record.source_ref,
                "related_ids": record.related_ids,
            }
            for record in uncertainties
        ],
    )


def request_created_at():
    from datetime import UTC, datetime

    return datetime.now(UTC)


def _normalize_source_passage(
    record: CandidateSourcePassage,
    request: RuleExtractionRequest,
    source_docs_by_id: dict[str, ScopedSourceDocument],
) -> dict[str, Any]:
    source_doc = source_docs_by_id.get(record.node_id)
    return {
        "node_id": record.node_id,
        "game_id": record.game_id or request.game_id,
        "document_id": record.document_id or (source_doc.document_id if source_doc else ""),
        "document_type": record.document_type or (source_doc.document_type if source_doc else ""),
        "authority_scope": record.authority_scope or (source_doc.authority_scope if source_doc else ""),
        "title": record.title or (source_doc.title if source_doc else record.node_id),
        "locator": record.locator or (source_doc.locator if source_doc else record.node_id),
        "page": record.page if record.page is not None else (source_doc.page if source_doc else None),
        "source_text": record.source_text,
        "citation_label": record.citation_label or (source_doc.citation_label if source_doc else record.node_id),
        "citation_short": record.citation_short or (source_doc.citation_short if source_doc else record.node_id),
        "language": record.language or "en",
        "subsystem": record.subsystem or request.subsystem,
        "content": record.content or record.source_text,
    }


def _normalize_canonical_rule(
    record: CandidateCanonicalRule,
    request: RuleExtractionRequest,
    expected_rules_by_id: dict[str, ExpectedCanonicalRule],
) -> dict[str, Any]:
    expected = expected_rules_by_id.get(record.node_id)
    return {
        "node_id": record.node_id,
        "game_id": record.game_id or request.game_id,
        "rule_kind": record.rule_kind or (expected.rule_kind if expected else "clarification"),
        "title": record.title or (expected.title if expected else record.node_id),
        "normalized_statement": record.normalized_statement,
        "scope": record.scope or (expected.scope if expected else request.scope),
        "subsystem": record.subsystem or request.subsystem,
        "content": record.content or record.normalized_statement,
    }


def _normalize_edge(
    record: CandidateRuleEdge,
    expected_edges_by_id: dict[str, ExpectedRuleEdge],
) -> dict[str, Any]:
    expected = expected_edges_by_id.get(record.edge_id)
    return {
        "edge_id": record.edge_id,
        "edge_type": record.edge_type or (expected.edge_type if expected else "supports"),
        "from_node_id": record.from_node_id,
        "to_node_id": record.to_node_id,
        "rationale": record.rationale or (expected.rationale if expected else ""),
        "description": record.description,
        "confidence": record.confidence,
    }
