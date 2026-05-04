"""Source-grounded audit checks for rule extraction bundles before graph import."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .contracts import RuleExtractionBundle
from .extraction import RuleExtractionRequest


RuleLoadAuditSeverity = Literal["info", "warning", "error"]
RuleLoadAuditLayer = Literal[
    "source_coverage",
    "grounding",
    "schema_completeness",
    "authority",
    "graph_integrity",
]


SOURCE_GROUNDING_EDGE_TYPES = {
    "supports",
    "clarifies",
    "modifies",
    "overrides",
    "requires",
    "prohibits",
    "defines",
    "enables",
    "applies_to",
    "applies_during",
}

HIGH_SIGNAL_RULE_TERMS = (
    " may ",
    " must ",
    " cannot ",
    " can't ",
    " can ",
    " cost ",
    " costs ",
    " pay ",
    " requires ",
    " required ",
    " if ",
    " when ",
    " before ",
    " after ",
    " during ",
)


class RuleLoadAuditFinding(BaseModel):
    """One source-load audit finding."""

    layer: RuleLoadAuditLayer
    severity: RuleLoadAuditSeverity
    code: str
    message: str
    related_ids: list[str] = Field(default_factory=list)


class RuleLoadSourceCoverageSummary(BaseModel):
    """Coverage counts for requested source documents versus extracted passages."""

    requested_source_documents: int = 0
    extracted_source_passages: int = 0
    requested_documents_represented: int = 0
    high_signal_unrepresented_documents: int = 0


class RuleLoadGraphSummary(BaseModel):
    """Graph counts for the candidate load."""

    source_passages: int
    canonical_rules: int
    edges: int
    source_grounded_rules: int


class RuleLoadAuditReport(BaseModel):
    """Machine-readable pre-import audit report for a rule extraction bundle."""

    seed_id: str
    game_id: str
    subsystem: str
    source_coverage: RuleLoadSourceCoverageSummary
    graph: RuleLoadGraphSummary
    findings: list[RuleLoadAuditFinding] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "warning")

    @property
    def passes_required_gates(self) -> bool:
        return self.error_count == 0


def audit_rule_extraction_bundle(
    bundle: RuleExtractionBundle,
    *,
    request: RuleExtractionRequest | None = None,
) -> RuleLoadAuditReport:
    """Audit one extracted rule bundle before materialization or graph import.

    The audit is intentionally game-neutral. It checks whether extracted rules
    are connected to source passages and whether provided source material is
    represented, but it does not encode game-specific rule vocabulary.
    """

    findings: list[RuleLoadAuditFinding] = []
    source_ids = [record.node_id for record in bundle.source_passages]
    rule_ids = [record.node_id for record in bundle.canonical_rules]
    node_ids = [*source_ids, *rule_ids]
    node_id_set = set(node_ids)
    source_id_set = set(source_ids)
    rule_id_set = set(rule_ids)

    _audit_duplicate_ids("node", node_ids, findings)
    _audit_duplicate_ids("edge", [record.edge_id for record in bundle.edges], findings)

    source_grounded_rule_ids: set[str] = set()
    for edge in bundle.edges:
        if edge.from_node_id not in node_id_set:
            findings.append(
                RuleLoadAuditFinding(
                    layer="graph_integrity",
                    severity="error",
                    code="edge_unknown_from_node",
                    message=f"Edge {edge.edge_id} references unknown from_node_id={edge.from_node_id}.",
                    related_ids=[edge.edge_id, edge.from_node_id],
                )
            )
        if edge.to_node_id not in node_id_set:
            findings.append(
                RuleLoadAuditFinding(
                    layer="graph_integrity",
                    severity="error",
                    code="edge_unknown_to_node",
                    message=f"Edge {edge.edge_id} references unknown to_node_id={edge.to_node_id}.",
                    related_ids=[edge.edge_id, edge.to_node_id],
                )
            )
        if not edge.rationale.strip():
            findings.append(
                RuleLoadAuditFinding(
                    layer="schema_completeness",
                    severity="warning",
                    code="edge_missing_rationale",
                    message=f"Edge {edge.edge_id} has no rationale.",
                    related_ids=[edge.edge_id],
                )
            )
        if (
            edge.from_node_id in source_id_set
            and edge.to_node_id in rule_id_set
            and edge.edge_type in SOURCE_GROUNDING_EDGE_TYPES
        ):
            source_grounded_rule_ids.add(edge.to_node_id)

    if request is not None:
        allowed_edge_types = set(request.allowed_edge_types)
        for edge in bundle.edges:
            if edge.edge_type not in allowed_edge_types:
                findings.append(
                    RuleLoadAuditFinding(
                        layer="graph_integrity",
                        severity="error",
                        code="edge_type_not_allowed_by_request",
                        message=f"Edge {edge.edge_id} uses edge_type={edge.edge_type!r} outside the request contract.",
                        related_ids=[edge.edge_id],
                    )
                )

    for passage in bundle.source_passages:
        if not passage.source_text.strip():
            findings.append(
                RuleLoadAuditFinding(
                    layer="schema_completeness",
                    severity="error",
                    code="source_passage_empty_text",
                    message=f"Source passage {passage.node_id} has empty source text.",
                    related_ids=[passage.node_id],
                )
            )
        for field_name, value in (
            ("document_id", passage.document_id),
            ("document_type", passage.document_type),
            ("authority_scope", passage.authority_scope),
            ("locator", passage.locator),
            ("citation_label", passage.citation_label),
            ("citation_short", passage.citation_short),
        ):
            if not str(value or "").strip():
                findings.append(
                    RuleLoadAuditFinding(
                        layer="authority" if field_name in {"authority_scope", "document_type"} else "schema_completeness",
                        severity="error",
                        code=f"source_passage_missing_{field_name}",
                        message=f"Source passage {passage.node_id} is missing {field_name}.",
                        related_ids=[passage.node_id],
                    )
                )

    for rule in bundle.canonical_rules:
        if not rule.rule_kind.strip():
            findings.append(_missing_rule_field(rule.node_id, "rule_kind"))
        if not rule.title.strip():
            findings.append(_missing_rule_field(rule.node_id, "title"))
        if not rule.normalized_statement.strip():
            findings.append(_missing_rule_field(rule.node_id, "normalized_statement"))
        if not rule.scope.strip():
            findings.append(_missing_rule_field(rule.node_id, "scope"))
        if rule.node_id not in source_grounded_rule_ids:
            findings.append(
                RuleLoadAuditFinding(
                    layer="grounding",
                    severity="error",
                    code="canonical_rule_missing_source_grounding",
                    message=f"Canonical rule {rule.node_id} has no direct source-passage grounding edge.",
                    related_ids=[rule.node_id],
                )
            )

    coverage_summary = _audit_source_coverage(bundle, request, findings)

    return RuleLoadAuditReport(
        seed_id=bundle.seed_id,
        game_id=bundle.game_id,
        subsystem=bundle.subsystem,
        source_coverage=coverage_summary,
        graph=RuleLoadGraphSummary(
            source_passages=len(bundle.source_passages),
            canonical_rules=len(bundle.canonical_rules),
            edges=len(bundle.edges),
            source_grounded_rules=len(source_grounded_rule_ids),
        ),
        findings=findings,
    )


def _audit_duplicate_ids(
    record_kind: str,
    values: list[str],
    findings: list[RuleLoadAuditFinding],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    for value in sorted(duplicates):
        findings.append(
            RuleLoadAuditFinding(
                layer="graph_integrity",
                severity="error",
                code=f"duplicate_{record_kind}_id",
                message=f"Duplicate {record_kind} id: {value}.",
                related_ids=[value],
            )
        )


def _missing_rule_field(rule_id: str, field_name: str) -> RuleLoadAuditFinding:
    return RuleLoadAuditFinding(
        layer="schema_completeness",
        severity="error",
        code=f"canonical_rule_missing_{field_name}",
        message=f"Canonical rule {rule_id} is missing {field_name}.",
        related_ids=[rule_id],
    )


def _audit_source_coverage(
    bundle: RuleExtractionBundle,
    request: RuleExtractionRequest | None,
    findings: list[RuleLoadAuditFinding],
) -> RuleLoadSourceCoverageSummary:
    if request is None:
        return RuleLoadSourceCoverageSummary(
            requested_source_documents=0,
            extracted_source_passages=len(bundle.source_passages),
            requested_documents_represented=0,
            high_signal_unrepresented_documents=0,
        )

    passages_by_id = {passage.node_id: passage for passage in bundle.source_passages}
    represented = 0
    high_signal_unrepresented = 0
    for source_document in request.source_documents:
        passage = passages_by_id.get(source_document.source_id)
        if passage is None:
            if _has_high_signal_rule_text(source_document.text):
                high_signal_unrepresented += 1
                findings.append(
                    RuleLoadAuditFinding(
                        layer="source_coverage",
                        severity="warning",
                        code="high_signal_source_document_unrepresented",
                        message=(
                            f"Source document {source_document.source_id} contains likely rule text "
                            "but has no extracted source passage."
                        ),
                        related_ids=[source_document.source_id],
                    )
                )
            continue

        represented += 1
        if not _normalized_contains(source_document.text, passage.source_text):
            findings.append(
                RuleLoadAuditFinding(
                    layer="grounding",
                    severity="error",
                    code="source_passage_text_not_in_requested_source",
                    message=(
                        f"Source passage {passage.node_id} text is not grounded in the requested source document."
                    ),
                    related_ids=[passage.node_id, source_document.source_id],
                )
            )

    return RuleLoadSourceCoverageSummary(
        requested_source_documents=len(request.source_documents),
        extracted_source_passages=len(bundle.source_passages),
        requested_documents_represented=represented,
        high_signal_unrepresented_documents=high_signal_unrepresented,
    )


def _has_high_signal_rule_text(value: str) -> bool:
    normalized = f" {value.lower()} "
    return any(term in normalized for term in HIGH_SIGNAL_RULE_TERMS)


def _normalized_contains(haystack: str, needle: str) -> bool:
    normalized_haystack = " ".join(haystack.split()).lower()
    normalized_needle = " ".join(needle.split()).lower()
    return normalized_needle in normalized_haystack
