"""Deterministic and live ruling assembly for the first SETI pilot slices."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..VectorGraphStore import VectorGraphStore
from ..schemas import SimilarNode
from .contracts import CanonicalRule, RuleExtractionBundle, SourcePassage, load_bundle_from_seed_records


class RuleCitation(BaseModel):
    """A source citation used in a user-facing ruling."""

    source_node_id: str
    citation_label: str
    citation_short: str
    title: str
    locator: str
    authority_scope: str
    source_excerpt: str


class RuleReference(BaseModel):
    """A canonical rule referenced in a ruling."""

    rule_node_id: str
    title: str
    rule_kind: str
    normalized_statement: str


class PrecedenceEntry(BaseModel):
    """One explicit precedence step in the ruling chain."""

    order: int
    summary: str
    rule_node_id: str | None = None
    source_node_id: str | None = None
    precedence_kind: Literal["primary", "modifier", "support", "authority"]


class RulesRulingRequest(BaseModel):
    """Input contract for the deterministic pilot ruling path."""

    question: str
    seed_id: str | None = None
    retrieval_limit: int = Field(default=6, ge=1, le=20)


class RulesRulingResult(BaseModel):
    """Structured rules-lawyer result intended for UI rendering."""

    question: str
    question_id: str
    seed_id: str
    subsystem: str
    ruling: str
    primary_rule: RuleReference | None = None
    primary_citation: RuleCitation | None = None
    modifying_rules: list[RuleReference] = Field(default_factory=list)
    modifying_citations: list[RuleCitation] = Field(default_factory=list)
    supporting_rules: list[RuleReference] = Field(default_factory=list)
    supporting_citations: list[RuleCitation] = Field(default_factory=list)
    precedence_order: list[PrecedenceEntry] = Field(default_factory=list)
    uncertainty: str | None = None
    abstain: bool = False
    backend: str = "deterministic-pilot"


class RetrievedEvidenceNode(BaseModel):
    """One live-retrieved node normalized for ruling selection."""

    storage_node_id: str
    logical_node_id: str
    node_kind: str
    title: str = ""
    content: str = ""
    rule_kind: str | None = None
    normalized_statement: str | None = None
    citation_label: str | None = None
    citation_short: str | None = None
    locator: str | None = None
    authority_scope: str | None = None
    seed_id: str | None = None
    subsystem: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    similarity_score: float | None = None

    @property
    def normalized_text(self) -> str:
        text = " ".join(
            value
            for value in (
                self.title,
                self.normalized_statement or "",
                self.citation_label or "",
                self.citation_short or "",
                self.locator or "",
                self.content,
            )
            if value
        )
        return _normalize_question(text)


class RetrievedEvidenceEdge(BaseModel):
    """One live-retrieved graph edge normalized for ranking and inspection."""

    edge_id: str
    relationship_type: str
    description: str = ""
    from_logical_node_id: str
    to_logical_node_id: str


class RetrievedRulingEvidence(BaseModel):
    """Typed intermediate evidence package for live ruling assembly."""

    question: str
    seed_id: str | None = None
    subsystem: str | None = None
    retrieved_node_ids: list[str] = Field(default_factory=list)
    expanded_node_ids: list[str] = Field(default_factory=list)
    nodes: list[RetrievedEvidenceNode] = Field(default_factory=list)
    edges: list[RetrievedEvidenceEdge] = Field(default_factory=list)

    @property
    def nodes_by_logical_id(self) -> dict[str, RetrievedEvidenceNode]:
        return {node.logical_node_id: node for node in self.nodes}

    @property
    def source_nodes(self) -> list[RetrievedEvidenceNode]:
        return [node for node in self.nodes if node.node_kind == "source_passage"]

    @property
    def rule_nodes(self) -> list[RetrievedEvidenceNode]:
        return [node for node in self.nodes if node.node_kind == "canonical_rule"]


class PilotSeedScore(BaseModel):
    """One candidate pilot seed scored against the normalized question."""

    seed_id: str
    score: float


class PilotSeedInference(BaseModel):
    """Typed seed-inference result for inspection and evaluation."""

    normalized_question: str
    selected_seed_id: str | None = None
    selected_score: float = 0.0
    candidates: list[PilotSeedScore] = Field(default_factory=list)


class PilotCaseMatch(BaseModel):
    """One candidate pilot case scored against question and retrieved evidence."""

    question_id: str
    seed_id: str
    question_score: float
    evidence_score: int
    total_score: float
    matched_reference_ids: list[str] = Field(default_factory=list)


class LivePilotRulingInspection(BaseModel):
    """Intermediate live ruling stages separated for testing and evaluation."""

    question: str
    normalized_question: str
    evidence: RetrievedRulingEvidence
    seed_inference: PilotSeedInference
    selected_seed_id: str | None = None
    selected_case: PilotCaseMatch | None = None
    candidate_cases: list[PilotCaseMatch] = Field(default_factory=list)


@dataclass(frozen=True)
class _PilotCaseSpec:
    question_id: str
    seed_id: str
    accepted_questions: tuple[str, ...]
    ruling: str
    primary_rule_id: str
    primary_source_id: str
    modifying_rule_ids: tuple[str, ...] = ()
    modifying_source_ids: tuple[str, ...] = ()
    supporting_rule_ids: tuple[str, ...] = ()
    supporting_source_ids: tuple[str, ...] = ()
    precedence_summaries: tuple[tuple[str, str | None, str | None, str], ...] = ()


def _normalize_question(value: str) -> str:
    lowered = value.strip().lower()
    replacements = {
        "another player": "opponent",
        "cheaper": "discount",
        "finish": "continue",
        "go back": "continue",
        "pause": "interrupt",
    }
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    collapsed = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_rule_bundle_from_seed_manifest(manifest_path: str | Path) -> RuleExtractionBundle:
    """Load one tracked seed manifest into the typed rules bundle."""

    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)
    node_records = _load_jsonl(Path(manifest["node_file"]))
    edge_records = _load_jsonl(Path(manifest["edge_file"]))
    return load_bundle_from_seed_records(manifest, node_records, edge_records)


def load_seti_pilot_bundles() -> dict[str, RuleExtractionBundle]:
    """Load the validated SETI pilot bundles from tracked fixtures."""

    fixture_dir = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "rag_eval"
    manifests = (
        fixture_dir / "seti_landing_orbiter_seed_v1_manifest.json",
        fixture_dir / "seti_free_action_authority_seed_v1_manifest.json",
    )
    return {
        bundle.seed_id: bundle
        for bundle in (load_rule_bundle_from_seed_manifest(path) for path in manifests)
    }


class DeterministicPilotRulingEngine:
    """Answer the frozen SETI pilot questions from validated rule bundles."""

    def __init__(self, bundles: dict[str, RuleExtractionBundle]):
        self.bundles = bundles
        self.case_specs = _build_case_specs()
        self._cases_by_seed_and_question: dict[tuple[str, str], _PilotCaseSpec] = {}
        for case in self.case_specs:
            for accepted_question in case.accepted_questions:
                self._cases_by_seed_and_question[(case.seed_id, _normalize_question(accepted_question))] = case
        self._validate_case_specs()

    @classmethod
    def for_seti_pilot(cls) -> "DeterministicPilotRulingEngine":
        """Build the engine over the tracked SETI pilot fixtures."""

        return cls(load_seti_pilot_bundles())

    def answer(self, request: RulesRulingRequest) -> RulesRulingResult:
        """Answer one frozen pilot question or abstain if unsupported."""

        normalized_question = _normalize_question(request.question)
        seed_ids = [request.seed_id] if request.seed_id is not None else list(self.bundles)

        for seed_id in seed_ids:
            case = self._cases_by_seed_and_question.get((seed_id, normalized_question))
            if case is not None:
                return self._build_result(case, request.question)

        target_seed_id = request.seed_id or seed_ids[0]
        bundle = self.bundles[target_seed_id]
        return RulesRulingResult(
            question=request.question,
            question_id="unsupported_question",
            seed_id=bundle.seed_id,
            subsystem=bundle.subsystem,
            ruling="I do not have a frozen pilot ruling for that question yet.",
            uncertainty=(
                "This deterministic pilot path only answers the tracked frozen SETI questions. "
                "Use one of the seed's curated pilot questions or extend the case library first."
            ),
            abstain=True,
        )

    def _build_result(self, case: _PilotCaseSpec, question: str) -> RulesRulingResult:
        bundle = self.bundles[case.seed_id]
        rules_by_id = {rule.node_id: rule for rule in bundle.canonical_rules}
        passages_by_id = {passage.node_id: passage for passage in bundle.source_passages}

        return RulesRulingResult(
            question=question,
            question_id=case.question_id,
            seed_id=bundle.seed_id,
            subsystem=bundle.subsystem,
            ruling=case.ruling,
            primary_rule=self._rule_reference(rules_by_id[case.primary_rule_id]),
            primary_citation=self._citation(passages_by_id[case.primary_source_id]),
            modifying_rules=[self._rule_reference(rules_by_id[rule_id]) for rule_id in case.modifying_rule_ids],
            modifying_citations=[self._citation(passages_by_id[source_id]) for source_id in case.modifying_source_ids],
            supporting_rules=[self._rule_reference(rules_by_id[rule_id]) for rule_id in case.supporting_rule_ids],
            supporting_citations=[self._citation(passages_by_id[source_id]) for source_id in case.supporting_source_ids],
            precedence_order=[
                PrecedenceEntry(
                    order=index,
                    summary=summary,
                    rule_node_id=rule_id,
                    source_node_id=source_id,
                    precedence_kind=precedence_kind,
                )
                for index, (summary, rule_id, source_id, precedence_kind) in enumerate(
                    case.precedence_summaries, start=1
                )
            ],
        )

    @staticmethod
    def _rule_reference(rule: CanonicalRule) -> RuleReference:
        return RuleReference(
            rule_node_id=rule.node_id,
            title=rule.title,
            rule_kind=rule.rule_kind,
            normalized_statement=rule.normalized_statement,
        )

    @staticmethod
    def _citation(passage: SourcePassage) -> RuleCitation:
        return RuleCitation(
            source_node_id=passage.node_id,
            citation_label=passage.citation_label,
            citation_short=passage.citation_short,
            title=passage.title,
            locator=passage.locator,
            authority_scope=passage.authority_scope,
            source_excerpt=passage.source_text,
        )

    def _validate_case_specs(self) -> None:
        for case in self.case_specs:
            if case.seed_id not in self.bundles:
                raise ValueError(f"Unknown ruling seed_id={case.seed_id}")
            bundle = self.bundles[case.seed_id]
            rule_ids = {rule.node_id for rule in bundle.canonical_rules}
            source_ids = {passage.node_id for passage in bundle.source_passages}

            referenced_rule_ids = {
                case.primary_rule_id,
                *case.modifying_rule_ids,
                *case.supporting_rule_ids,
                *(rule_id for _, rule_id, _, _ in case.precedence_summaries if rule_id is not None),
            }
            referenced_source_ids = {
                case.primary_source_id,
                *case.modifying_source_ids,
                *case.supporting_source_ids,
                *(source_id for _, _, source_id, _ in case.precedence_summaries if source_id is not None),
            }

            missing_rules = sorted(rule_id for rule_id in referenced_rule_ids if rule_id not in rule_ids)
            missing_sources = sorted(source_id for source_id in referenced_source_ids if source_id not in source_ids)
            if missing_rules or missing_sources:
                raise ValueError(
                    f"Invalid case spec {case.question_id}: missing_rules={missing_rules}, "
                    f"missing_sources={missing_sources}"
                )


class LivePilotRulingEngine:
    """Assemble pilot rulings from live vector retrieval and graph expansion."""

    _EDGE_PROJECTION_STEPS = (
        "bothE().project('edge_id','relationship_type','description','from_node_id','to_node_id')"
        ".by(values('edge_id')).by(values('relationship_type')).by(values('description'))"
        ".by(outV().values('node_id')).by(inV().values('node_id'))"
    )

    def __init__(self, store: VectorGraphStore, *, project_id: str):
        self.store = store
        self.project_id = project_id
        self.case_specs = _build_case_specs()
        self._cases_by_seed_and_question: dict[tuple[str, str], _PilotCaseSpec] = {}
        self._questions_to_seed: dict[str, str] = {}
        for case in self.case_specs:
            for accepted_question in case.accepted_questions:
                normalized_question = _normalize_question(accepted_question)
                self._cases_by_seed_and_question[(case.seed_id, normalized_question)] = case
                self._questions_to_seed[normalized_question] = case.seed_id

    def answer(self, request: RulesRulingRequest) -> RulesRulingResult:
        """Answer one frozen pilot question from live backend evidence."""

        inspection = self.inspect_request(request)
        evidence = inspection.evidence
        if not evidence.nodes:
            return self._abstain(
                question=request.question,
                seed_id=request.seed_id or "unknown",
                subsystem="unknown",
                question_id="unsupported_question",
                uncertainty="No live evidence was retrieved for this question from the current graph.",
            )
        seed_id = inspection.selected_seed_id
        if seed_id is None:
            return self._abstain(
                question=request.question,
                seed_id="unknown",
                subsystem=evidence.subsystem or "unknown",
                question_id="unsupported_question",
                uncertainty=(
                    "Live retrieval found evidence, but it could not be assigned to one supported pilot seed."
                ),
            )
        case = self._case_spec_by_question_id(inspection.selected_case.question_id) if inspection.selected_case else None
        if case is None:
            return self._abstain(
                question=request.question,
                seed_id=seed_id,
                subsystem=evidence.subsystem or self._subsystem_for_seed(seed_id),
                question_id="unsupported_question",
                uncertainty=(
                    "The retrieved evidence was not strong enough to map this question to one supported pilot ruling."
                ),
            )

        try:
            return self._build_live_result(case, request.question, evidence)
        except ValueError as exc:
            return self._abstain(
                question=request.question,
                seed_id=seed_id,
                subsystem=self._subsystem_for_seed(seed_id),
                question_id=case.question_id,
                uncertainty=str(exc),
            )

    def inspect_request(self, request: RulesRulingRequest) -> LivePilotRulingInspection:
        """Expose live retrieval and selection stages as typed intermediate output."""

        normalized_question = _normalize_question(request.question)
        evidence = self._retrieve_evidence(
            question=request.question,
            seed_id=request.seed_id,
            retrieval_limit=request.retrieval_limit,
        )
        seed_inference = self._infer_seed_from_question(normalized_question)
        selected_seed_id = request.seed_id or seed_inference.selected_seed_id or evidence.seed_id
        candidate_cases = (
            self._rank_cases(seed_id=selected_seed_id, question=request.question, evidence=evidence)
            if selected_seed_id is not None
            else []
        )
        selected_case = candidate_cases[0] if candidate_cases and candidate_cases[0].question_score >= 0.14 else None
        return LivePilotRulingInspection(
            question=request.question,
            normalized_question=normalized_question,
            evidence=evidence,
            seed_inference=seed_inference,
            selected_seed_id=selected_seed_id,
            selected_case=selected_case,
            candidate_cases=candidate_cases,
        )

    def _retrieve_evidence(
        self,
        *,
        question: str,
        seed_id: str | None,
        retrieval_limit: int,
    ) -> RetrievedRulingEvidence:
        similar_nodes = self.store.search_similar_nodes(
            content=question,
            limit=retrieval_limit,
            project_id=self.project_id,
        )
        filtered = self._filter_similar_nodes(similar_nodes, seed_id=seed_id)
        nodes_by_storage_id: dict[str, RetrievedEvidenceNode] = {
            node.node_id: self._node_from_similar_node(node)
            for node in filtered
        }
        retrieved_node_ids = sorted(node.logical_node_id for node in nodes_by_storage_id.values())

        edges_by_id: dict[str, RetrievedEvidenceEdge] = {}
        frontier = list(nodes_by_storage_id)
        visited: set[str] = set()

        for _ in range(2):
            next_frontier: list[str] = []
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)
                for row in self.store.traverse_from_node(node_id, self._EDGE_PROJECTION_STEPS):
                    if not isinstance(row, dict):
                        continue
                    edge = RetrievedEvidenceEdge(
                        edge_id=str(row["edge_id"]),
                        relationship_type=str(row["relationship_type"]),
                        description=str(row.get("description", "")),
                        from_logical_node_id=str(row["from_node_id"]),
                        to_logical_node_id=str(row["to_node_id"]),
                    )
                    edges_by_id[edge.edge_id] = edge
                    next_frontier.extend([edge.from_logical_node_id, edge.to_logical_node_id])

            missing_node_ids = [
                node_id for node_id in set(next_frontier)
                if node_id not in {node.logical_node_id for node in nodes_by_storage_id.values()}
            ]
            if missing_node_ids:
                for node_id, payload in self.store.get_nodes_batch(missing_node_ids).items():
                    if seed_id is not None and str(payload.get("seed_id", "")) != seed_id:
                        continue
                    normalized = self._node_from_payload(node_id, payload)
                    nodes_by_storage_id[normalized.storage_node_id] = normalized
            frontier = next_frontier

        resolved_seed_id = seed_id or _dominant_seed_id(nodes_by_storage_id.values())
        return RetrievedRulingEvidence(
            question=question,
            seed_id=resolved_seed_id,
            subsystem=self._subsystem_for_seed(resolved_seed_id) if resolved_seed_id else None,
            retrieved_node_ids=retrieved_node_ids,
            expanded_node_ids=sorted(node.logical_node_id for node in nodes_by_storage_id.values()),
            nodes=list(nodes_by_storage_id.values()),
            edges=list(edges_by_id.values()),
        )

    def _build_live_result(
        self,
        case: _PilotCaseSpec,
        question: str,
        evidence: RetrievedRulingEvidence,
    ) -> RulesRulingResult:
        sources = evidence.source_nodes
        rules = evidence.rule_nodes

        primary_rule = self._node_from_evidence(evidence, case.primary_rule_id)
        primary_citation = self._node_from_evidence(evidence, case.primary_source_id)
        if primary_rule is None or primary_citation is None:
            raise ValueError("Live evidence did not contain a complete primary rule and citation pair.")

        modifying_rules = self._preferred_nodes(evidence, case.modifying_rule_ids)
        modifying_citations = self._preferred_nodes(evidence, case.modifying_source_ids)
        supporting_rules = self._preferred_nodes(evidence, case.supporting_rule_ids)
        supporting_citations = self._preferred_nodes(evidence, case.supporting_source_ids)

        return RulesRulingResult(
            question=question,
            question_id=case.question_id,
            seed_id=case.seed_id,
            subsystem=evidence.subsystem or self._subsystem_for_seed(case.seed_id),
            ruling=case.ruling,
            primary_rule=self._rule_reference_from_live(primary_rule),
            primary_citation=self._citation_from_live(primary_citation),
            modifying_rules=[self._rule_reference_from_live(rule) for rule in modifying_rules],
            modifying_citations=[self._citation_from_live(source) for source in modifying_citations],
            supporting_rules=[self._rule_reference_from_live(rule) for rule in supporting_rules],
            supporting_citations=[self._citation_from_live(source) for source in supporting_citations],
            precedence_order=self._build_precedence(case, evidence),
            backend="live-pilot",
        )

    def _build_precedence(
        self,
        case: _PilotCaseSpec,
        evidence: RetrievedRulingEvidence,
    ) -> list[PrecedenceEntry]:
        result: list[PrecedenceEntry] = []
        for index, (summary, rule_id, source_id, precedence_kind) in enumerate(case.precedence_summaries, start=1):
            matched_rule = (
                self._node_from_evidence(evidence, rule_id)
                if rule_id is not None
                else None
            )
            matched_source = (
                self._node_from_evidence(evidence, source_id)
                if source_id is not None
                else None
            )
            result.append(
                PrecedenceEntry(
                    order=index,
                    summary=summary,
                    rule_node_id=matched_rule.logical_node_id if matched_rule is not None else None,
                    source_node_id=matched_source.logical_node_id if matched_source is not None else None,
                    precedence_kind=precedence_kind,
                )
            )
        return result

    @staticmethod
    def _rule_reference_from_live(node: RetrievedEvidenceNode) -> RuleReference:
        return RuleReference(
            rule_node_id=node.logical_node_id,
            title=node.title or node.logical_node_id,
            rule_kind=node.rule_kind or "",
            normalized_statement=node.normalized_statement or node.content,
        )

    @staticmethod
    def _citation_from_live(node: RetrievedEvidenceNode) -> RuleCitation:
        return RuleCitation(
            source_node_id=node.logical_node_id,
            citation_label=node.citation_label or node.logical_node_id,
            citation_short=node.citation_short or node.logical_node_id,
            title=node.title or node.logical_node_id,
            locator=node.locator or "",
            authority_scope=node.authority_scope or "",
            source_excerpt=node.content,
        )

    @staticmethod
    def _subsystem_for_seed(seed_id: str) -> str:
        if seed_id == "seti_landing_orbiter_seed_v1":
            return "landing_and_orbiter_interactions"
        if seed_id == "seti_free_action_authority_seed_v1":
            return "free_action_timing_and_authority"
        return "unknown"

    @staticmethod
    def _node_from_evidence(
        evidence: RetrievedRulingEvidence, node_id: str | None
    ) -> RetrievedEvidenceNode | None:
        if node_id is None:
            return None
        return evidence.nodes_by_logical_id.get(node_id)

    @classmethod
    def _preferred_nodes(
        cls,
        evidence: RetrievedRulingEvidence,
        node_ids: tuple[str, ...],
    ) -> list[RetrievedEvidenceNode]:
        return [
            node
            for node_id in node_ids
            if (node := cls._node_from_evidence(evidence, node_id)) is not None
        ]

    def _rank_cases(
        self,
        *,
        seed_id: str,
        question: str,
        evidence: RetrievedRulingEvidence,
    ) -> list[PilotCaseMatch]:
        candidates = [case for case in self.case_specs if case.seed_id == seed_id]
        if not candidates:
            return []
        question_text = _normalize_question(question)
        evidence_ids = set(evidence.nodes_by_logical_id)
        scored: list[PilotCaseMatch] = []
        for case in candidates:
            question_score = max(
                _token_overlap_score(question_text, _normalize_question(accepted))
                for accepted in case.accepted_questions
            )
            if question_score <= 0.0:
                continue
            reference_ids = {
                case.primary_rule_id,
                case.primary_source_id,
                *case.modifying_rule_ids,
                *case.modifying_source_ids,
                *case.supporting_rule_ids,
                *case.supporting_source_ids,
            }
            matched_reference_ids = sorted(ref_id for ref_id in reference_ids if ref_id in evidence_ids)
            evidence_score = len(matched_reference_ids)
            total = (question_score * 1000.0) + evidence_score
            scored.append(
                PilotCaseMatch(
                    question_id=case.question_id,
                    seed_id=case.seed_id,
                    question_score=question_score,
                    evidence_score=evidence_score,
                    total_score=total,
                    matched_reference_ids=matched_reference_ids,
                )
            )
        return sorted(scored, key=lambda item: item.total_score, reverse=True)

    def _infer_seed_from_question(self, normalized_question: str) -> PilotSeedInference:
        scored: list[PilotSeedScore] = []
        for case in self.case_specs:
            score = max(
                _token_overlap_score(normalized_question, _normalize_question(accepted))
                for accepted in case.accepted_questions
            )
            if score > 0.0:
                scored.append(PilotSeedScore(seed_id=case.seed_id, score=score))
        if not scored:
            return PilotSeedInference(normalized_question=normalized_question)
        deduped = sorted(
            {
                candidate.seed_id: candidate
                for candidate in sorted(scored, key=lambda item: item.score, reverse=True)
            }.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        best = deduped[0]
        return PilotSeedInference(
            normalized_question=normalized_question,
            selected_seed_id=best.seed_id if best.score >= 0.14 else None,
            selected_score=best.score,
            candidates=deduped,
        )

    def _case_spec_by_question_id(self, question_id: str) -> _PilotCaseSpec | None:
        for case in self.case_specs:
            if case.question_id == question_id:
                return case
        return None

    @staticmethod
    def _node_from_similar_node(node: SimilarNode) -> RetrievedEvidenceNode:
        payload = {"content": node.content, "node_type": node.node_type, **node.metadata}
        return LivePilotRulingEngine._node_from_payload(node.node_id, payload, similarity_score=node.similarity_score)

    @staticmethod
    def _node_from_payload(
        storage_node_id: str,
        payload: dict[str, Any],
        *,
        similarity_score: float | None = None,
    ) -> RetrievedEvidenceNode:
        return RetrievedEvidenceNode(
            storage_node_id=storage_node_id,
            logical_node_id=str(payload.get("logical_node_id", storage_node_id)),
            node_kind=str(payload.get("node_kind", payload.get("node_type", ""))),
            title=str(payload.get("title", "")),
            content=str(payload.get("content", "")),
            rule_kind=(str(payload["rule_kind"]) if "rule_kind" in payload else None),
            normalized_statement=(
                str(payload["normalized_statement"])
                if "normalized_statement" in payload
                else None
            ),
            citation_label=(str(payload["citation_label"]) if "citation_label" in payload else None),
            citation_short=(str(payload["citation_short"]) if "citation_short" in payload else None),
            locator=(str(payload["locator"]) if "locator" in payload else None),
            authority_scope=(str(payload["authority_scope"]) if "authority_scope" in payload else None),
            seed_id=(str(payload["seed_id"]) if "seed_id" in payload else None),
            subsystem=(str(payload["subsystem"]) if "subsystem" in payload else None),
            metadata={key: value for key, value in payload.items()},
            similarity_score=similarity_score,
        )

    @staticmethod
    def _filter_similar_nodes(
        similar_nodes: list[SimilarNode],
        *,
        seed_id: str | None,
    ) -> list[SimilarNode]:
        if seed_id is not None:
            filtered = [
                node
                for node in similar_nodes
                if str(node.metadata.get("seed_id", "")) == seed_id
            ]
            if filtered:
                return filtered
        return similar_nodes

    @staticmethod
    def _abstain(
        *,
        question: str,
        seed_id: str,
        subsystem: str,
        question_id: str,
        uncertainty: str,
    ) -> RulesRulingResult:
        return RulesRulingResult(
            question=question,
            question_id=question_id,
            seed_id=seed_id,
            subsystem=subsystem,
            ruling="I cannot assemble a sufficiently supported live ruling for that question.",
            uncertainty=uncertainty,
            abstain=True,
            backend="live-pilot",
        )


def _dominant_seed_id(nodes: Any) -> str | None:
    counts: dict[str, int] = {}
    for node in nodes:
        seed_id = getattr(node, "seed_id", None)
        if seed_id is None:
            continue
        counts[seed_id] = counts.get(seed_id, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return len(overlap) / max(len(left_tokens), len(right_tokens))


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "at",
        "can",
        "do",
        "does",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "s",
        "seti",
        "still",
        "that",
        "the",
        "there",
        "to",
        "what",
        "with",
        "you",
        "your",
    }
    return {token for token in text.split() if token and token not in stopwords}


def _build_case_specs() -> tuple[_PilotCaseSpec, ...]:
    return (
        _PilotCaseSpec(
            question_id="seti-rules-001-orbiter-cannot-land",
            seed_id="seti_landing_orbiter_seed_v1",
            accepted_questions=("Can an orbiter later land on the same planet?",),
            ruling="No. Once a probe becomes an orbiter, it remains an orbiter for the rest of the game and cannot later land on that planet.",
            primary_rule_id="rule_seti_orbiter_is_permanent",
            primary_source_id="src_seti_faq_q5_land_with_orbiter",
            supporting_rule_ids=("rule_seti_orbiter_status_change", "rule_seti_orbit_action_base"),
            supporting_source_ids=("src_seti_core_orbit_a_planet",),
            precedence_summaries=(
                (
                    "FAQ Q5 directly controls the later-landing question and states that an orbiter cannot land.",
                    "rule_seti_orbiter_is_permanent",
                    "src_seti_faq_q5_land_with_orbiter",
                    "primary",
                ),
                (
                    "The core orbit action establishes the state change from probe to orbiter that the FAQ clarification relies on.",
                    "rule_seti_orbiter_status_change",
                    "src_seti_core_orbit_a_planet",
                    "support",
                ),
            ),
        ),
        _PilotCaseSpec(
            question_id="seti-rules-002-opponent-orbiter-discount",
            seed_id="seti_landing_orbiter_seed_v1",
            accepted_questions=("Does an opponent's orbiter still reduce my landing cost?",),
            ruling="Yes. The base landing discount applies whenever an orbiter is present, and FAQ Q6 clarifies that the orbiter's owner does not matter.",
            primary_rule_id="rule_seti_landing_discount_if_orbiter_present",
            primary_source_id="src_seti_core_land_on_planet_or_moon",
            modifying_rule_ids=("rule_seti_landing_discount_not_owner_limited",),
            modifying_source_ids=("src_seti_faq_q6_opponent_orbiter_discount",),
            precedence_summaries=(
                (
                    "The core landing rule supplies the base discount when an orbiter is already at the planet.",
                    "rule_seti_landing_discount_if_orbiter_present",
                    "src_seti_core_land_on_planet_or_moon",
                    "primary",
                ),
                (
                    "FAQ Q6 resolves the ownership ambiguity by clarifying that the discount applies regardless of which player owns the orbiter.",
                    "rule_seti_landing_discount_not_owner_limited",
                    "src_seti_faq_q6_opponent_orbiter_discount",
                    "modifier",
                ),
            ),
        ),
        _PilotCaseSpec(
            question_id="seti-rules-003-moon-discount",
            seed_id="seti_landing_orbiter_seed_v1",
            accepted_questions=("Does an existing orbiter also reduce the cost to land on that planet's moon?",),
            ruling="Yes, if you are otherwise allowed to land on that moon. FAQ Q7 says moon landings inherit the planet's orbiter discount, while the core landing rule still requires a separate effect or tech to access moons.",
            primary_rule_id="rule_seti_moon_landing_inherits_planet_discount_logic",
            primary_source_id="src_seti_faq_q7_moon_discount",
            modifying_rule_ids=("rule_seti_moon_landing_requires_access",),
            modifying_source_ids=("src_seti_core_land_on_planet_or_moon",),
            supporting_rule_ids=("rule_seti_landing_discount_if_orbiter_present",),
            supporting_source_ids=("src_seti_core_land_on_planet_or_moon",),
            precedence_summaries=(
                (
                    "FAQ Q7 directly extends the orbiter discount logic from planets to moons.",
                    "rule_seti_moon_landing_inherits_planet_discount_logic",
                    "src_seti_faq_q7_moon_discount",
                    "primary",
                ),
                (
                    "The core landing rule remains a prerequisite because moons still need a separate access-enabling effect or tech.",
                    "rule_seti_moon_landing_requires_access",
                    "src_seti_core_land_on_planet_or_moon",
                    "modifier",
                ),
            ),
        ),
        _PilotCaseSpec(
            question_id="seti-rules-004-opponent-orbiter-moon-discount",
            seed_id="seti_landing_orbiter_seed_v1",
            accepted_questions=(
                "If another player's orbiter is at Jupiter and I have the tech that lets me land on moons, do I get the discount when landing on one of Jupiter's moons?",
            ),
            ruling="Yes. Once your tech lets you land on the moon, FAQ Q7 carries the orbiter discount over to moon landings and FAQ Q6 clarifies that another player's orbiter still counts for that discount.",
            primary_rule_id="rule_seti_moon_landing_inherits_planet_discount_logic",
            primary_source_id="src_seti_faq_q7_moon_discount",
            modifying_rule_ids=(
                "rule_seti_landing_discount_not_owner_limited",
                "rule_seti_moon_landing_requires_access",
            ),
            modifying_source_ids=(
                "src_seti_faq_q6_opponent_orbiter_discount",
                "src_seti_core_land_on_planet_or_moon",
            ),
            precedence_summaries=(
                (
                    "FAQ Q7 makes moon landings follow the same orbiter discount logic as the planet once moon access exists.",
                    "rule_seti_moon_landing_inherits_planet_discount_logic",
                    "src_seti_faq_q7_moon_discount",
                    "primary",
                ),
                (
                    "FAQ Q6 removes any ownership limit on the relevant orbiter.",
                    "rule_seti_landing_discount_not_owner_limited",
                    "src_seti_faq_q6_opponent_orbiter_discount",
                    "modifier",
                ),
                (
                    "The core landing rule remains a prerequisite because the moon still requires enabling tech or another effect.",
                    "rule_seti_moon_landing_requires_access",
                    "src_seti_core_land_on_planet_or_moon",
                    "modifier",
                ),
            ),
        ),
        _PilotCaseSpec(
            question_id="seti-rules-012-free-action-timing",
            seed_id="seti_free_action_authority_seed_v1",
            accepted_questions=(
                "Can I interrupt a main action with a free action, and can I interrupt one free action with another free action?",
            ),
            ruling="You can interrupt a main action with a free action, but you cannot interrupt one free action with another. A free action must fully resolve before the next free action begins.",
            primary_rule_id="rule_seti_free_actions_can_interrupt_main_action",
            primary_source_id="src_seti_faq_q4_free_action_timing",
            modifying_rule_ids=(
                "rule_seti_free_actions_cannot_interrupt_free_action",
                "rule_seti_free_action_must_resolve_before_next",
            ),
            modifying_source_ids=("src_seti_faq_q4_free_action_timing",),
            precedence_summaries=(
                (
                    "The FAQ timing paragraph authorizes a free action to interrupt a main action on your turn.",
                    "rule_seti_free_actions_can_interrupt_main_action",
                    "src_seti_faq_q4_free_action_timing",
                    "primary",
                ),
                (
                    "That same FAQ clarification forbids starting a second free action before the first one has fully resolved.",
                    "rule_seti_free_actions_cannot_interrupt_free_action",
                    "src_seti_faq_q4_free_action_timing",
                    "modifier",
                ),
            ),
        ),
        _PilotCaseSpec(
            question_id="seti-rules-021-scan-interrupt-publicity",
            seed_id="seti_free_action_authority_seed_v1",
            accepted_questions=(
                "During a Scan action, can I interrupt the Scan with a free action and then continue the Scan?",
            ),
            ruling="Yes. The Scan example in the FAQ shows that you can interrupt the Scan with a free action and then continue resolving the original main action afterward.",
            primary_rule_id="rule_seti_scan_action_can_resume_after_free_action_interrupt",
            primary_source_id="src_seti_faq_q4_scan_example",
            modifying_rule_ids=("rule_seti_free_actions_can_interrupt_main_action",),
            modifying_source_ids=("src_seti_faq_q4_free_action_timing",),
            precedence_summaries=(
                (
                    "The Scan worked example directly demonstrates that the interrupted main action can resume after the free action resolves.",
                    "rule_seti_scan_action_can_resume_after_free_action_interrupt",
                    "src_seti_faq_q4_scan_example",
                    "primary",
                ),
                (
                    "The broader FAQ timing rule supplies the general permission for free actions to interrupt main actions.",
                    "rule_seti_free_actions_can_interrupt_main_action",
                    "src_seti_faq_q4_free_action_timing",
                    "support",
                ),
            ),
        ),
        _PilotCaseSpec(
            question_id="seti-rules-022-free-action-income-nesting",
            seed_id="seti_free_action_authority_seed_v1",
            accepted_questions=(
                "If placing data triggers the income increase effect, can I place more data first and then resolve that income effect?",
            ),
            ruling="No. The triggered income-increase free action has to resolve before you begin another free action, so you cannot place more data first and return to that effect later.",
            primary_rule_id="rule_seti_income_increase_free_action_cannot_be_nested",
            primary_source_id="src_seti_faq_q4_income_example",
            modifying_rule_ids=("rule_seti_free_action_must_resolve_before_next",),
            modifying_source_ids=("src_seti_faq_q4_free_action_timing",),
            precedence_summaries=(
                (
                    "The income example directly answers the nesting question by requiring the triggered free action to resolve immediately.",
                    "rule_seti_income_increase_free_action_cannot_be_nested",
                    "src_seti_faq_q4_income_example",
                    "primary",
                ),
                (
                    "The general FAQ timing rule explains why: one free action must fully resolve before another begins.",
                    "rule_seti_free_action_must_resolve_before_next",
                    "src_seti_faq_q4_free_action_timing",
                    "support",
                ),
            ),
        ),
        _PilotCaseSpec(
            question_id="seti-rules-028-player-aid-free-action-summary-conflict",
            seed_id="seti_free_action_authority_seed_v1",
            accepted_questions=(
                'The player aid says free actions are "NO LIMIT". Does that allow one free action to interrupt another?',
            ),
            ruling='No. The player aid summary does not authorize free-action nesting. The FAQ timing clarification controls here and requires each free action to fully resolve before another one begins.',
            primary_rule_id="rule_seti_free_actions_cannot_interrupt_free_action",
            primary_source_id="src_seti_faq_q4_free_action_timing",
            modifying_rule_ids=("rule_seti_player_aid_free_actions_no_limit_summary",),
            modifying_source_ids=("src_seti_aid_free_actions_summary",),
            supporting_rule_ids=("rule_seti_free_action_must_resolve_before_next",),
            supporting_source_ids=("src_seti_faq_q4_free_action_timing",),
            precedence_summaries=(
                (
                    "The FAQ timing clarification is the controlling authority for whether one free action may interrupt another.",
                    "rule_seti_free_actions_cannot_interrupt_free_action",
                    "src_seti_faq_q4_free_action_timing",
                    "primary",
                ),
                (
                    'The player aid "NO LIMIT" line is summary-only and cannot override the FAQ timing restriction.',
                    "rule_seti_player_aid_free_actions_no_limit_summary",
                    "src_seti_aid_free_actions_summary",
                    "authority",
                ),
                (
                    "The FAQ also states that each free action must resolve before the next one begins, which explains the limit on nesting.",
                    "rule_seti_free_action_must_resolve_before_next",
                    "src_seti_faq_q4_free_action_timing",
                    "support",
                ),
            ),
        ),
    )
