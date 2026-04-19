"""Typed contracts for rules extraction and manual-seed interchange."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SourcePassage(BaseModel):
    """A cited source fragment that can support one or more canonical rules."""

    node_id: str
    game_id: str
    document_id: str
    document_type: str
    authority_scope: str
    title: str
    locator: str
    page: int | None = None
    source_text: str
    citation_label: str
    citation_short: str
    language: str = "en"
    subsystem: str | None = None
    content: str | None = None

    @property
    def rendered_content(self) -> str:
        return self.content or self.source_text


class CanonicalRule(BaseModel):
    """A normalized rule statement intended for graph reasoning."""

    node_id: str
    game_id: str
    rule_kind: str
    title: str
    normalized_statement: str
    scope: str
    subsystem: str | None = None
    content: str | None = None

    @property
    def rendered_content(self) -> str:
        return self.content or self.normalized_statement


class RuleEdge(BaseModel):
    """A directed semantic relationship between rule entities."""

    edge_id: str
    edge_type: str
    from_node_id: str
    to_node_id: str
    rationale: str
    description: str | None = None
    confidence: float | None = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def rendered_description(self) -> str:
        return self.description or self.rationale


class ExtractionUncertainty(BaseModel):
    """A parser-side ambiguity that should be surfaced rather than guessed through."""

    kind: str
    note: str
    source_ref: str | None = None
    related_ids: list[str] = Field(default_factory=list)


class RuleExtractionBundle(BaseModel):
    """The structured output contract for one bounded rules extraction run."""

    seed_id: str
    game_id: str
    scope: str = "rules_only"
    subsystem: str
    project_id: str
    source: str
    created_at: datetime
    updated_at: datetime
    embedding_model: str = "manual-seed-unembedded"
    related_docs: list[str] = Field(default_factory=list)
    frozen_questions: list[str] = Field(default_factory=list)
    source_passages: list[SourcePassage] = Field(default_factory=list)
    canonical_rules: list[CanonicalRule] = Field(default_factory=list)
    edges: list[RuleEdge] = Field(default_factory=list)
    uncertainties: list[ExtractionUncertainty] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_game_ids(self) -> "RuleExtractionBundle":
        for passage in self.source_passages:
            if passage.game_id != self.game_id:
                raise ValueError(
                    f"Source passage {passage.node_id} game_id={passage.game_id} "
                    f"does not match bundle game_id={self.game_id}"
                )
        for rule in self.canonical_rules:
            if rule.game_id != self.game_id:
                raise ValueError(
                    f"Canonical rule {rule.node_id} game_id={rule.game_id} "
                    f"does not match bundle game_id={self.game_id}"
                )
        return self

    @property
    def nodes(self) -> list[SourcePassage | CanonicalRule]:
        return [*self.source_passages, *self.canonical_rules]


def _isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def build_seed_node_records(bundle: RuleExtractionBundle) -> list[dict[str, Any]]:
    """Render a rules extraction bundle to the current manual-seed node JSONL shape."""

    records: list[dict[str, Any]] = []

    for passage in bundle.source_passages:
        custom_metadata: dict[str, Any] = {
            "seed_id": bundle.seed_id,
            "game_id": bundle.game_id,
            "node_kind": "source_passage",
            "document_id": passage.document_id,
            "document_type": passage.document_type,
            "authority_scope": passage.authority_scope,
            "title": passage.title,
            "locator": passage.locator,
            "page": passage.page,
            "citation_label": passage.citation_label,
            "citation_short": passage.citation_short,
            "language": passage.language,
        }
        if passage.subsystem is not None:
            custom_metadata["subsystem"] = passage.subsystem

        records.append(
            {
                "node_id": passage.node_id,
                "node_type": "source_passage",
                "content": passage.rendered_content,
                "created_at": _isoformat(bundle.created_at),
                "updated_at": _isoformat(bundle.updated_at),
                "source": bundle.source,
                "project_id": bundle.project_id,
                "embedding_model": bundle.embedding_model,
                "custom_metadata": custom_metadata,
            }
        )

    for rule in bundle.canonical_rules:
        custom_metadata = {
            "seed_id": bundle.seed_id,
            "game_id": bundle.game_id,
            "node_kind": "canonical_rule",
            "rule_kind": rule.rule_kind,
            "title": rule.title,
            "normalized_statement": rule.normalized_statement,
            "scope": rule.scope,
        }
        if rule.subsystem is not None:
            custom_metadata["subsystem"] = rule.subsystem

        records.append(
            {
                "node_id": rule.node_id,
                "node_type": "canonical_rule",
                "content": rule.rendered_content,
                "created_at": _isoformat(bundle.created_at),
                "updated_at": _isoformat(bundle.updated_at),
                "source": bundle.source,
                "project_id": bundle.project_id,
                "embedding_model": bundle.embedding_model,
                "custom_metadata": custom_metadata,
            }
        )

    return records


def build_seed_edge_records(bundle: RuleExtractionBundle) -> list[dict[str, Any]]:
    """Render a rules extraction bundle to the current manual-seed edge JSONL shape."""

    records: list[dict[str, Any]] = []

    for edge in bundle.edges:
        custom_metadata: dict[str, Any] = {
            "seed_id": bundle.seed_id,
            "game_id": bundle.game_id,
            "rationale": edge.rationale,
        }
        if bundle.subsystem:
            custom_metadata["subsystem"] = bundle.subsystem

        records.append(
            {
                "edge_id": edge.edge_id,
                "from_node_id": edge.from_node_id,
                "to_node_id": edge.to_node_id,
                "relationship_type": edge.edge_type,
                "description": edge.rendered_description,
                "created_at": _isoformat(bundle.created_at),
                "source": bundle.source,
                "project_id": bundle.project_id,
                "confidence": edge.confidence,
                "custom_metadata": custom_metadata,
            }
        )

    return records


def build_seed_manifest(
    bundle: RuleExtractionBundle,
    *,
    node_file: str,
    edge_file: str,
) -> dict[str, Any]:
    """Render bundle metadata to the current seed manifest shape."""

    return {
        "seed_id": bundle.seed_id,
        "game_id": bundle.game_id,
        "scope": bundle.scope,
        "subsystem": bundle.subsystem,
        "project_id": bundle.project_id,
        "source": bundle.source,
        "node_count": len(bundle.nodes),
        "edge_count": len(bundle.edges),
        "node_file": node_file,
        "edge_file": edge_file,
        "related_docs": bundle.related_docs,
        "frozen_questions": bundle.frozen_questions,
    }


def write_seed_fixture(
    bundle: RuleExtractionBundle,
    *,
    output_dir: str | Path,
    manifest_filename: str | None = None,
    node_filename: str | None = None,
    edge_filename: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write a typed bundle back to manifest/node/edge seed fixture files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_filename = manifest_filename or f"{bundle.seed_id}_manifest.json"
    node_filename = node_filename or f"{bundle.seed_id}_nodes.jsonl"
    edge_filename = edge_filename or f"{bundle.seed_id}_edges.jsonl"

    manifest_path = output_dir / manifest_filename
    node_path = output_dir / node_filename
    edge_path = output_dir / edge_filename

    node_records = build_seed_node_records(bundle)
    edge_records = build_seed_edge_records(bundle)
    manifest = build_seed_manifest(
        bundle,
        node_file=str(node_path),
        edge_file=str(edge_path),
    )

    node_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n" for record in node_records),
        encoding="utf-8",
    )
    edge_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n" for record in edge_records),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, node_path, edge_path


def load_bundle_from_seed_records(
    manifest: dict[str, Any],
    node_records: list[dict[str, Any]],
    edge_records: list[dict[str, Any]],
) -> RuleExtractionBundle:
    """Load the current seed fixture format into the typed extraction contract."""

    source_passages: list[SourcePassage] = []
    canonical_rules: list[CanonicalRule] = []

    created_at_values = {record["created_at"] for record in node_records}
    updated_at_values = {record["updated_at"] for record in node_records}
    source_values = {record["source"] for record in node_records}
    project_values = {record["project_id"] for record in node_records}
    embedding_values = {record["embedding_model"] for record in node_records}

    if len(created_at_values) != 1 or len(updated_at_values) != 1:
        raise ValueError("Seed nodes must currently share one created_at and updated_at value")
    if len(source_values) != 1 or len(project_values) != 1 or len(embedding_values) != 1:
        raise ValueError("Seed nodes must currently share one source, project_id, and embedding model")

    for record in node_records:
        custom_metadata = record.get("custom_metadata", {})
        if record["node_type"] == "source_passage":
            source_passages.append(
                SourcePassage(
                    node_id=record["node_id"],
                    game_id=custom_metadata["game_id"],
                    document_id=custom_metadata["document_id"],
                    document_type=custom_metadata["document_type"],
                    authority_scope=custom_metadata["authority_scope"],
                    title=custom_metadata["title"],
                    locator=custom_metadata["locator"],
                    page=custom_metadata.get("page"),
                    source_text=record["content"],
                    citation_label=custom_metadata["citation_label"],
                    citation_short=custom_metadata["citation_short"],
                    language=custom_metadata.get("language", "en"),
                    subsystem=custom_metadata.get("subsystem"),
                    content=record["content"],
                )
            )
            continue

        if record["node_type"] == "canonical_rule":
            canonical_rules.append(
                CanonicalRule(
                    node_id=record["node_id"],
                    game_id=custom_metadata["game_id"],
                    rule_kind=custom_metadata["rule_kind"],
                    title=custom_metadata["title"],
                    normalized_statement=custom_metadata["normalized_statement"],
                    scope=custom_metadata["scope"],
                    subsystem=custom_metadata.get("subsystem"),
                    content=record["content"],
                )
            )
            continue

        raise ValueError(f"Unsupported node_type in seed fixture: {record['node_type']}")

    edges = [
        RuleEdge(
            edge_id=record["edge_id"],
            edge_type=record["relationship_type"],
            from_node_id=record["from_node_id"],
            to_node_id=record["to_node_id"],
            rationale=record.get("custom_metadata", {}).get("rationale", record["description"]),
            description=record["description"],
            confidence=record.get("confidence"),
        )
        for record in edge_records
    ]

    return RuleExtractionBundle(
        seed_id=manifest["seed_id"],
        game_id=manifest["game_id"],
        scope=manifest.get("scope", "rules_only"),
        subsystem=manifest["subsystem"],
        project_id=next(iter(project_values)),
        source=next(iter(source_values)),
        created_at=datetime.fromisoformat(next(iter(created_at_values)).replace("Z", "+00:00")),
        updated_at=datetime.fromisoformat(next(iter(updated_at_values)).replace("Z", "+00:00")),
        embedding_model=next(iter(embedding_values)),
        related_docs=list(manifest.get("related_docs", [])),
        frozen_questions=list(manifest.get("frozen_questions", [])),
        source_passages=source_passages,
        canonical_rules=canonical_rules,
        edges=edges,
    )
