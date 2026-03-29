#!/usr/bin/env python3
"""Import a manual seed fixture into Vector Graph Memory.

This script reads a seed manifest plus JSONL node and edge files, then loads the
records into the existing Qdrant + JanusGraph backend through VectorGraphStore.

The importer is intentionally idempotent at the JanusGraph layer:

- nodes are skipped if a vertex with the same `node_id` already exists
- edges are skipped if an edge with the same `edge_id` already exists

Usage:
    uv run python scripts/import_manual_seed.py
    uv run python scripts/import_manual_seed.py \
        --manifest tests/fixtures/rag_eval/seti_landing_orbiter_seed_v1_manifest.json
    uv run python scripts/import_manual_seed.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gremlin_python.driver import client as gremlin_client  # type: ignore[import-untyped]
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from qdrant_client import QdrantClient

from vgm.VectorGraphStore import VectorGraphStore
from vgm.schemas import EdgeMetadata, NodeMetadata


DEFAULT_MANIFEST = Path(
    "tests/fixtures/rag_eval/seti_landing_orbiter_seed_v1_manifest.json"
)
SEED_NAMESPACE_PREFIX = "vgm-manual-seed"


def _parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO datetime with optional trailing Z."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
    return records


def _janus_exists(janus: gremlin_client.Client, gremlin_query: str) -> bool:
    result = janus.submit(gremlin_query).all().result()
    if not result:
        return False
    first = result[0]
    try:
        return int(first) > 0
    except (TypeError, ValueError):
        return bool(first)


def _stable_uuid(seed_id: str, logical_id: str) -> str:
    """Create a deterministic UUID for a logical seed identifier."""
    return str(uuid5(NAMESPACE_URL, f"{SEED_NAMESPACE_PREFIX}:{seed_id}:{logical_id}"))


def _node_exists(janus: gremlin_client.Client, node_id: str) -> bool:
    escaped = node_id.replace("\\", "\\\\").replace("'", "\\'")
    query = f"g.V().has('node_id', '{escaped}').limit(1).count()"
    return _janus_exists(janus, query)


def _edge_exists(janus: gremlin_client.Client, edge_id: str) -> bool:
    escaped = edge_id.replace("\\", "\\\\").replace("'", "\\'")
    query = f"g.E().has('edge_id', '{escaped}').limit(1).count()"
    return _janus_exists(janus, query)


def _build_node_metadata(record: dict[str, Any]) -> NodeMetadata:
    seed_id = record.get("custom_metadata", {}).get("seed_id", "manual_seed")
    logical_node_id = record["node_id"]
    custom_metadata = dict(record.get("custom_metadata", {}))
    custom_metadata["logical_node_id"] = logical_node_id

    return NodeMetadata(
        node_id=_stable_uuid(seed_id, logical_node_id),
        node_type=record["node_type"],
        content=record["content"],
        created_at=_parse_iso_datetime(record["created_at"]),
        updated_at=_parse_iso_datetime(record["updated_at"]),
        source=record["source"],
        project_id=record["project_id"],
        embedding_model=record["embedding_model"],
        custom_metadata=custom_metadata,
    )


def _build_edge_metadata(record: dict[str, Any]) -> EdgeMetadata:
    seed_id = record.get("custom_metadata", {}).get("seed_id", "manual_seed")
    logical_edge_id = record["edge_id"]
    logical_from = record["from_node_id"]
    logical_to = record["to_node_id"]
    custom_metadata = dict(record.get("custom_metadata", {}))
    custom_metadata["logical_edge_id"] = logical_edge_id
    custom_metadata["logical_from_node_id"] = logical_from
    custom_metadata["logical_to_node_id"] = logical_to

    return EdgeMetadata(
        edge_id=_stable_uuid(seed_id, logical_edge_id),
        from_node_id=_stable_uuid(seed_id, logical_from),
        to_node_id=_stable_uuid(seed_id, logical_to),
        relationship_type=record["relationship_type"],
        description=record.get("description", ""),
        created_at=_parse_iso_datetime(record["created_at"]),
        source=record["source"],
        project_id=record["project_id"],
        confidence=record.get("confidence"),
        custom_metadata=custom_metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to the seed manifest JSON (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report what would be imported without writing anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = _load_json(manifest_path)
    node_path = Path(manifest["node_file"])
    edge_path = Path(manifest["edge_file"])

    if not node_path.exists():
        raise FileNotFoundError(f"Node file not found: {node_path}")
    if not edge_path.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_path}")

    nodes = _load_jsonl(node_path)
    edges = _load_jsonl(edge_path)

    expected_node_count = manifest.get("node_count")
    expected_edge_count = manifest.get("edge_count")
    if expected_node_count is not None and len(nodes) != expected_node_count:
        raise ValueError(
            f"Manifest node_count={expected_node_count}, but loaded {len(nodes)} nodes"
        )
    if expected_edge_count is not None and len(edges) != expected_edge_count:
        raise ValueError(
            f"Manifest edge_count={expected_edge_count}, but loaded {len(edges)} edges"
        )

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
    janusgraph_host = os.getenv("JANUSGRAPH_HOST", "localhost")
    janusgraph_port = int(os.getenv("JANUSGRAPH_PORT", "8182"))
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    if args.dry_run:
        print(f"Dry run for seed: {manifest['seed_id']}")
        print(f"Manifest: {manifest_path}")
        print(f"Nodes: {len(nodes)} from {node_path}")
        print(f"Edges: {len(edges)} from {edge_path}")
        print(f"Qdrant target: {qdrant_host}:{qdrant_port}")
        print(f"JanusGraph target: {janusgraph_host}:{janusgraph_port}")
        print(f"Embedding model: {embedding_model_name}")
        return 0

    print(f"Importing seed: {manifest['seed_id']}")
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
    print(f"Connecting to JanusGraph at {janusgraph_host}:{janusgraph_port}")

    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    janus = gremlin_client.Client(
        f"ws://{janusgraph_host}:{janusgraph_port}/gremlin", "g"
    )
    embedding_model = OpenAIEmbeddingModel(embedding_model_name)
    store = VectorGraphStore(
        qdrant_client=qdrant,
        janus_client=janus,
        embedding_model=embedding_model,
    )

    inserted_nodes = 0
    skipped_nodes = 0
    inserted_edges = 0
    skipped_edges = 0

    try:
        for record in nodes:
            seed_id = record.get("custom_metadata", {}).get("seed_id", "manual_seed")
            logical_node_id = record["node_id"]
            node_id = _stable_uuid(seed_id, logical_node_id)
            if _node_exists(janus, node_id):
                skipped_nodes += 1
                print(f"Skip node: {logical_node_id} -> {node_id}")
                continue

            metadata = _build_node_metadata(record)
            store.add_node(metadata)
            inserted_nodes += 1
            print(f"Add node: {logical_node_id} -> {node_id}")

        for record in edges:
            seed_id = record.get("custom_metadata", {}).get("seed_id", "manual_seed")
            logical_edge_id = record["edge_id"]
            edge_id = _stable_uuid(seed_id, logical_edge_id)
            if _edge_exists(janus, edge_id):
                skipped_edges += 1
                print(f"Skip edge: {logical_edge_id} -> {edge_id}")
                continue

            metadata = _build_edge_metadata(record)
            store.add_edge(metadata)
            inserted_edges += 1
            print(f"Add edge: {logical_edge_id} -> {edge_id}")
    finally:
        janus.close()

    print("Import complete.")
    print(f"Nodes inserted: {inserted_nodes}")
    print(f"Nodes skipped: {skipped_nodes}")
    print(f"Edges inserted: {inserted_edges}")
    print(f"Edges skipped: {skipped_edges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
