#!/usr/bin/env python3
"""Verify that a manual seed fixture is present in Vector Graph Memory.

This script checks that the expected nodes and edges from a seed manifest exist
in the live Qdrant + JanusGraph backend. It also verifies that the four frozen
SETI landing/orbiter support paths are structurally satisfiable by checking that
the required node IDs are present.

Usage:
    uv run python scripts/verify_manual_seed.py
    uv run python scripts/verify_manual_seed.py \
        --manifest tests/fixtures/rag_eval/seti_landing_orbiter_seed_v1_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gremlin_python.driver import client as gremlin_client  # type: ignore[import-untyped]
from qdrant_client import QdrantClient


DEFAULT_MANIFEST = Path(
    "tests/fixtures/rag_eval/seti_landing_orbiter_seed_v1_manifest.json"
)
SEED_NAMESPACE_PREFIX = "vgm-manual-seed"

EXPECTED_SUPPORT_PATHS: dict[str, list[str]] = {
    "q1_orbiter_can_it_land": [
        "rule_seti_orbiter_status_change",
        "rule_seti_orbiter_is_permanent",
        "src_seti_faq_q5_land_with_orbiter",
    ],
    "q2_opponent_orbiter_discount": [
        "rule_seti_landing_discount_if_orbiter_present",
        "rule_seti_landing_discount_not_owner_limited",
        "src_seti_faq_q6_opponent_orbiter_discount",
        "src_seti_core_land_on_planet_or_moon",
    ],
    "q3_moon_discount": [
        "rule_seti_moon_landing_inherits_planet_discount_logic",
        "rule_seti_landing_discount_if_orbiter_present",
        "rule_seti_moon_landing_requires_access",
        "src_seti_faq_q7_moon_discount",
    ],
    "q4_opponent_orbiter_moon_discount": [
        "rule_seti_landing_discount_not_owner_limited",
        "rule_seti_moon_landing_inherits_planet_discount_logic",
        "rule_seti_moon_landing_requires_access",
        "rule_seti_landing_discount_if_orbiter_present",
        "src_seti_faq_q6_opponent_orbiter_discount",
        "src_seti_faq_q7_moon_discount",
    ],
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _stable_uuid(seed_id: str, logical_id: str) -> str:
    """Create a deterministic UUID for a logical seed identifier."""
    return str(uuid5(NAMESPACE_URL, f"{SEED_NAMESPACE_PREFIX}:{seed_id}:{logical_id}"))


def _janus_count(janus: gremlin_client.Client, gremlin_query: str) -> int:
    result = janus.submit(gremlin_query).all().result()
    if not result:
        return 0
    return int(result[0])


def _node_exists_qdrant(qdrant: QdrantClient, collection: str, node_id: str) -> bool:
    return bool(qdrant.retrieve(collection_name=collection, ids=[node_id]))


def _node_exists_janus(janus: gremlin_client.Client, node_id: str) -> bool:
    escaped = node_id.replace("\\", "\\\\").replace("'", "\\'")
    query = f"g.V().has('node_id', '{escaped}').limit(1).count()"
    return _janus_count(janus, query) > 0


def _edge_exists_janus(janus: gremlin_client.Client, edge_id: str) -> bool:
    escaped = edge_id.replace("\\", "\\\\").replace("'", "\\'")
    query = f"g.E().has('edge_id', '{escaped}').limit(1).count()"
    return _janus_count(janus, query) > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to the seed manifest JSON (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--collection",
        default="vgm_memory",
        help="Qdrant collection name to verify against (default: vgm_memory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _load_json(args.manifest)
    node_records = _load_jsonl(Path(manifest["node_file"]))
    edge_records = _load_jsonl(Path(manifest["edge_file"]))

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
    janusgraph_host = os.getenv("JANUSGRAPH_HOST", "localhost")
    janusgraph_port = int(os.getenv("JANUSGRAPH_PORT", "8182"))

    print(f"Verifying seed: {manifest['seed_id']}")
    print(f"Qdrant target: {qdrant_host}:{qdrant_port}")
    print(f"JanusGraph target: {janusgraph_host}:{janusgraph_port}")

    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    janus = gremlin_client.Client(
        f"ws://{janusgraph_host}:{janusgraph_port}/gremlin", "g"
    )

    missing_qdrant_nodes: list[str] = []
    missing_janus_nodes: list[str] = []
    missing_janus_edges: list[str] = []

    try:
        for record in node_records:
            seed_id = record.get("custom_metadata", {}).get("seed_id", manifest["seed_id"])
            logical_node_id = record["node_id"]
            node_id = _stable_uuid(seed_id, logical_node_id)
            if not _node_exists_qdrant(qdrant, args.collection, node_id):
                missing_qdrant_nodes.append(logical_node_id)
            if not _node_exists_janus(janus, node_id):
                missing_janus_nodes.append(logical_node_id)

        for record in edge_records:
            seed_id = record.get("custom_metadata", {}).get("seed_id", manifest["seed_id"])
            logical_edge_id = record["edge_id"]
            edge_id = _stable_uuid(seed_id, logical_edge_id)
            if not _edge_exists_janus(janus, edge_id):
                missing_janus_edges.append(logical_edge_id)

        all_present_node_ids = {
            record["node_id"]
            for record in node_records
            if record["node_id"] not in missing_qdrant_nodes
            and record["node_id"] not in missing_janus_nodes
        }

        missing_support_paths: dict[str, list[str]] = {}
        for path_name, required_node_ids in EXPECTED_SUPPORT_PATHS.items():
            missing = [node_id for node_id in required_node_ids if node_id not in all_present_node_ids]
            if missing:
                missing_support_paths[path_name] = missing
    finally:
        janus.close()

    print(f"Expected nodes: {len(node_records)}")
    print(f"Expected edges: {len(edge_records)}")
    print(f"Missing Qdrant nodes: {len(missing_qdrant_nodes)}")
    print(f"Missing Janus nodes: {len(missing_janus_nodes)}")
    print(f"Missing Janus edges: {len(missing_janus_edges)}")

    if missing_qdrant_nodes:
        print("Qdrant node misses:")
        for node_id in missing_qdrant_nodes:
            print(f"  - {node_id}")

    if missing_janus_nodes:
        print("Janus node misses:")
        for node_id in missing_janus_nodes:
            print(f"  - {node_id}")

    if missing_janus_edges:
        print("Janus edge misses:")
        for edge_id in missing_janus_edges:
            print(f"  - {edge_id}")

    if missing_support_paths:
        print("Missing support-path requirements:")
        for path_name, node_ids in missing_support_paths.items():
            print(f"  - {path_name}: {', '.join(node_ids)}")

    if missing_qdrant_nodes or missing_janus_nodes or missing_janus_edges or missing_support_paths:
        return 1

    print("Verification successful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
