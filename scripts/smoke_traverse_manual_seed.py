#!/usr/bin/env python3
"""Run live traversal smoke checks for the SETI manual seed.

This script exercises the seeded graph for the first two frozen landing/orbiter
questions by traversing from the controlling rule nodes to the expected support
nodes and source passages.

Usage:
    uv run python scripts/smoke_traverse_manual_seed.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gremlin_python.driver import client as gremlin_client  # type: ignore[import-untyped]
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from qdrant_client import QdrantClient

from vgm.VectorGraphStore import VectorGraphStore


SEED_ID = "seti_landing_orbiter_seed_v1"
SEED_NAMESPACE_PREFIX = "vgm-manual-seed"


def _stable_uuid(logical_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{SEED_NAMESPACE_PREFIX}:{SEED_ID}:{logical_id}"))


def _logical_ids_from_results(results: list[dict]) -> set[str]:
    logical_ids: set[str] = set()
    for record in results:
        if not isinstance(record, dict):
            continue
        logical_node_id = record.get("logical_node_id")
        if logical_node_id:
            logical_ids.add(logical_node_id)
            continue
        metadata = record.get("custom_metadata", {})
        logical_node_id = metadata.get("logical_node_id")
        if logical_node_id:
            logical_ids.add(logical_node_id)
    return logical_ids


def main() -> int:
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
    janusgraph_host = os.getenv("JANUSGRAPH_HOST", "localhost")
    janusgraph_port = int(os.getenv("JANUSGRAPH_PORT", "8182"))

    print(f"Traversal smoke using Qdrant at {qdrant_host}:{qdrant_port}")
    print(f"Traversal smoke using JanusGraph at {janusgraph_host}:{janusgraph_port}")

    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    janus = gremlin_client.Client(
        f"ws://{janusgraph_host}:{janusgraph_port}/gremlin", "g"
    )
    embedding_model = OpenAIEmbeddingModel(os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    store = VectorGraphStore(
        qdrant_client=qdrant,
        janus_client=janus,
        embedding_model=embedding_model,
    )

    checks = [
        {
            "name": "q1_orbiter_can_it_land",
            "start_logical_id": "rule_seti_orbiter_is_permanent",
            "gremlin_steps": "both().values('node_id')",
            "expected_logical_ids": {
                "src_seti_faq_q5_land_with_orbiter",
                "rule_seti_orbiter_status_change",
            },
        },
        {
            "name": "q2_opponent_orbiter_discount",
            "start_logical_id": "rule_seti_landing_discount_if_orbiter_present",
            "gremlin_steps": "both().values('node_id')",
            "expected_logical_ids": {
                "src_seti_core_land_on_planet_or_moon",
                "rule_seti_landing_discount_not_owner_limited",
            },
        },
        {
            "name": "q3_moon_discount",
            "start_logical_id": "rule_seti_moon_landing_inherits_planet_discount_logic",
            "gremlin_steps": "both().values('node_id')",
            "expected_logical_ids": {
                "rule_seti_landing_discount_if_orbiter_present",
                "src_seti_faq_q7_moon_discount",
            },
        },
        {
            "name": "q4_opponent_orbiter_moon_discount",
            "start_logical_id": "rule_seti_moon_landing_inherits_planet_discount_logic",
            "gremlin_steps": "both().values('node_id')",
            "expected_logical_ids": {
                "rule_seti_landing_discount_if_orbiter_present",
                "src_seti_faq_q7_moon_discount",
            },
            "additional_checks": [
                {
                    "start_logical_id": "rule_seti_landing_discount_if_orbiter_present",
                    "expected_logical_ids": {
                        "rule_seti_landing_discount_not_owner_limited",
                    },
                },
                {
                    "start_logical_id": "rule_seti_moon_landing_requires_access",
                    "expected_logical_ids": {
                        "src_seti_faq_q7_moon_discount",
                        "src_seti_core_land_on_planet_or_moon",
                    },
                },
            ],
        },
    ]

    failures: list[str] = []
    try:
        for check in checks:
            start_uuid = _stable_uuid(check["start_logical_id"])
            results = store.traverse_from_node(
                node_id=start_uuid,
                gremlin_steps=check["gremlin_steps"],
            )
            found_logical_ids = _logical_ids_from_results(results)
            missing = sorted(check["expected_logical_ids"] - found_logical_ids)

            print(f"Check: {check['name']}")
            print(f"  Start: {check['start_logical_id']} -> {start_uuid}")
            print(f"  Found: {sorted(found_logical_ids)}")

            if missing:
                failures.append(f"{check['name']}: missing {', '.join(missing)}")
                print(f"  Missing: {missing}")
            else:
                print("  Result: ok")

            for extra_check in check.get("additional_checks", []):
                extra_start_logical_id = extra_check["start_logical_id"]
                extra_start_uuid = _stable_uuid(extra_start_logical_id)
                extra_results = store.traverse_from_node(
                    node_id=extra_start_uuid,
                    gremlin_steps="both().values('node_id')",
                )
                extra_found_logical_ids = _logical_ids_from_results(extra_results)
                extra_missing = sorted(
                    extra_check["expected_logical_ids"] - extra_found_logical_ids
                )

                print(f"  Additional start: {extra_start_logical_id} -> {extra_start_uuid}")
                print(f"  Additional found: {sorted(extra_found_logical_ids)}")

                if extra_missing:
                    failures.append(
                        f"{check['name']} via {extra_start_logical_id}: missing {', '.join(extra_missing)}"
                    )
                    print(f"  Additional missing: {extra_missing}")
                else:
                    print("  Additional result: ok")
    finally:
        janus.close()

    if failures:
        print("Traversal smoke failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Traversal smoke successful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
