#!/usr/bin/env python3
"""Run the live SETI pilot ruling eval suite against the local graph stack."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gremlin_python.driver import client as gremlin_client  # type: ignore[import-untyped]
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from qdrant_client import QdrantClient

from vgm.VectorGraphStore import VectorGraphStore
from vgm.rules import LivePilotRulingEngine, PilotRulingEvaluator


DEFAULT_SUITE = Path("tests/fixtures/rag_eval/seti_rules_ruling_eval_v1.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE,
        help=f"Path to the ruling eval JSONL suite (default: {DEFAULT_SUITE})",
    )
    parser.add_argument(
        "--project-id",
        default="seti_rules_lawyer",
        help="Project identifier used to scope vector retrieval.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_HTTP_PORT", "8111"))
    janusgraph_host = os.getenv("JANUSGRAPH_HOST", "localhost")
    janusgraph_port = int(os.getenv("JANUSGRAPH_PORT", "8182"))

    print(f"Ruling eval suite: {args.suite}")
    print(f"Qdrant target: {qdrant_host}:{qdrant_port}")
    print(f"JanusGraph target: {janusgraph_host}:{janusgraph_port}")

    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    janus = gremlin_client.Client(f"ws://{janusgraph_host}:{janusgraph_port}/gremlin", "g")
    try:
        embedding_model = OpenAIEmbeddingModel(os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
        store = VectorGraphStore(
            qdrant_client=qdrant,
            janus_client=janus,
            embedding_model=embedding_model,
        )
        evaluator = PilotRulingEvaluator.from_suite(args.suite)
        report = evaluator.evaluate_engine(LivePilotRulingEngine(store, project_id=args.project_id))
    except Exception as exc:
        print(f"Ruling eval failed: {exc}", file=sys.stderr)
        return 2
    finally:
        janus.close()

    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=True))
    return 0 if report.failed_cases == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
