#!/usr/bin/env python3
"""Materialize a validated RuleExtractionBundle JSON into seed fixture files.

Usage:
    uv run python scripts/materialize_rule_extraction_seed.py \
        --bundle /tmp/seti_extraction.json \
        --output-dir /tmp/seti_seed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgm.rules import RuleExtractionBundle, write_seed_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to a RuleExtractionBundle JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the manifest, nodes, and edges files should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = RuleExtractionBundle.model_validate(json.loads(args.bundle.read_text()))
    manifest_path, node_path, edge_path = write_seed_fixture(
        bundle,
        output_dir=args.output_dir,
    )
    print(f"Manifest: {manifest_path}")
    print(f"Nodes: {node_path}")
    print(f"Edges: {edge_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
