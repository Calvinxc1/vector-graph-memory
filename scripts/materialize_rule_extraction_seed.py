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

from vgm.rules import RuleExtractionBundle, audit_rule_extraction_bundle, write_seed_fixture


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
    parser.add_argument(
        "--audit-report",
        type=Path,
        help="Optional path to write the pre-materialization audit report JSON.",
    )
    parser.add_argument(
        "--allow-audit-fail",
        action="store_true",
        help="Write seed files even when the audit has blocking errors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = RuleExtractionBundle.model_validate(json.loads(args.bundle.read_text()))
    audit_report = audit_rule_extraction_bundle(bundle)
    if args.audit_report is not None:
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(audit_report.model_dump_json(indent=2), encoding="utf-8")
    if not audit_report.passes_required_gates and not args.allow_audit_fail:
        print(
            "Rule-load audit failed; refusing to materialize seed fixture. "
            "Use --allow-audit-fail only for local debugging.",
            file=sys.stderr,
        )
        print(f"Errors: {audit_report.error_count}; warnings: {audit_report.warning_count}", file=sys.stderr)
        return 1
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
