#!/usr/bin/env python3
"""Audit a RuleExtractionBundle or seed manifest before loading it into the graph.

Usage:
    uv run python scripts/audit_rule_load.py \
        --manifest tests/fixtures/rag_eval/seti_landing_orbiter_seed_v1_manifest.json

    uv run python scripts/audit_rule_load.py \
        --bundle /tmp/rule_extraction.json \
        --report /tmp/rule_load_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgm.rules import (  # noqa: E402
    RuleExtractionBundle,
    audit_rule_extraction_bundle,
    build_request_from_reference_bundle,
    load_bundle_from_seed_records,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="Path to a RuleExtractionBundle JSON file.")
    source.add_argument("--manifest", type=Path, help="Path to a seed manifest JSON file.")
    parser.add_argument("--report", type=Path, help="Optional path to write the audit report JSON.")
    parser.add_argument(
        "--no-reference-request",
        action="store_true",
        help="When auditing a manifest, skip source-coverage checks built from the same reference bundle.",
    )
    return parser.parse_args()


def _load_bundle_from_manifest(path: Path) -> RuleExtractionBundle:
    manifest = _load_json(path)
    node_records = _load_jsonl(Path(manifest["node_file"]))
    edge_records = _load_jsonl(Path(manifest["edge_file"]))
    return load_bundle_from_seed_records(manifest, node_records, edge_records)


def main() -> int:
    args = parse_args()
    if args.bundle is not None:
        bundle = RuleExtractionBundle.model_validate(_load_json(args.bundle))
        request = None
    else:
        bundle = _load_bundle_from_manifest(args.manifest)
        request = None if args.no_reference_request else build_request_from_reference_bundle(bundle)

    report = audit_rule_extraction_bundle(bundle, request=request)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print(f"Seed: {report.seed_id}")
    print(f"Graph: {report.graph.source_passages} source passages, {report.graph.canonical_rules} rules, {report.graph.edges} edges")
    print(f"Source-grounded rules: {report.graph.source_grounded_rules}/{report.graph.canonical_rules}")
    print(f"Errors: {report.error_count}")
    print(f"Warnings: {report.warning_count}")
    for finding in report.findings:
        print(f"{finding.severity.upper()} {finding.code}: {finding.message}")

    return 0 if report.passes_required_gates else 1


if __name__ == "__main__":
    raise SystemExit(main())
