#!/usr/bin/env python3
"""Run the model-backed SETI rules extractor and compare against the manual seed.

Usage:
    uv run python scripts/run_rule_extraction.py --model openai:gpt-4o-mini
    uv run python scripts/run_rule_extraction.py --model ollama:qwen2.5:14b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgm.rules import (
    PydanticAIRuleExtractionPredictor,
    RawPydanticAIRuleExtractionPredictor,
    RuleExtractionRunner,
    build_request_from_reference_bundle,
    compare_rule_extractions,
    load_bundle_from_seed_records,
)


DEFAULT_MANIFEST = Path(
    "tests/fixtures/rag_eval/seti_landing_orbiter_seed_v1_manifest.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="PydanticAI model string to use")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to the seed manifest JSON (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--dump-output",
        type=Path,
        help="Optional path to write the extracted RuleExtractionBundle JSON",
    )
    parser.add_argument(
        "--dump-raw-on-error",
        type=Path,
        help="Optional path to write raw model output JSON if validation fails",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest = _load_json(args.manifest)
    node_records = _load_jsonl(Path(manifest["node_file"]))
    edge_records = _load_jsonl(Path(manifest["edge_file"]))
    reference_bundle = load_bundle_from_seed_records(manifest, node_records, edge_records)
    request = build_request_from_reference_bundle(reference_bundle)

    predictor = PydanticAIRuleExtractionPredictor(model=args.model)
    runner = RuleExtractionRunner(predictor=predictor)
    try:
        extracted_bundle = runner.extract(request)
    except Exception:
        if args.dump_raw_on_error:
            raw_predictor = RawPydanticAIRuleExtractionPredictor(model=args.model)
            raw_output = raw_predictor(
                runner.build_system_prompt(request),
                runner.build_user_prompt(request),
            )
            args.dump_raw_on_error.write_text(
                json.dumps(raw_output, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        raise

    if args.dump_output:
        args.dump_output.write_text(
            extracted_bundle.model_dump_json(indent=2),
            encoding="utf-8",
        )

    comparison = compare_rule_extractions(reference_bundle, extracted_bundle)

    print(f"Seed: {reference_bundle.seed_id}")
    print(f"Model: {args.model}")
    print(f"Exact match: {comparison.is_exact_match}")
    print(f"Missing source passages: {comparison.missing_source_passage_ids}")
    print(f"Extra source passages: {comparison.extra_source_passage_ids}")
    print(f"Missing canonical rules: {comparison.missing_canonical_rule_ids}")
    print(f"Extra canonical rules: {comparison.extra_canonical_rule_ids}")
    print(f"Missing edges: {comparison.missing_edge_ids}")
    print(f"Extra edges: {comparison.extra_edge_ids}")
    print(f"Field mismatches: {len(comparison.field_mismatches)}")

    for mismatch in comparison.field_mismatches:
        print(
            f"  - {mismatch.record_kind}:{mismatch.record_id}:{mismatch.field_name} "
            f"expected={mismatch.expected!r} actual={mismatch.actual!r}"
        )

    return 0 if comparison.is_exact_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
