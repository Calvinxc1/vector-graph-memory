#!/usr/bin/env python3
"""Compile a DSPy rules extractor against one seed manifest or a seed suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgm.rules import RuleExtractionEvaluator, compile_rule_extractor


DEFAULT_MANIFEST = Path(
    "tests/fixtures/rag_eval/seti_landing_orbiter_seed_v1_manifest.json"
)
DEFAULT_SUITE = Path("tests/fixtures/rag_eval/seti_rules_extraction_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="DSPy/OpenAI model string")
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help=(
            "Seed manifest path. Repeat to compile against multiple manifests. "
            f"If omitted, --suite is used when provided, otherwise {DEFAULT_MANIFEST}."
        ),
    )
    parser.add_argument(
        "--suite",
        type=Path,
        help=f"Seed suite JSON path (default behavior target: {DEFAULT_SUITE})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest and args.suite:
        raise ValueError("Pass either --manifest or --suite, not both")

    if args.suite:
        evaluator = RuleExtractionEvaluator.from_seed_suite(args.suite)
        target_label = str(args.suite)
    elif args.manifest:
        evaluator = RuleExtractionEvaluator.from_manifest_paths(args.manifest)
        target_label = ", ".join(str(path) for path in args.manifest)
    else:
        evaluator = RuleExtractionEvaluator.from_seed_suite(DEFAULT_SUITE)
        target_label = str(DEFAULT_SUITE)

    compile_rule_extractor(llm_model=args.model, evaluator=evaluator)
    print(f"Compiled rule extractor for {args.model} against {target_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
