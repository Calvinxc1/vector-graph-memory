#!/usr/bin/env python3
"""Run the deterministic SETI pilot ruling path for one frozen question."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgm.rules import DeterministicPilotRulingEngine, RulesRulingRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Frozen pilot question to answer.")
    parser.add_argument("--seed-id", help="Optional seed_id to scope the lookup.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = DeterministicPilotRulingEngine.for_seti_pilot()
    result = engine.answer(
        RulesRulingRequest(
            question=args.question,
            seed_id=args.seed_id,
        )
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=True))
    return 0 if not result.abstain else 1


if __name__ == "__main__":
    raise SystemExit(main())
