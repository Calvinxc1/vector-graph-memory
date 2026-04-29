# RAG Eval Fixtures

This directory holds tracked eval metadata for DSPy synthesis benchmarking.

Current suite:

- `seti_rules_reference_v1.jsonl`
  - `20` rubric-based rules-reference cases for `SETI`
  - frozen by `document_id + page + locator` against local source documents
  - mostly single-turn, with a small number of short synthetic multi-turn cases
  - includes abstention cases with intentionally insufficient retrieved context
- `seti_rules_ruling_eval_v1.jsonl`
  - `8` frozen live-ruling evaluation cases for the bounded `SETI` pilot
  - scores retrieval nodes, expanded evidence, seed inference, case selection, primary citation, modifier selection, precedence assembly, and abstention separately
  - current v1 acceptance uses hard gates on abstain behavior, primary citation, and precedence, plus suite thresholds on the aggregate component averages
  - includes broader paraphrase coverage, authority-conflict coverage, and one intentional abstain case
- `seti_rules_ruling_eval_heldout_v1.jsonl`
  - `34` typed held-out questions for the same bounded `SETI` pilot surface
  - organized into:
    - `6` `seen_regression` cases
    - `12` `heldout_supported` cases
    - `10` `near_miss_abstain` cases
    - `6` `out_of_scope_abstain` cases
  - carries typed metadata for `bucket`, `split`, `manual_candidate`, and `expected_abstain_kind`
  - intended usage:
    - tune on `seen_regression` + `heldout_supported/dev` + `near_miss_abstain/dev`
    - final automated check on `seen_regression` + `heldout_supported/validation` + `near_miss_abstain/validation` + all `out_of_scope_abstain`
    - select a stable manual human subset from the `manual_candidate=true` rows

Manual seed fixtures:

- `seti_landing_orbiter_seed_v1_manifest.json`
  - manifest for the first hand-built `SETI` rules-lawyer seed slice
- `seti_landing_orbiter_seed_v1_nodes.jsonl`
  - `14` machine-readable node records shaped to align with current `NodeMetadata` usage
  - includes `5` source-passage nodes and `9` canonical-rule nodes
- `seti_landing_orbiter_seed_v1_edges.jsonl`
  - `15` machine-readable edge records shaped to align with current `EdgeMetadata` usage
  - captures source-to-rule support and clarification edges plus rule-to-rule reasoning edges
- `seti_free_action_authority_seed_v1_manifest.json`
  - manifest for the second hand-built `SETI` rules-lawyer seed slice
- `seti_free_action_authority_seed_v1_nodes.jsonl`
  - `11` machine-readable node records for free-action timing and player-aid authority
- `seti_free_action_authority_seed_v1_edges.jsonl`
  - `11` machine-readable edge records for timing clarifications and authority constraints
- `seti_rules_extraction_v1.json`
  - suite manifest that groups both tracked seed slices for extraction evaluation and DSPy compilation
- `scripts/import_manual_seed.py`
  - importer for loading a manual seed fixture into the current Qdrant plus JanusGraph backend
  - skips existing `node_id` and `edge_id` records to avoid duplicate JanusGraph inserts on rerun
- `scripts/materialize_rule_extraction_seed.py`
  - converts a validated `RuleExtractionBundle` JSON into manifest plus node/edge JSONL files
  - intended bridge between accepted extractor output and the existing importer
- `scripts/verify_manual_seed.py`
  - verifier for checking that a manual seed fixture is actually present in live Qdrant plus JanusGraph storage
  - also checks that the required node set for each frozen landing-orbiter support path is present
  - when used against the local Docker Compose stack, it now defaults to the host-exposed ports:
    - Qdrant `localhost:8111`
    - JanusGraph `localhost:8182`
- `scripts/smoke_traverse_manual_seed.py`
  - live traversal smoke check for the first two frozen landing-orbiter questions
  - verifies that the controlling rule nodes can reach their expected support nodes through graph traversal
- `scripts/run_pilot_ruling_eval.py`
  - live evaluator for the tracked pilot ruling suite
  - emits a typed JSON report over `inspect_request()` plus final ruling assembly

Seed scope:

- rules-only
- bounded subsystems:
  - landing and orbiter interactions
  - free-action timing and player-aid authority
- intended as hand-built reference targets for later extraction comparisons and graph seeding

Authority model for `SETI` v1:

- alien addendum wins for species-specific exceptions
- FAQ wins for FAQ-specific clarifications
- core rulebook is authoritative otherwise
- player aid is summary-only and should not be treated as authoritative on its own

Local source documents:

- expected in `tests/fixtures/rag_eval/source_documents/`
- kept out of git on purpose
- exact retrieval text is resolved locally from those ignored files
