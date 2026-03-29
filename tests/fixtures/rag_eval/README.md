# RAG Eval Fixtures

This directory holds tracked eval metadata for DSPy synthesis benchmarking.

Current suite:

- `seti_rules_reference_v1.jsonl`
  - `20` rubric-based rules-reference cases for `SETI`
  - frozen by `document_id + page + locator` against local source documents
  - mostly single-turn, with a small number of short synthetic multi-turn cases
  - includes abstention cases with intentionally insufficient retrieved context

Manual seed fixtures:

- `seti_landing_orbiter_seed_v1_manifest.json`
  - manifest for the first hand-built `SETI` rules-lawyer seed slice
- `seti_landing_orbiter_seed_v1_nodes.jsonl`
  - `14` machine-readable node records shaped to align with current `NodeMetadata` usage
  - includes `5` source-passage nodes and `9` canonical-rule nodes
- `seti_landing_orbiter_seed_v1_edges.jsonl`
  - `15` machine-readable edge records shaped to align with current `EdgeMetadata` usage
  - captures source-to-rule support and clarification edges plus rule-to-rule reasoning edges
- `scripts/import_manual_seed.py`
  - importer for loading a manual seed fixture into the current Qdrant plus JanusGraph backend
  - skips existing `node_id` and `edge_id` records to avoid duplicate JanusGraph inserts on rerun
- `scripts/verify_manual_seed.py`
  - verifier for checking that a manual seed fixture is actually present in live Qdrant plus JanusGraph storage
  - also checks that the required node set for each frozen landing-orbiter support path is present
- `scripts/smoke_traverse_manual_seed.py`
  - live traversal smoke check for the first two frozen landing-orbiter questions
  - verifies that the controlling rule nodes can reach their expected support nodes through graph traversal

Seed scope:

- rules-only
- first bounded subsystem: landing and orbiter interactions
- intended as the hand-built reference target for later extraction comparisons and graph seeding

Authority model for `SETI` v1:

- alien addendum wins for species-specific exceptions
- FAQ wins for FAQ-specific clarifications
- core rulebook is authoritative otherwise
- player aid is summary-only and should not be treated as authoritative on its own

Local source documents:

- expected in `tests/fixtures/rag_eval/source_documents/`
- kept out of git on purpose
- exact retrieval text is resolved locally from those ignored files
