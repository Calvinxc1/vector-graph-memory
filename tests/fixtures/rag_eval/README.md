# RAG Eval Fixtures

This directory holds tracked eval metadata for DSPy synthesis benchmarking.

Current suite:

- `seti_rules_reference_v1.jsonl`
  - `20` rubric-based rules-reference cases for `SETI`
  - frozen by `document_id + page + locator` against local source documents
  - mostly single-turn, with a small number of short synthetic multi-turn cases
  - includes abstention cases with intentionally insufficient retrieved context

Authority model for `SETI` v1:

- alien addendum wins for species-specific exceptions
- FAQ wins for FAQ-specific clarifications
- core rulebook is authoritative otherwise
- player aid is summary-only and should not be treated as authoritative on its own

Local source documents:

- expected in `tests/fixtures/rag_eval/source_documents/`
- kept out of git on purpose
- exact retrieval text is resolved locally from those ignored files
