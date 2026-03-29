# DSPy Grounded Synthesis Implementation Plan

This document is the working implementation plan for the DSPy-backed grounded-answer path in Vector Graph Memory.

This is not the full rules-lawyer plan. It covers the narrower synthesis layer that sits between retrieved context and an answer. The broader rules-lawyer strategy, roadmap, and `SETI` pilot work are tracked in separate planning documents.

Status:

- partially implemented
- deterministic `RagContext` seam exists
- baseline DSPy synthesis path exists behind feature flags
- local compile and artifact-cache scaffold exists
- offline evaluation scaffold exists and uses a tracked `SETI` rules-reference fixture

## Purpose

The purpose of this plan is to improve the quality and model-specific behavior of the `retrieved context -> grounded answer` step.

The scope of this document is deliberately narrow:

- DSPy is used for synthesis, evaluation, and compilation
- retrieval remains native to Vector Graph Memory
- this plan does not define the full game-specific ingestion or rules-lawyer adjudication workflow

## Relationship To The Rules-Lawyer Direction

The repository now has a broader strategic direction around a rules-lawyer product for tabletop games.

That direction depends partly on this DSPy layer, but it should not be collapsed into it:

- this document is about answer synthesis over retrieved context
- the rules-lawyer work adds canonical rule modeling, precedence, citations, ingestion, and product UX requirements
- a successful DSPy eval run is useful evidence, but it does not prove the rules-lawyer product is complete

Related planning docs:

- [rules-lawyer-strategy.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/rules-lawyer-strategy.md)
- [rules-lawyer-roadmap.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/rules-lawyer-roadmap.md)
- [seti-pilot-next-steps.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/seti-pilot-next-steps.md)

## Current Constraints

The current request path still has structural limitations:

- `src/vgm/api/server.py` is the integration point for the OpenAI-compatible request path
- `src/vgm/MemoryAgent.py` still owns the memory-oriented tool loop
- retrieval for the baseline product remains centered on memory-oriented behavior rather than a dedicated ruling engine
- the storage layer already exposes reusable retrieval primitives through `VectorGraphStore`

These constraints matter because DSPy needs a stable contract if it is going to be tuned and evaluated reliably.

## Target Architecture

The intended grounded-answer path is:

1. receive an OpenAI-compatible chat request
2. preserve message history and current user question as separate fields
3. build deterministic retrieval context using native Vector Graph Memory retrieval
4. pass the structured retrieval payload into a DSPy synthesis module
5. return grounded answer text and source metadata
6. keep memory-review behavior separate from synthesis as much as possible

Main internal components:

- `RagContext` builder
- baseline DSPy synthesizer
- eval and compile manager
- runtime selector for baseline versus compiled artifacts
- trace logging for later analysis

## Scope And Non-Goals

In scope:

- DSPy-based grounded answer synthesis over retrieved context
- model-specific compilation and caching
- offline evaluation and guarded rollout
- feature-flagged API integration

Out of scope:

- replacing vector or graph retrieval with DSPy
- full retrieval planning or multi-hop orchestration in DSPy
- declaring the repository already has a complete rules-lawyer pipeline
- using Open WebUI feedback as a launch dependency

## Implementation Phases

### Phase 1: Establish A Clean RAG Seam

Objective:

- separate answer synthesis from the mixed memory-management loop enough to give DSPy a stable target

Concrete work:

- introduce a dedicated internal `retrieve -> synthesize` answer path
- preserve chat history as structured messages instead of flattening everything into one prompt string
- build a typed retrieval payload from native store calls
- keep the current memory proposal and confirmation workflow intact

Definition of done:

- the application can build structured retrieval payloads without depending on the mixed tool-output path
- answer synthesis has a stable input contract

Current status:

- implemented in initial form

### Phase 2: Add A Baseline DSPy Synthesizer

Objective:

- introduce DSPy in the narrowest useful place before more aggressive optimization

Concrete work:

- define a baseline DSPy module for answer synthesis
- feed the module structured retrieval inputs
- return grounded answer text and source identifiers

Suggested contract shape:

- inputs:
  - `conversation_history`
  - `question`
  - `passages`
  - `graph_facts`
  - `use_case`
- outputs:
  - `answer`
  - `cited_source_ids`
  - `abstain`

Definition of done:

- the API can answer through the baseline DSPy module with no compilation step
- the baseline path is feature-flagged and can fall back safely

Current status:

- implemented in baseline form

### Phase 3: Build The Evaluation Set And Metric

Objective:

- define the optimization target before relying on compilation results

Concrete work:

- assemble a representative evaluation set
- freeze retrieval payloads so synthesis can be evaluated independently of retrieval drift
- encode expected facts, abstention behavior, and citation expectations

Metric priorities:

- groundedness to retrieved evidence
- source alignment
- abstention when evidence is weak
- output-format compliance

Current implementation note:

- the repository includes `tests/fixtures/rag_eval/seti_rules_reference_v1.jsonl`
- tracked eval cases are currently built around `SETI` rules-reference examples
- validation tests cover fixture structure and source-locator handling when local ignored source documents are present

Definition of done:

- there is a repeatable offline evaluation set for synthesis quality
- baseline and candidate programs can be compared on explicit metrics

Current status:

- implemented in initial form

### Phase 4: Add DSPy Compilation And Artifact Caching

Objective:

- optimize synthesis per model without making startup or first request brittle

Concrete work:

- compile candidate DSPy programs against the frozen eval suite
- cache artifacts by model identity and retrieval or program version
- use baseline synthesis immediately for unseen model identities
- promote compiled artifacts only when they beat baseline

Runtime policy:

- do not block normal chat startup on full compile
- do not assume successful compile implies production readiness
- recompile only when the model or schema changes materially

Definition of done:

- model-specific compiled synthesis artifacts can be created, selected, and reused safely

Current status:

- scaffold exists
- promotion is still intentionally conservative

### Phase 5: Integrate Into The API Behind Feature Flags

Objective:

- roll out the DSPy path without breaking the existing memory-oriented API

Concrete work:

- add feature flags for the DSPy synthesis path
- update the OpenAI-compatible chat endpoint to preserve structured history, build retrieval context, call the selected synthesizer, and fall back explicitly when needed

Definition of done:

- the API can serve either the baseline path or the DSPy-backed path deterministically
- fallback behavior is explicit and testable

Current status:

- partially implemented

### Phase 6: Separate Memory Review From Answer Synthesis

Objective:

- prevent memory-review behavior from contaminating the synthesis optimization target

Concrete work:

- remove the current pattern where memory-review guidance is injected into every turn under `ai_determined`
- treat memory review as a separate internal concern from answer synthesis

Definition of done:

- answer synthesis can be tuned independently from memory-management behavior

Current status:

- not complete

### Phase 7: Add Better Trace Logging

Objective:

- preserve enough run data to inspect regressions, compare compiled artifacts, and support future feedback loops

Concrete work:

- record model ID, retrieval payload metadata, synthesis version, answer text, source IDs, and trace identifiers
- make proof-run artifacts easy to inspect locally

Definition of done:

- local proof runs and production-like traces are inspectable enough to debug synthesis regressions

Current status:

- partial support exists through local run logging

## Current Risks

Main risks for this DSPy path:

- contract drift between `RagContext` and the DSPy module
- overfitting to the current `SETI` eval fixture
- treating answer quality metrics as a substitute for full adjudication quality
- allowing memory-review behavior to pollute synthesis evaluation
- assuming compiled artifacts imply acceptance readiness

## Validation Status

What has been validated:

- the repository contains tests for the DSPy context, synthesis, evaluation, compile-manager, and API integration seams
- the tracked eval fixture exists and is exercised by tests

What remains unvalidated at the broader product level:

- full rules-lawyer output contracts
- game-specific ingestion quality
- graph-backed precedence reasoning for real adjudication flows
- local-model viability for the end-user rules-lawyer experience

## Next Practical Steps

For this DSPy plan specifically, the next high-value work is:

1. keep the synthesis contract narrow and stable
2. avoid mixing rules-lawyer ambitions into the synthesis layer prematurely
3. use the `SETI` fixture to improve grounded synthesis while the separate `SETI` pilot work defines canonical rule modeling
4. tighten traceability between retrieved evidence and generated outputs

That separation matters. The repository needs both:

- a sound synthesis layer
- a sound rules-lawyer architecture

They are related, but they are not the same task.
