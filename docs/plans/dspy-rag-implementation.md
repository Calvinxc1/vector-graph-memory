# DSPy RAG Implementation Plan

This document is the working implementation plan for adding DSPy-backed RAG prompt support to Vector Graph Memory.

Status:

- Partially implemented
- Phase 1 seam creation is in place
- A baseline Phase 2 DSPy synthesis path exists behind feature flags
- Phase 3 eval scaffolding exists as a local SETI rules-reference dataset with tracked rubrics
- Scope in this document is limited to answer synthesis over retrieved context
- Retrieval stays native to Vector Graph Memory
- Open WebUI feedback is a later optimization input, not a launch dependency

## Goal

Improve the quality and model-specific behavior of the "retrieved context -> grounded answer" step by introducing a DSPy synthesis layer that can be evaluated, compiled, cached, and rolled out safely.

This plan does not treat DSPy as the retrieval engine. Vector and graph retrieval remain implemented by Vector Graph Memory.

## Current Constraints

The current request path mixes several responsibilities that should be separated before DSPy is introduced:

- `src/vgm/api/server.py` flattens multi-turn chat history into a single prompt string before dispatch
- `src/vgm/MemoryAgent.py` combines answer generation, memory-review prompting, and memory-management tools in one loop
- Retrieval for answer generation is currently exposed primarily through LLM tools that return formatted text
- The storage layer already provides structured retrieval primitives through `VectorGraphStore`

These constraints make prompt optimization harder because the synthesis step does not yet have a stable, typed contract.

## Target Architecture

The planned answer path is:

1. Receive OpenAI-compatible chat request
2. Preserve message history and current user question as separate fields
3. Build deterministic retrieval context using native Vector Graph Memory retrieval
4. Pass structured retrieval output into a DSPy synthesis module
5. Return grounded answer and source metadata
6. Run memory-review logic separately from answer synthesis

The main internal components should be:

- `RagContext` builder:
  - Normalizes chat history, current question, similar nodes, graph expansions, source IDs, scores, and any retrieval metadata needed by synthesis
- Baseline DSPy synthesizer:
  - A non-compiled DSPy module that defines the answer-generation contract before any optimization is attempted
- Eval and compile manager:
  - Runs DSPy optimizers against a frozen evaluation set and persists model-specific compiled artifacts
- Runtime selector:
  - Chooses baseline or compiled synthesizer based on feature flags and cache availability
- Trace logger:
  - Records model ID, retrieval payload, synthesis version, answer, and trace ID for later analysis

## Scope And Non-Goals

In scope for the first implementation:

- DSPy-based answer synthesis over retrieved context
- Model-specific compilation and caching
- Offline evaluation and guarded rollout
- Feature-flagged API integration

Explicitly out of scope for the first implementation:

- Replacing vector or graph retrieval with DSPy
- Full retrieval-planning or multi-hop orchestration in DSPy
- Using thumbs-up or thumbs-down as the primary optimization metric
- Blocking first-use chat requests on a full compile

## Implementation Phases

### Phase 1: Establish A Clean RAG Seam

Objective:

- Separate the answer path from the memory-management path so DSPy can target one stable unit of behavior

Concrete work:

- Introduce a dedicated internal `retrieve -> synthesize` path for answering
- Preserve chat history as structured messages instead of collapsing everything into one string
- Build a typed retrieval payload from native store calls
- Keep the current memory proposal and confirmation workflow intact for now

Expected code direction:

- Add a small RAG-focused package or module namespace, for example `src/vgm/rag/`
- Move answer-time retrieval assembly out of the mixed `MemoryAgent.run(...)` path
- Reuse `VectorGraphStore.search_similar_nodes(...)` and graph traversal directly instead of routing answer retrieval through tool-formatted strings

Definition of done:

- The application can build a structured retrieval payload without invoking the current mixed tool loop
- Answer synthesis has a stable input contract

Phase-specific risks to guard against:

- Memory-review behavior leaking back into the answer path and polluting the synthesis target
- Over-designing the first `RagContext` schema before the synthesizer proves what data it actually needs
- Pulling graph traversal in too aggressively and drowning synthesis in low-signal context on day one

### Phase 2: Add A Baseline DSPy Synthesizer

Objective:

- Introduce DSPy in the narrowest useful place before any optimization logic is added

Concrete work:

- Add DSPy as a dependency
- Create a baseline DSPy module with a clear signature for answer synthesis
- Feed the module structured retrieval inputs rather than one large context blob
- Return answer text and source identifiers from the module output

Suggested signature shape:

- Inputs:
  - `conversation_history`
  - `question`
  - `passages`
  - `graph_facts`
  - `use_case`
- Outputs:
  - `answer`
  - `cited_source_ids`
  - `abstain`

Definition of done:

- The API can answer through the baseline DSPy module with no compilation step
- The baseline path is feature-flagged and can fall back to the current answer path

Phase-specific risks to guard against:

- Contract drift between `RagContext` and the baseline DSPy signature before the input shape stabilizes
- Falling back to one giant prompt blob and losing the structured seam created in Phase 1
- Losing provenance between retrieved evidence and the answer by failing to return or log cited source IDs
- Reintroducing `MemoryAgent` memory-review behavior into the DSPy synthesis path
- Weak fallback behavior if DSPy initialization or synthesis fails at runtime

### Phase 3: Build The Evaluation Set And Metric

Objective:

- Create the optimization target before introducing compilation or auto-tuning

Concrete work:

- Assemble a small representative evaluation set from the repository's real use cases
- Freeze retrieval payloads for each example so synthesis is evaluated independently of retrieval drift
- Capture expected behavior per example:
  - facts that must be used
  - facts that must not be invented
  - expected source usage
  - expected abstention behavior
  - format requirements

Metric priorities:

- Groundedness to retrieved evidence only
- Correct source attachment or citation behavior
- Preference for graph-derived facts when relevant
- Refusal to invent when evidence is weak
- Output-format compliance

Definition of done:

- There is a repeatable offline eval set for synthesis quality
- The baseline DSPy module is measured against explicit metrics rather than subjective inspection alone

Current implementation note:

- The repository now includes a tracked JSONL rules-reference fixture for `SETI` under `tests/fixtures/rag_eval/`
- Exact source documents remain local and gitignored; tracked eval cases freeze retrieval by document, page, and locator
- The repository now defines a v1 single-score contract that weights groundedness highest, followed by abstention correctness, source alignment, and completeness
- Validation tests check fixture structure in all environments and verify local source locators when the ignored corpus is present

### Phase 4: Add DSPy Compilation And Artifact Caching

Objective:

- Optimize the synthesis program per model without making startup or first request brittle

Concrete work:

- Add a compile manager around the DSPy synthesizer
- Start with `dspy.MIPROv2` for prompt optimization
- Cache compiled artifacts by:
  - provider
  - model ID
  - model version if available
  - retrieval schema version
  - synthesis program version
- Use the baseline synthesizer immediately for an unseen model
- Run a short compile in the background for new model configurations
- Promote a compiled artifact only if it beats the baseline on the eval set

Runtime policy:

- Do not block normal chat startup on a full compile
- Do not assume a successful compile implies production readiness
- Recompile only when the model or retrieval schema changes materially

Definition of done:

- Model-specific compiled synthesis artifacts can be created, stored, selected, and reused safely

### Phase 5: Integrate Into The API Behind A Feature Flag

Objective:

- Roll out the new answer path without breaking the current API behavior

Concrete work:

- Add feature flags or configuration toggles for DSPy-backed synthesis
- Update the OpenAI-compatible chat endpoint to:
  - preserve structured history
  - build retrieval context
  - call the selected synthesizer
  - attach any source metadata needed for tracing
- Keep a clean fallback to the current path during rollout

Definition of done:

- The API can serve either the current path or the DSPy-backed path deterministically
- Fallback behavior is explicit and testable

### Phase 6: Separate Memory Review From Answer Synthesis

Objective:

- Prevent memory-proposal behavior from contaminating the synthesis optimization target

Concrete work:

- Remove the current pattern where memory-review instructions are injected into every answer turn under `ai_determined`
- Treat memory review as a second pass or separate internal component
- Keep user-confirmed memory proposal behavior unchanged from the user's perspective

Definition of done:

- Answer synthesis can be tuned independently from memory-management behavior

### Phase 7: Add Trace Logging For Future Feedback Loops

Objective:

- Capture the metadata needed to evaluate future prompt variants and connect them to user feedback later

Concrete work:

- Log:
  - trace ID
  - model ID
  - retrieval payload summary or source IDs
  - synthesis program version
  - whether the baseline or compiled path was used
  - answer text or answer digest
- Keep the trace shape stable enough to join with external feedback later

Definition of done:

- Each answer can be tied back to the exact synthesis path and prompt-program version used

### Phase 8: Explore Open WebUI Feedback As A Weak Signal

Objective:

- Use Open WebUI feedback to improve prompts carefully, without overfitting to noisy thumbs data

Concrete work:

- Investigate how reliably Open WebUI feedback can be exported or joined to backend traces
- Treat thumbs-up, thumbs-down, ratings, and text comments as weak labels only
- Use feedback to prioritize examples for review and dataset expansion
- Avoid direct optimization on raw thumbs alone

Why this is later:

- Negative feedback may reflect retrieval quality, latency, formatting, or user preference rather than synthesis prompt quality
- Prompt tuning should first rest on an explicit offline eval set

Definition of done:

- Feedback is available as an analysis input for future dataset growth and retraining decisions

## Configuration Changes Expected

The implementation will likely need new configuration separate from the current memory settings, for example:

- `RAG_SYNTHESIS_BACKEND`
- `RAG_DSPY_ENABLED`
- `RAG_DSPY_COMPILE_ON_START`
- `RAG_DSPY_CACHE_DIR`
- `RAG_DSPY_PROGRAM_VERSION`
- `RAG_RETRIEVAL_SCHEMA_VERSION`

Exact names are not final and should be chosen during implementation.

## Validation Plan

Low-risk validation:

- Unit tests for retrieval-context assembly
- Unit tests for feature-flag and cache-selection behavior
- Basic serialization tests for compiled-artifact metadata

Moderate-risk validation:

- Offline eval comparison between baseline and compiled synthesizers
- Manual smoke testing through the OpenAI-compatible API
- Manual Open WebUI verification for answer quality and regression checks

High-risk validation before broader rollout:

- Repeated eval runs across more than one target model
- Failure-mode testing for missing caches, compile failures, and fallback selection
- Verification that memory proposal behavior is unchanged when DSPy synthesis is enabled

## Order Of Implementation

Recommended execution order:

1. Build the retrieval-context seam
2. Add the baseline DSPy synthesizer
3. Create the frozen evaluation set and metric
4. Add compilation and model-specific caching
5. Integrate the new answer path behind a feature flag
6. Separate memory review from answer synthesis
7. Add trace logging
8. Explore Open WebUI feedback as a later optimization signal

## Assumptions

- The first target is answer synthesis only, not retrieval replacement
- The selected LLM may change across environments, so model-specific compilation is useful
- The first implementation should optimize reliability and observability before optimization depth
- Open WebUI feedback is useful, but not reliable enough to be the only quality signal

## Remaining Open Questions

- Where compiled DSPy artifacts should be stored in local and containerized deployments
- Whether source IDs should be surfaced in the OpenAI-compatible response immediately or only logged internally
- How retrieval payloads should represent graph traversals for synthesis without overloading token budgets
- Which evaluation examples best represent the intended primary use cases for this repository
