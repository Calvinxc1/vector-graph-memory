# SETI Pilot Next Steps

This document defines the immediate execution plan for the `SETI` pilot slice of VGM Rules Lawyer.

Status:

- Planning document
- Intended to connect the current repository state to the first rules-lawyer milestone
- Focused on one bounded `SETI` subsystem as the near-term proving ground
- As of March 29, 2026, the pilot has moved beyond a single subsystem plan:
  - two manual seed slices exist in `tests/fixtures/rag_eval/`
  - typed extraction contracts exist under `src/vgm/rules/`
  - the landing/orbiter extraction path has reproduced the reference seed under the current comparison rules
  - seed import and verification scripts now work against the live Qdrant plus JanusGraph stack
  - direct DB inspection is available at `http://localhost:8111/dashboard/` and `http://localhost:8112/`

## Pilot Goal

Use `SETI` to prove that Vector Graph Memory can return a defensible rules ruling by retrieving and traversing stored rule structure rather than synthesizing an answer from scratch.

The first milestone is not “support all of `SETI`.” The first milestone is:

- choose one subsystem
- build a manual ground-truth graph slice for it
- run automated extraction against the same source material
- compare the extraction output to the hand-built reference
- expose ruling output with structural citations and precedence order

## Where We Are Now

Current repository position based on the tracked docs and fixtures:

- Vector Graph Memory already provides the hybrid storage substrate: vector retrieval plus graph traversal.
- A Dockerized local stack already exists with Open WebUI integration.
- There is an existing DSPy-backed RAG seam and evaluation scaffold in the repository.
- The repository already contains a tracked `SETI` rules-reference evaluation fixture under `tests/fixtures/rag_eval/`.
- Existing DSPy work is centered on grounded answer synthesis over retrieved context, not yet on a dedicated rules-lawyer output contract.

What this means in practice:

- the storage and local deployment foundation already exists
- some `SETI`-specific evaluation material already exists
- the current repository is still optimized for general grounded answering, not formal rule adjudication

## Where We Need To Be

For the `SETI` pilot to count as a meaningful rules-lawyer milestone, the system needs to support all of the following:

- a bounded `SETI` subsystem chosen as the canonical pilot slice
- a clear schema for source passages, canonical rules, and rule relationships
- a manual reference graph for that subsystem
- an automated extraction path that produces comparable node and edge candidates
- a ruling output format that explicitly returns:
  - ruling
  - primary citation
  - modifying citations
  - precedence order
  - uncertainty markers when needed
- an evaluation rubric that scores citation correctness and precedence correctness, not only answer plausibility
- a local-model test path through Ollama or equivalent local inference

## Gap Analysis

The main gaps between current state and pilot target are:

### 1. Pilot scope is not yet tightly bounded

The repository has `SETI` evaluation artifacts, but the first subsystem and its exact success criteria are not yet captured in a focused execution document.

### 2. Rules-lawyer schema is not yet explicit

The current DSPy RAG planning is about grounded synthesis. It does not yet define the rule-specific contracts needed for:

- canonical rule identity
- source-passage to rule mapping
- precedence representation
- structured ruling output

### 3. Ground-truth comparison workflow is not yet documented

The key pilot insight is to compare automated extraction against a hand-built seed. That workflow is not yet fully specified.

### 4. Validation still leans toward answer quality over adjudication quality

Existing evaluation scaffolding is useful, but the `SETI` pilot needs metrics that punish:

- missing controlling rules
- incorrect precedence ordering
- unsupported citations
- confident output where the rule graph is incomplete

### 5. Local-model validation is still open

The strategic direction assumes small local models are sufficient once constrained, but the repository does not yet demonstrate that with `SETI` rules-lawyer outputs.

## Immediate Next Steps

### Step 1: Choose And Freeze The First SETI Subsystem

Status:

- Completed for `landing_and_orbiter_interactions`
- Expanded with a second manual slice for `free_action_timing_and_authority`

Action:

- select one `SETI` subsystem narrow enough to model manually in full

Selection criteria:

- contains a meaningful rule interaction rather than only simple definitions
- has a manageable amount of source text
- is familiar enough for fast error diagnosis
- is representative of the kinds of precedence or dependency issues expected later

Deliverable:

- a short subsystem brief naming the exact source sections included in scope

Why this comes first:

- without a hard boundary, ingestion and evaluation will sprawl

### Step 2: Define The Pilot Rule Model

Status:

- Completed as an explicit typed contract in `src/vgm/rules/contracts.py`

Action:

- specify the data model for the pilot slice before scaling extraction

The model should distinguish at minimum:

- source passages
- canonical rule nodes
- citation metadata
- rule relationships

The relationship set should include only what the pilot actually needs, likely starting with:

- defines
- requires
- modifies
- overrides
- applies during
- contributes to

Deliverable:

- a small schema note for `SETI` pilot nodes, edges, and citation fields

Why this matters:

- the first extraction run needs a target representation, not just raw JSON blobs

### Step 3: Build A Manual Ground-Truth Graph Slice

Status:

- Completed for two `SETI` slices:
  - `seti_landing_orbiter_seed_v1`
  - `seti_free_action_authority_seed_v1`

Action:

- hand-author the canonical nodes, edge relationships, and citations for the chosen subsystem

The hand-built slice should be treated as the adjudication reference, not as throwaway setup.

Deliverable:

- one manually validated graph slice for the pilot subsystem

Validation expectation:

- inspect every node and edge for citation accuracy and graph logic

### Step 4: Define The Automated Extraction Contract

Status:

- Completed for the current pilot contract
- `PydanticAI` is now the required structured-output boundary for programmatic LLM extraction in this repo

Action:

- formalize the extraction task that DSPy or LLM-assisted parsing must perform against the same scoped source material

The extraction contract should cover:

- when to create a new canonical rule node
- when to attach a source passage to an existing canonical rule
- how to propose edges
- how to represent uncertain or ambiguous links

Deliverable:

- one extraction signature or structured contract that can be evaluated against the manual graph slice

Why this matters:

- the hard part is not text chunking, it is deciding canonical rule identity and relation semantics

### Step 5: Create A Comparison Rubric

Status:

- Partially completed
- extraction comparison and DSPy-oriented evaluation seams exist, but the broader acceptance rubric is still light and seed-centric

Action:

- evaluate extraction output against the manual slice with rule-aware metrics

The rubric should score at least:

- node recall for canonical rules
- citation correctness
- edge correctness
- duplicate or over-split rule creation
- missed merges across fragmented passages

Deliverable:

- a pilot comparison rubric with pass or fail thresholds for each category

### Step 6: Define The Ruling Output Contract

Status:

- Partially completed
- a typed deterministic pilot ruling contract now exists for the frozen `SETI` pilot questions
- the first implementation is bundle-backed and fixture-scoped, not yet a general live retrieval path

Action:

- specify the user-facing `SETI` ruling schema separately from the extraction schema

Required output fields:

- ruling
- primary citation
- modifying citations
- precedence order
- uncertainty or abstain flag

Preferred presentation:

- structured markdown intended for Open WebUI rendering

Deliverable:

- one rules-lawyer response schema for the `SETI` pilot
- first deterministic ruling engine covering the two tracked pilot slices

### Step 7: Wire The Pilot Retrieval Path

Status:

- Partially completed
- import into live Qdrant plus JanusGraph is working
- direct DB inspection is working
- a pilot inspection API now exists for retrieving the structured live ruling trace directly
- deterministic ruling assembly over the tracked pilot bundles now exists for the frozen question set
- a first live graph-backed ruling path now exists for the frozen pilot questions through Qdrant retrieval plus JanusGraph expansion
- retrieval and ruling assembly are still tightly scoped to the frozen `SETI` pilot question set, not yet generalized beyond it
- a typed live-ruling eval suite now exists for paraphrases, authority conflicts, and abstain behavior, with separate scoring for retrieval nodes, expanded evidence, seed inference, case selection, citation choice, modifier choice, and precedence assembly

Action:

- connect the manual or extracted `SETI` graph slice to a deterministic retrieval path that can gather controlling rules for a pilot question

The first version does not need full automation. It needs inspectable behavior.

Priorities:

- retrieve semantically relevant rule material
- expand via graph edges to connected controlling rules
- preserve the source metadata needed for exact citations

Deliverable:

- one end-to-end pilot retrieval path for the chosen subsystem

### Step 8: Test With Local Models

Status:

- Still open
- hosted-model baseline work has been done with `openai:gpt-4o-mini`
- local-model comparison has not yet been validated

Action:

- run the constrained ruling task against target local models through Ollama

Test for:

- adherence to output schema
- citation attachment discipline
- refusal behavior when evidence is weak
- degradation on multi-rule interactions

Deliverable:

- a simple comparison of at least one baseline hosted model versus one or more local models on the pilot cases

Why this matters:

- the product assumption is local viability, so this cannot stay theoretical for long

### Step 9: Put The Pilot In Front Of A Human Workflow

Action:

- exercise the `SETI` slice through Open WebUI with a small set of realistic questions

Evaluate:

- whether the output is understandable during live adjudication
- whether the citation chain feels sufficient to trust
- whether ambiguity handling is visible enough

Deliverable:

- operator feedback notes on usability, not just technical correctness

## Suggested Work Sequence

Recommended order:

1. freeze subsystem scope
2. define pilot rule model
3. build manual ground-truth graph
4. define extraction contract
5. build comparison rubric
6. define ruling output schema
7. wire retrieval path
8. test local models
9. run human workflow trial

This order keeps the pilot anchored to a known reference before investing in automation.

## Assumptions

This plan assumes:

- the current VGM storage layer is sufficient for a first pilot without major substrate redesign
- existing DSPy evaluation scaffolding can be extended rather than replaced
- one bounded subsystem is enough to expose the dominant ingestion and adjudication problems
- local-model viability depends more on output constraint quality than on open-ended reasoning ability

## Validation Expectations

Before calling the `SETI` pilot successful, validate at least:

- the manual graph slice is internally coherent
- automated extraction can be measured directly against it
- pilot rulings include correct citations on representative questions
- at least one local model can stay inside the output contract at acceptable quality

If those checks are not complete, the pilot may still be implemented, but it should not be presented as validated.

## What Success Looks Like

The `SETI` slice is successful if:

- a user can ask a scoped question and receive a traceable ruling
- the citation chain points to the right source passages and canonical rules
- precedence is explicit rather than implied
- the system abstains or flags uncertainty when the graph or evidence is weak
- the lessons learned are concrete enough to guide both `Arkham Horror LCG` and `Stellar Horizons`

The `SETI` slice is not yet successful if:

- answers look plausible but the controlling rules are missing
- the graph cannot explain why one rule controls another
- the extraction output is too noisy to compare against the manual seed
- local models only work when the task quietly falls back to broad synthesis
