# VGM Rules Lawyer Roadmap

This document outlines the staged game roadmap for proving Vector Graph Memory as a rules-lawyer system.

Status:

- Planning document
- Ordered for capability growth rather than popularity alone
- Assumes `SETI` remains the immediate execution focus

## Roadmap Logic

The roadmap should increase complexity one major dimension at a time.

The intended sequence is:

1. `SETI`
2. `Arkham Horror LCG`
3. `Stellar Horizons`
4. `Magic: The Gathering`

This order is useful because each step introduces a new class of difficulty:

- `SETI`: clean baseline rule extraction and ruling presentation
- `Arkham Horror LCG`: card text interacting with a rules reference backbone
- `Stellar Horizons`: fragmented and poorly indexed source material that requires reconstruction
- `MTG`: extreme scale, dense precedence interactions, and a mature rules hierarchy

`Stellar Horizons` is intentionally treated as a branch after `SETI`, not a mandatory gate before `Arkham Horror LCG`. The goal is to learn from two different directions before committing to `MTG`:

- one path tests card-plus-core-rule interaction (`Arkham`)
- one path tests fragmented manual reconstruction (`Stellar Horizons`)

Both should be proven before `MTG`.

## Phase 1: SETI

Why it is chosen:

- the rulebook is comparatively clean and modern
- secondary sources and explanatory material exist
- the maintainer already knows the game well enough to debug mistakes
- it is a good environment for separating ingestion mistakes from domain misunderstanding

What `SETI` should prove:

- one subsystem can be modeled as canonical rule nodes plus supporting passages
- graph edges can represent definitions, timing, requirements, and modifications cleanly
- the system can produce a structured ruling with citations and visible precedence
- DSPy-based output constraints can keep the LLM inside the intended role

Key facets to implement:

- source-passage ingestion with stable locators
- canonical rule-node design for one subsystem
- basic rule edge taxonomy applied consistently
- structured ruling schema in markdown
- baseline evaluation cases for standard and hard interactions
- manual ground-truth graph slice for comparison against automated extraction

Exit criteria:

- at least one `SETI` subsystem has a manually validated graph seed
- automated extraction can be measured against that seed
- the chat output shows correct citation paths on representative pilot questions

## Phase 2: Arkham Horror LCG

Why it is chosen:

- it introduces a card-interaction layer on top of a rules backbone
- the Rules Reference format is closer to `MTG` than most board games
- the Core Set provides a bounded initial corpus
- the game has meaningful interaction complexity without the scale of `MTG`

What `Arkham Horror LCG` should prove:

- the architecture can support card text as a second knowledge layer
- rulings can combine card effects with global rules-reference entries
- the system can distinguish between baseline rules and scenario or card-specific modifications

Key facets to implement:

- separate schema treatment for rules-reference entries and card text
- card-to-rule and card-to-card relation modeling
- specificity and exception handling between card effects and core rules
- Core Set scoped corpus before any expansion-cycle growth
- evaluation cases focused on timing, triggers, and exception handling

Exit criteria:

- card text and rules-reference entries can be retrieved and ranked together
- the ruling output can cite both backbone rules and card text without muddling precedence
- the Core Set slice is stable enough to test repeated adjudication patterns

## Phase 3: Stellar Horizons

Why it is chosen:

- the rulebook is a deliberate stress test for poor source structure
- rules are scattered, fragmented, and often difficult to assemble manually
- the community already experiences this as a real pain point
- there is a plausible early-adopter and beta-tester pool

What `Stellar Horizons` should prove:

- VGM can reconstruct a coherent rule view from fragmented source passages
- the graph architecture can represent a single practical rule through multiple source fragments
- retrieval can bridge weak indexing and poor document organization

Key facets to implement:

- fragment clustering and canonicalization workflow
- explicit support for one canonical rule depending on multiple source passages
- conflict and deduplication handling for overlapping extractions
- stronger ingestion review tooling than `SETI` requires
- evaluation cases focused on “find the whole rule across multiple locations”

Exit criteria:

- the system can assemble complete rulings from dispersed manual sections
- operators can inspect why certain passages were merged or kept separate
- community-facing pilot questions show practical value over manual lookup

## Phase 4: Magic: The Gathering

Why it is chosen:

- it is the ultimate target and the highest-value proof of concept
- the Comprehensive Rules are dense, explicit, and richly cross-referenced
- Scryfall Oracle text provides a practical card-text layer
- the ecosystem has a well-developed notion of rule hierarchy and edge-case adjudication

What `MTG` should prove:

- the system scales to a large, highly interconnected rules corpus
- graph traversal and citation structure remain usable at much larger rule density
- card text, Oracle updates, and comprehensive rules can coexist in one adjudication workflow
- the approach remains viable when multiple interacting layers and exceptions stack together

Key facets to implement:

- parser pipeline for Comprehensive Rules plain text into canonical nodes and edges
- card layer built from Oracle text rather than printed text
- support for explicit layer and dependency semantics where relevant
- robust precedence modeling between CR backbone and card-specific effects
- evaluation focused on complex multi-rule interactions rather than simple lookups

Exit criteria:

- the system can support nontrivial `MTG` rulings with correct rule hierarchy
- graph and retrieval performance remain acceptable on local hardware
- the ingestion model is stable enough to handle corpus updates over time

## Cross-Game Implementation Themes

Across all four games, the roadmap should preserve a common backbone:

- stable rule-node and source-passage modeling
- explicit rule relationship taxonomy
- deterministic citation metadata
- structured ruling output
- evaluation sets with hard cases
- visible uncertainty and abstention behavior

What should vary by game:

- ingestion heuristics
- game-specific edge types
- canonical rule granularity
- precedence logic extensions
- corpus update workflows

## Decision Gates Before Advancing

Before moving from one game to the next, confirm:

- the current game added the intended new capability rather than just more content
- the evaluation set exposes the dominant failure modes for that stage
- at least one validation path is executable and repeatable
- unresolved ingest problems are understood well enough not to contaminate the next stage

The critical gate before `MTG` is especially strict:

- both `Arkham Horror LCG` and `Stellar Horizons` should demonstrate success
- one proves card-plus-rules interaction
- the other proves fragmented-manual reconstruction
- if either remains weak, `MTG` risk goes up sharply and the roadmap should pause
