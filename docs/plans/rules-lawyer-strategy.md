# VGM Rules Lawyer Strategy

This document frames the product direction for using Vector Graph Memory as a rules-lawyer system for complex tabletop games.

Status:

- Strategic planning draft
- Focused on product direction rather than implementation detail
- Assumes the near-term proving ground is a fully local, air-gapped deployment built around Open WebUI

## Product Definition

VGM Rules Lawyer is a rule-retrieval and rule-precedence system, not a freeform answer bot.

The core value proposition is:

- retrieve relevant rules and rule fragments
- assemble them into a defensible ruling path
- expose the citation chain and precedence order explicitly
- keep the language model in a constrained identification and ranking role

The intended output is structured and inspectable:

- ruling
- primary rule citation
- modifying or overriding rules
- precedence order
- confidence or uncertainty markers when evidence is incomplete

This matters because tournament and adjudication workflows do not need persuasive prose. They need traceable reasoning with structural citations.

## Customer Base

Primary customer segments:

- local game stores running events without consistent access to expert judges
- tournament organizers for niche or judge-light game communities
- play groups handling complex edge cases in campaign or competitive environments
- content creators, rules explainers, and moderators who repeatedly answer the same high-friction questions

Secondary customer segments:

- serious hobby groups that want an offline adjudication tool
- convention organizers running events across multiple games
- publishers or community maintainers who want to validate that rules can be navigated consistently

The first commercial wedge is local game stores because they have recurring need, clear pain, and practical tolerance for a tool that augments human judgment rather than replacing it.

## Core Use Cases

High-priority use cases:

- resolve a disputed interaction during live play
- cite the exact rules that control the interaction
- identify whether a more specific rule overrides a general one
- surface the minimum rule set needed to justify a ruling
- explain uncertainty when the rules corpus is ambiguous or fragmented

Operational use cases:

- train staff on common edge cases
- build a searchable record of frequently asked rulings
- support post-game or between-round adjudication without internet access
- provide a structured starting point for escalation to a human expert

Non-goals for the early product:

- replacing certified judges in high-stakes events
- generating unofficial policy documents from scratch
- acting as a general conversational assistant for broad hobby topics
- giving authoritative rulings when the source corpus is incomplete or contradictory without exposing that limitation

## Market Segment

This product fits a narrow but real niche:

- rules-dense tabletop games
- communities where mistakes are costly in time, trust, or tournament integrity
- environments where internet access is unreliable, undesirable, or impossible
- ecosystems with insufficient human expert coverage at point of play

Why this segment is attractive:

- the pain is acute rather than abstract
- the success criteria are clearer than for generic AI assistants
- traceable citations are valued more than stylistic fluency
- modest local hardware is acceptable if the model role stays constrained

The likely adoption pattern is not mass-consumer first. It is specialist adoption through communities with existing rule-friction and strong word-of-mouth channels.

## Product Principles

The product should be guided by the following principles:

- Retrieval over synthesis: the system should expose stored rule structure rather than fabricate explanations.
- Structural citations: citations should be attached from node metadata and graph traversal, not generated ad hoc by the model.
- Precedence is first class: the output must show why one rule controls another.
- Local-first operation: the stack should run in Docker on modest machines with no external dependency.
- Human-augmenting posture: the tool should support judgment, not overclaim authority.
- Narrow model responsibility: the LLM should identify, rank, and format within a constrained schema.
- Failure visibility: ambiguity, missing evidence, and conflicting passages must be surfaced rather than hidden.

## Product Requirements That Follow From Those Principles

To support the intended use case, the product will need:

- a rule-node schema that separates canonical rules from raw source passages when needed
- edge types for definition, modification, override, timing, requirement, contribution, conflict, and domain-specific precedence
- source metadata rich enough to reconstruct exact citations in the UI
- a retrieval path that can collect both semantically similar passages and graph-connected controlling rules
- a response schema that preserves ruling structure instead of flattening everything into answer prose
- an evaluation framework that scores citation correctness, precedence correctness, and abstention quality
- Open WebUI presentation that makes the ruling chain legible to a non-technical user

## Delivery Assumptions

Current strategic assumptions:

- deployment is fully local through Docker Compose
- Open WebUI is the primary operator interface
- Ollama-hosted local models are the target inference path
- early wins come from domain-constrained pilots, not from broad generalization
- DSPy signatures and evaluation harnesses are the main control mechanism for smaller-model output discipline

These are reasonable assumptions for the pilot phase, but they still need validation against real user behavior.

## Risks And Constraints

Product risks:

- users may expect definitive authority even when the corpus is weak or fragmented
- some communities may reject non-official tooling unless the citation trail is exceptionally clear
- if result formatting is too technical, store staff may not trust or use the output in the moment

Technical risks:

- ingestion quality may dominate system accuracy more than model quality
- implicit cross-references may be missed in human-written manuals
- fragmented rules may resist canonicalization without game-specific logic
- poor edge semantics will make graph traversal look sophisticated while producing weak rulings
- local small models may struggle on multi-rule interactions without tight signatures and examples

Operational risks:

- data preparation per game may be more expensive than initially expected
- some rulebooks or card databases may have licensing or redistribution constraints
- keeping game-specific corpora current may require ongoing ingest maintenance

## Product Development Guidance

Near-term product development should optimize for:

- one clearly successful pilot over shallow support for many games
- explicit evidence chains over answer elegance
- correctness on hard edge cases over breadth of conversational capability
- tooling for ingestion review and graph inspection, not just end-user chat
- evaluation discipline that separates retrieval quality from synthesis quality

This implies a practical build order:

1. Prove that one clean subsystem can be represented correctly in graph form.
2. Prove that the system can return a ruling with structural citations and visible precedence.
3. Prove that a small local model can reliably fill the constrained output schema.
4. Prove that game-specific ingestion can scale past a hand-built seed.

## Open Questions

Product questions:

- who is the actual initial buyer: store owners, event organizers, community moderators, or advanced hobbyists?
- what level of authority language is acceptable without implying official certification?
- what result format is fastest to trust during live play: terse ruling blocks, expandable reasoning trees, or both?
- should the system default to abstaining more often in tournament mode than in casual mode?

Go-to-market questions:

- is the first viable offering a local tool sold to stores, a community-supported package, or a consulting-style pilot with one game community?
- what support burden comes with multi-game deployment in small retail environments?
- which early adopter community is most likely to provide rapid, structured feedback?

Technical questions:

- what is the correct canonical unit of a rule in each target game: numbered rule, paragraph, bullet, clause, or synthesized composite?
- when should fragmented passages remain separate nodes versus being merged into one canonical rule node?
- how much game-specific logic belongs in ingestion compared to the generic VGM substrate?
- what graph edge taxonomy should remain universal across games, and what should be game-specific extensions?
- what minimum local model quality is acceptable before the user experience degrades at point of play?

Evaluation questions:

- how should correctness be scored when the ruling text is acceptable but the citation chain is incomplete?
- what is the right threshold for abstention versus best-effort ranking?
- how many hand-labeled edge cases are needed before a pilot is trustworthy?

## Success Criteria

The strategy is working if the pilot can demonstrate:

- users can obtain a traceable ruling with correct citations on representative hard cases
- the system abstains or flags uncertainty when evidence is weak
- the output is fast enough and clear enough for live adjudication
- the ingestion workflow can be repeated for a second game without redesigning the entire stack

The strategy is not yet working if:

- the system gives plausible but weakly supported rulings
- citation chains are present but not decision-relevant
- the graph schema becomes game-specific in an uncontrolled way
- ingestion remains so manual that expansion beyond one pilot is unrealistic
