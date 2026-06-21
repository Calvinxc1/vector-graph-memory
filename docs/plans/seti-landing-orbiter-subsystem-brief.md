# SETI Landing And Orbiter Subsystem Brief

This document defines the first bounded `SETI` subsystem for the VGM Rules Lawyer pilot.

Status:

- Pilot subsystem brief
- Intended as the concrete scope freeze for the first manual graph seed
- Focused on landing and orbiter interactions, with a tracked extension point for card-based exceptions

## Why This Subsystem

This subsystem is a strong first pilot target because it is:

- small enough to model manually without excessive setup
- rich enough to require multi-rule composition instead of single-rule lookup
- already represented in the tracked `SETI` eval fixture
- easy to sanity-check with domain knowledge
- structurally similar to the kinds of specificity and exception handling the broader project needs

It exercises several important adjudication behaviors at once:

- status change from probe to orbiter
- permanence of that status
- cost modification based on board state
- ownership-independent applicability
- inheritance of landing discount from planet to moon
- separate prerequisite for moon landing access

## In-Scope Questions

The first-pass subsystem should answer at least these questions:

1. Can an orbiter later land on the same planet?
2. Does an opponent's orbiter still reduce my landing cost?
3. Does that same landing discount apply to moons?
4. If I can land on moons and another player's orbiter is already at the planet, do I get the discount on that moon?

These are sufficient to prove that the pilot can combine multiple stored rule fragments into one ruling with explicit support.

## Source Scope

Primary source passages:

- `seti-faq.txt`:
  - `Q5. Can I land with an orbiter?`
  - `Q6. Do I get the landing discount for an opponent's orbiter?`
  - `Q7. Can landing on moons be discounted?`

Supporting core-rule passages:

- `seti-rules-en.txt`:
  - `ORBIT A PLANET`
  - `LAND ON A PLANET OR MOON`

Relevant source locations already visible in the local extracted texts:

- [seti-faq.txt](../../tests/fixtures/rag_eval/source_documents/extracted/seti-faq.txt#L76)
- [seti-rules-en.txt](../../tests/fixtures/rag_eval/source_documents/extracted/seti-rules-en.txt#L444)

## Authority Model For This Subsystem

Authority order for the pilot slice:

1. species-specific addendum or card-specific exception, if one explicitly applies
2. FAQ clarification when it directly clarifies the interaction in question
3. core rulebook for the base action and reward structure
4. player aid is summary-only and is not authoritative on its own

For the initial manual graph seed, the dominant sources are the FAQ clarifications plus supporting core-rule text.

## Initial Scope Boundary

In scope:

- turning a probe into an orbiter
- orbiter permanence
- standard landing cost
- landing discount from an existing orbiter
- ownership-independence of that discount
- moon landing inheriting the same discount logic
- separate moon-access prerequisite

Out of scope for the first manual seed:

- general probe movement
- full moon-tech acquisition rules
- species-specific overrides unrelated to landing discounts
- broad card corpus ingestion
- all landing reward details unrelated to the landing-discount interaction

This keeps the pilot focused on adjudication logic rather than broad coverage.

## Candidate Canonical Rule Nodes

The first manual graph seed should likely include at least these canonical rule nodes:

1. `orbit_action_base`
   - A probe on a planet space can be turned into an orbiter as a main action.

2. `orbiter_status_change`
   - Once the figure is moved to orbit, it becomes an orbiter rather than a probe on the solar system board.

3. `orbiter_is_permanent`
   - An orbiter remains in orbit for the rest of the game and cannot later become a lander.

4. `landing_action_base`
   - Landing on a planet or moon turns a probe into a lander and has a standard energy cost.

5. `landing_discount_if_orbiter_present`
   - Landing at a planet costs less if an orbiter is already present there.

6. `landing_discount_not_owner_limited`
   - The landing discount applies regardless of which player owns the orbiter.

7. `moon_landing_requires_access`
   - Landing on a moon requires a specific enabling effect or technology.

8. `moon_landing_inherits_planet_discount_logic`
   - For discount purposes, landing on a moon is treated the same as landing on that moon's planet.

9. `moon_landing_inherits_other_landing_discounts`
   - Landing discounts from other applicable effects also apply to moons when moon access exists.

This is intentionally more granular than the final schema may be. For the pilot, slightly over-separating the rules is safer than merging distinct conditions too early.

## Candidate Source Passage Nodes

Separate source-passage nodes should be captured for at least:

- `faq_q5_land_with_orbiter`
- `faq_q6_opponent_orbiter_discount`
- `faq_q7_moon_discount`
- `core_orbit_a_planet`
- `core_land_on_a_planet_or_moon`

These source nodes should preserve:

- document ID
- document type
- page number where available
- locator text
- authority scope

## Candidate Edges

The first-pass graph should likely include edges such as:

- `core_orbit_a_planet -> supports -> orbit_action_base`
- `core_orbit_a_planet -> supports -> orbiter_status_change`
- `faq_q5_land_with_orbiter -> clarifies -> orbiter_is_permanent`
- `orbiter_status_change -> contributes_to -> orbiter_is_permanent`
- `core_land_on_a_planet_or_moon -> supports -> landing_action_base`
- `core_land_on_a_planet_or_moon -> supports -> landing_discount_if_orbiter_present`
- `faq_q6_opponent_orbiter_discount -> clarifies -> landing_discount_not_owner_limited`
- `landing_discount_not_owner_limited -> modifies -> landing_discount_if_orbiter_present`
- `faq_q7_moon_discount -> clarifies -> moon_landing_inherits_planet_discount_logic`
- `faq_q7_moon_discount -> requires -> moon_landing_requires_access`
- `moon_landing_inherits_planet_discount_logic -> contributes_to -> landing_discount_if_orbiter_present`
- `moon_landing_requires_access -> applies_during -> landing_action_base`
- `orbiter_is_permanent -> conflicts_with -> any_rule_claiming_orbiters_can_later_land`

The exact edge labels may change once the pilot schema is finalized, but the semantic relationships above should survive that renaming.

## First Frozen Pilot Question Set

The first frozen question set for this subsystem should be:

1. `Can an orbiter later land on the same planet?`
2. `Does an opponent's orbiter still reduce my landing cost?`
3. `Does an existing orbiter also reduce the cost to land on that planet's moon?`
4. `If another player's orbiter is at Jupiter and I have the tech that lets me land on moons, do I get the discount when landing on one of Jupiter's moons?`

These map cleanly onto the currently tracked eval cases:

- `seti-rules-004-can-orbiter-land`
- `seti-rules-005-opponent-orbiter-discount`
- `seti-rules-006-moon-discount`
- `seti-rules-027-moon-discount-opponent-orbiter`

## Manual Graph Seed Acceptance Criteria

Before treating the hand-built graph slice as the pilot reference, confirm:

- every canonical rule node is backed by at least one explicit source passage
- every rule-to-rule edge can be justified in plain language
- the authority model is encoded consistently across source nodes
- the composed moon-plus-opponent-orbiter case can be answered from the graph without ad hoc unstated reasoning
- the graph can also support abstention if the moon-access prerequisite is missing from the retrieved context

## Known Extension Point: Card-Based Exceptions

There is at least one known card-level interaction that bends normal landing occupancy or first-arrival reward logic by allowing a lander to share a spot and gain the same reward even when it was not first there.

Current status of that exception:

- known from maintainer domain knowledge
- not yet pinned to an exact card ID or source passage in the local extracted documents used during this pass
- should be treated as a tracked extension point, not as part of the first manual seed unless the exact source is identified first

Why it matters:

- it is a useful follow-up test for exception handling over the base landing rules
- it likely belongs in phase two of this subsystem, after the base rules are modeled cleanly

Follow-up action:

- identify the exact card, its text, and any FAQ clarification
- add it as a rule-specific exception node that overrides the standard occupancy or first-arrival reward logic

## Recommended Build Order For This Subsystem

1. capture the five in-scope source passages
2. create source-passage nodes with full citation metadata
3. create the base canonical rule nodes
4. wire the core support and FAQ clarification edges
5. test the four frozen pilot questions against the manual graph
6. only then add the first verified card-based exception

## Why This Is A Better First Slice Than The Alternatives

Compared with `Analyze Data`:

- this slice has stronger rule composition and more obvious future exception handling

Compared with `Scan`:

- this slice is narrower and less operationally noisy for the first manual seed

Compared with `Missions`:

- this slice is easier to debug and explain while still proving multi-rule interaction

Compared with `Free actions vs player aid`:

- this slice provides a broader adjudication structure, while authority-conflict modeling can be added later as a second pilot slice or a supplemental authority test
