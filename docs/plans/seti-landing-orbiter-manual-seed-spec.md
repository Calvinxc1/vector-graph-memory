# SETI Landing And Orbiter Manual Seed Spec

This document defines the first manual graph seed for the `SETI` landing and orbiter subsystem.

Status:

- Manual seed specification
- Rules-only scope
- Intended to be the hand-built reference for the first pilot graph slice

Related planning docs:

- [seti-landing-orbiter-subsystem-brief.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/seti-landing-orbiter-subsystem-brief.md)
- [seti-pilot-next-steps.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/seti-pilot-next-steps.md)

## Purpose

The purpose of this seed spec is to define one explicit, reproducible hand-built graph slice that can be used as:

- the adjudication reference for the first `SETI` pilot subsystem
- the comparison target for automated extraction
- the backing structure for the first frozen pilot question set

This is not the final schema for the full product. It is a deliberately narrow reference slice.

## Scope

In scope:

- orbit action as the status change from probe to orbiter
- orbiter permanence
- standard landing action and baseline landing discount from an existing orbiter
- ownership-independence of the orbiter-based landing discount
- moon landing using the same discount logic as the planet
- separate moon-access prerequisite

Out of scope:

- card-based landing exceptions
- general movement rules
- broader probe-limit logic
- species-specific overrides
- detailed landing reward modeling beyond what is needed to support the discount questions

## Frozen Pilot Questions

This seed must support these four questions:

1. `Can an orbiter later land on the same planet?`
2. `Does an opponent's orbiter still reduce my landing cost?`
3. `Does an existing orbiter also reduce the cost to land on that planet's moon?`
4. `If another player's orbiter is at Jupiter and I have the tech that lets me land on moons, do I get the discount when landing on one of Jupiter's moons?`

## Seed Shape

The manual seed should include three kinds of records:

- source-passage nodes
- canonical rule nodes
- edges between source passages and rules, and between rules themselves

The first-pass seed should prefer explicitness over compression.

## Source-Passage Node Spec

Required fields for every source-passage node:

- `node_id`
- `node_type`
- `game_id`
- `document_id`
- `document_type`
- `authority_scope`
- `title`
- `locator`
- `page`
- `source_text`
- `citation_label`
- `citation_short`
- `language`
- `seed_version`

Recommended fixed values:

- `node_type`: `source_passage`
- `game_id`: `seti`
- `language`: `en`
- `seed_version`: `seti_landing_orbiter_seed_v1`

### Source Nodes

#### 1. `src_seti_faq_q5_land_with_orbiter`

- `node_type`: `source_passage`
- `document_id`: `seti-faq`
- `document_type`: `faq`
- `authority_scope`: `rule_clarification`
- `title`: `Q5. Can I land with an orbiter?`
- `locator`: `Q5. Can I land with an orbiter?`
- `page`: `3`
- `citation_label`: `SETI FAQ Q5`
- `citation_short`: `FAQ Q5`
- `source_text_summary`: orbiter remains in orbit, cannot return to the solar system board, and cannot land later

#### 2. `src_seti_faq_q6_opponent_orbiter_discount`

- `node_type`: `source_passage`
- `document_id`: `seti-faq`
- `document_type`: `faq`
- `authority_scope`: `rule_clarification`
- `title`: `Q6. Do I get the landing discount for an opponent's orbiter?`
- `locator`: `Q6. Do I get the landing discount for an opponent's orbiter?`
- `page`: `3`
- `citation_label`: `SETI FAQ Q6`
- `citation_short`: `FAQ Q6`
- `source_text_summary`: landing discount applies as long as an orbiter is present, regardless of ownership

#### 3. `src_seti_faq_q7_moon_discount`

- `node_type`: `source_passage`
- `document_id`: `seti-faq`
- `document_type`: `faq`
- `authority_scope`: `rule_clarification`
- `title`: `Q7. Can landing on moons be discounted?`
- `locator`: `Q7. Can landing on moons be discounted?`
- `page`: `3`
- `citation_label`: `SETI FAQ Q7`
- `citation_short`: `FAQ Q7`
- `source_text_summary`: moon landings use the same discount logic as the planet, but still require moon-landing access

#### 4. `src_seti_core_orbit_a_planet`

- `node_type`: `source_passage`
- `document_id`: `seti-rules-en`
- `document_type`: `core_rules`
- `authority_scope`: `general_rules`
- `title`: `ORBIT A PLANET`
- `locator`: `Pay 1 credit and 1 energy to turn one of your probes into an orbiter`
- `page`: `10`
- `citation_label`: `SETI Core Rules p.10`
- `citation_short`: `Core p.10`
- `source_text_summary`: orbit action turns a probe into an orbiter, grants base orbit rewards, and removes the figure from probe-limit counting on the solar system board

#### 5. `src_seti_core_land_on_planet_or_moon`

- `node_type`: `source_passage`
- `document_id`: `seti-rules-en`
- `document_type`: `core_rules`
- `authority_scope`: `general_rules`
- `title`: `LAND ON A PLANET OR MOON`
- `locator`: `Pay 3 energy to turn one of your probes into a lander`
- `page`: `11`
- `citation_label`: `SETI Core Rules p.11`
- `citation_short`: `Core p.11`
- `source_text_summary`: landing is a main action, moon landing requires enabling access, and an existing orbiter reduces the normal landing cost

## Canonical Rule Node Spec

Required fields for every canonical rule node:

- `node_id`
- `node_type`
- `game_id`
- `rule_kind`
- `title`
- `normalized_statement`
- `scope`
- `seed_version`

Recommended fixed values:

- `node_type`: `canonical_rule`
- `game_id`: `seti`
- `seed_version`: `seti_landing_orbiter_seed_v1`

### Canonical Rule Nodes

#### 1. `rule_seti_orbit_action_base`

- `rule_kind`: `action_rule`
- `title`: `Orbit Action Base`
- `normalized_statement`: a probe on a planet space may be turned into an orbiter as a main action
- `scope`: `general`

#### 2. `rule_seti_orbiter_status_change`

- `rule_kind`: `state_transition`
- `title`: `Probe Becomes Orbiter`
- `normalized_statement`: when a probe is moved to orbit on the planetary board, it becomes an orbiter rather than remaining a probe on the solar system board
- `scope`: `general`

#### 3. `rule_seti_orbiter_is_permanent`

- `rule_kind`: `restriction`
- `title`: `Orbiter Permanence`
- `normalized_statement`: once a figure becomes an orbiter, it stays there for the rest of the game and cannot later land or return to the solar system board
- `scope`: `general`

#### 4. `rule_seti_landing_action_base`

- `rule_kind`: `action_rule`
- `title`: `Landing Action Base`
- `normalized_statement`: landing turns a probe into a lander as a main action and normally costs 3 energy
- `scope`: `general`

#### 5. `rule_seti_landing_discount_if_orbiter_present`

- `rule_kind`: `cost_modifier`
- `title`: `Landing Discount From Existing Orbiter`
- `normalized_statement`: if an orbiter is already at the planet, landing there costs 1 less energy than normal
- `scope`: `general`

#### 6. `rule_seti_landing_discount_not_owner_limited`

- `rule_kind`: `clarification`
- `title`: `Orbiter Ownership Does Not Matter`
- `normalized_statement`: the landing discount applies if any orbiter is present at the planet, regardless of which player owns it
- `scope`: `general`

#### 7. `rule_seti_moon_landing_requires_access`

- `rule_kind`: `prerequisite`
- `title`: `Moon Landing Requires Access`
- `normalized_statement`: a player cannot land on a moon unless some effect or technology explicitly allows it
- `scope`: `general`

#### 8. `rule_seti_moon_landing_inherits_planet_discount_logic`

- `rule_kind`: `clarification`
- `title`: `Moon Landing Uses Planet Discount Logic`
- `normalized_statement`: for discount purposes, landing on a moon is treated the same as landing on that moon's planet
- `scope`: `general`

#### 9. `rule_seti_moon_landing_inherits_other_landing_discounts`

- `rule_kind`: `clarification`
- `title`: `Other Landing Discounts Also Apply To Moons`
- `normalized_statement`: other landing-discount effects that apply to the planet also apply to the moon when moon landing is allowed
- `scope`: `general`

## Edge Spec

Required fields for every edge:

- `edge_id`
- `edge_type`
- `from_node_id`
- `to_node_id`
- `game_id`
- `seed_version`
- `rationale`

Recommended fixed values:

- `game_id`: `seti`
- `seed_version`: `seti_landing_orbiter_seed_v1`

### Source-To-Rule Edges

#### 1. `edge_src_core_orbit_supports_rule_orbit_action_base`

- `edge_type`: `supports`
- `from_node_id`: `src_seti_core_orbit_a_planet`
- `to_node_id`: `rule_seti_orbit_action_base`
- `rationale`: core rules define orbit as the action that creates an orbiter

#### 2. `edge_src_core_orbit_supports_rule_orbiter_status_change`

- `edge_type`: `supports`
- `from_node_id`: `src_seti_core_orbit_a_planet`
- `to_node_id`: `rule_seti_orbiter_status_change`
- `rationale`: core rules describe moving the figure from the solar system board to planetary orbit

#### 3. `edge_src_faq_q5_clarifies_rule_orbiter_is_permanent`

- `edge_type`: `clarifies`
- `from_node_id`: `src_seti_faq_q5_land_with_orbiter`
- `to_node_id`: `rule_seti_orbiter_is_permanent`
- `rationale`: FAQ Q5 explicitly states the orbiter cannot later land or return

#### 4. `edge_src_core_land_supports_rule_landing_action_base`

- `edge_type`: `supports`
- `from_node_id`: `src_seti_core_land_on_planet_or_moon`
- `to_node_id`: `rule_seti_landing_action_base`
- `rationale`: core rules define landing as a main action with standard energy cost

#### 5. `edge_src_core_land_supports_rule_landing_discount_if_orbiter_present`

- `edge_type`: `supports`
- `from_node_id`: `src_seti_core_land_on_planet_or_moon`
- `to_node_id`: `rule_seti_landing_discount_if_orbiter_present`
- `rationale`: core rules define the reduced cost when an orbiter is already present

#### 6. `edge_src_faq_q6_clarifies_rule_landing_discount_not_owner_limited`

- `edge_type`: `clarifies`
- `from_node_id`: `src_seti_faq_q6_opponent_orbiter_discount`
- `to_node_id`: `rule_seti_landing_discount_not_owner_limited`
- `rationale`: FAQ Q6 resolves the ownership condition directly

#### 7. `edge_src_core_land_supports_rule_moon_landing_requires_access`

- `edge_type`: `supports`
- `from_node_id`: `src_seti_core_land_on_planet_or_moon`
- `to_node_id`: `rule_seti_moon_landing_requires_access`
- `rationale`: core rules state moon landing requires some enabling effect or tech

#### 8. `edge_src_faq_q7_clarifies_rule_moon_landing_requires_access`

- `edge_type`: `clarifies`
- `from_node_id`: `src_seti_faq_q7_moon_discount`
- `to_node_id`: `rule_seti_moon_landing_requires_access`
- `rationale`: FAQ Q7 preserves the access prerequisite while discussing discounts

#### 9. `edge_src_faq_q7_clarifies_rule_moon_landing_inherits_planet_discount_logic`

- `edge_type`: `clarifies`
- `from_node_id`: `src_seti_faq_q7_moon_discount`
- `to_node_id`: `rule_seti_moon_landing_inherits_planet_discount_logic`
- `rationale`: FAQ Q7 states that moon landing should be treated as landing on the moon's planet for discount purposes

#### 10. `edge_src_faq_q7_clarifies_rule_moon_landing_inherits_other_discounts`

- `edge_type`: `clarifies`
- `from_node_id`: `src_seti_faq_q7_moon_discount`
- `to_node_id`: `rule_seti_moon_landing_inherits_other_landing_discounts`
- `rationale`: FAQ Q7 extends other landing discounts to moons as well

### Rule-To-Rule Edges

#### 11. `edge_rule_orbit_status_contributes_to_orbiter_permanence`

- `edge_type`: `contributes_to`
- `from_node_id`: `rule_seti_orbiter_status_change`
- `to_node_id`: `rule_seti_orbiter_is_permanent`
- `rationale`: the permanence rule applies after the status transition to orbiter

#### 12. `edge_rule_owner_irrelevant_modifies_orbiter_discount`

- `edge_type`: `modifies`
- `from_node_id`: `rule_seti_landing_discount_not_owner_limited`
- `to_node_id`: `rule_seti_landing_discount_if_orbiter_present`
- `rationale`: the clarification expands the base discount rule to all orbiters, not just the player's own

#### 13. `edge_rule_moon_logic_contributes_to_orbiter_discount`

- `edge_type`: `contributes_to`
- `from_node_id`: `rule_seti_moon_landing_inherits_planet_discount_logic`
- `to_node_id`: `rule_seti_landing_discount_if_orbiter_present`
- `rationale`: the moon rule reuses the planet discount rule for moon landings

#### 14. `edge_rule_moon_access_applies_during_landing`

- `edge_type`: `applies_during`
- `from_node_id`: `rule_seti_moon_landing_requires_access`
- `to_node_id`: `rule_seti_landing_action_base`
- `rationale`: moon access is a prerequisite constraint on the landing action

#### 15. `edge_rule_other_moon_discounts_modifies_landing_base`

- `edge_type`: `modifies`
- `from_node_id`: `rule_seti_moon_landing_inherits_other_landing_discounts`
- `to_node_id`: `rule_seti_landing_action_base`
- `rationale`: other landing discounts become relevant when the landing action targets a moon

## Citation Metadata Shape

The ruling layer for this seed should be able to pull citations directly from source nodes using at least:

- `source_node_id`
- `citation_label`
- `citation_short`
- `document_id`
- `document_type`
- `page`
- `locator`
- `authority_scope`

Example citation object shape:

```json
{
  "source_node_id": "src_seti_faq_q6_opponent_orbiter_discount",
  "citation_label": "SETI FAQ Q6",
  "citation_short": "FAQ Q6",
  "document_id": "seti-faq",
  "document_type": "faq",
  "page": 3,
  "locator": "Q6. Do I get the landing discount for an opponent's orbiter?",
  "authority_scope": "rule_clarification"
}
```

## Expected Ruling Support Per Question

### Question 1

`Can an orbiter later land on the same planet?`

Minimum supporting rule path:

- `rule_seti_orbiter_status_change`
- `rule_seti_orbiter_is_permanent`
- primary citation: `src_seti_faq_q5_land_with_orbiter`

### Question 2

`Does an opponent's orbiter still reduce my landing cost?`

Minimum supporting rule path:

- `rule_seti_landing_discount_if_orbiter_present`
- `rule_seti_landing_discount_not_owner_limited`
- primary citation: `src_seti_faq_q6_opponent_orbiter_discount`
- supporting citation: `src_seti_core_land_on_planet_or_moon`

### Question 3

`Does an existing orbiter also reduce the cost to land on that planet's moon?`

Minimum supporting rule path:

- `rule_seti_moon_landing_inherits_planet_discount_logic`
- `rule_seti_landing_discount_if_orbiter_present`
- `rule_seti_moon_landing_requires_access`
- primary citation: `src_seti_faq_q7_moon_discount`

### Question 4

`If another player's orbiter is at Jupiter and I have the tech that lets me land on moons, do I get the discount when landing on one of Jupiter's moons?`

Minimum supporting rule path:

- `rule_seti_landing_discount_not_owner_limited`
- `rule_seti_moon_landing_inherits_planet_discount_logic`
- `rule_seti_moon_landing_requires_access`
- `rule_seti_landing_discount_if_orbiter_present`
- primary citations:
  - `src_seti_faq_q6_opponent_orbiter_discount`
  - `src_seti_faq_q7_moon_discount`

## Acceptance Criteria For The Manual Seed

Before treating this seed as the reference slice, verify:

- all five source-passage nodes are populated with exact citation metadata
- all nine canonical rule nodes are represented
- all fifteen defined edges are represented or intentionally adjusted with documented rationale
- each frozen pilot question can be answered from the graph with no unstated bridging logic
- the answer path for the moon questions includes the moon-access prerequisite explicitly

## Known Simplifications

This seed intentionally simplifies some details:

- it does not model every landing reward on every planet or moon
- it does not model card-specific exceptions
- it does not encode every possible synonym for probe, orbiter, and lander
- it treats the rules-only slice as game-general rather than planet-specific

Those simplifications are acceptable for the first pilot as long as they are explicit.
