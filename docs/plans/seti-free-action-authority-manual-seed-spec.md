# SETI Free Action And Player Aid Authority Manual Seed Spec

This document defines the second manual graph seed for the `SETI` free-action timing and player-aid authority subsystem.

Status:

- Manual seed specification
- Rules-only scope
- Intended to be the hand-built reference for the second pilot graph slice

Related planning docs:

- [seti-free-action-authority-subsystem-brief.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/seti-free-action-authority-subsystem-brief.md)
- [seti-pilot-next-steps.md](/home/jcherry/Documents/storage/git/vector-graph-memory/docs/plans/seti-pilot-next-steps.md)

## Purpose

The purpose of this seed spec is to define one explicit, reproducible hand-built graph slice that can be used as:

- the adjudication reference for the second `SETI` pilot subsystem
- the comparison target for automated extraction
- the backing structure for timing and authority-sensitive pilot questions

## Scope

In scope:

- free actions only during your turn
- free actions interrupting main actions
- prohibition on interrupting one free action with another
- full resolution requirement for a free action before another begins
- scan-example timing
- income-increase-example timing
- player-aid free-action summary wording

Out of scope:

- broader card timing interactions
- species-specific exceptions
- unrelated player-aid interpretation questions

## Frozen Pilot Questions

This seed must support these four questions:

1. `Can I interrupt a main action with a free action, and can I interrupt one free action with another free action?`
2. `During a Scan action, can I interrupt the Scan with a free action and then continue the Scan?`
3. `If placing data triggers the income increase effect, can I place more data first and then resolve that income effect?`
4. `The player aid says free actions are "NO LIMIT". Does that allow one free action to interrupt another?`

## Seed Shape

The manual seed should include three kinds of records:

- source-passage nodes
- canonical rule nodes
- edges between source passages and rules, and between rules themselves

## Source Nodes

The second seed should capture:

1. `src_seti_faq_q4_free_action_timing`
2. `src_seti_faq_q4_scan_example`
3. `src_seti_faq_q4_income_example`
4. `src_seti_aid_free_actions_summary`

## Canonical Rule Nodes

The second seed should include:

1. `rule_seti_free_actions_during_turn_only`
2. `rule_seti_free_actions_can_interrupt_main_action`
3. `rule_seti_free_actions_cannot_interrupt_free_action`
4. `rule_seti_free_action_must_resolve_before_next`
5. `rule_seti_scan_action_can_resume_after_free_action_interrupt`
6. `rule_seti_income_increase_free_action_cannot_be_nested`
7. `rule_seti_player_aid_free_actions_no_limit_summary`

## Rule Semantics

- the FAQ timing paragraph should drive the core timing rules
- the scan example should clarify that a main action can resume after a free-action interruption
- the income example should clarify that one free action must finish before another begins
- the player-aid summary should be represented, but it should not override the FAQ timing rule

## Edge Semantics

The second seed should likely include:

- FAQ support edges into the timing rules
- example clarification edges into the example-specific rules
- `contributes_to` edges from example-specific rules into the broader timing rules
- a `modifies` edge from the FAQ non-nesting rule into the player-aid `NO LIMIT` summary rule

## Acceptance Criteria

Before treating the hand-built graph slice as the reference, confirm:

- the player-aid summary is represented without being treated as authoritative over the FAQ
- the graph can answer the direct timing question and the authority-conflict question
- the scan and income examples both reinforce the base timing rules
