# SETI Free Action And Player Aid Authority Subsystem Brief

This document defines the second bounded `SETI` subsystem for the VGM Rules Lawyer pilot.

Status:

- Pilot subsystem brief
- Intended as the second manual graph seed after landing/orbiter interactions
- Focused on free-action timing plus authority handling between the FAQ and the player aid

## Why This Subsystem

This subsystem is a strong second pilot target because it is:

- small enough to model manually without widening the game scope too far
- different from the landing/orbiter slice because it tests timing and source authority
- already represented in the tracked `SETI` eval fixture
- useful for proving that summary material should not silently override authoritative clarification

It exercises several important adjudication behaviors at once:

- main-action interruption by free actions
- prohibition on interrupting one free action with another
- requirement to fully resolve a free action before starting another
- explicit worked examples that constrain timing interpretation
- authority preference for FAQ clarification over player-aid summary wording

## In-Scope Questions

The first-pass subsystem should answer at least these questions:

1. Can I interrupt a main action with a free action?
2. Can I interrupt one free action with another free action?
3. During a Scan action, can I interrupt the action with a free action and then resume the Scan?
4. Does the player aid text `FREE ACTIONS (NO LIMIT)` mean free actions can nest inside each other?

## Source Scope

Primary source passages:

- `seti-faq.txt`:
  - `Q4. When exactly can I play free actions? Can they interrupt a main action?`
  - `Example 1: When performing a Scan action`
  - `Example 2: You would place a data token on the fourth space in your computer`

Supporting summary passage:

- `seti-player-aid-en.txt`:
  - `FREE ACTIONS (NO LIMIT)`

Relevant source locations already visible in the local extracted texts:

- [seti-faq.txt](../../tests/fixtures/rag_eval/source_documents/extracted/seti-faq.txt#L55)
- [seti-player-aid-en.txt](../../tests/fixtures/rag_eval/source_documents/extracted/seti-player-aid-en.txt#L1)

## Authority Model For This Subsystem

Authority order for this pilot slice:

1. species-specific addendum or card-specific exception, if one explicitly applies
2. FAQ clarification when it directly clarifies the interaction in question
3. core rulebook for the base timing structure
4. player aid is summary-only and is not authoritative on its own

For this manual seed, the key authority relationship is between the FAQ clarification and the player-aid summary.

## Initial Scope Boundary

In scope:

- free actions only during your turn
- free actions interrupting main actions
- free actions not interrupting other free actions
- full resolution requirement for a free action before another begins
- the Scan worked example
- the income-increase worked example
- player-aid summary wording and its limited authority

Out of scope:

- all other timing questions in the game
- full card corpus timing interactions
- mission timing beyond the worked example shape
- species-specific or card-specific overrides

## Frozen Pilot Question Set

The first frozen question set for this subsystem should be:

1. `Can I interrupt a main action with a free action, and can I interrupt one free action with another free action?`
2. `During a Scan action, can I interrupt the Scan with a free action and then continue the Scan?`
3. `If placing data triggers the income increase effect, can I place more data first and then resolve that income effect?`
4. `The player aid says free actions are "NO LIMIT". Does that allow one free action to interrupt another?`

These map cleanly onto the currently tracked eval cases:

- `seti-rules-012-free-action-timing`
- `seti-rules-021-scan-interrupt-publicity`
- `seti-rules-022-free-action-income-nesting`
- `seti-rules-028-player-aid-free-action-summary-conflict`

## Candidate Canonical Rule Nodes

The second manual graph seed should likely include at least these canonical rule nodes:

1. `rule_seti_free_actions_during_turn_only`
2. `rule_seti_free_actions_can_interrupt_main_action`
3. `rule_seti_free_actions_cannot_interrupt_free_action`
4. `rule_seti_free_action_must_resolve_before_next`
5. `rule_seti_scan_action_can_resume_after_free_action_interrupt`
6. `rule_seti_income_increase_free_action_cannot_be_nested`
7. `rule_seti_player_aid_free_actions_no_limit_summary`

## Candidate Source Passage Nodes

The manual seed should capture at least:

- `src_seti_faq_q4_free_action_timing`
- `src_seti_faq_q4_scan_example`
- `src_seti_faq_q4_income_example`
- `src_seti_aid_free_actions_summary`

## Manual Graph Seed Acceptance Criteria

Before treating the hand-built graph slice as the pilot reference, confirm:

- every canonical rule node is backed by at least one explicit source passage
- the FAQ-derived timing rules remain separate from the player-aid summary rule
- the summary wording can be present in the graph without overriding the FAQ clarification
- the scan and income examples strengthen the timing rules without replacing them
