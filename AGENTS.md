# AGENTS.md

This file defines required behavior for coding agents working in this repository.
These instructions apply to the entire repo tree.

## 1) GitFlow Requirements (Mandatory)

- Follow GitFlow branch roles:
  - `main`: production-ready history only.
  - `dev`: integration branch for upcoming work.
  - `feature/*`: branch from `dev`, merge back into `dev`.
  - `release/*`: branch from `dev`, merge into `main` only.
  - `hotfix/*`: branch from `main`, merge into `main` only.
- Contribution scope:
  - Community contributors are welcome to propose changes through `feature/* -> dev` pull requests.
  - `release/*` and `hotfix/*` branches and pull requests are core-developer managed.
- Never commit directly to `main` or `dev`.
- Use pull requests for all merges.
- Create pull requests as draft PRs by default; this is a recommended default, not a mandatory enforcement. Developers may open regular/open PRs when they judge it appropriate.
- Keep branches scoped to one purpose; avoid mixing unrelated changes.
- Keep commits scoped to one logical change whenever possible.
- Avoid mixing unrelated code, tests, docs, or config updates in a single commit unless they are required for one atomic change.
- Use semantic commit messages (Conventional Commits), for example:
  - `feat: ...`
  - `fix: ...`
  - `refactor: ...`
  - `chore: ...`
  - `ci: ...`
- CI trigger policy:
  - Do not run CI on `feature/*` push events.
  - Run CI on pull requests to `dev`/`main` only.
- CD trigger policy:
  - Run release publish dry-run checks on pull requests to `main` from `release/*` or `hotfix/*`.
  - Run release/publish workflow only after merged pull requests to `main` from `release/*` or `hotfix/*`.
  - Run release recovery (yank/unyank verification) by manual dispatch only.

## 1.1) Semantic Versioning (Mandatory)

- Follow Semantic Versioning (`MAJOR.MINOR.PATCH`) for all release versions.
- Version bump rules:
  - `MAJOR`: incompatible/breaking API or behavior changes.
  - `MINOR`: backward-compatible feature additions.
  - `PATCH`: backward-compatible bug fixes or small internal corrections.
- Do not change version numbers arbitrarily; bump only when release scope warrants it.
- If release impact is unclear, ask the user which SemVer level should be applied.

## 1.2) Enforcement vs Discretion

- Policies enforced by branch protection and required status checks are mandatory controls.
- Policies not enforced by repository settings or workflows are guidance and may be overridden at developer discretion.
- Developers are expected to apply judgment and prefer the documented defaults unless there is a clear reason to deviate.

## 1.3) AI Usage Scope (Mandatory)

- This repository is intentionally maintained as a fully vibe-coded, AI-generated codebase under human direction.
- AI tooling may be used for implementation, refactoring, test authoring, documentation drafting/editing, GitHub Actions/workflow authoring and maintenance, and development planning/decision support.
- Human developers retain final responsibility for correctness, validation, release decisions, and policy interpretation.
- Permission to use AI broadly in this repository does not reduce validation requirements or imply acceptance of AI-generated output.

## 1.4) AI-Only Repository Controls (Mandatory)

- Treat all existing code, documentation, and workflow configuration as potentially AI-generated and therefore potentially polished but incorrect, inconsistent, weakly validated, or partially hallucinated.
- Do not treat existing repository patterns as authoritative merely because they already exist.

### AI-Code Skepticism

- Verify that an existing pattern is internally consistent and correctly applied across the repo before extending it.
- Do not use prior AI-generated code as the sole justification for architecture, API, or implementation decisions.
- If code, docs, config, and workflows disagree, surface the conflict explicitly instead of silently choosing one interpretation.

### Validation Requirements

- For low-risk changes, perform at least a brief sanity check appropriate to the change.
- For moderate-risk changes, complete at least one concrete validation step before presenting the result as ready for acceptance.
- For high-risk changes, complete the most relevant available validation steps and state any remaining validation gap explicitly.
- Prefer executable validation when available, including tests, linting, type checking, builds, and workflow verification.
- If executable validation is available but not run, do not present the result as fully verified.
- Do not recommend acceptance of nontrivial code based only on reasoning or superficial plausibility when direct validation is feasible.

### Structured LLM Output Policy

- In this repository, any LLM output that is intended for programmatic use must be routed through `PydanticAI` with explicit typed models wherever practical.
- Treat freeform or weakly structured model output as a temporary debugging aid, not as an acceptable steady-state interface for application logic.
- When an LLM output feeds parsing, graph construction, evaluation, workflow control, API contracts, configuration generation, or persistence, define a typed `Pydantic` model first and make that model the acceptance boundary.
- If a provider or framework requires an intermediate looser shape, normalize it immediately into the typed model before downstream validation or storage.
- Prefer adding alias-handling, normalization, and retries at the `PydanticAI` boundary over adding ad hoc string parsing later in the pipeline.
- DSPy may optimize prompts or signatures around a task, but it should not replace the repository's typed `PydanticAI` acceptance boundary for LLM output that the code consumes.

### Review Standard For AI-Generated Code

- When reviewing or modifying the repo, check specifically for hallucinated abstractions, dead code paths, configuration drift, API/documentation mismatch, inconsistent data models, shallow error handling, unused complexity, and assumptions copied across files without verification.
- Prefer simplification over speculative extensibility unless the user explicitly asks for broader design.

### Change Traceability

- For substantive changes, state what assumptions the change relies on, what was validated, and what remains unvalidated.
- Keep implementation status, validation status, and acceptance status separate.
- Do not imply that passing checks proves correctness beyond the scope of those checks.

### Documentation Drift Control

- Treat documentation drift as a real maintenance risk in this repository because AI-generated code and AI-generated docs can diverge while both still appear polished and plausible.
- For any substantive change to code, configuration, workflows, API behavior, setup steps, or product direction, assess whether existing documentation may now be inaccurate, incomplete, or misleading.
- Check the most relevant documentation before finishing the task. Typical files include `README.md`, `API.md`, `AGENTS.md`, `.env.example`, workflow documentation, and related files under `docs/`.
- In the final summary, state whether documentation impact was reviewed and whether any drift was found.
- If documentation is out of date and the user has authorized documentation changes, update the affected docs in the same change when practical.
- If documentation appears out of date but the user has not authorized doc edits, do not edit it silently; explicitly report the likely drift and identify the affected files.
- During review tasks, treat documentation drift as a reportable finding when behavior, setup, contracts, status claims, or roadmap claims no longer match the repository.
- Keep the drift check active over time, but monitor for noise: do not generate speculative, low-confidence, or low-value documentation warnings on trivial changes.
- Prefer targeted recommendations tied to concrete impact. Name the specific doc files and the specific behavior or claim that may now be out of alignment.
- If the change is too small to justify documentation updates, say so briefly instead of manufacturing doc churn.

## 2) Explicit-Instruction-Only Mode (Mandatory)

- Do not edit, create, rename, or delete any file unless the user explicitly asks for that action.
- Do not run any shell/system command unless the user explicitly asks for that command or explicitly asks you to perform an action that clearly requires commands.
- Do not infer permission from context, prior turns, or "best next step".
- Treat proposal-style language (for example: "how about", "what if", "should we", "would it make sense") as discussion by default, not execution permission.
- Treat question-form phrasing (for example: "can you", "could you", "is it possible to") as discussion by default, not execution permission.
- Treat declarative requirement statements (for example: "it should...", "the action should...", "this needs to...") as non-executable unless accompanied by a separate explicit execution cue.
- Require a separate explicit execution cue (for example: "implement this", "go ahead and make this change") before making changes after proposal/question discussion.
- For question-form prompts, do not execute edits or commands even if a task is described; require a follow-up explicit execution cue in a separate interaction.
- Before executing any change after proposal/question discussion, send a preflight confirmation message: "Execution confirmation required. No changes made yet."
- After an explicit execution cue is received, execute without requesting another confirmation unless requirements changed materially or became ambiguous.
- If a user message mixes question framing with an implied task, treat it as non-executable until explicit confirmation is received.
- If intent is ambiguous, ask a short confirmation question before making changes.
- If a request is ambiguous, ask a clarifying question before taking any action.
- Default behavior is read-only discussion and planning until explicit user direction is given.

## 3) Safety and Transparency

- Before any change, state exactly what you will do.
- For bug/failure remediation (for example CI/workflow errors), first explain the proposed fix and ask for explicit confirmation before making file edits.
- When changing GitHub Actions/workflow behavior, verify `AGENTS.md` policy text matches the realized workflow triggers and rules; if not aligned, update `AGENTS.md` in the same change.
- Use `uv` as the Python package manager for this repository and prefer `uv run ...` for Python command execution and tests.
- `uv.lock` is intentionally developer-local and not tracked in git for this repository. Do not commit it.
- Guardrails may be bypassed only after explicit user verification. This verification must be a separate interaction beyond the original action request, where the user explicitly confirms the bypass.
- Any bypass confirmation request must include a brief overview of the specific guardrail(s) being bypassed.
- After any change, summarize exactly what changed and where.
- If requested action conflicts with these rules, ask for confirmation and explain the conflict.
- Successful execution, existing precedent, or passing checks does not by itself establish design quality or release readiness in this AI-generated repository.
