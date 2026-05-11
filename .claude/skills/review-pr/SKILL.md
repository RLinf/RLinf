---
name: review-pr
description: Reviews a pull request from a PR URL by directly fetching the URL content (no `gh` dependency) and verifies compliance with CONTRIBUTING.md. Use when the user asks for a PR review, to review changes before merge, or to check contribution guidelines.
---

# Review PR (From PR URL)

Reviews the changes in a specific GitHub pull request. **The primary focus is code correctness and design-pattern consistency with the existing codebase.** PR formatting, commit conventions, and user-facing documentation are checked but should not dominate the review. See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution rules referenced below.

## 1. Input: PR URL

Require a PR URL (for example: `https://github.com/RLinf/RLinf/pull/123`).

## 2. Fetch PR data and the main branch

Fetch PR details directly from the URL:

- Open/fetch the PR page itself for title, description, and metadata.
- Fetch unified diff via URL forms:
  - `<PR_URL>.diff` (preferred)
  - `<PR_URL>.patch` (fallback)
- If needed, fetch related pages directly from URL for comments/checks.

**All cross-references must be against `origin/main`, not the local working tree.** The current checkout may be on an unrelated branch or contain WIP changes; Glob/Grep over the working tree does not show the upstream state. Before any cross-reference lookup:

- Run `git fetch origin main` to refresh.
- Read files via `git show origin/main:<path>` (or browse `https://github.com/RLinf/RLinf/blob/main/<path>`).
- Use `git log origin/main..<pr-head>` to scope what the PR actually adds.
- Never treat Glob/Grep results over the working tree as "current state of the project" — they reflect whatever branch is checked out, not main.

If the PR page is private and URL fetch is blocked, report that access is unavailable and ask the user to provide exported diff/details.

## 3. Review priorities

Findings are ordered by severity. Categories are listed in priority order — most of the review should be on (a) and (b); (c)–(e) are checked but should not pad the output.

### (a) Correctness, bugs, and edge cases — primary

For every changed function, branch, and config path, look for:

- **Logic bugs**: off-by-one, inverted conditions, wrong operator/default, swapped args, mutated shared state, missing await/sync.
- **Edge cases**: empty/None/NaN, single-element batches, zero-size tensors, world-size=1, first/last iter, eval-only paths, resume-from-checkpoint, multi-node vs single-node paths.
- **Concurrency / distributed**: collectives that must run on every rank, device placement, non-deterministic ordering across workers, blocking calls in async paths, races on Ray actors / channels.
- **Resource & lifecycle**: GPU memory leaks (uncleared tensors, missing `enable_offload`), file handles, Ray actor lifetime, missed cleanup on exceptions, double-init.
- **Numerical**: dtype mismatches (fp16/bf16/fp32), in-place ops on grad-required tensors, unsafe casts, accumulator precision, loss-mask correctness.
- **Error handling**: silent `except`, exceptions on hot paths, missing input validation at boundaries; conversely, *over*-validation of internally-trusted state.
- **Refactor regressions**: read both the `origin/main` version and the new version side-by-side; partial refactors often change semantics by accident.

Cite file:line in the diff and the matching `origin/main:file:line` when relevant.

### (b) Design and pattern consistency vs `origin/main` — primary

The PR must match how RLinf already does things. Mismatches are usually defects from writing in isolation, not style nits.

- **Find the closest sibling in `origin/main`** — comparable model under `rlinf/models/embodiment/`, env under `rlinf/envs/`, runner under `rlinf/runners/`, worker under `rlinf/workers/`, advantage/loss/reward under `rlinf/algorithms/`. Compare structure, naming, registration, base-class usage, and config wiring.
- **Registry wiring**: new advantage/loss/reward must use `register_advantage` / `register_policy_loss` / `register_reward`; new model/env must extend `SupportedModel` / `SupportedEnvType` and update `get_env_cls()` and `validate_cfg`. Flag ad-hoc bypasses.
- **Worker conventions**: subclass `Worker`, implement `initialize`, use `self.log_info` / `log_warning` / `log_error` (not `print` or stdlib `logging`), launch via `create_group(...).launch(...)`.
- **Base class / interface**: embodied policies must inherit `BasePolicy` and implement the documented forwards (`default_forward`, `predict_action_batch`, plus algorithm-specific). Flag re-implementations of base behavior.
- **Config layout**: new YAML must be copied from a sibling in `examples/` and follow the same key hierarchy; no calculations or dynamic values in YAML; fields read-only in code.
- **Reuse vs duplication**: if the change reimplements a helper that already exists in `rlinf/utils/` (placement, checkpoint, distributed, data-iter, logging), point to the existing helper with file:line.
- **Simplicity**: prefer the approach the codebase already uses; flag clumsy / over-engineered alternatives with a concrete simpler suggestion.
- **No hardcoded paths/hacks**: machine-specific paths, sleep-based sync, monkey-patches → propose a config/env-driven version.

### (c) Code ↔ docs consistency — required when code OR docs change

Both directions matter, and EN/ZH parity must be checked explicitly:

- **Docs → code**: every config key, CLI flag, env var, file path, function/class name, and supported model/env name mentioned in changed docs must exist in `origin/main` + this PR. Verify with `git show origin/main:rlinf/...`. Stale references = finding.
- **Code → docs**: when this PR adds, removes, or renames a public-facing config key, model, env, runner, script, env var, or supported feature, the corresponding doc page **must be updated in the same PR**. If missing, list the exact doc files (EN and ZH) that need edits.
- **EN ↔ ZH parity** (do this explicitly, even when only one language was touched): paired pages under `docs/source-en/` and `docs/source-zh/` must agree on setup commands, paths, env vars, config keys, supported models/envs/algorithms, capability claims, reported numbers (metrics, table values, dataset sizes, trial counts), and section structure/order. If only one side is updated, name the matching file that also needs the change.
- **Sibling style**: cross-check with sibling pages in the same area (e.g. `opensora.rst`) for section naming/order, code-block conventions, table/link style.
- Each docs finding must include a concrete suggested wording / structure fix and exact file references.

### (d) Tests and CI integration

- **User-facing changes** must have **tests** (unit or e2e). Reviewer must be able to validate reproducibility.
- **Dependencies / CI**: new env/model needs install-script update, Docker stage, and CI/e2e coverage — cross-check with the [add-install-docker-ci-e2e skill](../add-install-docker-ci-e2e/SKILL.md).
- New CI-relevant YAML must be referenced in the e2e test matrix.
- Large/new dependencies (docker, models, datasets) → maintainer ping noted.

### (e) Style, commit & PR metadata — secondary

Mention only if there are real issues; do not pad the review.

- Google Python Style; passes `pre-commit run --all-files`.
- Public classes/methods have Google-style docstrings; type hints on parameters; return type when not deducible.
- Assertions/exceptions have meaningful messages (no empty or `xxx != yyy` restatements).
- `logging` / `self.log_*` not `print`.
- Every commit `Signed-off-by`; messages follow Conventional Commits `<type>(<scope>): <description>`.
- PR title in Conventional Commits format; PR Description and Checklist sections filled; testing results if performance/stability is affected.

## 4. Explain the PR first

Before listing issues, start with a brief explanation of the PR:

- What problem it tries to solve.
- Main change categories (e.g., packaging, docs, CI, refactor).
- Potential impact/risk areas.

Keep this concise (3-6 bullets), then move to findings.

## 5. Output format

- Open with the PR URL and the brief explanation from section 4.
- List **findings**, ordered by severity (highest first), as bullets:
  - `Severity` + `Area/File`: issue summary
  - `Suggested fix`: concrete action
  - `Reference`: file:line in the diff, plus `origin/main:file:line` when the finding came from a main-branch cross-check
- The bulk of findings should be from categories (a) and (b). If you find none in those categories, say so explicitly — do not fabricate.
- Group docs (c) findings together; tests/CI (d) together; style/PR-metadata (e) at the end.
- Explicitly label findings discovered via **main-branch cross-check** (not only direct diff lines).
- Do not include findings that simply restate that the PR description is well-formed; only flag *problems*.

For a concise checklist, see [reference.md](reference.md).
