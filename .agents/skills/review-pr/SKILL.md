---
name: review-pr
description: Reviews a pull request from a PR URL by directly fetching the URL content (no `gh` dependency) and verifies compliance with CONTRIBUTING.md. Use when the user asks for a PR review, to review changes before merge, or to check contribution guidelines.
---

# Review PR (From PR URL)

Reviews the changes in a specific GitHub pull request. **The primary focus is code correctness, clean abstractions and code reuse, and eliminating unnecessary changes that deviate from the existing codebase.** PR formatting, commit conventions, and user-facing documentation are checked but should not dominate the review. See [AGENTS.md](../../../AGENTS.md) and [CONTRIBUTING.md](../../../CONTRIBUTING.md) for the design and contribution rules referenced below.

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

### (b) Design, code reuse, and unnecessary changes vs `origin/main` — primary

The PR must match how RLinf already does things and follow the [Design and review principles in `AGENTS.md`](../../../AGENTS.md#design-and-review-principles). Mismatches, bloated workarounds, and dead boilerplate are design defects, not style nits.

- **Compare against the *newest* idiomatic sibling in `origin/main`** — comparable model under `rlinf/models/embodiment/`, env under `rlinf/envs/`, runner under `rlinf/runners/`, worker under `rlinf/workers/`, advantage/loss/reward under `rlinf/algorithms/`. Check `git log` on siblings so you compare against current conventions rather than legacy boilerplate.
- **Clean, explicit abstractions**:
  - Prefer small abstractions with one responsibility and one stable name (`AGENTS.md`). Flag parallel vocabularies, pass-through wrapper classes/functions that hide ownership, and dynamic machinery where a direct constructor, method, or existing registry (`register_model`, `register_advantage`, `register_policy_loss`, `register_reward`, `get_env_cls()`) suffices.
  - Design invalid states out of the API: explicit ownership, lifecycle order, and idempotent cleanup; keep the common local path direct and use the same API in local and remote configurations.
  - Embodied policies must inherit `BasePolicy` and implement the documented forwards (`default_forward`, `predict_action_batch`, plus algorithm-specific); workers must subclass `Worker`, implement `initialize`, use `self.log_info` / `log_warning` / `log_error`, and launch via `create_group(...).launch(...)`.
- **Code reuse over duplication**:
  - Check whether new logic duplicates helpers or abstractions that already exist anywhere in the stack — `rlinf/utils/` (placement, checkpoint, distributed, data-iter, logging), `rlinf/models/embodiment/` (shared transforms, image parsing, action packing), `rlinf/envs/` (`prepare_actions`, wrappers), `rlinf/data/` (dataset writers/loaders), or `requirements/install.sh` helpers (`clone_or_reuse_repo`, `install_lerobot`, `install_flash_attn`). Always cite the existing helper with `origin/main:file:line` and propose reusing or extending it.
- **Unnecessary changes, dead code, and bloat**:
  - **Dead / unreferenced code**: grep `origin/main` + the PR diff for every newly added function, method, class, dataclass field, and config key. Flag anything with zero callers or consumers (e.g. copy-pasted legacy helpers like `make_*_example()`, `generate_observations()`, unused dataclass fields, or defensive branches for shapes/types that the caller pipeline never produces).
  - **Redundant YAML config keys**: new YAML must start from a current sibling in `examples/`, keep all values static, and treat user-settable fields as read-only in code. When a YAML file inherits a base config via Hydra `defaults` (e.g. `model/pi0_5@actor.model`), flag blocks that re-declare unchanged default keys instead of overriding only task-specific fields.
  - **Over-engineered install / Docker / CI changes**:
    - In `requirements/install.sh`, flag bespoke subshells, inline `pyproject.toml` patching, or per-env `TORCH_VERSION` overrides when a small `case` branch calling shared helpers (`install_<env>_env`) or a pinned fork under `github.com/RLinf/<repo>` can express it cleanly (see [install-check](../install-check/SKILL.md)).
    - In `docker/Dockerfile` and `.github/workflows/docker-build.yml`, check whether comparable targets on `origin/main` actually build Docker images (e.g. real-world hardware envs like `yam`, `so101`, `piper`, `dosw1`, `gim_arm`, `xsquare_turtle2` intentionally do not).
    - In `.github/workflows/ci-tests.yml` and e2e workflows, flag single-file path filter entries that are already covered by existing directory globs (`rlinf/models/embodiment/**`, `evaluations/**`, `examples/sft/**`, etc.) and flag e2e steps that duplicate coverage already running in another workflow.
  - **Test layout bloat**: `tests/unit_tests/` has one file per component (`test_models.py`, `test_worker.py`, `test_robotics.py`, etc.). Flag standalone test files named after a single model, env, symptom, or function, and flag mock-heavy tests that only prove mocks were called (`AGENTS.md` § *Tests*).
  - **No hardcoded paths/hacks**: machine-specific paths, sleep-based sync, monkey-patches → propose a config/env-driven version.

### (c) Code ↔ docs consistency — required when code OR docs change

Both directions matter, and EN/ZH parity must be checked explicitly:

- **If this PR changes documentation**, follow the [docs-check skill](../docs-check/SKILL.md) to drive this review: it cross-checks docs against code and against each other (commands, config keys, paths, model/env names) and enforces EN↔ZH parity.
- **Docs → code**: every config key, CLI flag, env var, file path, function/class name, and supported model/env name mentioned in changed docs must exist in `origin/main` + this PR. Verify with `git show origin/main:rlinf/...`. Stale references = finding.
- **Code → docs**: when this PR adds, removes, or renames a public-facing config key, model, env, runner, script, env var, or supported feature, the corresponding doc page **must be updated in the same PR**. If missing, list the exact doc files (EN and ZH) that need edits.
- **Example-config model paths ↔ docs**: for changed example configs under `examples/embodiment/config/*.yaml`, every model-weight path field (`model_path`, `lora_path`, `backbone_model_path`, `wan_wm_hf_ckpt_path`) must satisfy all three:
  1. **Form**: it is `/path/to/<repo-name>` where the trailing path component equals the repo in its `# https://huggingface.co/<org>/<repo>` comment (the docs convention is `hf download <org>/<repo> --local-dir <repo>`, so basename == repo name). Flag stale/placeholder basenames (`RLinf-Pi0-SFT`, `model`, `openpi`, …) and a missing leading slash.
  2. **Exists**: each distinct referenced repo resolves on Hugging Face — `curl -sL -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/<org>/<repo>` returns `200` (use `/api/datasets/` for datasets). Flag 401/404. Also flag *redirecting* names: if the API's returned `id` differs from the requested name, the repo was renamed — point at the canonical `id` (a `200` via redirect is still a stale reference).
  3. **Matches the docs' per-variant model**: the repo equals the one the corresponding env recipe page prescribes for that exact environment + task suite + model family — not just any existing repo. Cross-check the recipe doc's model table/download block (e.g. LIBERO spatial/object/goal + π₀ → `RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT`, LIBERO-10/Long + π₀ → `RLinf-Pi0-LIBERO-Long-SFT`, any LIBERO suite + π₀.₅ → `RLinf-Pi05-LIBERO-SFT`, GR00T N1.5 per-suite `RLinf-Gr00t-SFT-{Spatial,Object,Goal,10}`). Watch for base-vs-adapter mixups (`model_path` = full base model, `lora_path` = LoRA adapter) and casing drift vs the doc/canonical name. Intentional user-supplied placeholders (e.g. DAgger `student_model`/`expert_model`, "any pi05 checkpoint") are not findings — confirm against the recipe page before flagging.
- **EN ↔ ZH parity** (do this explicitly, even when only one language was touched): paired pages under `docs/source-en/` and `docs/source-zh/` must agree on setup commands, paths, env vars, config keys, supported models/envs/algorithms, capability claims, reported numbers (metrics, table values, dataset sizes, trial counts), and section structure/order. If only one side is updated, name the matching file that also needs the change.
- **Sibling style**: cross-check with sibling pages in the same area (e.g. `opensora.rst`) for section naming/order, code-block conventions, table/link style.
- Each docs finding must include a concrete suggested wording / structure fix and exact file references.

### (d) Tests and CI integration

- **User-facing changes** must have **tests** (unit or e2e). Reviewer must be able to validate reproducibility.
- **If this PR changes the install script** (`requirements/install.sh`, `requirements/embodied/`, or `docker/Dockerfile`), follow the [install-check skill](../install-check/SKILL.md) to review the changes: reuse of common utilities, system deps kept in `sys_deps.sh`, pinned/forked git deps, no ad-hoc `pyproject.toml`/core-dep hacks, and a matching Dockerfile build stage for every new model/env.
- **Dependencies / CI**: new env/model needs install-script update, Docker stage, and CI/e2e coverage — cross-check with the [add-install-docker-ci-e2e skill](../add-install-docker-ci-e2e/SKILL.md).
- New CI-relevant YAML must be referenced in the e2e test matrix.
- Large/new dependencies (docker, models, datasets) → maintainer ping noted.

### (e) Style, commit & PR metadata — secondary

Mention only if there are real issues; do not pad the review.

- Google Python Style; passes `pre-commit run --all-files`.
- Public classes/methods have Google-style docstrings; type hints on parameters; return type when not deducible.
- Assertions/exceptions have meaningful messages (no empty or `xxx != yyy` restatements).
- `logging` / `self.log_*` not `print`.
- **License header (newly-added files)**: every file *added* by this PR must carry the standard RLinf header (`# Copyright <YEAR> The RLinf Authors.`), and `<YEAR>` must equal the current calendar year. Determine the current year (e.g. `date +%Y`), list added files with `git diff --name-status origin/main...<pr-head>` (status `A`), and flag any new file whose header is missing or whose copyright year is not the current year. Files vendored from third parties keep their upstream copyright line — apply this only to the RLinf Authors header.
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
