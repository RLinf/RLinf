# PR review checklist

Checklist for step 5 of [SKILL.md](SKILL.md), ordered by finding category. Design rules live in [code/patterns.md](../code/patterns.md); area checks live in `code`, `install`, and `docs`. Cross-reference `main` via `git show "$BASE:<path>"` and `git grep <pattern> "$BASE"`, citing `path:line` on `$PR_HEAD` (plus `main:path:line` for cross-checks).

## correctness

- [ ] **Logic**: off-by-one, inverted conditions, wrong defaults, swapped args, mutated shared state, missing `await`/sync.
- [ ] **Edge cases**: empty/`None`/NaN inputs, batch size 1, zero-size tensors, world size 1, first/last step, eval-only, checkpoint resume, single-node vs. multi-node.
- [ ] **Concurrency & distributed**: rank-asymmetric collectives, device placement, worker ordering races, blocking calls in async paths, Ray actor/channel races.
- [ ] **Resources & lifecycle**: GPU memory leaks (missing offload/free), leaked handles/actors, exception-path cleanup, double init/free.
- [ ] **Numerics**: dtype mismatches (fp16/bf16/fp32), in-place ops breaking autograd, unsafe casts, accumulator precision, loss masks.
- [ ] **Error handling**: silent `except`, hot-path exceptions, missing boundary validation, or over-validating internal state ([patterns.md](../code/patterns.md#no-unnecessary-protection-code)).
- [ ] **Refactor regressions**: compare every changed function against `main` across all affected backends, envs, models, and tasks.
- [ ] **Caller & consumer sweep**: when a signature, return shape, base-class contract, shared helper, or config key changes, grep all callers, subclasses, YAMLs, tests, and docs at `$PR_HEAD` for broken untouched consumers.
- [ ] **Claims vs. diff**: verify PR description claims (no behavior change, test coverage, metrics/baselines) against the actual diff and tests.

## design

A mismatch with RLinf conventions, bloated workaround, or dead boilerplate is a design defect, not a style nit.

- [ ] **Sibling**: matches the newest idiomatic sibling on `main` (`git log "$BASE" -- <dir>`), not legacy boilerplate.
- [ ] **Patterns**: follows [patterns.md § Design](../code/patterns.md#design) (no component branches in shared code, contained component files, single-owner abstractions, fix at owning layer, direct common path, preserved behavior).
- [ ] **Test-driven guards**: no product code weakened for incomplete test mocks/stubs (replacing `obj.attr` with `getattr`/`hasattr` or adding fallback branches); fix the test instead.
- [ ] **Extension points & interfaces**: plugs into existing registries/enums (`register_*`, `SupportedModel`, `SupportedEnvType`, `validate_cfg`) and sibling contracts (`Worker`, `BasePolicy`).
- [ ] **Dead code & scope creep**: every new function, field, and config key has a consumer; no unrelated reformatting, renames, or extra install/Docker/CI edits.
- [ ] **YAML & hacks**: static YAML copied from a current sibling, overriding only non-default keys; user fields stay read-only; no hardcoded paths/GPUs, sleep sync, or monkey-patches.

## reuse

- [ ] **Duplication**: reuses existing `main` helpers (`rlinf/utils/`, `rlinf/models/embodiment/`, `rlinf/envs/`, `rlinf/data/`, `requirements/install.sh` helpers, and shared doc setup pages/partials like `video_models.rst`, `_setup_common.rst`, `_model_path.rst`).
- [ ] **Citation**: cite the existing helper as `main:<path>:<line>` when flagging reuse; name the closest comparison when nothing applies.

## tests, docs, and docs-style

Area scripts and checklists run in step 6 ([code § Check](../code/SKILL.md#check), [install § Check](../install/SKILL.md#check), [docs § Check](../docs/SKILL.md#check)). Review these PR-wide checks alongside them:

- [ ] **Coverage & completeness**: user-facing changes include a test that fails without the change; new models/envs ship install, Docker (where comparable targets have it), e2e config, CI filter, EN/ZH docs, and README highlight (ping maintainers for large new assets/images).
- [ ] **Docs readability (fresh agent)**: run a fresh-agent pass (no `rlinf/` context) following [docs § Check](../docs/SKILL.md#check) for plain human prose, native Chinese (no direct machine translation), concise explanations, no internal code citations, clean commands, and no negative-contrast restatement ("not A, not B, but C"). Ask to fix YAML defaults/scripts or link Reference rather than adding prose workarounds.
- [ ] **Sibling alignment**: compare recipes side by side with closest model/domain-family siblings on `main`; flag duplicated shared setup or non-standard commands even when reproduction passes.

## style

Flag only real problems; lint, formatting, `__init__.py`, and license headers come from `code/check.py`, and code conventions from [patterns.md § Code](../code/patterns.md#code).

- [ ] **Commits & PR metadata**: `Signed-off-by:` and Conventional Commits on commits; valid PR title, filled Description/Checklist, and performance/training curves when applicable (`lint_pr.py lint --pr <N>`).
