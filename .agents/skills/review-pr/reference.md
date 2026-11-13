# PR review checklist

The checks for step 5 of [SKILL.md](SKILL.md), by finding category in priority order. Most findings should come from correctness, design, and reuse. Tests, docs, and docs style come from the area checks of step 6, and PR formatting and commit conventions are checked but must not pad the review. The design rules behind these checks are in [code/patterns.md](../code/patterns.md).

Read `main` with `git show "$BASE:<path>"` (or `https://github.com/RLinf/RLinf/blob/main/<path>`) and search it with `git grep <pattern> "$BASE"`, never the working tree. Cite each finding as `path:line` on the PR head, plus `main:path:line` when it came from a main-branch cross-check.

## correctness (question 4)

- [ ] Logic: off-by-one, inverted conditions, wrong operator or default, swapped arguments, mutated shared state, missing `await` or sync.
- [ ] Edge cases: empty, `None`, or NaN input; single-element batches; zero-size tensors; world size 1; first and last iteration; eval-only paths; resume from checkpoint; single-node versus multi-node paths.
- [ ] Concurrency and distributed: collectives that must run on every rank, device placement, non-deterministic ordering across workers, blocking calls in async paths, races on Ray actors and channels.
- [ ] Resources and lifecycle: GPU memory leaks (tensors never freed, a missing `enable_offload`), file handles, Ray actor lifetime, cleanup skipped on exceptions, double init or double free.
- [ ] Numerics: dtype mismatches (fp16, bf16, fp32), in-place ops on tensors that need grad, unsafe casts, accumulator precision, loss-mask correctness.
- [ ] Error handling: silent `except`, exceptions raised on hot paths, missing validation at boundaries, and the opposite, over-validation of state RLinf already guarantees ([patterns.md](../code/patterns.md#no-unnecessary-protection-code)).
- [ ] Refactor regressions: read the `main` and PR versions of every changed function side by side, because partial refactors often change semantics by accident, and compare every affected backend, env, model, and task.

## design (question 5)

A mismatch with how RLinf already does things, a bloated workaround, or dead boilerplate is a design defect, not a style nit.

- [ ] Sibling: the PR follows the newest idiomatic implementation of the same kind on `main`: a model under `rlinf/models/embodiment/`, an env under `rlinf/envs/`, a runner under `rlinf/runners/`, a worker under `rlinf/workers/`, or an advantage, loss, or reward under `rlinf/algorithms/`. Check `git log "$BASE" -- <dir>` so you compare against current conventions rather than legacy boilerplate.
- [ ] Patterns: every rule in [patterns.md § Design](../code/patterns.md#design) holds: no feature-specific branches in shared code, component code contained in its own files, small single-owner abstractions, fixes at the owning layer, no unnecessary protection code, a direct common path, and preserved behavior.
- [ ] Extension points: the PR extends the registry or enum its component kind already has (`register_model`, `register_advantage`, `register_policy_loss`, `register_reward`, `SupportedModel`, `SupportedEnvType` with `get_env_cls()`, `validate_cfg`) instead of adding a parallel lookup or dynamic machinery.
- [ ] Interfaces: a new component implements the interface its siblings implement. Workers subclass `Worker`, implement `initialize`, log with `self.log_info`, `self.log_warning`, and `self.log_error`, and launch through `create_group(...).launch(...)`. Embodied policies subclass `BasePolicy` and implement `default_forward`, `predict_action_batch`, and the algorithm-specific forwards they need.
- [ ] Dead code: grep `main` and the PR for every new function, method, class, dataclass field, and config key, and flag those with no caller or consumer, such as copied legacy helpers (`make_*_example()`, `generate_observations()`), unused dataclass fields, and defensive branches for shapes or types the pipeline never produces.
- [ ] Unnecessary changes: every changed line serves the PR's goal. Flag reformatting or renames in untouched code and install, Docker, or CI changes beyond what the new component needs.
- [ ] YAML: a new file starts from a current sibling in `examples/` and keeps every value static, and code treats user-settable fields as read-only. A file that inherits a base config through Hydra `defaults` (for example `model/pi0_5@actor.model`) overrides only task-specific fields and does not re-declare unchanged defaults.
- [ ] Hacks: no machine-specific paths, sleep-based synchronization, or monkey-patches; ask for a config- or env-driven version.

## reuse (question 6)

- [ ] Duplication: new logic does not duplicate a helper or abstraction that already exists anywhere in the stack on `main`, such as `rlinf/utils/` (placement, checkpoint, distributed, data iterators, logging), `rlinf/models/embodiment/` (shared transforms, image parsing, action packing), `rlinf/envs/` (`prepare_actions`, wrappers), `rlinf/data/` (dataset writers and loaders), or the `requirements/install.sh` helpers (`clone_or_reuse_repo`, `install_lerobot`, `install_flash_attn`).
- [ ] Citation: each reuse finding cites the existing helper as `main:<path>:<line>` and proposes reusing or extending it.
- [ ] Comparison: when nothing should be reused, the review names the implementation it compared against.

## tests, docs, and docs-style (questions 3, 7, and 8)

These come from the area checks of step 6: [code § Check](../code/SKILL.md#check) for unit tests, mocks, markers, and CI path filters; [install § Check](../install/SKILL.md#check) for install, Docker, and e2e wiring; and [docs § Check](../docs/SKILL.md#check) for docs against code, EN/ZH parity, README highlights, asset references, and the reading of every changed page against the style guide. Two checks stay with the review, because they need the whole PR in view:

- [ ] Coverage: a user-facing change has a unit or e2e test that fails without the change, so a reviewer can validate that it works.
- [ ] Completeness: a new model or env comes with an install path, a Docker stage where comparable targets have one, an e2e config, CI coverage, a recipe page in both languages, and a README highlight; a large new dependency (a Docker image, model, or dataset) comes with a maintainer ping.

## style (question 9)

Flag only real problems; do not pad the review. Lint, formatting, package layout, and license headers come from the code check, and code conventions from [patterns.md § Code](../code/patterns.md#code).

- [ ] Commits: every commit has a `Signed-off-by:` trailer and a Conventional Commits subject, `<type>(<scope>): <description>`.
- [ ] PR title and description: the title follows Conventional Commits, the Description and Checklist sections are filled, and a change that can affect performance or stability shows its test results (curves for training changes). `python3 .agents/skills/create-pr/lint_pr.py lint --pr <N>` checks the form.
