# PR review checklist

The checks for step 4 of [SKILL.md](SKILL.md), grouped by finding category in priority order. Most findings should come from correctness, design, and reuse. Docs style has a section of its own, and PR formatting and commit conventions are checked but must not pad the review. The design rules behind these checks are in [patterns.md](patterns.md).

Read `main` with `git show "$BASE:<path>"` (or `https://github.com/RLinf/RLinf/blob/main/<path>`) and search it with `git grep <pattern> "$BASE"`, never the working tree. Cite each finding as `path:line` on the PR head, plus `main:path:line` when it came from a main-branch cross-check.

## correctness (question 4)

- [ ] Logic: off-by-one, inverted conditions, wrong operator or default, swapped arguments, mutated shared state, missing `await` or sync.
- [ ] Edge cases: empty, `None`, or NaN input; single-element batches; zero-size tensors; world size 1; first and last iteration; eval-only paths; resume from checkpoint; single-node versus multi-node paths.
- [ ] Concurrency and distributed: collectives that must run on every rank, device placement, non-deterministic ordering across workers, blocking calls in async paths, races on Ray actors and channels.
- [ ] Resources and lifecycle: GPU memory leaks (tensors never freed, a missing `enable_offload`), file handles, Ray actor lifetime, cleanup skipped on exceptions, double init or double free.
- [ ] Numerics: dtype mismatches (fp16, bf16, fp32), in-place ops on tensors that need grad, unsafe casts, accumulator precision, loss-mask correctness.
- [ ] Error handling: silent `except`, exceptions raised on hot paths, missing validation at boundaries, and the opposite, over-validation of state RLinf already guarantees ([patterns.md](patterns.md#no-unnecessary-protection-code)).
- [ ] Refactor regressions: read the `main` and PR versions of every changed function side by side, because partial refactors often change semantics by accident, and compare every affected backend, env, model, and task.

## design (question 5)

A mismatch with how RLinf already does things, a bloated workaround, or dead boilerplate is a design defect, not a style nit.

- [ ] Sibling: the PR follows the newest idiomatic implementation of the same kind on `main`: a model under `rlinf/models/embodiment/`, an env under `rlinf/envs/`, a runner under `rlinf/runners/`, a worker under `rlinf/workers/`, or an advantage, loss, or reward under `rlinf/algorithms/`. Check `git log "$BASE" -- <dir>` so you compare against current conventions rather than legacy boilerplate.
- [ ] Patterns: every rule in [patterns.md § Design](patterns.md#design) holds: no feature-specific branches in shared code, component code contained in its own files, small single-owner abstractions, fixes at the owning layer, no unnecessary protection code, a direct common path, and preserved behavior.
- [ ] Extension points: the PR extends the registry or enum its component kind already has (`register_model`, `register_advantage`, `register_policy_loss`, `register_reward`, `SupportedModel`, `SupportedEnvType` with `get_env_cls()`, `validate_cfg`) instead of adding a parallel lookup or dynamic machinery.
- [ ] Interfaces: a new component implements the interface its siblings implement. Workers subclass `Worker`, implement `initialize`, log with `self.log_info`, `self.log_warning`, and `self.log_error`, and launch through `create_group(...).launch(...)`. Embodied policies subclass `BasePolicy` and implement `default_forward`, `predict_action_batch`, and the algorithm-specific forwards they need.
- [ ] Dead code: grep `main` and the PR for every new function, method, class, dataclass field, and config key, and flag those with no caller or consumer, such as copied legacy helpers (`make_*_example()`, `generate_observations()`), unused dataclass fields, and defensive branches for shapes or types the pipeline never produces.
- [ ] Unnecessary changes: every changed line serves the PR's goal. Flag reformatting or renames in untouched code and install, Docker, or CI changes beyond what the new component needs (the checks under tests).
- [ ] YAML: a new file starts from a current sibling in `examples/` and keeps every value static, and code treats user-settable fields as read-only. A file that inherits a base config through Hydra `defaults` (for example `model/pi0_5@actor.model`) overrides only task-specific fields and does not re-declare unchanged defaults.
- [ ] Hacks: no machine-specific paths, sleep-based synchronization, or monkey-patches; ask for a config- or env-driven version.

## reuse (question 6)

- [ ] Duplication: new logic does not duplicate a helper or abstraction that already exists anywhere in the stack on `main`, such as `rlinf/utils/` (placement, checkpoint, distributed, data iterators, logging), `rlinf/models/embodiment/` (shared transforms, image parsing, action packing), `rlinf/envs/` (`prepare_actions`, wrappers), `rlinf/data/` (dataset writers and loaders), or the `requirements/install.sh` helpers (`clone_or_reuse_repo`, `install_lerobot`, `install_flash_attn`).
- [ ] Citation: each reuse finding cites the existing helper as `main:<path>:<line>` and proposes reusing or extending it.
- [ ] Comparison: when nothing should be reused, the review names the implementation it compared against.

## tests (question 7)

- [ ] Coverage: a user-facing change has a unit or e2e test that fails without the change, so a reviewer can validate that it works.
- [ ] Layout: `tests/unit_tests/` holds one file per component (`test_models.py`, `test_worker.py`, `test_robotics.py`, and so on). New cases go into the component's file; flag a standalone test file named after a single model, env, symptom, fix, or function.
- [ ] Mocks: tests use real objects through the public contract, and mocks cover only vendor SDKs, hardware, and remote services (`tests/robot_mocks/` for robots). Flag mock-heavy tests that only prove the mocks were called and tests that no named change would break.
- [ ] Markers: a test that hosts parts in scheduler workers carries `@pytest.mark.placement`.
- [ ] New model or env: it comes with an install path, a Docker stage where comparable targets have one, an e2e config, and CI coverage; cross-check with the [add-install-docker-ci-e2e](../add-install-docker-ci-e2e/SKILL.md) skill.
- [ ] Install: a change to `requirements/install.sh`, `requirements/embodied/`, or `docker/Dockerfile` passes the [install-check](../install-check/SKILL.md) skill: common utilities reused, system deps kept in `sys_deps.sh`, git deps pinned or forked, no ad-hoc `pyproject.toml` or core-dep hacks. Flag bespoke subshells, inline `pyproject.toml` patching, and per-env `TORCH_VERSION` overrides where a small `case` branch calling shared helpers (`install_<env>_env`) or a pinned fork under `github.com/RLinf/<repo>` expresses it cleanly.
- [ ] Docker: in `docker/Dockerfile` and `.github/workflows/docker-build.yml`, a new target builds an image only when comparable targets on `main` do. Real-world hardware envs (`yam`, `so101`, `piper`, `dosw1`, `gim_arm`, `xsquare_turtle2`) intentionally have none.
- [ ] CI filters: every new file matches a path filter in `.github/workflows/ci-tests.yml`. Flag a single-file entry that an existing directory glob (`rlinf/models/embodiment/**`, `evaluations/**`, `examples/sft/**`) already covers, and an e2e step that duplicates coverage another workflow already runs.
- [ ] E2E matrix: a new CI-relevant YAML, such as an e2e config under `tests/e2e_tests/`, is referenced in the matrix of the e2e workflow that runs it.
- [ ] Large dependencies: a new Docker image, model, or dataset comes with a maintainer ping in the PR.

## docs (questions 3 and 7)

These checks apply whenever code or docs change, in both directions.

- [ ] Docs check: a PR that changes docs passes the [docs-check](../docs-check/SKILL.md) skill, which cross-checks docs against code and against each other.
- [ ] Docs to code: every config key, CLI flag, env var, file path, function or class name, and supported model or env named in changed docs exists on `main` plus the PR (`git show "$BASE:rlinf/..."`). A stale reference is a finding.
- [ ] Code to docs: when the PR adds, removes, or renames a public config key, model, env, runner, script, env var, or supported feature, the docs change in the same PR. Name the exact files under `docs/source-en/` and `docs/source-zh/` that still need it.
- [ ] README highlight: a new feature worth announcing, such as a model, env, algorithm, rollout or training backend, accelerator, robot, or major capability, is highlighted in `README.md` and `README.zh-CN.md` in the same PR. It adds a dated entry at the top of "What's NEW!" (`最新动态` in the Chinese README) in the form `- [YYYY/MM] 🔥 RLinf supports <feature> <with what, where>. Doc: [<page>](<docs link>).`, and an item in the matching support list (the embodied Simulators, Models, and Algorithms lists, the Hardware Support table, or the Agentic AI table) that links to its docs page. The Chinese entries link to the `zh-cn` docs. Bug fixes, refactors with no user-visible change, and internal tooling need no entry.
- [ ] EN and ZH parity, checked explicitly even when only one language changed: paired pages agree on setup commands, paths, env vars, config keys, supported models, envs, and algorithms, capability claims, reported numbers (metrics, table values, dataset sizes, trial counts), and section structure and order. No paragraph is duplicated, missing, or conflicting between them without a reason. If only one side changed, name the matching file that also needs the change.
- [ ] Asset references: every reference to a downloadable model, adapter, or dataset in changed configs and docs passes the three checks below. In `examples/embodiment/config/*.yaml` these are the weight fields `model_path`, `lora_path`, `backbone_model_path`, and `wan_wm_hf_ckpt_path`.
  1. Form: the path is `/path/to/<repo-name>` with a leading slash, and its last component equals the repo in the `# https://huggingface.co/<org>/<repo>` comment above it, because the docs download with `hf download <org>/<repo> --local-dir <repo>`. Flag stale or placeholder basenames (`RLinf-Pi0-SFT`, `model`, `openpi`, and the like) and a missing leading slash.
  2. Exists: `curl -sL -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/<org>/<repo>` returns `200` (`/api/datasets/` for datasets); flag `401` and `404`. When the returned `id` differs from the requested name, the repo was renamed: point to the canonical `id`, since a `200` through a redirect is still a stale reference.
  3. Matches the docs: the repo is the one the env's recipe page prescribes for that env, task suite, and model family, not just any repo that exists. For example, LIBERO spatial, object, or goal with π₀ uses `RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT`; LIBERO-10 (Long) with π₀ uses `RLinf-Pi0-LIBERO-Long-SFT`; any LIBERO suite with π₀.₅ uses `RLinf-Pi05-LIBERO-SFT`; GR00T N1.5 uses the per-suite `RLinf-Gr00t-SFT-{Spatial,Object,Goal,10}`. Watch for base-versus-adapter mixups (`model_path` is the full base model, `lora_path` the LoRA adapter) and casing that drifts from the docs or the canonical name. Intentional user-supplied placeholders, such as DAgger's `student_model` and `expert_model` or "any pi05 checkpoint", are not findings; confirm against the recipe page before flagging.
- [ ] Fixes: each docs finding names the exact file and gives a concrete wording or structure fix.

## docs-style (question 8)

These are the rules of [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md), which the [refine-docs](../refine-docs/SKILL.md) skill applies when writing and this pass checks. Read every page the PR adds or changes in full, in both languages, as a reader who is new to RLinf and has not seen its code, and classify it first, because the rules differ by page type. README text the PR adds follows the voice rules. Cite the section of the guide that a finding breaks.

- [ ] First-time reader: a reader new to RLinf can tell from the page what the feature does, how it works, what to prepare, and how to run it. Mechanisms are explained through what the reader observes in a run. Flag a mechanism explained by listing modules, classes, functions, or file paths, and code or paths the reader never types, runs, opens, or edits; config examples, config keys, run commands, and, on Extending pages, the files a contributor edits belong on the page.
- [ ] Lead: the first prose sentence after the title, or after a leading figure, states what the page explains, enables, routes, or lets the reader look up. The introduction sets out the result, scope, and reading order, and each section opens by connecting to the one before and naming the question it resolves, not with a code block, table, or API name.
- [ ] Flow: the page follows one line of thought. Each sentence follows from the one before, each paragraph builds on the last, and the introduction plus the first paragraph of each section reads as a coherent outline. Flag a sentence, paragraph, or section that could be dropped or moved without changing the explanation.
- [ ] Voice: second person and imperative, outcome first, and none of the tells the guide lists: throat-clearing ("This section provides a comprehensive guide to …"), "it is worth noting", "simply", "seamlessly", "powerful", "leverage", "robust", "in order to", "a wide range of", paragraphs that end by praising the design, and "Notice that …" announcements.
- [ ] Current state: the page describes what exists now. Flag sentences whose only content is that something is not needed, not included, or no longer used, and the history of a change, which belongs in the PR description.
- [ ] Explain before naming: headings and opening sentences make sense before the reader knows the implementation, and an API term is explained in prose before it appears in a heading (reference pages excepted). The primary example's public calls are explained in the order a caller makes them, and the example closes the loop from declaration to use.
- [ ] Commands: every non-trivial command is followed by what it does and where to configure it further.
- [ ] Cadence, on concept and guide pages: sentences finish their thought, paragraphs carry into one another with real transitions, and paragraph lengths vary.
- [ ] Guide pages: plain and technical, with simple task headings, one config example, one worked example, and only the config keys a user writes; no equations or tensor-shape tables.
- [ ] Recipe pages: the page anatomy of the guide, in order: figure and intro, `Overview` with four aligned cards (`.. grid:: 2 4 4 4`) in the card schema of its gallery, `Tasks` and `Observation and Action` as `list-table`s, then `Installation`, `Download the Model`, `Run It`, and `Visualization and Results`. No Env type card and no generic Algorithm section, and metrics and standalone evaluation are linked rather than explained. Compare with the closest sibling in the same gallery, such as `opensora.rst` for embodied example pages, in section names and order, code blocks, tables, and links.
- [ ] Hardware backends: a recipe that runs on a non-CUDA accelerator has one `Run on Different Hardware Backends` section with a `_<recipe>-hardware` anchor, one H3 per platform (`AMD ROCm`, `Huawei Ascend CANN`, `Moore Threads MUSA`), container and install blocks in `_<platform>_<env>.rst` partials, and a matching row in the `vla_wam_index` hardware table.
- [ ] Index pages: a one-line outcome, then a card grid or `list-table` with a `:hidden:` toctree; never bullet lists in the body.
- [ ] Placement: the page sits in exactly one of the eight axes, chosen by the reader's starting question, appears in one gallery index only, has a bare-name nav caption (`LIBERO <embodied/libero>`), and is added to the EN and ZH toctrees together.
- [ ] Headings and admonitions: Title Case, one H1 per page, the standard section names, footguns in a `warning`, and side information in a `note`.
- [ ] Reuse: reference material (full config tables, the metrics list, evaluation workflows) is linked rather than inlined, and a block repeated on three or more pages lives in an underscore-prefixed partial.
- [ ] Images: every figure and media URL returns `200` directly (`raw.githubusercontent.com`, not a redirecting `github.com/<org>/<repo>/raw/...`), static images are preferred over large GIFs, and captions are specific to the page.
- [ ] Native phrasing: each language reads as if an engineer in the field wrote it in that language, with natural word order and the field's established terms. Flag English calques from Chinese such as "video memory" for GPU memory, "realize the function" for implement, and "carry out training" for train, and Chinese translationese from English: 「一个」「它」「你的」 added for English articles and pronouns, 「当……时」「对于……来说」「通过……的方式」 for every when, for, and through, 「允许你……」「使……能够……」「为……提供支持」 for English verb structures, 「被」 passives, English word order, and literal translations of established terms (rollout as 「展开」, policy as 「策略」).
- [ ] Chinese writing: the ZH page reads as Chinese rather than tracking the EN page sentence for sentence. It uses full-width punctuation, a space between Chinese and English or digits, one source line per paragraph, untranslated identifiers, policy rather than 策略 for an RL policy, and no `**bold**` glued between CJK characters, and it avoids empty verbs (进行, 实现, 提供) and both chatty and bureaucratic phrasing.
- [ ] Build: `python3 .agents/skills/docs-check/build_docs.py` builds both languages with no new warnings, and `python3 .agents/skills/docs-check/check_rst_markup.py` reports no broken inline markup. When the build cannot run, the review says so.
- [ ] Fixes: each finding quotes the line, names the guide section it breaks, and gives the rewritten line as a `suggestion` block, or the new section order for a structural finding.

## style (question 9)

Flag only real problems; do not pad the review.

- [ ] Code style: the code follows the Google Python Style Guide and passes `pre-commit run --all-files` (Ruff lint and format).
- [ ] Docstrings and types: public classes and methods have Google-style docstrings, parameters have type hints, and the return type is annotated when static analysis cannot deduce it.
- [ ] Messages: assertions and exceptions have meaningful messages, never empty ones or restatements such as `xxx != yyy`.
- [ ] Logging and comments: logging goes through `self.log_*` or `get_logger()`, never `print`, and comments follow [patterns.md § Code](patterns.md#code).
- [ ] License header: take the current year from `date +%Y`, list the files the PR adds with `git diff --name-status "$BASE...$PR_HEAD"` (status `A`), and flag each one whose `# Copyright <YEAR> The RLinf Authors.` header is missing or whose `<YEAR>` is not the current year. Files vendored from third parties keep their upstream copyright line, and moved or renamed files keep their year.
- [ ] Commits: every commit has a `Signed-off-by:` trailer and a Conventional Commits subject, `<type>(<scope>): <description>`.
- [ ] PR title and description: the title follows Conventional Commits, the Description and Checklist sections are filled, and a change that can affect performance or stability shows its test results (curves for training changes). Check with `python3 .agents/skills/create-pr/lint_pr.py lint --pr <N>`.
