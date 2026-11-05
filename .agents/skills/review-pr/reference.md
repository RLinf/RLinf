# PR review checklist

The checks for step 4 of [SKILL.md](SKILL.md), by finding category, in priority
order. Most findings should come from correctness, design, and reuse; the later
categories are checked but must not pad the review. The design rules behind
these checks are in [patterns.md](patterns.md). Read every file on `main` with
`git show "$BASE:<path>"`, never from the working tree.

## correctness (question 4)

- [ ] Logic: off-by-one, inverted conditions, wrong operator or default, swapped
      arguments, mutated shared state, missing `await` or sync.
- [ ] Edge cases: empty, `None`, or NaN input; single-element batches;
      zero-size tensors; world size 1; first and last iteration; eval-only
      paths; resume from checkpoint; single-node and multi-node paths.
- [ ] Distributed: collectives run on every rank; device placement; ordering
      across workers is deterministic; no blocking call in an async path; no
      race on Ray actors or channels.
- [ ] Lifecycle and resources: GPU memory, offload, file handles, Ray actor
      lifetime, cleanup on exceptions, no double init or double free.
- [ ] Numerics: dtype mismatches (fp16, bf16, fp32), in-place ops on tensors
      that need grad, unsafe casts, accumulator precision, loss masks.
- [ ] Error handling: validation at boundaries with messages that say what was
      expected and what was found; no silent `except`.
- [ ] Refactor parity: read the `main` and PR versions side by side for every
      changed function; compare every affected backend, env, model, and task.

## design (question 5)

- [ ] The closest recent sibling on `main` is identified (`git log "$BASE" --
      <dir>`), and the PR follows its structure and naming, not legacy
      boilerplate.
- [ ] Each pattern in [patterns.md § Design](patterns.md#design) holds: no
      feature-specific branches in shared code, component code contained in its
      own files, small single-owner abstractions, fixes at the owning layer, no
      unnecessary protection code, a direct common path, preserved behavior.
- [ ] Extension points are used and updated: `SupportedModel`,
      `SupportedEnvType`, `get_env_cls()`, `validate_cfg`, the registries.
- [ ] Workers subclass `Worker`, log with `self.log_*`, and launch through
      `create_group(...).launch(...)`; embodied policies subclass `BasePolicy`
      and implement `default_forward` and `predict_action_batch` plus the
      algorithm-specific forwards they need.
- [ ] YAML starts from a current sibling, holds static values, overrides only
      what differs from its Hydra `defaults`, and code never writes a
      user-settable field.
- [ ] No hardcoded machine paths, sleep-based synchronization, or
      monkey-patches.
- [ ] No unnecessary changes: grep `main` and the PR for every new function,
      method, class, dataclass field, and config key, and flag those with no
      caller or consumer, including copied legacy helpers and defensive branches
      for inputs the pipeline never produces.

## reuse (question 6)

- [ ] New logic does not duplicate a helper on `main`: `rlinf/utils/`
      (placement, checkpoint, distributed, data iterators, logging),
      `rlinf/models/embodiment/` (transforms, image parsing, action packing),
      `rlinf/envs/` (action formatting, wrappers), `rlinf/data/`, or the
      `requirements/install.sh` helpers (`clone_or_reuse_repo`,
      `install_flash_attn`, and others).
- [ ] Each reuse finding cites the existing helper as `main:<path>:<line>` and
      says how to reuse or extend it.
- [ ] When there is nothing to reuse, the review names the implementation it
      compared against.

## tests (question 7)

- [ ] User-facing changes have a unit or e2e test that fails without the change.
- [ ] Unit tests are added to the component's existing
      `tests/unit_tests/test_<component>.py`; no file per model, env, fix, or
      function.
- [ ] Tests use real objects through the public contract; mocks cover only
      vendor SDKs, hardware, and remote services (`tests/robot_mocks/` for
      robots). Mock-heavy tests and tests no named change would break are
      findings.
- [ ] Tests that host parts in scheduler workers carry
      `@pytest.mark.placement`.
- [ ] A new model or env has an install path, a Docker stage where comparable
      targets have one, an e2e config, and CI coverage; use the
      [add-install-docker-ci-e2e](../add-install-docker-ci-e2e/SKILL.md) skill.
      Real-world hardware envs (`yam`, `so101`, `piper`, `dosw1`, `gim_arm`,
      `xsquare_turtle2`) intentionally ship no Docker image.
- [ ] Changes to `requirements/install.sh`, `requirements/embodied/`, or
      `docker/Dockerfile` pass the [install-check](../install-check/SKILL.md)
      skill: shared helpers reused, system deps in `sys_deps.sh`, git deps
      pinned or forked under `github.com/RLinf/`, no bespoke subshells, inline
      `pyproject.toml` patching, or per-env Torch pins.
- [ ] New files match a path filter in `.github/workflows/ci-tests.yml`; no
      single-file entry already covered by a directory glob; no e2e step that
      duplicates another workflow.
- [ ] A new e2e config under `tests/e2e_tests/` appears in the matrix of the
      e2e workflow that runs it.
- [ ] Large new dependencies (Docker images, models, datasets) have a
      maintainer ping in the PR.

## docs (questions 3 and 7)

- [ ] Changed docs pass the [docs-check](../docs-check/SKILL.md) skill.
- [ ] Docs to code: every config key, CLI flag, env var, path, class or function
      name, and supported model or env named in changed docs exists on `main`
      plus the PR.
- [ ] Code to docs: a public config key, model, env, runner, script, env var,
      or feature that the PR adds, renames, or removes is updated in the docs
      of the same PR; name the EN and ZH files that still need it.
- [ ] EN and ZH pages agree on commands, paths, env vars, config keys,
      supported models and envs, claims, reported numbers, and section order,
      even when only one language changed.
- [ ] Pages follow [patterns.md § Docs](patterns.md#docs) and match their
      siblings in section names, order, code blocks, tables, and links.
- [ ] Model-weight paths in changed `examples/embodiment/config/*.yaml`
      (`model_path`, `lora_path`, `backbone_model_path`, `wan_wm_hf_ckpt_path`):
  1. Form: `/path/to/<repo>`, where `<repo>` matches the
     `# https://huggingface.co/<org>/<repo>` comment above it.
  2. Exists: `curl -sL -o /dev/null -w '%{http_code}'
     https://huggingface.co/api/models/<org>/<repo>` returns `200`
     (`/api/datasets/` for datasets), and the returned `id` equals the
     requested name, so the repo was not renamed.
  3. Matches the recipe page: the repo is the one the env's doc page prescribes
     for that env, task suite, and model family (for example, LIBERO-10 with
     π₀ uses `RLinf-Pi0-LIBERO-Long-SFT`), with no base-versus-LoRA mixup
     and no casing drift. Intentional placeholders, such as DAgger's
     `student_model` and `expert_model`, are not findings.
- [ ] Each docs finding gives the exact file and a concrete wording or
      structure fix.

## style (question 8)

Flag only real problems.

- [ ] `pre-commit run --all-files` passes (Ruff lint and format).
- [ ] Public classes and methods have Google-style docstrings and type hints.
- [ ] No `print` in product code; comments follow
      [patterns.md § Code](patterns.md#code).
- [ ] Files the PR adds (`git diff --name-status "$BASE...$HEAD"`, status `A`)
      carry `# Copyright <current year> The RLinf Authors.`
- [ ] Every commit has `Signed-off-by:` and a Conventional Commits subject.
- [ ] The PR title and description pass
      `python3 .agents/skills/create-pr/lint_pr.py lint --pr <N>`.
