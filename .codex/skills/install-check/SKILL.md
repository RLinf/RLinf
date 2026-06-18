---
name: install-check
description: Check, fix, or extend requirements/install.sh and its docker/Dockerfile coverage when adding a new embodied model or environment in RLinf, so the install logic reuses common utilities, keeps system deps in sys_deps.sh, pins/forks git deps, avoids ad-hoc pyproject/core-dep hacks, and every new model/env gets a matching Dockerfile build stage. Use when writing or cleaning up an install_<model>_model / install_<env>_env function, fixing a flaky or non-portable install path, or reviewing an install.sh / Dockerfile diff for convention compliance.
---

# Install check: install.sh + Dockerfile conventions for a new model/env

`requirements/install.sh` installs every embodied model + env combo, and
`docker/Dockerfile` builds an image per target. New contributions tend to
copy-paste an existing block and hack it until it works on one machine —
re-cloning repos by hand, `apt-get`-ing a package, pinning `ray`, `sed`-ing
`pyproject.toml`, and forgetting the Dockerfile entirely. This skill is the
convention checklist plus a linter that flags those anti-patterns.

> Sibling skill: [add-install-docker-ci-e2e](../add-install-docker-ci-e2e/SKILL.md)
> *wires up* a new model/env (register, Dockerfile stage, CI jobs). This skill
> checks the *quality* of the install function and that the Dockerfile keeps up.

All paths below are relative to the repo root.

## Run the check (start here)

The harness is [check.sh](check.sh). Run it before and after editing
`requirements/install.sh` or `docker/Dockerfile`:

```bash
bash .codex/skills/install-check/check.sh
```

It scans the install script for each convention, then verifies every model in
`SUPPORTED_MODELS` and env in `SUPPORTED_ENVS` has a matching `--model` /
`--env` install invocation in `docker/Dockerfile`. It prints line numbers + fix
hints and exits non-zero if anything is flagged. Paths can be passed explicitly:

```bash
bash .codex/skills/install-check/check.sh requirements/install.sh docker/Dockerfile
```

Heuristics **over-report** — every hit is a line to *review*, not an automatic
bug. Some are legitimate exceptions (see Gotchas). The goal when fixing is: each
flagged item is either resolved or consciously kept with a one-line reason.

## The conventions

1. **Reuse common utilities — don't rebuild them.** Clone with
   `clone_or_reuse_repo`, install flash-attn with `install_flash_attn`, apex
   with `install_apex`, base deps with `install_common_embodied_deps`. If a
   utility *almost* fits, **enhance the utility**, don't fork a near-duplicate.
2. **A model has one `install_<model>_model()`** that `case`s on `$ENV_NAME` and
   calls the matching `install_<env>_env()`. An env with **no** model is
   installed through `install_env_only` (add a branch there).
3. **System packages go in `requirements/embodied/sys_deps.sh`**, never
   `apt-get`/`dnf`/`yum`/`pacman` inside `install.sh`. `sys_deps.sh` covers
   apt/dnf/yum/pacman — add the package to **every** `install_deps_<mgr>`
   function so all distros are handled.
4. **Pin every git dependency** to a commit/tag/branch (`@<rev>` or `-b
   <branch>`), **or** fork it into `github.com/RLinf/<repo>`. No floating
   `main`.
5. **Don't touch RLinf's own `pyproject.toml` from `install.sh`** except through
   `apply_torch_override`. If an override pin conflicts with a dependency you're
   installing, `cd`/`pushd` into that dependency's checkout and install from
   there (editing the *cloned* repo's pyproject after `cd` is fine).
6. **Don't override core deps (`ray`, and avoid loose `torch` pins).** Before
   adding a `uv pip install x==y`, check whether the common embodied/agentic
   deps or `pyproject.toml` already provide it. Fewer overrides = fewer
   cross-env breakages.
7. **Every new model/env install needs a matching `docker/Dockerfile` change.**
   Adding a model to `SUPPORTED_MODELS` or env to `SUPPORTED_ENVS` (and its
   `install_*` function) is not complete until `docker/Dockerfile` has a build
   stage that installs it. Add `base-image-embodied-<target>`, a `FROM
   embodied-common-image AS embodied-<target>-image` stage, the `RUN bash
   requirements/install.sh … --model <m> --env <e>` line, and the default-venv
   `.bashrc` line. (See [add-install-docker-ci-e2e](../add-install-docker-ci-e2e/SKILL.md)
   for the full stage + CI pattern.)
8. **Keep it clean and simple.** Prefer a small `case` branch that calls shared
   helpers over a long bespoke block of `sed`/`uv pip` hacks.
9. **When uncertain, ask.** If you can't tell whether an override is required,
   whether to fork-vs-pin, or whether a target genuinely needs a Docker image —
   surface the question instead of guessing.

## Common utilities to reuse (in `install.sh`)

| Need | Use | Don't |
|---|---|---|
| Clone a repo | `clone_or_reuse_repo ENV_VAR DEFAULT_DIR URL [git args…]` (reuse via `*_PATH`, mirror prefix, corruption re-clone) | raw `git clone` |
| flash-attn | `install_flash_attn` (platform-aware, prebuilt wheels, source fallback) | hand-built wheel URLs |
| apex | `install_apex` | building apex inline |
| venv + base sync | `create_and_sync_venv` then `install_common_embodied_deps` | re-implementing `uv venv` / `uv sync` |
| torch version change | `--torch` flag → `apply_torch_override` | `sed` on `pyproject.toml` |
| system libs / EGL / Vulkan | `requirements/embodied/sys_deps.sh` (+ per-platform render config) | `apt-get`, writing ICD json inline |
| CUDA / ROCm detection | `detect_cuda_major_minor`, `detect_rocm_version` | re-parsing `nvcc`/`rocminfo` |
| mirror-aware git | `GITHUB_PREFIX` prefix + `setup_mirror` | hardcoding mirror URLs |

## Workflow

**Adding an env to a model:** add a branch to the model's `case "$ENV_NAME"`:
`create_and_sync_venv` → `install_common_embodied_deps` → `install_<env>_env`
(+ `install_flash_attn` if the model needs it). Register the env name in
`SUPPORTED_ENVS`. **Then add the Dockerfile stage (Req 7).**

**Adding a model:** add `install_<model>_model()` (case on `$ENV_NAME`), a
`case "$MODEL"` branch in `main()`, and the name in `SUPPORTED_MODELS`. Put
model-specific pip pins in `requirements/embodied/models/<model>.txt` rather
than inline when there are more than a couple. **Then add the Dockerfile stage.**

**Env with no model:** add a branch in `install_env_only`.

After editing, run the check and resolve/annotate every new hit, then
syntax-check both files:

```bash
bash -n requirements/install.sh && echo "syntax OK"
```

## Gotchas (real findings from the current tree)

- **`apt-get install git-lfs` inside `install_gr00t_n1d6_model`** (around line
  1248) is exactly the Req-3 violation: git-lfs should be added to
  `sys_deps.sh` across all four package managers, not `apt-get`-ed inline (which
  breaks on non-Debian hosts).
- **`torch==` pins: behavior is fine, genesis is a real candidate.**
  `install_behavior_env` pins `torch==2.5.1` but wraps it in `pushd ~ … popd` —
  installing from `~` so RLinf's pyproject overrides don't apply (sanctioned).
  `install_genesis_env` pins `torch==2.8.0` **inline with no `cd`** — review it.
  Same Req-6 flag, opposite verdicts — read the context.
- **A clone can be "pinned later."** `NVIDIA/Isaac-GR00T` is cloned unpinned but
  the function then does `git checkout 7d5a455…` — effectively pinned. The
  linter still flags the clone line; acceptable. Prefer pinning at the
  `clone_or_reuse_repo` call when possible.
- **The franka catkin workspace** uses raw `git clone` for three ROS repos
  building one workspace — a genuine multi-repo build, not a single dep. Keep
  it.
- **Editing a *cloned* repo's `pyproject.toml`** (gr00t's `peft` pin) is fine
  because it happens after `cd "$gr00t_path"`. The linter can't tell whose
  pyproject it is — confirm it's not RLinf's.
- **Docker coverage gaps are real.** The check currently flags models
  `gr00t_n1d6`, `dreamzero`, `qwen3_vl` and envs `genesis`, `habitat`,
  `xsquare_turtle2`, `franka-dexhand` as having no Dockerfile stage. New
  additions like these should get one. A handful of utility/real-robot targets
  (`dummy`, `d4rl`, `gim_arm`, `dosw1`) intentionally have no image — keep them
  out *consciously*, don't let the gap pass silently.

## Gotchas (about the harness)

- It's grep/awk static analysis: it cannot see runtime `cd`, so it can't always
  tell a sanctioned cloned-repo edit from an RLinf-pyproject edit. Read the
  surrounding function.
- Docker matching is by `--model <name>` / `--env <name>` with a name boundary,
  so `--env libero` won't match `--env liberopro` and `--env franka;` (trailing
  `;`) still matches. If you wire a target in via a non-standard invocation, the
  check may report a false gap.
- The ANSI bold headers come out as escape codes when piped; strip with
  `sed 's/\x1b\[[0-9;]*m//g'` if you need plain text.

## When to stop and ask

Ask the user (Req 9) when: a needed dep genuinely conflicts with a core pin and
neither fork-vs-pin nor a `cd` workaround is obviously right; a model's set of
supported envs is unclear; or whether a new target should ship a Docker image at
all. Don't silently pick — a wrong pin or a missing image breaks installs/CI for
every other env.
