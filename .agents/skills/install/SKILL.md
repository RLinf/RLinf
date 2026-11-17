---
name: install
description: "RLinf install, Docker images, and their CI (requirements/, pyproject.toml, docker/, tests/e2e_tests/, and the Docker, install, and e2e workflows): add or change how a model or env installs, gets a Docker image, and runs in CI e2e; check install.sh and Dockerfile changes against the conventions; build a venv and run the matching e2e test. Use before editing any of these files, when adding a new embodied model or env, and to verify an install or an e2e config."
paths:
  - requirements/**
  - pyproject.toml
  - docker/**
  - tests/e2e_tests/**
  - .github/workflows/docker-build.yml
  - .github/workflows/install.yml
  - ".github/workflows/*e2e*.yml"
  - .github/scripts/**
---

# Install

`requirements/install.sh` installs every embodied model and env combination, `docker/Dockerfile` builds an image per target, and the e2e workflows install a target and train it for a few steps on a CI runner. This skill covers adding a target to all three, the conventions the install script follows, a check for changes to it, and a driver that builds the venv and runs the e2e test the way CI does. All paths are relative to the repository root, and [reference.md](reference.md) has the code shapes.

## Conventions

1. **Reuse the common utilities.** If one almost fits, extend it rather than writing a near-duplicate.

   | Need | Use | Instead of |
   |---|---|---|
   | Clone a repo | `clone_or_reuse_repo ENV_VAR DEFAULT_DIR URL [git args…]` (reuse through `*_PATH`, mirror prefix, re-clone when corrupt) | raw `git clone` |
   | flash-attn | `install_flash_attn` (platform-aware, prebuilt wheels, source fallback) | hand-built wheel URLs |
   | apex | `install_apex` | building apex inline |
   | venv and base sync | `create_and_sync_venv`, then `install_common_embodied_deps` | `uv venv` or `uv sync` by hand |
   | A torch version | the `--torch` flag, applied by `apply_torch_override` | `sed` on `pyproject.toml` |
   | System libs, EGL, Vulkan | `requirements/embodied/sys_deps.sh` and the per-platform render config | `apt-get`, or ICD JSON written inline |
   | CUDA or ROCm version | `detect_cuda_major_minor`, `detect_rocm_version` | parsing `nvcc` or `rocminfo` |
   | Mirror-aware git | the `GITHUB_PREFIX` prefix and `setup_mirror` | hardcoded mirror URLs |

2. **One `install_<model>_model()` per model**, which `case`s on `$ENV_NAME` and calls the matching `install_<env>_env()`. An env with no model installs through a branch in `install_env_only`.
3. **System packages go in `requirements/embodied/sys_deps.sh`**, added to every `install_deps_<manager>` function so apt, dnf, yum, and pacman hosts all get them. `install.sh` never calls a package manager.
4. **Every git dependency is pinned** to a commit, tag, or branch (`@<rev>` or `-b <branch>`), or forked into `github.com/RLinf/<repo>`.
5. **RLinf's own `pyproject.toml` changes only through `apply_torch_override`.** When an override conflicts with a dependency, `cd` into that dependency's checkout and install from there; editing the cloned repo's `pyproject.toml` after the `cd` is fine.
6. **Core dependencies stay as they are.** Never pin `ray` per env, and avoid loose `torch` pins. Before adding `uv pip install x==y`, check whether the common deps or `pyproject.toml` already provide it.
7. **A new model or env gets a Docker stage** when comparable targets on `main` have one. Real-world hardware envs (`yam`, `so101`, `piper`, `dosw1`, `gim_arm`, `xsquare_turtle2`) have none.
8. **Keep it small.** A short `case` branch that calls shared helpers beats a long block of `sed` and `uv pip` workarounds.

## Write

### Add a model or env

1. **Install script.** Add the name to `SUPPORTED_MODELS` or `SUPPORTED_ENVS`.
   - A new model gets `install_<model>_model()` and a `<model>)` branch in the `case "$MODEL"` of `main()`. Model-specific pins go in `requirements/embodied/models/<model>.txt` once there are more than a couple.
   - A new env for an existing model is a branch in that model's `case "$ENV_NAME"` that calls `create_and_sync_venv`, `install_common_embodied_deps`, and `install_<env>_env`; the model's own packages install after the `case`.
   - An env with no model is a branch in `install_env_only`.
2. **Dockerfile.** Add a `base-image-embodied-<target>` line if the target needs another base, and a `FROM embodied-common-image AS embodied-<target>-image` stage that runs `install.sh` and sets the default venv in `~/.bashrc`. A stage that installs several venvs chains every `install.sh` call in one `RUN` with `&&`, because uv hardlinks from its cache only within one layer. The new stage is a valid `BUILD_TARGET` with no change to the final stage.
3. **Docker build CI.** Copy a `build-embodied-*` job in `.github/workflows/docker-build.yml` and change the job id, step name, `BUILD_TARGET`, and tag.
4. **E2e test.** Add `tests/e2e_tests/embodied/<env>_<algo>_<model>.yaml`, started from a current sibling and following the code skill's config rules ([patterns.md § Configuration](../code/patterns.md#configuration)). Add a job to `.github/workflows/embodied-e2e-tests.yml` that installs the target, runs `tests/e2e_tests/embodied/run.sh <config>` (or `run_async.sh`), and removes `.venv`. Jobs run on `runs-on: embodied`, and none runs `uv cache prune` against the shared cache.
5. **Path filters and docs.** New files go into the path filters of `.github/workflows/ci-tests.yml` ([code](../code/SKILL.md)), and the model or env gets its recipe page and README highlight ([docs](../docs/SKILL.md)).

Run `bash -n requirements/install.sh` after each edit, and the check below before you finish.

## Check

```bash
python3 .agents/skills/install/check.py            # files changed against main
python3 .agents/skills/install/check.py FILE ...
```

The line rules report only the lines the change adds to `install.sh`, and the Docker stage rule only the names it adds, so exceptions already on `main` do not repeat; `--all-lines` audits the whole file instead. The rules are heuristics: each hit is a line to review against the conventions. Findings use the shared format of [README.md](../README.md#findings):

| Rule | Severity | Reports |
|---|---|---|
| `install/syntax` | critical | a shell script under `requirements/` that fails `bash -n` |
| `install/sys-deps` | major | a package-manager call in `install.sh` (convention 3) |
| `install/git-pin` | major | a git dependency neither pinned nor forked into RLinf (convention 4) |
| `install/core-dep` | major or minor | a `ray` pin (major) or an explicit `torch` pin (minor) (convention 6) |
| `install/pyproject` | minor | a write to a `pyproject.toml` outside `apply_torch_override` (convention 5) |
| `install/clone` | minor | a raw `git clone` (convention 1) |
| `install/docker-stage` | major | a model or env added to `SUPPORTED_*` with no `--model` or `--env` install in the Dockerfile (convention 7) |
| `install/docker-layers` | major | a changed Docker stage that calls `install.sh` from more than one `RUN` |
| `install/e2e-wiring` | major | a new e2e config that no workflow runs |

Some hits are sanctioned: a `torch` pin installed from `~` inside `pushd ~ … popd`, a clone that a later `git checkout <hash>` in the same function pins, a multi-repo build such as the Franka catkin workspace, and an edit to a cloned repo's `pyproject.toml` after `cd` into it. Keep such a line and say why in the PR.

Then judge what the script cannot:

- [ ] The change is a small branch that calls shared helpers, and each new override is one a common dependency does not already provide.
- [ ] A new Docker image, model, or dataset comes with a maintainer ping in the PR.
- [ ] A new e2e job does not duplicate coverage another workflow already runs.
- [ ] Install, Docker, or CI changes stay within what the new component needs.

## Verify

[driver.py](driver.py) reads the install command, env vars, and test configs from `.github/workflows/embodied-e2e-tests.yml`, so it reproduces CI without a copy that drifts. It runs on the embodied CI runner or an equivalent host: NVIDIA GPUs, the shared `/workspace/dataset/` tree, `uv`, and a `python3` with PyYAML (it re-runs itself under `uv run --with pyyaml` otherwise).

Start with the read-only subcommands, which return at once:

```bash
python3 .agents/skills/install/driver.py list                       # combos CI covers and their configs
python3 .agents/skills/install/driver.py resolve <model> <env>      # CI's install command, env vars, configs
python3 .agents/skills/install/driver.py check-paths <config>       # do the config's model paths exist here?
python3 .agents/skills/install/driver.py check-all                  # check-paths over every e2e config
```

`check-paths` classifies each absolute path as `[model/input]` (must exist), `[output dir]` (created by the run), or `[path]` (informational). A missing `[model/input]` fails the run before training; on a host without the dataset tree that is a staging gap, not an install bug.

`run` installs, checks paths, and runs the test. Preview it with `--dry-run`, which prints the exact shell, then drop the flag:

```bash
python3 .agents/skills/install/driver.py run <model> <env> --venv /workspace/test-venvs/<model> --dry-run
python3 .agents/skills/install/driver.py install <model> <env> --venv /workspace/test-venvs/<model>
python3 .agents/skills/install/driver.py test <config> --venv /workspace/test-venvs/<model>
```

`install` and `test` run the two halves separately, so one install serves many tests. A run is done when fresh TensorBoard files appear under the config's `log_path`. Afterwards `rm -rf` the `--venv` you built unless the user wants to keep it; leave the shared uv caches and the prebuilt `/opt/venv/*` alone. When a model or env has no e2e job, ask the user what to run; `test <config> --runner run|run_async|run_offline` runs a config that CI does not. [reference.md § Driver notes](reference.md#driver-notes) has the gotchas and the troubleshooting.

## When to stop and ask

Ask the user when a dependency conflicts with a core pin and neither a pin, a fork, nor a `cd` workaround is clearly right; when the set of envs a model supports is unclear; or when it is unclear whether a target should ship a Docker image. A wrong pin or a missing image breaks installs or CI for every other env.
