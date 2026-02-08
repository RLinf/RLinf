---
name: add-install-docker-ci-e2e
description: Adds install command in install script, Docker build stage in Dockerfile, and CI jobs for docker build, install script, and embodied e2e test when introducing a new model or environment in RLinf. Use when adding a new embodied model (e.g. dexbotic), new env (e.g. maniskill_libero), or new model+env combination that should be installable, dockerized, and tested in CI.
---

# Add Install, Docker Build, and CI for a New Model or Environment

Use this skill when adding a **new model** or **new environment** (or combination) to RLinf so that: (1) users can install it via `requirements/install.sh`, (2) a Docker image can be built for it (optional), (3) CI runs install, Docker build, and an end-to-end test.

---

## 1. Install script (`requirements/install.sh`)

- **Register model or env**
  - New **model**: add to `SUPPORTED_MODELS` (e.g. `"dexbotic"`).
  - New **environment**: add to `SUPPORTED_ENVS` (e.g. `"maniskill_libero"`).

- **Implement install logic**
  - **New model**: add `install_<model>_model()` that switches on `ENV_NAME` and for each supported env: create venv, install common embodied deps, env-specific deps, and the model. Call it from the main `case "$MODEL"` (add a new `model_name)` branch that runs `install_<model>_model`).
  - **New env only** (no new model): either add a new env branch inside an existing `install_*_model()` or add `install_<env>_env()` and call it from the relevant model installers. If the env is used by `install_env_only`, add a branch in `install_env_only` for that env.

- **Help text**  
  `print_help` shows `SUPPORTED_MODELS` and `SUPPORTED_ENVS`; no change needed if you only added to those arrays.

See [reference.md](reference.md) for exact variable names and code patterns.

---

## 2. Dockerfile (`docker/Dockerfile`)

- **Base image**  
  If the combo needs a different base (e.g. Ubuntu 20 for ROS/Franka), add:
  `FROM <base> AS base-image-embodied-<target>`  
  Otherwise reuse: `FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image-embodied-<target>`.

- **Build stage**  
  Add a stage:
  - `FROM embodied-common-image AS embodied-<target>-image`
  - **Single RUN for all installs**: If the image installs **multiple** envs (multiple model+env or venvs), chain every `install.sh` call in **one** `RUN` with `&&`. Splitting installs across multiple `RUN` layers breaks uv’s hardlink mode (`UV_LINK_MODE=hardlink`), because the cache from the previous layer is not in the same layer for hardlinking. Example: `RUN bash requirements/install.sh embodied --venv openvla --model openvla --env maniskill_libero && \` then `    bash requirements/install.sh embodied --venv openpi --model openpi --env maniskill_libero`.
  - Any asset download/link in the same or a following RUN; then `RUN echo "source \${UV_PATH}/<venv>/bin/activate" >> ~/.bashrc` for default env.

- **Final stage**  
  The last stage is `FROM ${BUILD_TARGET}-image AS final-image`. Valid `BUILD_TARGET` values are those that have a matching `*-image` stage (e.g. `reason`, `embodied-maniskill_libero`, `embodied-dexbotic-maniskill_libero`). Adding a new stage makes the new target valid; no change to the final stage line.

Naming: `BUILD_TARGET` is typically `embodied-<env>` (e.g. `embodied-maniskill_libero`) or `embodied-<env>-<model>` when one image combines multiple models (e.g. behavior-openvlaoft). Match the pattern used by existing stages.

---

## 3. CI: Docker build (`.github/workflows/docker-build.yml`)

Add a job that builds the new image:

- Job id: `build-embodied-<target>` (same `<target>` as in Dockerfile stage name, e.g. `build-embodied-maniskill_libero`).
- Reuse the same steps as existing jobs: maximize storage, checkout, setup Docker Buildx, then build with `BUILD_TARGET=embodied-<target>`, `NO_MIRROR=true`, `outputs: type=cacheonly`, and a tag like `rlinf:embodied-<target>`.

Copy an existing `build-embodied-*` job and replace the target name. See [reference.md](reference.md).

---

## 4. CI: Install script (`.github/workflows/install.yml`)

Add an “Install <model>-<env>” step (or “Install <model>” with one or more envs) in the `build` job:

- `pip install uv` (and `uv cache prune --ci` if desired).
- `bash requirements/install.sh embodied --model <model> --env <env>` (add `TEST_BUILD=1` only if the install script is designed to support it for that target).
- `rm -rf .venv` before the next install.

For multiple envs for the same model, use multiple `install.sh` calls, each followed by `rm -rf .venv`. For special runners (e.g. Franka on Ubuntu 20.04), follow the existing `build-franka` pattern (container image, env vars, loop over versions if any).

---

## 5. CI: Embodied e2e test (`.github/workflows/embodied-e2e-tests.yml`)

- **Test config**  
  Add a YAML config under `tests/e2e_tests/embodied/` (e.g. `<env>_<algo>_<model>.yaml`). The e2e runner is `train_embodied_agent.py` with `--config-name <name>`; the config name is the filename without `.yaml`.

- **Workflow job**  
  Add a job (e.g. `embodied-<model>-<env>-test`):
  - Checkout.
  - Create embodied environment: set `UV_*`, any required path env vars (e.g. `LIBERO_PATH`, `GR00T_PATH`), then `bash requirements/install.sh embodied --model <model> --env <env>`.
  - Run test: `source .venv/bin/activate`, set `REPO_PATH`, then `bash tests/e2e_tests/embodied/run.sh <config_name>` (or `run_async.sh` if the test is async). Use a reasonable `timeout-minutes`.
  - Clean up: `rm -rf .venv`, `uv cache prune`, and any test-specific cleanup.

Use `runs-on: embodied` so the job runs on a runner with GPU/datasets. See existing jobs in the file for env vars and step order.

---

## Checklist

- [ ] **Install script**: Model in `SUPPORTED_MODELS` and/or env in `SUPPORTED_ENVS`; `install_*` function and `case "$MODEL"` (or env) updated.
- [ ] **Dockerfile**: `base-image-embodied-<target>` if needed; `embodied-<target>-image` stage with `install.sh` and default venv. If multiple envs: all install.sh calls chained in one RUN (for uv hardlink).
- [ ] **docker-build.yml**: New job `build-embodied-<target>` with `BUILD_TARGET=embodied-<target>`.
- [ ] **install.yml**: New install step(s) for the new model/env.
- [ ] **E2e**: Config YAML in `tests/e2e_tests/embodied/`; new job in `embodied-e2e-tests.yml` (install env, run `run.sh <config_name>`, clean up).
