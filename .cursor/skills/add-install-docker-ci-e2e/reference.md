# Reference: Install, Docker, and CI Locations

## 1. Install script — `requirements/install.sh`

**Arrays (near top):**
```bash
SUPPORTED_MODELS=("openvla" "openvla-oft" "openpi" "gr00t")
SUPPORTED_ENVS=("behavior" "maniskill_libero" "metaworld" "calvin" "isaaclab" "robocasa" "franka" "frankasim" "robotwin" "habitat" "opensora")
```

**New model installer pattern:** Implement `install_<model>_model()` with `case "$ENV_NAME" in ... esac`, then in `main()` inside `case "$MODEL" in` add:
```bash
model_name)
    install_<model>_model
    ;;
```

**New env in existing model:** In `install_<model>_model()`, add a new `env_name)` branch: create_and_sync_venv, install_common_embodied_deps, env-specific install (and optional `install_<env>_env`), then model-specific pip/install.

**New env only:** In `install_env_only`, add a branch for the new `ENV_NAME`; if you introduce `install_<env>_env()`, call it from the model installers that support that env.

---

## 2. Dockerfile — `docker/Dockerfile`

**Base image (top of file):**
```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image-embodied-<target>
```
For Ubuntu 20 / ROS use: `FROM ubuntu:20.04 AS base-image-embodied-<target>`.

**New embodied stage (after embodied-common-image):**

Single env:
```dockerfile
##################################################################################################
# Embodied: <Description>
##################################################################################################
FROM embodied-common-image AS embodied-<target>-image

# Install <model> env
RUN bash requirements/install.sh embodied --venv <venv> --model <model> --env <env>

# Optional: download/link assets, then link_assets
# RUN source switch_env <venv> && download_assets --dir /opt/assets --assets ...
# RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/<venv>/bin/activate" >> ~/.bashrc
```

**Multiple envs in one image:** Chain all install commands in a **single RUN** so uv hardlinking works (uv uses `UV_LINK_MODE=hardlink`; cross-layer cache breaks hardlinks).
```dockerfile
# Install multiple envs in one layer (required for uv hardlink)
RUN bash requirements/install.sh embodied --venv <venv1> --model <model1> --env <env> && \
    bash requirements/install.sh embodied --venv <venv2> --model <model2> --env <env>
```

The final stage is already `FROM ${BUILD_TARGET}-image AS final-image`; ensure `<target>` matches the name used in `BUILD_TARGET` (e.g. `embodied-<target>`).

---

## 3. Docker build CI — `.github/workflows/docker-build.yml`

Duplicate an existing job (e.g. `build-embodied-frankasim`). Change:
- Job id: `build-embodied-<target>`
- Step name: `Build embodied-<target>`
- `build-args`: `BUILD_TARGET=embodied-<target>`, `NO_MIRROR=true`
- `tags`: e.g. `rlinf:embodied-<target>`

All jobs use the same “Maximize storage space” and “Checkout” / “Set up Docker Buildx” steps; only the build step name and build-args/tags differ.

---

## 4. Install CI — `.github/workflows/install.yml`

In the `build` job, add a step:
```yaml
- name: Install <model>-<env>
  run: |
    pip install uv
    uv cache prune --ci   # optional
    bash requirements/install.sh embodied --model <model> --env <env>
    rm -rf .venv
```

For multiple envs for one model, repeat the last two lines (install then rm -rf .venv) for each env. Use `TEST_BUILD=1` only where the install script is designed to use it (e.g. reason).

---

## 5. E2e test — config and workflow

**Config:** Add `tests/e2e_tests/embodied/<name>.yaml`. The runner is:
```bash
python ${REPO_PATH}/examples/embodiment/train_embodied_agent.py --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name <name>
```
So `run.sh <name>` runs the config `<name>.yaml`.

**Workflow:** In `.github/workflows/embodied-e2e-tests.yml`, add a job patterned on existing ones:
- Job id: `embodied-<model>-<env>-test`
- “Create embodied environment”: set `UV_PATH`, `UV_LINK_MODE`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, and any path vars (`LIBERO_PATH`, `GR00T_PATH`, `BEHAVIOR_PATH`, etc.), then `bash requirements/install.sh embodied --model <model> --env <env>`
- “<Description> test”: `source .venv/bin/activate`, `export REPO_PATH=$(pwd)`, then `bash tests/e2e_tests/embodied/run.sh <config_name>` (or `run_async.sh` for async)
- “Clean up”: `rm -rf .venv`, `uv cache prune`

All e2e jobs use `runs-on: embodied`.
