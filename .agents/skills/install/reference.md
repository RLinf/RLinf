# Install reference

Code shapes for the steps in [SKILL.md § Write](SKILL.md#write), and notes on [driver.py](driver.py). Read the current `SUPPORTED_MODELS` and `SUPPORTED_ENVS` in `requirements/install.sh` rather than a copy.

## Install script

A new model installer `case`s on the env, and `main()` dispatches to it:

```bash
install_<model>_model() {
    case "$ENV_NAME" in
        <env>)
            create_and_sync_venv
            install_common_embodied_deps
            install_<env>_env
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for <Model> model." >&2
            exit 1
            ;;
    esac
    # the model's own packages, pinned
    install_flash_attn        # if the model needs it
}

# in main(), inside case "$MODEL" in
    <model>)
        install_<model>_model
        ;;
```

A new env for an existing model is one more branch in that model's `case "$ENV_NAME"`. An env with no model is a branch in `install_env_only`; if it introduces `install_<env>_env()`, the model installers that support the env call it too.

## Dockerfile

A target that needs another base adds it at the top:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image-embodied-<target>
# Ubuntu 20.04 and ROS targets use: FROM ubuntu:20.04 AS base-image-embodied-<target>
```

The stage follows `embodied-common-image`:

```dockerfile
##################################################################################################
# Embodied: <Description>
##################################################################################################
FROM embodied-common-image AS embodied-<target>-image

RUN bash requirements/install.sh embodied --venv <venv> --model <model> --env <env>

# Optional: download or link assets
# RUN source switch_env <venv> && download_assets --dir /opt/assets --assets ...
# RUN link_assets

RUN echo "source ${UV_PATH}/<venv>/bin/activate" >> ~/.bashrc
```

A stage with several venvs installs them in one `RUN`:

```dockerfile
RUN bash requirements/install.sh embodied --venv <venv1> --model <model1> --env <env> && \
    bash requirements/install.sh embodied --venv <venv2> --model <model2> --env <env>
```

`BUILD_TARGET` is `embodied-<env>`, or `embodied-<env>-<model>` when one image combines several models; match the existing stages. The last stage, `FROM ${BUILD_TARGET}-image AS final-image`, needs no change.

## Docker build job

Copy an existing `build-embodied-*` job in `.github/workflows/docker-build.yml`. Every job shares the storage, checkout, and Buildx steps; change only:

- the job id, `build-embodied-<target>`, and the build step name;
- `build-args`: `BUILD_TARGET=embodied-<target>` and `NO_MIRROR=true`;
- `tags`: `rlinf:embodied-<target>`.

## E2e test

`run.sh <name>` runs `tests/e2e_tests/embodied/<name>.yaml` through:

```bash
python ${REPO_PATH}/examples/embodiment/train_embodied_agent.py \
    --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name <name>
```

The job in `.github/workflows/embodied-e2e-tests.yml`, `embodied-<model>-<env>-test`, follows its neighbors:

- **Create embodied environment:** export `UV_PATH`, `UV_LINK_MODE`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, and any path variables (`GR00T_PATH`, `BEHAVIOR_PATH`), then `bash requirements/install.sh embodied --model <model> --env <env>`.
- **Test:** `source .venv/bin/activate`, `export REPO_PATH=$(pwd)`, then `bash tests/e2e_tests/embodied/run.sh <config>` (or `run_async.sh`), with a `timeout-minutes` that fits.
- **Clean up:** `rm -rf .venv` and any test-specific files.

## Driver notes

- `install` and `run` need `--venv <path>`, passed to `install.sh --venv`. Use an absolute path; a bare name lands under the repository root. `test` sources `<path>/bin/activate`.
- `--venv` reuses an existing venv at that path after `install.sh` validates its Python version. For a clean install, use a fresh path or remove the old one first.
- The driver runs CI's shell as written, including its `UV_PATH=/workspace/dataset/.uv` exports, so an install writes into the shared uv cache the way CI does.
- Every install gets `--use-mirror` unless the CI job already has it; `--no-mirror` turns it off.
- `resolve` shows each job's `--platform`; `amd` and `ascend` jobs install only on those runners.
- Some test steps do extra setup, such as copying ManiSkill assets into the repository or exporting `ROBOT_PLATFORM`. The driver replays the whole step, which assumes those asset directories exist under `/workspace/dataset/`.
- A config listed twice by `list` runs twice with different flags (FSDP on and off).

Troubleshooting:

- `No CI job for model=… env=…`: that combination is not in the workflow. `list` shows the valid pairs; names match `install.sh`'s `--model` and `--env` exactly (`maniskill_libero`, not `libero`).
- `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during a test is protobuf noise from the workers. The Ray `Placement(...)` lines and the rollout progress bar show real progress.
- `error: run this from inside the RLinf repo`: run the driver from the repository root.
- A test that exits 0 at once without training was backgrounded with `&`, and the launcher returned while training detached. Run it in the foreground.
- An install that hangs at `uv sync` with no CPU use and no cache writes usually has a dead `http(s)_proxy` variable, such as a local proxy that no longer forwards. Unset the proxy variables and retry.
- `setup_mirror` failing with `cannot overwrite multiple values ... insteadOf` means interrupted runs left duplicate `url.<mirror>.insteadOf` entries in the global git config. Clear them with `git config --global --unset-all url."https://gh-proxy.com/github.com/".insteadOf` and retry.
