## Building Docker Images

RLinf provides a unified Dockerfile for both the math reasoning image and the various embodied images. Use the `BUILD_TARGET` build argument to select which image to build:

- `reason` — math reasoning image
- `embodied-<env>` — embodied image for a specific environment (and optionally a specific model when multiple model flavors exist for the same env)

To build the Docker image, run the following command **in the RLinf root directory**:

```shell
export BUILD_TARGET=reason # or one of the embodied-* targets defined in the Dockerfile
docker build -f docker/Dockerfile --build-arg BUILD_TARGET=$BUILD_TARGET -t rlinf:$BUILD_TARGET .
```

### Available `BUILD_TARGET` values

Each `BUILD_TARGET` maps to a build stage in [`Dockerfile`](Dockerfile). To see the full, up-to-date list of targets and the venvs each one installs, look at the stage names (`FROM ... AS <target>-image`) and the `install.sh` invocations inside them — the Dockerfile is the source of truth, so this README does not duplicate the list.

### Additional build arguments

- `PLATFORM` (default `nvidia`) — hardware platform: `nvidia` (CUDA), `amd` (ROCm), `ascend` (CANN), or `musa` (Moore Threads). Selects the base image and is also recorded as `RLINF_PLATFORM` in the final image. The `embodied-franka` target ignores `PLATFORM` and always uses a plain `ubuntu:20.04` base.
- Per-platform runtime versions: `CUDA_VER`, `ROCM_VER`, `ROCM_ARCHS`, `CANN_VER`, `MUSA_VER`, `UBUNTU_VER`. Override any of these to bump versions without changing the rest of the build. For a fully custom base, set `NVIDIA_BASE_IMAGE`, `AMD_BASE_IMAGE`, `ASCEND_BASE_IMAGE`, or `MUSA_BASE_IMAGE` directly.
- `NO_MIRROR` — set to `1` to skip the USTC apt/pypi mirror rewrites (recommended outside of mainland China).

Example with non-default args:

```shell
docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-metaworld \
    --build-arg PLATFORM=nvidia \
    --build-arg CUDA_VER=12.4.1 \
    --build-arg NO_MIRROR=1 \
    -t rlinf:embodied-metaworld .
```

### Building for Moore Threads (MUSA)

`PLATFORM=musa` builds on top of the Moore Threads training suite image
(`registry.mthreads.com/mcctest/ai/training-suite:$MUSA_VER`), which already
carries a MUSA-built torch plus `torch-musa`. `install.sh` therefore installs no
torch of its own — it creates the venv with `--system-site-packages` on the
image's interpreter and skips every CUDA-only package (flash-attn, apex, and the
vLLM/SGLang kernels). The `embodied-maniskill_libero` target builds the subset of
models that need none of them (`openpi` and `gr00t`) when `PLATFORM=musa`. Build
and run it with the `mthreads` container runtime:

Build with BuildKit — the legacy builder resolves every `FROM` in the
Dockerfile, including the CUDA and ROCm bases on Docker Hub that a MUSA host
often cannot reach.

```shell
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-maniskill_libero \
    --build-arg PLATFORM=musa \
    -t rlinf:embodied-maniskill_libero .

docker run -it --runtime=mthreads --ipc=host --shm-size=100g \
    -e MTHREADS_VISIBLE_DEVICES=all \
    rlinf:embodied-maniskill_libero bash
```

# Using the Docker Image

The built Docker image contains one or more Python virtual environments (venvs) under `/opt/venv/`. Which venvs are present, and which one is activated by default in new shells, depends on the `BUILD_TARGET` — see the corresponding build stage in the [`Dockerfile`](Dockerfile).

To switch between venvs, use the built-in `switch_env` script:

```shell
source switch_env <env_name> # e.g., source switch_env openvla-oft, source switch_env openpi, etc.
```