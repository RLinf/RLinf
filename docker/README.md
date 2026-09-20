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

- `PLATFORM` (default `nvidia`) — hardware platform: `nvidia` (CUDA), `amd` (ROCm), `ascend` (CANN), `musa` (Moore Threads), or `kunlun` (Kunlunxin). Selects the base image and is also recorded as `RLINF_PLATFORM` in the final image.
- Per-platform runtime versions: `CUDA_VER`, `ROCM_VER`, `ROCM_ARCHS`, `CANN_VER`, `MUSA_VER`, `KUNLUN_VER`, `UBUNTU_VER`. Override any of these to bump versions without changing the rest of the build. For a fully custom base, set `NVIDIA_BASE_IMAGE`, `AMD_BASE_IMAGE`, `ASCEND_BASE_IMAGE`, `MUSA_BASE_IMAGE`, or `KUNLUN_BASE_IMAGE` directly.
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

### Building for Franka

The `embodied-franka` target builds on the standard platform base image. Its
default `franky` venv runs the Franky backend (libfranka 0.19.0) with the
dexterous-hand dependencies, and the `openvla`, `openvla-oft`, `openpi`, and
`gr00t` venvs add those policies to the same stack. Its venvs carry CUDA PyTorch,
so one container can run the actor, rollout, and robot control; run it with
`--gpus all` on a host with the NVIDIA driver and NVIDIA Container Toolkit. See the
[Franka example](../docs/source-en/rst_source/examples/embodied/franka.rst).

### Building for Other Hardware Platforms

Every `PLATFORM` builds the same image for a given `BUILD_TARGET`; only the base image and the platform packages that `install.sh` adds differ. Support for a model on a platform is listed in the documentation's hardware support table. The sections below cover the base image, build arguments, and container runtime for each non-NVIDIA platform.

### Building for AMD (ROCm)

`PLATFORM=amd` builds on `rocm/dev-ubuntu-$UBUNTU_VER:$ROCM_VER-complete`, which is published for `linux/amd64` only. Set `ROCM_VER` to the host's ROCm release and `ROCM_ARCHS` to the `gfx` architectures of the target GPUs. GPUs are not visible during `docker build`, so extensions such as flash-attn compile for exactly the architectures listed there. The default `ROCM_VER=7.2` installs prebuilt flash-attn wheels for Python 3.11 and 3.12; other ROCm releases compile flash-attn from source, which takes much longer.

```shell
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-maniskill_libero \
    --build-arg PLATFORM=amd \
    --build-arg ROCM_VER=7.2 \
    --build-arg 'ROCM_ARCHS=gfx90a;gfx942' \
    -t rlinf:embodied-maniskill_libero-rocm7.2 .
```

Run the image with the AMD kernel and render devices:

```shell
docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 20g --network host \
    rlinf:embodied-maniskill_libero-rocm7.2 bash
```

### Building for Huawei Ascend (CANN)

`PLATFORM=ascend` builds on `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:$CANN_VER-ubuntu$UBUNTU_VER-py3.11`. `CANN_VER` carries the SoC suffix of the base image tag, such as `9.1.1-910b` (the default) or `9.1.1-950`. The base images are published for both `linux/amd64` and `linux/arm64`. Build with BuildKit so that the CUDA and ROCm bases on Docker Hub are not pulled.

On an Ascend host, build for the host's own architecture:

```shell
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-maniskill_libero \
    --build-arg PLATFORM=ascend \
    --build-arg CANN_VER=9.1.1-910b \
    -t rlinf:embodied-maniskill_libero-cann9.1 .
```

Most Ascend servers are aarch64. To build an aarch64 image on an x86_64 machine, register QEMU emulation for arm64 once, then build with `docker buildx` and `--platform linux/arm64`. Every `RUN` step, including the Python dependency installs, then runs under emulation and takes several times longer than a native build.

```shell
docker run --privileged --rm tonistiigi/binfmt --install arm64

docker buildx build --platform linux/arm64 -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-maniskill_libero \
    --build-arg PLATFORM=ascend \
    --build-arg CANN_VER=9.1.1-910b \
    -t rlinf:embodied-maniskill_libero-cann9.1-arm64 .
```

On some hosts, notably WSL2, statically linked arm64 programs such as `ldconfig` crash intermittently under emulation with `qemu: uncaught target signal 11 (Segmentation fault)`, depending on where the kernel randomly places their memory. `apt` runs `ldconfig` while installing packages, so a crash can fail the build. Disable address randomization while the build runs, then restore it:

```shell
sudo sysctl kernel.randomize_va_space=0
# run the cross-build above
sudo sysctl kernel.randomize_va_space=2
```

Building on an aarch64 host avoids emulation altogether.

With Docker's default builder, the arm64 image lands in the local image store like a native build. A builder created with `docker buildx create` keeps results in its own cache: add `--load` to import the image, or `--push` with a registry tag to publish it. Move a local image to an Ascend host with `docker save` and `docker load`.

Run the image with the host's NPU driver mounted:

```shell
docker run -it --rm \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    --device=/dev/davinci0 \
    --ipc=host --shm-size 20g --network host \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    rlinf:embodied-maniskill_libero-cann9.1 bash
```

Add a `--device=/dev/davinciN` entry for each NPU the container should use.

### Building for Moore Threads (MUSA)

`PLATFORM=musa` builds on top of the Moore Threads training suite image
(`registry.mthreads.com/mcctest/ai/training-suite:$MUSA_VER`), which already
carries a MUSA-built torch plus `torch-musa`. `install.sh` therefore installs no
torch of its own — it creates the venv with `--system-site-packages` on the
image's interpreter and skips every CUDA-only package (flash-attn, apex, and the
vLLM/SGLang kernels). Build and run the `embodied-maniskill_libero` image with the `mthreads` container
runtime:

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

### Building for Kunlunxin (KUNLUN)

`PLATFORM=kunlun` builds on top of the Kunlunxin image
(`hub.kunlunxin.com/public/kunlita/xav-rlinf:$KUNLUN_VER`). The
image supplies the vendor Torch runtime, and `install.sh` clones that Python
environment before installing RLinf dependencies.

Install Buildx and build the image from the RLinf root directory.

```shell
apt-get install -y docker-buildx

docker buildx build \
    -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-maniskill_libero \
    --build-arg PLATFORM=kunlun \
    -t rlinf:embodied-maniskill_libero-kunlun .
```

# Using the Docker Image

The built Docker image contains one or more Python virtual environments (venvs) under `/opt/venv/`. Which venvs are present, and which one is activated by default in new shells, depends on the `BUILD_TARGET` — see the corresponding build stage in the [`Dockerfile`](Dockerfile).

To switch between venvs, use the built-in `switch_env` script:

```shell
source switch_env <env_name> # e.g., source switch_env openvla-oft, source switch_env openpi, etc.
```
