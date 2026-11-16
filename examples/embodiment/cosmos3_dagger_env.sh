#!/usr/bin/env bash
# Source this file before launching Cosmos3 DAgger.
cosmos_dagger_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$cosmos_dagger_repo/.venv/bin/activate" || return
export REPO_PATH="$cosmos_dagger_repo"
# Cosmos3 SGLang eval uses its own Torch/Transformers stack.
export RLINF_SGLANG_EVAL_VENV="$cosmos_dagger_repo/.venv-sglang"

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$VIRTUAL_ENV/system-libs/usr/bin:$VIRTUAL_ENV/system-libs/bin:$PATH"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/system-libs/usr/lib/x86_64-linux-gnu:$VIRTUAL_ENV/system-libs/lib/x86_64-linux-gnu:$VIRTUAL_ENV/system-libs/usr/lib:$VIRTUAL_ENV/system-libs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VIRTUAL_ENV/system-libs/usr/share/glvnd/egl_vendor.d/10_nvidia.json:$VIRTUAL_ENV/system-libs/usr/share/glvnd/egl_vendor.d/50_mesa.json"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export LIBERO_CONFIG_PATH="$VIRTUAL_ENV/libero_config"
export LIBERO_ASSET_PATH=/mnt/public/datasets/libero_pro/pro/source/libero/libero/assets
export LIBERO_ROOT=/mnt/public/rhy/dataset/LIBERO_LeRobot_v3/libero_10
export COSMOS_FRAMEWORK_PATH="$VIRTUAL_ENV/cosmos-framework"
export PYTHONPATH="$REPO_PATH:$COSMOS_FRAMEWORK_PATH"
export COSMOS_TOKENIZER_PATH=/mnt/public/rhy/dataset/Cosmos3-Edge
export WAN_VAE_PATH=/mnt/public/rhy/dataset/Wan2.2-TI2V-5B/Wan2.2_VAE.pth

export RAY_ADDRESS=local
export RAY_AUTH_MODE=token
export RAY_TMPDIR="/tmp/rhy_dagger_$$"
export HF_HOME=/mnt/public/rhy/dataset/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_CACHE_DIR=/mnt/public/rhy/dataset/uv-cache
export UV_OFFLINE=1
export UV_PYTHON_DOWNLOADS=never

unset cosmos_dagger_repo
