#!/usr/bin/env bash

# Source this file from the PPO checkout; no dependencies or weights are installed.
_wam_ppo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
_wam_ppo_venv="${COSMOS3_PPO_VENV:-/mnt/public/rhy/cosmos_dagger/RLinf/.venv}"
if [[ ! -f "$_wam_ppo_venv/bin/activate" ]]; then
    printf 'WAM PPO environment not found: %s\n' "$_wam_ppo_venv" >&2
    return 1
fi
source "$_wam_ppo_venv/bin/activate"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$VIRTUAL_ENV/system-libs/usr/bin:$VIRTUAL_ENV/system-libs/bin:$PATH"
export PYTHONPATH="$_wam_ppo_root"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/system-libs/usr/lib/x86_64-linux-gnu:$VIRTUAL_ENV/system-libs/lib/x86_64-linux-gnu:$VIRTUAL_ENV/system-libs/usr/lib:$VIRTUAL_ENV/system-libs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VIRTUAL_ENV/system-libs/usr/share/glvnd/egl_vendor.d/10_nvidia.json:$VIRTUAL_ENV/system-libs/usr/share/glvnd/egl_vendor.d/50_mesa.json"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export LIBERO_CONFIG_PATH="$VIRTUAL_ENV/libero_config"
export LIBERO_ASSET_PATH=/mnt/public/rhy/WAM_create/third_party/LIBERO/libero/libero/assets
export LIBERO_ROOT=/mnt/public/rhy/dataset/LIBERO_LeRobot_v3/libero_10
export COSMOS_FRAMEWORK_PATH="$VIRTUAL_ENV/cosmos-framework"
export COSMOS_TRAINING=1
export COSMOS_DEVICE=cuda
export WAN_VAE_PATH=/mnt/public/rhy/dataset/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export HF_HOME=/mnt/public/rhy/dataset/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_PYTHON_DOWNLOADS=never
export RAY_ADDRESS=local
export RAY_AUTH_MODE=token
export RAY_TMPDIR=/tmp/rhy_wam_ppo
unset _wam_ppo_root _wam_ppo_venv
