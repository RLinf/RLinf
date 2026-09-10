#!/usr/bin/env bash

# Install the Ascend NPU environment used by the reasoning and agent examples.
#
# The script uses uv to create or reuse a Python virtual environment, then
# installs CPU PyTorch plus torch_npu and builds the NPU-enabled SGLang stack.

set -eo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11.14}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.2}"
SGLANG_DIR="${SGLANG_DIR:-${VENV_DIR}/src/sglang}"
SGL_KERNEL_NPU_DIR="${SGL_KERNEL_NPU_DIR:-${VENV_DIR}/src/sgl-kernel-npu}"
SGL_KERNEL_NPU_REF="${SGL_KERNEL_NPU_REF:-20251214}"
TORCH_WHEEL_URL="${TORCH_WHEEL_URL:-https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl}"
# Override this URL with the torch_npu wheel matching the installed CANN version.
TORCH_NPU_WHEEL_URL="${TORCH_NPU_WHEEL_URL:-https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_28_aarch64.whl}"
WHEEL_DIR="${WHEEL_DIR:-${VENV_DIR}/wheels}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${VENV_DIR}/cache}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
TEST_PYPI_INDEX_URL="${TEST_PYPI_INDEX_URL:-https://test.pypi.org/simple}"

install_uv() {
    if ! command -v uv &> /dev/null; then
        echo "uv command not found. Installing uv..."
        pip install uv
    fi
}

python_major_minor() {
    python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")"
}

download_wheel() {
    local wheel_url="$1"
    local wheel_name="${wheel_url##*/}"
    wheel_name="${wheel_name//%2B/+}"
    local wheel_path="${WHEEL_DIR}/${wheel_name}"
    local partial_path="${wheel_path}.part"

    mkdir -p "${WHEEL_DIR}"
    if [[ ! -f "${wheel_path}" ]]; then
        wget --show-progress -O "${partial_path}" "${wheel_url}"
        mv "${partial_path}" "${wheel_path}"
    fi
    printf '%s\n' "${wheel_path}"
}

create_or_reuse_venv() {
    local py_major py_minor _
    IFS=. read -r py_major py_minor _ <<< "${PYTHON_VERSION}"
    local required_python_mm="${py_major}.${py_minor}"

    install_uv
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        local active_python_mm
        active_python_mm="$(python_major_minor)"
        if [[ "${active_python_mm}" == "${required_python_mm}" ]]; then
            echo "Reusing active virtual environment at ${VIRTUAL_ENV}"
            VENV_DIR="${VIRTUAL_ENV}"
            return
        fi
        echo "Active virtual environment uses Python ${active_python_mm}; expected ${required_python_mm}.x."
    fi

    if [[ -d "${VENV_DIR}" && -f "${VENV_DIR}/bin/activate" ]]; then
        source "${VENV_DIR}/bin/activate"
        local venv_python_mm
        venv_python_mm="$(python_major_minor)"
        if [[ "${venv_python_mm}" == "${required_python_mm}" ]]; then
            echo "Reusing existing virtual environment at ${VENV_DIR}"
            return
        fi

        echo "Recreating ${VENV_DIR} for Python ${required_python_mm}.x"
        deactivate || true
        rm -rf "${VENV_DIR}"
    fi

    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
    source "${VENV_DIR}/bin/activate"
}

command -v git >/dev/null || { echo "git is required." >&2; exit 1; }
command -v wget >/dev/null || { echo "wget is required." >&2; exit 1; }
create_or_reuse_venv
export UV_CACHE_DIR

uv pip install --upgrade pip
uv pip install \
    "numpy>=1.19.2,<2" "decorator>=4.4.0" "sympy>=1.5.1" \
    cffi attrs cython pyyaml scipy requests psutil absl-py pybind11 \
    "omegaconf==2.4.0.dev4" "hydra-core==1.4.0.dev1" einops nvitop \
    "setuptools>=69.5.1,<75.9" ninja tensorboard "swanlab>=0.6.11"

TORCH_WHEEL_PATH="$(download_wheel "${TORCH_WHEEL_URL}")"
TORCH_NPU_WHEEL_PATH="$(download_wheel "${TORCH_NPU_WHEEL_URL}")"
uv pip install --no-deps "${TORCH_WHEEL_PATH}" "${TORCH_NPU_WHEEL_PATH}"
uv pip install "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install -i "${TEST_PYPI_INDEX_URL}" "triton-ascend<3.2.0rc"

if [[ ! -d "${SGL_KERNEL_NPU_DIR}/.git" ]]; then
    git clone --recursive --branch "${SGL_KERNEL_NPU_REF}" \
        https://github.com/sgl-project/sgl-kernel-npu.git "${SGL_KERNEL_NPU_DIR}"
fi
git -C "${SGL_KERNEL_NPU_DIR}" submodule update --init --recursive
(
    cd "${SGL_KERNEL_NPU_DIR}"
    bash build.sh
    uv pip install output/sgl*.whl output/torch*.whl output/deepep*.whl
)

DEEP_EP_SITE="$(python -c "import os, deep_ep; print(os.path.dirname(os.path.dirname(deep_ep.__file__)))")"
for library in "${DEEP_EP_SITE}"/deep_ep/deep_ep_cpp*.so; do
    [[ -e "${library}" ]] || continue
    ln -sf "${library}" "${DEEP_EP_SITE}/"
done

if [[ ! -d "${SGLANG_DIR}/.git" ]]; then
    git clone --branch "v${SGLANG_VERSION}" \
        https://github.com/sgl-project/sglang.git "${SGLANG_DIR}"
fi
uv pip install -e "${SGLANG_DIR}/python[srt_npu]" --no-build-isolation

uv pip install \
    accelerate "ray[default]>=2.47.0" pylatexenc datasets \
    "latex2sympy2 @ git+https://github.com/RLinf/latex2sympy2.git" \
    sentencepiece torchdata "wandb<0.25.1" word2number regex \
    "transformers==4.56.1" -i "${PYPI_INDEX_URL}"

# SGLang NPU packages provide triton-ascend instead of the CUDA triton wheel.
uv pip uninstall triton triton-ascend pynvml || true
uv pip install -i "${TEST_PYPI_INDEX_URL}" "triton-ascend<3.2.0rc"

echo "Ascend environment installation completed."
