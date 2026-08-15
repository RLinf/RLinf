#! /bin/bash

set -eo pipefail

TARGET=""

MODEL=""
ENV_NAME=""
VENV_DIR=".venv"
PYTHON_VERSION="3.11.14"
# Set by --python; MUSA only defaults PYTHON_VERSION when it is unset.
USER_SET_PYTHON=0
LEROBOT_COMMIT="0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
TORCH_VERSION=""
SGLANG_VERSION=""
ENGINE=""
TRANSFORMERS_VERSION=""
XGRAMMAR_VERSION=""
PLATFORM="nvidia"
ROCM_VERSION=""
# PEP 440 local-version segment (including the leading '+') that
# apply_torch_override appends to torch/torchvision/torchaudio overrides so uv
# is forced to fetch the platform-specific wheel instead of the bare PyPI one.
# Empty for nvidia (PyPI CUDA wheels match `==X.Y.Z` directly). Set by the
# per-platform configure_<platform> hooks.
PLATFORM_TORCH_STR=""
# URL of the platform-specific PyTorch wheel index. When non-empty,
# apply_torch_override injects [[tool.uv.index]] + [tool.uv.sources] blocks
# into pyproject.toml so `uv sync` resolves torch/torchvision/torchaudio from
# this index (UV_TORCH_BACKEND alone only affects `uv pip install` /  `uv add`).
PLATFORM_TORCH_INDEX=""
# Package names routed through PLATFORM_TORCH_INDEX. Must include any transitive
# deps that only live on the platform-specific index (e.g. pytorch-triton-rocm
# for ROCm). Set per-platform by configure_<platform>.
PLATFORM_TORCH_PACKAGES=()
# Lines appended to the venv's bin/activate by embodied installers (each is a
# full shell statement, e.g. `export VK_DRIVER_FILES=...`). Populated per-
# platform by configure_<platform>; other targets ignore the array.
PLATFORM_VENV_EXPORTS=()
# Whether the platform supports flash-attn at all. When 0, install_flash_attn
# returns immediately without installing or building anything (e.g. Ascend
# where the kernels are CUDA-only and no NPU equivalent ships in the package).
PLATFORM_FLASH_ATTN_INSTALL=1
# Whether the platform has prebuilt flash-attn wheels available on the
# Dao-AILab GitHub releases. When 0, install_flash_attn skips the wheel and
# does a `uv pip install flash-attn==<ver> --no-build-isolation` source build.
# Only consulted when PLATFORM_FLASH_ATTN_INSTALL=1.
PLATFORM_FLASH_ATTN_PREBUILT=0
# User-level opt-out, set by --no-flash-attn. Wins over the platform default
# so the user can skip flash-attn on platforms where it would otherwise
# install (e.g. when build deps aren't available on the host).
DISABLE_FLASH_ATTN=0
# User-level opt-out for apex, set by --no-apex. Wins over the platform default.
DISABLE_APEX=0
# Platform torchcodec pin; when set it wins over the version-derived one (the
# derived pin has no wheels on e.g. Ascend/aarch64). Set by configure_<platform>.
PLATFORM_TORCHCODEC_SPEC=""
# Whether apply_torch_override should rewrite the pyproject.toml `torchcodec`
# pin from ==0.2 to >=0.5. The ==0.2 line in override-dependencies has wheels
# only for x86_64 + torch 2.5/2.6, so it breaks on AMD (torch 2.8 from rocm
# index) and on Ascend (aarch64). Set per-platform by configure_<platform>.
PLATFORM_RELAX_TORCHCODEC=0
# Extra entries (full PEP 508 specifiers) inserted into the pyproject.toml
# `override-dependencies` array by apply_torch_override. Use this for
# platform-specific transitive pins that aren't in the original file
# (e.g. `"evdev<1.9"` on Ascend where newer evdev fails to build against
# older kernel headers). Set per-platform by configure_<platform>.
PLATFORM_EXTRA_OVERRIDES=()
# Extra flags appended to every `uv sync`. Set by configure_<platform>.
PLATFORM_UV_SYNC_ARGS=()
# Whether the venv exposes the interpreter's system site-packages.
PLATFORM_SYSTEM_SITE_PACKAGES=0
# Name of a function run after the venv is created and activated, before the
# first `uv sync`. Set by configure_<platform>; empty means no hook.
PLATFORM_VENV_HOOK=""
# ERE matching lines to drop from embodied/envs/common.txt, for platforms where
# some of those wheels are unusable. Set by configure_<platform>.
PLATFORM_COMMON_REQ_EXCLUDE_RE=""
# Default torch-backend per platform; user can override by exporting
# UV_TORCH_BACKEND before invoking this script.
DEFAULT_BACKEND_NVIDIA="auto"
# AMD composes UV_TORCH_BACKEND=rocm<version>; --rocm picks the version. When
# unset, configure_amd detects the system's ROCm version and auto-picks the
# minimum torch version on https://download.pytorch.org/whl/torch/ that has a
# matching +rocm<version> wheel.
# Add new platforms by extending SUPPORTED_PLATFORMS, defining
# configure_<platform> + install_<platform>_extras, and routing in their
# respective dispatchers below.
SUPPORTED_PLATFORMS=("nvidia" "amd" "ascend" "musa")
TEST_BUILD=${TEST_BUILD:-0}
# Absolute path to this script (resolves symlinks)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
USE_MIRRORS=0
GITHUB_PREFIX=""
NO_ROOT=0
NO_INSTALL_RLINF_CMD="--no-install-project"
SUPPORTED_TARGETS=("embodied" "agentic" "docs")
SUPPORTED_ENGINES=("sglang" "vllm")
SUPPORTED_MODELS=("openvla" "openvla-oft" "openpi" "gr00t" "gr00t_n1d6" "gr00t_n1d7" "dexbotic" "starvla" "lingbotvla" "dreamzero" "qwen3_vl" "abot_m0" "molmoact2" "evo1")
SUPPORTED_ENVS=("behavior" "maniskill_libero" "libero" "metaworld" "calvin" "isaaclab" "robocasa" "robocasa365" "franka" "franka-dexhand" "franka-franky" "frankasim" "robotwin" "habitat" "opensora" "wan" "genesis" "xsquare_turtle2" "liberopro" "liberoplus" "roboverse" "embodichain" "d4rl" "dosw1" "gim_arm" "dummy" "polaris")

#=======================Utility Functions=======================

print_help() {
        cat <<EOF
Usage: bash install.sh <target> [options]

Targets:
    embodied               Install embodied model and envs (default).
    agentic                Install agentic stack (Megatron etc.).
    docs                   Install documentation requirements.

Options (for target=embodied):
    --model <name>         Embodied model to install: ${SUPPORTED_MODELS[*]}.
    --env <name>           Single environment to install: ${SUPPORTED_ENVS[*]}.

Common options:
    -h, --help             Show this help message and exit.
    --venv <dir>           Virtual environment directory name (default: .venv).
    --torch <version>      Override torch version (e.g., 2.7.0). torchvision/torchaudio are derived
                           automatically (torchvision=0.<minor+15>.<patch>, torchaudio=<torch>).
                           torchcodec is left untouched. Patches pyproject.toml in place for the
                           duration of the install; the original is restored on exit. On
                           --platform amd, defaults to the lowest torch version with a matching
                           +rocm<version> wheel on https://download.pytorch.org/whl/torch/.
    --engine <name>        Rollout engine for target=agentic: ${SUPPORTED_ENGINES[*]}
                           (default: sglang). One venv holds one engine -- they pin the
                           same kernel libraries to different versions, so run install.sh
                           twice with different --venv to get both.
    --sglang <version>     Override sglang version (e.g., 0.5.4). Must be one of the
                           versions in requirements/agentic/; torch is derived from it.
    --vllm <version>       Override vllm version (e.g., 0.23.0). Must be one of the
                           versions in requirements/agentic/.
    --transformers <version> Override transformers version (e.g., 4.57.1). Patches
                           the == pinned version in agentic extras; restored on exit.
    --platform <name>      Hardware platform: nvidia (default, fully tested), amd (experimental,
                           ROCm), ascend (experimental, NPU), or musa (experimental, Moore
                           Threads). Sets UV_TORCH_BACKEND where applicable
                           (auto / rocm<version> / cpu); export UV_TORCH_BACKEND yourself to
                           bypass (e.g. UV_TORCH_BACKEND=cu124). Ascend uses CPU torch from PyPI
                           and adds torch-npu in install_ascend_extras. MUSA installs no torch at
                           all: run it inside the Moore Threads training-suite image and it
                           reuses that image's torch/torch-musa via a --system-site-packages venv.
    --rocm <version>       ROCm version for --platform amd. When unset, auto-detected from the
                           system (/opt/rocm/.info/version, hipconfig, rocminfo). Composes
                           UV_TORCH_BACKEND=rocm<version>. Ignored on other platforms.
    --python <version>     Python version for the venv (e.g. 3.11.14). Defaults to 3.11.14.
                           Must be >=3.10. Some envs (behavior, d4rl) require 3.10 and will override this.
    --use-mirror           Use mirrors for faster downloads.
    --no-root              Avoid system dependency installation for non-root users. Only use this if you are certain system dependencies are already installed.
    --no-flash-attn        Skip flash-attn install. Useful when the host lacks a CUDA build
                           toolchain or when the platform has no flash-attn support
                           (Ascend/MUSA).
    --no-apex              Skip apex install. Useful when Megatron-LM is not needed and
                           CUDA toolchain mismatch prevents download apex of the right version.
    --install-rlinf        Install RLinf itself into the python.
EOF
}

parse_args() {
    if [ "$#" -eq 0 ]; then
        print_help
        exit 0
    fi

    while [ "$#" -gt 0 ]; do
        case "$1" in
            -h|--help)
                print_help
                exit 0
                ;;
            --venv)
                if [ -z "${2:-}" ]; then
                    echo "--venv requires a directory name argument." >&2
                    exit 1
                fi
                VENV_DIR="${2:-}"
                shift 2
                ;;
            --python)
                if [ -z "${2:-}" ]; then
                    echo "--python requires a version argument (e.g. 3.11.14)." >&2
                    exit 1
                fi
                PYTHON_VERSION="${2:-}"
                USER_SET_PYTHON=1
                shift 2
                ;;
            --torch)
                if [ -z "${2:-}" ]; then
                    echo "--torch requires a version argument (e.g. 2.7.0)." >&2
                    exit 1
                fi
                TORCH_VERSION="${2:-}"
                shift 2
                ;;
            --sglang)
                if [ -z "${2:-}" ]; then
                    echo "--sglang requires a version argument (e.g. 0.5.4)." >&2
                    exit 1
                fi
                SGLANG_VERSION="${2:-}"
                shift 2
                ;;
            --vllm)
                if [ -z "${2:-}" ]; then
                    echo "--vllm requires a version argument (e.g. 0.23.0)." >&2
                    exit 1
                fi
                VLLM_VERSION="${2:-}"
                shift 2
                ;;
            --engine)
                if [[ ! " ${SUPPORTED_ENGINES[*]} " =~ " ${2:-} " ]]; then
                    echo "--engine requires one of: ${SUPPORTED_ENGINES[*]}." >&2
                    exit 1
                fi
                ENGINE="${2:-}"
                shift 2
                ;;
            --transformers)
                if [ -z "${2:-}" ]; then
                    echo "--transformers requires a version argument (e.g. 4.57.1)." >&2
                    exit 1
                fi
                TRANSFORMERS_VERSION="${2:-}"
                shift 2
                ;;
            --platform)
                if [ -z "${2:-}" ]; then
                    echo "--platform requires one of: ${SUPPORTED_PLATFORMS[*]}." >&2
                    exit 1
                fi
                PLATFORM="${2:-}"
                shift 2
                ;;
            --rocm)
                if [ -z "${2:-}" ]; then
                    echo "--rocm requires a version argument (e.g. 6.3)." >&2
                    exit 1
                fi
                ROCM_VERSION="${2:-}"
                shift 2
                ;;
            --model)
                if [ -z "${2:-}" ]; then
                    echo "--model requires a model name argument." >&2
                    exit 1
                fi
                MODEL="${2:-}"
                shift 2
                ;;
            --env)
                if [ -n "$ENV_NAME" ]; then
                    echo "Only one --env can be specified." >&2
                    exit 1
                fi
                ENV_NAME="${2:-}"
                shift 2
                ;;
            --use-mirror)
                USE_MIRRORS=1
                shift
                ;;
            --no-root)
                NO_ROOT=1
                shift
                ;;
            --install-rlinf)
                NO_INSTALL_RLINF_CMD=""
                shift
                ;;
            --no-flash-attn)
                DISABLE_FLASH_ATTN=1
                shift
                ;;
            --no-apex)
                DISABLE_APEX=1
                shift
                ;;
            --*)
                echo "Unknown option: $1" >&2
                echo "Use --help to see available options." >&2
                exit 1
                ;;
            *)
                if [ -z "$TARGET" ]; then
                    TARGET="$1"
                    shift
                else
                    echo "Unexpected positional argument: $1" >&2
                    echo "Use --help to see usage." >&2
                    exit 1
                fi
                ;;
        esac
    done

    if [ -z "$TARGET" ]; then
        TARGET="embodied"
    fi
}

validate_python_version() {
    # Reject malformed versions (must be X.Y or X.Y.Z with numeric components).
    if [[ ! "$PYTHON_VERSION" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
        echo "--python must be of form X.Y or X.Y.Z (got '$PYTHON_VERSION')." >&2
        exit 1
    fi

    # Soft-check against pyproject.toml's requires-python = ">=3.10".
    local py_major py_minor _py_patch
    IFS='.' read -r py_major py_minor _py_patch <<< "$PYTHON_VERSION"
    local mm="${py_major}.${py_minor}"
    if [ "$(printf '%s\n3.10\n' "$mm" | sort -V | head -n1)" != "3.10" ]; then
        echo "[install.sh] WARNING: Python ${PYTHON_VERSION} is below the pyproject.toml requires-python minimum (>=3.10). The install may fail." >&2
    fi
}

#=======================PLATFORM CONFIG=======================
# Per-platform runtime env-var configuration. Each configure_<platform> runs
# before any uv operation, so set everything that affects how dependencies
# resolve here (UV_TORCH_BACKEND, indexes, build flags, etc.). All functions
# respect a pre-existing UV_TORCH_BACKEND from the caller's environment.

# Detect installed ROCm version. Prints major.minor on success, returns 1 on
# failure. Probes the standard locations in order of reliability.
detect_rocm_version() {
    local raw=""
    if [ -f /opt/rocm/.info/version ]; then
        raw=$(head -n1 /opt/rocm/.info/version 2>/dev/null)
    fi
    if [ -z "$raw" ] && command -v hipconfig &>/dev/null; then
        raw=$(hipconfig --version 2>/dev/null | head -n1)
    fi
    if [ -z "$raw" ] && command -v rocminfo &>/dev/null; then
        raw=$(rocminfo 2>/dev/null | grep -i 'ROCm Version' | head -n1)
    fi
    [ -z "$raw" ] && return 1

    local mm
    mm=$(echo "$raw" | grep -oE '[0-9]+\.[0-9]+' | head -n1)
    [ -z "$mm" ] && return 1
    echo "$mm"
}

# Find a torch version on the PyTorch wheel index that has a +rocm<rocm_ver>
# Linux x86_64 wheel matching PYTHON_VERSION's cpXY tag. Prefers the smallest
# version >= 2.5; falls back to the highest available wheel if no >= 2.5 wheel
# exists. Uses the NJU mirror (per-ROCm subdir) when --use-mirror is set,
# otherwise the upstream universal index. Echoes X.Y.Z on success, returns 1
# on failure.
detect_torch_for_rocm() {
    local rocm_ver="$1"

    if ! command -v curl &>/dev/null; then
        echo "[install.sh] curl not found; cannot auto-detect torch version." >&2
        return 1
    fi

    local url
    if [ "$USE_MIRRORS" -eq 1 ]; then
        url="https://mirrors.nju.edu.cn/pytorch/whl/rocm${rocm_ver}/torch/"
    else
        url="https://download.pytorch.org/whl/torch/"
    fi

    # Python ABI tag (e.g. 3.11.14 -> cp311). The venv hasn't been created yet
    # at this point, so derive it from the PYTHON_VERSION script global.
    local py_major py_minor _py_patch
    IFS='.' read -r py_major py_minor _py_patch <<< "$PYTHON_VERSION"
    local py_tag="cp${py_major}${py_minor}"

    local html
    html=$(curl -fsSL --max-time 30 "$url" 2>/dev/null) || {
        echo "[install.sh] Failed to fetch ${url}." >&2
        return 1
    }

    # Wheel filenames look like:
    #   torch-2.8.0+rocm6.4-cp311-cp311-manylinux_2_28_x86_64.whl
    # The abi tag may have a trailing 't' (free-threaded build); the platform
    # tag covers manylinux_*_x86_64 / manylinux<digits>_x86_64 / linux_x86_64
    # (NJU and upstream both stick to manylinux_2_28 for recent ROCm wheels,
    # but allow the older tags for forward-compat).
    local rocm_re="${rocm_ver//./\\.}"
    local versions
    versions=$(echo "$html" \
        | grep -oE "torch-[0-9]+\.[0-9]+\.[0-9]+\+rocm${rocm_re}(\.[0-9]+)?-${py_tag}-${py_tag}t?-(manylinux[^-]*|linux)_x86_64\.whl" \
        | sed -E 's/torch-([0-9]+\.[0-9]+\.[0-9]+).*/\1/' \
        | sort -uV)
    [ -z "$versions" ] && return 1

    # Prefer the smallest version >= 2.5.0; otherwise take the highest
    # available wheel (which the project may still reject at install time, but
    # surfaces a usable starting point).
    local picked=""
    while IFS= read -r v; do
        if [ "$(printf '%s\n2.5.0\n' "$v" | sort -V | head -n1)" = "2.5.0" ]; then
            picked="$v"
            break
        fi
    done <<< "$versions"
    if [ -z "$picked" ]; then
        picked=$(echo "$versions" | tail -n1)
    fi
    echo "$picked"
}

# Prints "MAJOR MINOR" (e.g. "12 4") on success, returns 1 if no CUDA is
# available. Probes torch.version.cuda first (safe None check), then falls
# back to nvcc so callers work both before and after the venv is populated.
detect_cuda_major_minor() {
    local mm
    if mm=$(python - <<'EOF' 2>/dev/null
import torch, sys
v = torch.version.cuda
if v is None:
    sys.exit(1)
parts = v.split(".")
print(parts[0], parts[1] if len(parts) > 1 else "0")
EOF
    ); then
        echo "$mm"
        return 0
    fi

    local nvcc_exe=""
    if command -v nvcc &>/dev/null; then
        nvcc_exe=$(command -v nvcc)
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        nvcc_exe="/usr/local/cuda/bin/nvcc"
    fi
    [ -z "$nvcc_exe" ] && return 1
    local ver
    ver=$("$nvcc_exe" --version | grep 'Cuda compilation tools' | awk '{print $5}' | tr -d ',')
    [ -z "$ver" ] && return 1
    echo "${ver%%.*} ${ver#*.}"
}

detect_nvidia_driver_max_cuda() {
    command -v nvidia-smi &>/dev/null || return 1
    local ver maj min
    ver=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+')
    [ -n "$ver" ] || return 1
    maj=${ver%%.*}
    min=${ver#*.}
    echo "$(( maj * 10 + min ))"
}

detect_nvidia_torch_cuda_tag() {
    local torch_ver="$1" index_base="$2" driver_num="$3"
    local ver_re n listing
    ver_re=$(printf '%s' "$torch_ver" | sed 's/\./\\./g')
    for n in 130 129 128 126 124 121 118; do
        [ "$n" -le "$driver_num" ] || continue
        listing=$(curl -fsSL --max-time 60 "${index_base}/cu${n}/torch/" 2>/dev/null) || continue
        if grep -qE "torch-${ver_re}(%2B|\+)cu${n}-" <<< "$listing"; then
            echo "cu${n}"
            return 0
        fi
    done
    return 1
}

configure_nvidia() {
    PLATFORM_TORCH_STR=""
    PLATFORM_TORCH_INDEX=""
    PLATFORM_TORCH_PACKAGES=()
    PLATFORM_VENV_EXPORTS=(
        "export NVIDIA_DRIVER_CAPABILITIES=all"
        "export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json"
        "export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json"
    )
    PLATFORM_FLASH_ATTN_INSTALL=1
    PLATFORM_FLASH_ATTN_PREBUILT=1
    PLATFORM_RELAX_TORCHCODEC=0
    PLATFORM_TORCHCODEC_SPEC=""
    PLATFORM_EXTRA_OVERRIDES=()
    PLATFORM_UV_SYNC_ARGS=()
    PLATFORM_SYSTEM_SITE_PACKAGES=0
    PLATFORM_VENV_HOOK=""
    PLATFORM_COMMON_REQ_EXCLUDE_RE=""
    PLATFORM_CUDA_TAG=""
    local _uvtb_user_set=1
    if [ -z "${UV_TORCH_BACKEND:-}" ]; then
        _uvtb_user_set=0
        export UV_TORCH_BACKEND="$DEFAULT_BACKEND_NVIDIA"
    fi

    local _torch_ver _tmaj _tmin _rest _driver_num _index_base _cuda_tag
    _torch_ver="$TORCH_VERSION"
    if [ -z "$_torch_ver" ] && [ -f "$PYPROJECT_FILE" ]; then
        _torch_ver=$(sed -nE 's/.*"torch==([^"+]+).*".*/\1/p' "$PYPROJECT_FILE" | head -1)
    fi
    _tmaj=${_torch_ver%%.*}
    _rest=${_torch_ver#*.}
    _tmin=${_rest%%.*}
    if [ -n "$_torch_ver" ] && [[ "$_tmaj" =~ ^[0-9]+$ ]] && [[ "$_tmin" =~ ^[0-9]+$ ]] \
        && { [ "$_tmaj" -gt 2 ] || { [ "$_tmaj" -eq 2 ] && [ "$_tmin" -ge 11 ]; }; }; then
        local _cpu_only=0
        # An explicit UV_TORCH_BACKEND=cuXXX wins over driver detection, for
        # hosts with CUDA forward-compatibility libraries installed.
        if [ "$_uvtb_user_set" -eq 1 ] && [[ "$UV_TORCH_BACKEND" =~ ^cu[0-9]+$ ]]; then
            _driver_num="${UV_TORCH_BACKEND#cu}"
            echo "[install.sh] UV_TORCH_BACKEND=${UV_TORCH_BACKEND} was set explicitly; using it instead of the detected driver CUDA version."
        elif ! _driver_num=$(detect_nvidia_driver_max_cuda); then
            local _cmm _cmaj _cmin
            if _cmm=$(detect_cuda_major_minor); then
                read -r _cmaj _cmin <<< "$_cmm"
                _driver_num=$(( _cmaj * 10 + _cmin ))
                echo "[install.sh] No NVIDIA driver detected; using CUDA toolkit ${_cmaj}.${_cmin} (cu${_driver_num}) for the torch build (e.g. docker build)."
            else
                _cpu_only=1
                echo "[install.sh] No NVIDIA driver or CUDA toolkit detected; using CPU-only torch."
            fi
        fi
        if [ "$USE_MIRRORS" -eq 1 ]; then
            _index_base="https://mirrors.tencent.com/pytorch-wheels/whl"
        else
            _index_base="https://download.pytorch.org/whl"
        fi
        if [ "$_cpu_only" -eq 1 ]; then
            PLATFORM_TORCH_STR=""
            PLATFORM_TORCH_INDEX="${_index_base}/cpu"
            PLATFORM_TORCH_PACKAGES=("torch" "torchvision" "torchaudio")
            if [ "$_uvtb_user_set" -eq 0 ]; then
                export UV_TORCH_BACKEND="cpu"
            fi
            echo "[install.sh] Routing torch ${_torch_ver} (cpu) through ${PLATFORM_TORCH_INDEX} (UV_TORCH_BACKEND=${UV_TORCH_BACKEND})."
        elif _cuda_tag=$(detect_nvidia_torch_cuda_tag "$_torch_ver" "$_index_base" "$_driver_num"); then
            PLATFORM_CUDA_TAG="$_cuda_tag"
            PLATFORM_TORCH_STR="+${_cuda_tag}"
            PLATFORM_TORCH_INDEX="${_index_base}/${_cuda_tag}"
            PLATFORM_TORCH_PACKAGES=("torch" "torchvision" "torchaudio")
            if [ "$_uvtb_user_set" -eq 0 ]; then
                export UV_TORCH_BACKEND="$_cuda_tag"
            fi
            echo "[install.sh] Driver supports CUDA <= ${_driver_num}; routing torch ${_torch_ver} (${_cuda_tag}) through ${PLATFORM_TORCH_INDEX} (UV_TORCH_BACKEND=${UV_TORCH_BACKEND})."
        else
            echo "[install.sh] Could not resolve a driver-compatible CUDA wheel for torch ${_torch_ver}; leaving torch on the default index (UV_TORCH_BACKEND=${UV_TORCH_BACKEND})." >&2
        fi
    fi
}

configure_amd() {
    if [ -z "$ROCM_VERSION" ]; then
        ROCM_VERSION=$(detect_rocm_version) || {
            echo "[install.sh] Could not auto-detect ROCm version; pass --rocm explicitly." >&2
            exit 1
        }
        echo "[install.sh] Auto-detected ROCm version: ${ROCM_VERSION}"
    fi

    if [[ ! "$ROCM_VERSION" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
        echo "--rocm must be of form X.Y or X.Y.Z (got '$ROCM_VERSION')." >&2
        exit 1
    fi

    if [ -z "$TORCH_VERSION" ]; then
        TORCH_VERSION=$(detect_torch_for_rocm "$ROCM_VERSION") || {
            echo "[install.sh] No compatible torch wheels found for ROCm ${ROCM_VERSION} (Python ${PYTHON_VERSION}). Pass --torch explicitly." >&2
            exit 1
        }
        echo "[install.sh] Auto-selected torch version for ROCm ${ROCM_VERSION}: ${TORCH_VERSION}"
    fi

    PLATFORM_TORCH_STR="+rocm${ROCM_VERSION}"
    if [ "$USE_MIRRORS" -eq 1 ]; then
        PLATFORM_TORCH_INDEX="https://mirrors.nju.edu.cn/pytorch/whl/rocm${ROCM_VERSION}"
    else
        PLATFORM_TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"
    fi
    # All four packages are routed through the ROCm index (and only that
    # index — see explicit=true on [[tool.uv.index]]). torchvision/torchaudio
    # arrive transitively via vllm/etc.; pytorch-triton-rocm arrives
    # transitively via torch. apply_torch_override promotes them to direct
    # deps in [project.dependencies] so [tool.uv.sources] mappings actually
    # take effect (uv only applies sources to direct deps).
    PLATFORM_TORCH_PACKAGES=("torch" "torchvision" "torchaudio" "pytorch-triton-rocm" "triton-rocm")
    PLATFORM_VENV_EXPORTS=(
        "export AMD_VULKAN_ICD=RADV"
        "export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
        "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
    )
    PLATFORM_FLASH_ATTN_INSTALL=1
    PLATFORM_FLASH_ATTN_PREBUILT=0
    PLATFORM_RELAX_TORCHCODEC=1
    PLATFORM_TORCHCODEC_SPEC=""
    PLATFORM_EXTRA_OVERRIDES=()
    PLATFORM_UV_SYNC_ARGS=()
    PLATFORM_SYSTEM_SITE_PACKAGES=0
    PLATFORM_VENV_HOOK=""
    PLATFORM_COMMON_REQ_EXCLUDE_RE=""
    if [ -z "${UV_TORCH_BACKEND:-}" ]; then
        export UV_TORCH_BACKEND="rocm${ROCM_VERSION}"
    fi
}

configure_ascend() {
    # Ascend NPU uses CPU torch from PyPI plus torch-npu installed via
    # install_ascend_extras. No platform-specific wheel index is needed
    # because there's no ascend-tagged torch on PyTorch's index — torch-npu
    # is the standalone package that adds the NPU backend at runtime.
    PLATFORM_TORCH_STR=""
    PLATFORM_TORCH_INDEX=""
    PLATFORM_TORCH_PACKAGES=()
    PLATFORM_VENV_EXPORTS=()
    # flash-attn is CUDA-only; skip the install entirely on Ascend instead
    # of trying (and failing) to build it from source.
    PLATFORM_FLASH_ATTN_INSTALL=0
    PLATFORM_FLASH_ATTN_PREBUILT=0
    PLATFORM_RELAX_TORCHCODEC=1
    # The derived pin (==0.2 for torch 2.6) is x86_64-only; Ascend is aarch64.
    # Keep a loose pin so uv can *resolve* torchcodec (embodied extra lists it),
    # but do not install the wheel — see PLATFORM_UV_SYNC_ARGS below.
    PLATFORM_TORCHCODEC_SPEC="torchcodec>=0.5"
    PLATFORM_EXTRA_OVERRIDES=()
    # PyPI torchcodec>=0.11 ships CUDA wheels by default (0.16.0 dlopens
    # libnvrtc.so.13). Ascend is CPU torch + torch-npu, so that import raises
    # OSError. GR00T n1.5 only catches ImportError/RuntimeError around
    # `import torchcodec`, which aborts libero_spatial_ppo_gr00t. Skip the
    # package; n1.5 uses decord (built from source on aarch64) instead.
    PLATFORM_UV_SYNC_ARGS=("--no-install-package" "torchcodec")
    PLATFORM_SYSTEM_SITE_PACKAGES=0
    PLATFORM_VENV_HOOK=""
    PLATFORM_COMMON_REQ_EXCLUDE_RE=""
    # torch-npu tracks torch 1:1 and needs a matching CANN (2.11.0 wants CANN
    # 8.5.0), so Ascend stays on torch 2.6. Bump with the hosts' CANN.
    if [ -z "$TORCH_VERSION" ]; then
        TORCH_VERSION="2.6.0"
        echo "[install.sh] ascend: pinning torch ${TORCH_VERSION} to match torch-npu/CANN (pass --torch to override)."
    fi
    if [ -z "${UV_TORCH_BACKEND:-}" ]; then
        # `cpu` keeps `uv pip install torch ...` calls fetching the CPU build
        # from download.pytorch.org/whl/cpu instead of PyPI's CUDA wheel.
        export UV_TORCH_BACKEND="cpu"
    fi
    # evdev's generated ecodes.c references KEY_* constants without including
    # <linux/input-event-codes.h>. On systems where userspace kernel headers
    # split input-event-codes.h out of input.h, the build fails with
    # "KEY_ALL_APPLICATIONS undeclared" etc. Force-include the header for all
    # C compilations during this install so the constants are always visible.
    if [ -f /usr/include/linux/input-event-codes.h ]; then
        export CFLAGS="${CFLAGS:+$CFLAGS }-include /usr/include/linux/input-event-codes.h"
    fi
}

configure_musa() {
    # torch-musa is not on PyPI: it ships preinstalled in the vendor
    # training-suite image. So MUSA installs no torch of its own and instead
    # reuses the image's interpreter and site-packages.
    if [ "$USER_SET_PYTHON" -eq 0 ]; then
        PYTHON_VERSION=$(python - <<'EOF'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
EOF
)
        echo "[install.sh] musa: reusing the image interpreter, python ${PYTHON_VERSION}"
        validate_python_version
    fi
    # uv would otherwise pick a managed interpreter, which has no torch-musa.
    export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-system}"
    # Pin torch to the image's version so the seeded metadata satisfies it.
    if [ -z "$TORCH_VERSION" ]; then
        # Not `import torch`: that needs a device, which a build has not got.
        local _torch_probe
        _torch_probe=$(python - <<'EOF' 2>&1 || true
import importlib.metadata as metadata

print(metadata.version("torch").split("+")[0])
EOF
)
        if [[ "$_torch_probe" =~ ^[0-9]+\.[0-9]+ ]]; then
            TORCH_VERSION="$_torch_probe"
            echo "[install.sh] musa: pinning torch ${TORCH_VERSION} to match the image."
        else
            echo "[install.sh] musa: no torch found for the image interpreter ($(command -v python)); is this a Moore Threads training-suite container?" >&2
            echo "[install.sh] ${_torch_probe}" >&2
            exit 1
        fi
    fi
    PLATFORM_TORCH_STR=""
    # torch is resolved but never installed; the CPU index keeps the
    # multi-GB nvidia-* CUDA wheels out of the resolution.
    if [ "$USE_MIRRORS" -eq 1 ]; then
        PLATFORM_TORCH_INDEX="https://mirrors.tencent.com/pytorch-wheels/whl/cpu"
    else
        PLATFORM_TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    fi
    PLATFORM_TORCH_PACKAGES=("torch" "torchvision" "torchaudio")
    if [ -z "${UV_TORCH_BACKEND:-}" ]; then
        export UV_TORCH_BACKEND="cpu"
    fi
    # RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES comes from MUSAGPUManager, the
    # MUSA libraries are already on the image's LD_LIBRARY_PATH, and the
    # renderer is selected per run by the e2e/example scripts.
    PLATFORM_VENV_EXPORTS=()
    # The image ships MUSA builds of flash-attn and apex, so there is nothing to
    # install; the CUDA sources would not build here anyway.
    PLATFORM_FLASH_ATTN_INSTALL=0
    PLATFORM_FLASH_ATTN_PREBUILT=0
    PLATFORM_RELAX_TORCHCODEC=1
    PLATFORM_TORCHCODEC_SPEC=""
    # The image's torch-musa is built against numpy 1.x.
    PLATFORM_EXTRA_OVERRIDES=("numpy<2")
    # Either MUSA builds that only exist in the image, or CUDA-only kernels:
    # resolve them, never write them into the venv.
    PLATFORM_UV_SYNC_ARGS=("--inexact")
    local pkg
    for pkg in torch torchvision torchaudio torchcodec triton flash-attn \
        deepspeed vllm sglang xgrammar liger-kernel transformer-engine \
        torch-memory-saver ray; do
        PLATFORM_UV_SYNC_ARGS+=("--no-install-package" "$pkg")
    done
    PLATFORM_SYSTEM_SITE_PACKAGES=1
    PLATFORM_VENV_HOOK=seed_musa_torch_metadata
    # The nvidia-* wheels pull a CUDA torch in behind them.
    PLATFORM_COMMON_REQ_EXCLUDE_RE='^[[:space:]]*nvidia-'
}

# uv does not see packages inherited through --system-site-packages, so any
# `uv pip install` needing torch or flash-attn would pull one in and shadow the
# image's MUSA build. Copy their *metadata* (never the files) so uv treats them
# as installed. RECORD is left empty so an uninstall cannot delete the originals.
seed_musa_torch_metadata() {
    VENV_DIR="$VENV_DIR" python - <<'EOF'
import importlib.metadata as metadata
import os
import pathlib
import shutil
import subprocess
import sys

venv_python = pathlib.Path(os.environ["VENV_DIR"]) / "bin" / "python"
venv_site = pathlib.Path(
    subprocess.run(
        [str(venv_python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
)

KEEP = ("METADATA", "WHEEL", "INSTALLER", "top_level.txt", "entry_points.txt")

for name in ("torch", "torch_musa", "torchvision", "torchaudio", "flash_attn"):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    # locate_file("") is the site-packages dir holding the distribution; the
    # dist-info spelling varies (torch-musa vs torch_musa), so glob for it.
    root = pathlib.Path(dist.locate_file(""))
    matches = [
        d
        for variant in {name, name.replace("_", "-"), name.replace("-", "_")}
        for d in root.glob(f"{variant}-{dist.version}.dist-info")
        if d.is_dir()
    ]
    if not matches:
        continue
    src = matches[0]
    if src.parent == venv_site:
        continue  # already the venv's own copy
    dst = venv_site / src.name
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for meta in KEEP:
        if (src / meta).is_file():
            shutil.copy2(src / meta, dst / meta)
    (dst / "RECORD").write_text("")
    print(f"[install.sh] musa: reusing the image's {name}=={dist.version}", file=sys.stderr)
EOF
}


# Envs that need a different torch than the project default (Isaac Sim /
# OmniGibson need 2.5.1) declare it here, so configure_platform and
# apply_torch_override re-point TORCH_VERSION, the wheel index and
# UV_TORCH_BACKEND together instead of a mid-install `uv pip install torch==...`
# that leaves a mixed torch tree. An explicit --torch always wins.
apply_env_default_torch() {
    [ -n "$TORCH_VERSION" ] && return 0
    local env_torch=""
    case "$ENV_NAME" in
        behavior) env_torch="2.5.1" ;;
    esac
    if [ -n "$env_torch" ]; then
        TORCH_VERSION="$env_torch"
        echo "[install.sh] Environment '${ENV_NAME}' pins torch ${TORCH_VERSION}; overriding the project default."
    fi
}

configure_platform() {
    if [[ ! " ${SUPPORTED_PLATFORMS[*]} " =~ " $PLATFORM " ]]; then
        echo "--platform must be one of: ${SUPPORTED_PLATFORMS[*]} (got '$PLATFORM')." >&2
        exit 1
    fi

    if [ -n "$ROCM_VERSION" ] && [ "$PLATFORM" != "amd" ]; then
        echo "[install.sh] WARNING: --rocm is only meaningful with --platform amd; ignoring on platform=${PLATFORM}." >&2
        ROCM_VERSION=""
    fi

    case "$PLATFORM" in
        nvidia)  configure_nvidia ;;
        amd)     configure_amd ;;
        ascend)  configure_ascend ;;
        musa)    configure_musa ;;
    esac
    echo "[install.sh] platform=${PLATFORM}, UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-<unset>}"
}

#=======================PLATFORM EXTRAS=======================
# Per-platform post-install hooks. Each install_<platform>_extras runs after
# the target-specific case finishes (venv populated, target deps installed).
# Keep these symmetric — add platform-specific runtime libs / drivers / kernel
# packages here rather than sprinkling them through target installers.

install_nvidia_extras() {
    : # CUDA torch from PyPI works out of the box; flash-attn/apex are wired
      # into target installers where they are actually used.
}

install_amd_extras() {
    # Some downstream packages (vllm and friends) import `triton` directly even
    # when running on ROCm. pytorch-triton-rocm provides the ROCm runtime but
    # is not importable as `triton`, so install the `triton` shim package at
    # the matching version to satisfy `import triton`. Skipping is safe if
    # pytorch-triton-rocm isn't present — that just means torch-based packages
    # haven't been installed for this target.
    local triton_ver
    triton_ver=$(python - <<'EOF' 2>/dev/null || true
try:
    import importlib.metadata as m
    print(m.version("pytorch-triton-rocm"))
except Exception:
    pass
EOF
)
    if [ -z "$triton_ver" ]; then
        echo "[install.sh] pytorch-triton-rocm not installed; skipping matching triton install."
        return 0
    fi
    echo "[install.sh] Installing triton==${triton_ver} to match pytorch-triton-rocm"
    uv pip install "triton==${triton_ver}" amdsmi
}

install_ascend_extras() {
    # Ascend NPU support comes from torch-npu, a side-car package that
    # registers an NPU backend on torch import. The package version must
    # match the installed torch (torch-npu 2.X.Y → torch 2.X.Y). Skip if
    # torch isn't present (e.g. docs target), so this hook is safe to run
    # for every ascend target.
    local torch_ver
    torch_ver=$(python - <<'EOF' 2>/dev/null || true
try:
    import torch
    print(torch.__version__.split("+")[0])
except Exception:
    pass
EOF
)
    if [ -z "$torch_ver" ]; then
        echo "[install.sh] torch not installed; skipping torch-npu install."
        return 0
    fi
    # torch-npu imports a few packages at runtime (`yaml`, `decorator`) but
    # doesn't declare them in its wheel metadata, so install them explicitly.
    uv pip install pyyaml decorator
    echo "[install.sh] Installing torch-npu==${torch_ver} to match torch"
    uv pip install "torch-npu==${torch_ver}" \
        || (echo "[install.sh] Pinned torch-npu==${torch_ver} failed; falling back to latest compatible build." >&2 \
            && uv pip install torch-npu)
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
        echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> "$VENV_DIR/bin/activate"
    fi
    # A later `uv pip install` (lerobot, GR00T extras, …) may still pull a
    # CUDA torchcodec wheel. Uninstall it when import fails so GR00T's
    # optional `import torchcodec` raises ImportError (caught) rather than
    # OSError: libnvrtc.so.13 (not caught). A wheel that does import is kept.
    if python -c "import torchcodec" >/dev/null 2>&1; then
        echo "[install.sh] torchcodec imports; keeping it."
    elif python -c "import importlib.metadata as m; m.version('torchcodec')" >/dev/null 2>&1; then
        echo "[install.sh] torchcodec is installed but does not import (likely a CUDA wheel without libnvrtc); uninstalling."
        uv pip uninstall torchcodec || true
    fi
}

install_musa_extras() {
    # Nothing to install; just fail here rather than mid-training.
    python - <<'EOF'
import importlib.metadata as metadata
import sys

# Checkable without a device, unlike `import torch`.
missing = []
for name in ("torch", "torch_musa"):
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        missing.append(name)
if missing:
    print(
        "[install.sh] --platform musa requires "
        f"{' and '.join(missing)} to be installed for the image's interpreter. "
        "Run this inside a Moore Threads training-suite container.",
        file=sys.stderr,
    )
    sys.exit(1)

# Only possible where a device is visible, i.e. not in a docker build.
try:
    import torch
    import torch_musa  # noqa: F401

    available = torch.musa.is_available()
except Exception as exc:
    print(
        f"[install.sh] musa: torch {metadata.version('torch')} present; "
        f"skipping the device check ({type(exc).__name__}). This is expected "
        "during a docker build.",
        file=sys.stderr,
    )
else:
    if available:
        print(
            f"[install.sh] musa: torch {torch.__version__}, "
            f"{torch.musa.device_count()} device(s)"
        )
    else:
        print(
            "[install.sh] WARNING: torch_musa is installed but reports no "
            "available MUSA device. At runtime that means the container was "
            "started without `--runtime=mthreads`.",
            file=sys.stderr,
        )
EOF

    # These are renamed between torch releases (nvidia-cuda-runtime-cu12 ->
    # nvidia-cuda-runtime -> ...), so sweep by prefix. nvidia-ml-py is a
    # pure-python NVML binding others import defensively, so keep it.
    local cuda_pkgs
    cuda_pkgs=$(uv pip list --format json 2>/dev/null \
        | grep -oE '"name":"[^"]+"' \
        | sed -e 's/^"name":"//' -e 's/"$//' \
        | grep -E '^(nvidia|cuda)[-_]' \
        | grep -vx 'nvidia-ml-py' \
        | tr '\n' ' ' || true)
    if [ -n "$cuda_pkgs" ]; then
        echo "[install.sh] musa: removing CUDA-only wheels: ${cuda_pkgs}"
        # shellcheck disable=SC2086
        uv pip uninstall $cuda_pkgs || true
    fi
}

install_platform_extras() {
    case "$PLATFORM" in
        nvidia)  install_nvidia_extras ;;
        amd)     install_amd_extras ;;
        ascend)  install_ascend_extras ;;
        musa)    install_musa_extras ;;
    esac
}

PYPROJECT_FILE="$(dirname "$SCRIPT_DIR")/pyproject.toml"
PYPROJECT_BACKUP=""

restore_pyproject() {
    if [ -n "$PYPROJECT_BACKUP" ] && [ -f "$PYPROJECT_BACKUP" ]; then
        mv -f "$PYPROJECT_BACKUP" "$PYPROJECT_FILE"
        PYPROJECT_BACKUP=""
    fi
}

AGENTIC_DEFAULT_ENGINE="sglang"

agentic_latest_version() {
    ls "$SCRIPT_DIR/agentic/" 2>/dev/null \
        | sed -nE "s/^$1_(.*)_(cu1[23])\.txt\$/\1/p" \
        | sort -Vu \
        | tail -n1
}

effective_engine() {
    printf '%s\n' "${ENGINE:-$AGENTIC_DEFAULT_ENGINE}"
}

effective_engine_version() {
    local engine
    engine=$(effective_engine)
    case "$engine" in
        sglang) printf '%s\n' "${SGLANG_VERSION:-$(agentic_latest_version sglang)}" ;;
        vllm)   printf '%s\n' "${VLLM_VERSION:-$(agentic_latest_version vllm)}" ;;
    esac
}

agentic_torch_for_engine() {
    case "$(effective_engine):$(effective_engine_version)" in
        sglang:0.4.6.post5)        echo "2.6.0" ;;
        sglang:0.5.2|sglang:0.5.4) echo "2.8.0" ;;
        vllm:0.8.5)                echo "2.6.0" ;;
        *)                         echo "-" ;;
    esac
}

engine_needs_torch211() {
    [ "$(agentic_torch_for_engine)" = "-" ]
}

agentic_cuda_line() {
    case "${PLATFORM_CUDA_TAG:-}" in
        cu13*) echo "cu13" ;;
        *)     echo "cu12" ;;
    esac
}

agentic_requirements_file() {
    # $1 = engine (sglang|vllm), $2 = version
    local line path
    line=$(agentic_cuda_line)
    path="$SCRIPT_DIR/agentic/$1_$2_${line}.txt"
    if [ ! -f "$path" ]; then
        echo "[install.sh] ERROR: no $1 $2 build for ${line}. Available:" >&2
        ls "$SCRIPT_DIR/agentic/" | sed -nE "s/^$1_(.*)_(cu1[23])\.txt$/  $1 \1 (\2)/p" >&2
        exit 1
    fi
    printf '%s\n' "$path"
}

install_engine_requirements() {
    local req="$1" engine_specs engine_req
    engine_specs=$(sed -n 's/^# engine: //p' "$req")
    local index_args=()
    if [ -n "${PLATFORM_TORCH_INDEX:-}" ]; then
        index_args=(--extra-index-url "$PLATFORM_TORCH_INDEX" --index-strategy unsafe-best-match)
    fi
    env -u UV_TORCH_BACKEND uv pip install "${index_args[@]}" -r "$req"
    if [ -n "$engine_specs" ]; then
        engine_req=$(mktemp)
        printf '%s\n' "$engine_specs" > "$engine_req"
        env -u UV_TORCH_BACKEND uv pip install "${index_args[@]}" --no-deps -r "$engine_req"
        rm -f "$engine_req"
    fi
}

derive_torchcodec_spec() {
    local tmaj="$1" tmin="$2"
    [ "$tmaj" = "2" ] || { echo ""; return 0; }
    case "$tmin" in
        6)  echo "torchcodec==0.2" ;;
        7)  echo "torchcodec>=0.3,<0.6" ;;
        8)  echo "torchcodec>=0.6,<0.8" ;;
        9)  echo "torchcodec>=0.8,<0.10" ;;
        10) echo "torchcodec>=0.10,<0.11" ;;
        11) echo "torchcodec>=0.11,<0.12" ;;
        *)
            if [ "$tmin" -le 5 ] 2>/dev/null; then
                echo "torchcodec==0.1"
            elif [ "$tmin" -ge 12 ] 2>/dev/null; then
                echo "torchcodec>=0.12"
            else
                echo ""
            fi
            ;;
    esac
}

VLLM_VERSION=""

apply_agentic_torch_default() {
    [ "$TARGET" = "agentic" ] || return 0
    # --torch wins; on AMD, configure_amd derives torch from the ROCm version.
    [ -z "$TORCH_VERSION" ] || return 0
    [ "$PLATFORM" != "amd" ] || return 0
    local torch_ver
    torch_ver=$(agentic_torch_for_engine)
    [ "$torch_ver" != "-" ] || return 0
    TORCH_VERSION="$torch_ver"
    echo "[install.sh] agentic: $(effective_engine) $(effective_engine_version) needs torch ${TORCH_VERSION}; pinning it (pass --torch to override)."
}

apply_torch_override() {
    # Fires when --torch is given (rewrite versions), PLATFORM_TORCH_STR is
    # non-empty (append a PEP 440 local segment so uv picks the platform-specific
    # wheel rather than PyPI's CUDA build), PLATFORM_TORCH_INDEX is non-empty
    # (route torch* through a dedicated index for `uv sync`, which doesn't honor
    # UV_TORCH_BACKEND), PLATFORM_RELAX_TORCHCODEC is set (rewrite the
    # torchcodec pin for non-x86_64 / non-CUDA torch combos), or
    # PLATFORM_EXTRA_OVERRIDES has entries (insert extra override pins).

    local _eff_torch="${TORCH_VERSION}"
    if [ -z "$_eff_torch" ] && [ -f "$PYPROJECT_FILE" ]; then
        _eff_torch=$(sed -nE 's/.*"torch==([^"+]+).*".*/\1/p' "$PYPROJECT_FILE" | head -1)
    fi
    local _torchcodec_spec="" _cur_torchcodec=""
    if [ -n "$PLATFORM_TORCHCODEC_SPEC" ]; then
        _torchcodec_spec="$PLATFORM_TORCHCODEC_SPEC"
    elif [ -n "$_eff_torch" ]; then
        local _tmaj _tmin _tpatch
        IFS='.' read -r _tmaj _tmin _tpatch <<< "$_eff_torch"
        _torchcodec_spec=$(derive_torchcodec_spec "$_tmaj" "$_tmin")
    fi
    if [ -f "$PYPROJECT_FILE" ]; then
        _cur_torchcodec=$(sed -nE 's/.*"(torchcodec[<>=!][^"]*)".*/\1/p' "$PYPROJECT_FILE" | head -1)
    fi
    if [ -n "$_torchcodec_spec" ] && [ "$_torchcodec_spec" != "$_cur_torchcodec" ]; then
        PLATFORM_RELAX_TORCHCODEC=1
    fi

    local needs_torch_rewrite=0
    if [ -n "$TORCH_VERSION" ] || [ -n "$PLATFORM_TORCH_STR" ] || [ -n "$PLATFORM_TORCH_INDEX" ]; then
        needs_torch_rewrite=1
    fi
    if [ "$needs_torch_rewrite" -eq 0 ] \
        && [ "$PLATFORM_RELAX_TORCHCODEC" -ne 1 ] \
        && [ ${#PLATFORM_EXTRA_OVERRIDES[@]} -eq 0 ]; then
        return 0
    fi

    if [ ! -f "$PYPROJECT_FILE" ]; then
        echo "Cannot locate pyproject.toml at $PYPROJECT_FILE" >&2
        exit 1
    fi

    PYPROJECT_BACKUP="${PYPROJECT_FILE}.rlinf-torch-bak.$$"
    cp "$PYPROJECT_FILE" "$PYPROJECT_BACKUP"
    trap 'restore_pyproject' EXIT INT TERM HUP

    if [ "$PLATFORM_RELAX_TORCHCODEC" -eq 1 ] && [ -n "$_torchcodec_spec" ]; then
        sed -i -E "s/\"torchcodec[<>=!][^\"]*\"/\"${_torchcodec_spec}\"/" "$PYPROJECT_FILE"
        echo "[install.sh] Set torchcodec override to ${_torchcodec_spec} for torch ${_eff_torch}"
    fi

    if [ ${#PLATFORM_EXTRA_OVERRIDES[@]} -gt 0 ]; then
        # Insert each extra override right after the opening bracket of the
        # override-dependencies array. Done in reverse so the final order
        # matches the array order. The trap restores the original on exit.
        local i
        for (( i=${#PLATFORM_EXTRA_OVERRIDES[@]}-1; i>=0; i-- )); do
            local entry="${PLATFORM_EXTRA_OVERRIDES[i]}"
            sed -i "/^override-dependencies = \\[\$/a\\    \"${entry}\"," "$PYPROJECT_FILE"
        done
        echo "[install.sh] Added override-dependencies entries: ${PLATFORM_EXTRA_OVERRIDES[*]}"
    fi

    if [ "$needs_torch_rewrite" -eq 0 ]; then
        echo "[install.sh] Original pyproject.toml will be restored on exit."
        return 0
    fi

    local torch_version torchvision_version torchaudio_version
    if [ -n "$TORCH_VERSION" ]; then
        local torch_major torch_minor torch_patch
        IFS='.' read -r torch_major torch_minor torch_patch <<< "$TORCH_VERSION"
        if [ "$torch_major" != "2" ] || [ -z "$torch_minor" ] || [ -z "$torch_patch" ]; then
            echo "--torch must be of form 2.Y.Z (got '$TORCH_VERSION')." >&2
            exit 1
        fi
        case "$torch_minor$torch_patch" in
            *[!0-9]*)
                echo "--torch components must be numeric (got '$TORCH_VERSION')." >&2
                exit 1
                ;;
        esac
        local tv_minor=$((torch_minor + 15))
        torch_version="$TORCH_VERSION"
        torchvision_version="0.${tv_minor}.${torch_patch}"
        torchaudio_version="$TORCH_VERSION"
    else
        # Reuse the public versions already pinned in pyproject.toml, stripping
        # any pre-existing local segment so PLATFORM_TORCH_STR can be re-applied cleanly.
        torch_version=$(sed -nE 's/.*"torch==([^"+]+).*".*/\1/p' "$PYPROJECT_FILE" | head -1)
        torchvision_version=$(sed -nE 's/.*"torchvision==([^"+]+).*".*/\1/p' "$PYPROJECT_FILE" | head -1)
        torchaudio_version=$(sed -nE 's/.*"torchaudio==([^"+]+).*".*/\1/p' "$PYPROJECT_FILE" | head -1)
        if [ -z "$torch_version" ] || [ -z "$torchvision_version" ] || [ -z "$torchaudio_version" ]; then
            echo "Could not parse existing torch/torchvision/torchaudio pins from $PYPROJECT_FILE" >&2
            exit 1
        fi
    fi

    local torch_pin="${torch_version}${PLATFORM_TORCH_STR}"
    local torchvision_pin="${torchvision_version}${PLATFORM_TORCH_STR}"
    local torchaudio_pin="${torchaudio_version}${PLATFORM_TORCH_STR}"

    sed -i \
        -e "s/\"torch==[^\"]*\"/\"torch==${torch_pin}\"/" \
        -e "s/\"torchvision==[^\"]*\"/\"torchvision==${torchvision_pin}\"/" \
        -e "s/\"torchaudio==[^\"]*\"/\"torchaudio==${torchaudio_pin}\"/" \
        "$PYPROJECT_FILE"

    echo "[install.sh] Patched pyproject.toml override-dependencies: torch==${torch_pin}, torchvision==${torchvision_pin}, torchaudio==${torchaudio_pin}"

    if [ -n "$PLATFORM_TORCH_INDEX" ]; then
        # `uv sync` does not honor UV_TORCH_BACKEND for resolution, so register
        # the platform-specific wheel index and pin every torch-family package
        # to it. `explicit = true` keeps unrelated packages (e.g. cmake) from
        # being shadowed by stale copies on the PyTorch index. Because
        # [tool.uv.sources] only applies to direct deps, also promote each
        # mapped package to a direct dep in [project.dependencies] (skipping
        # any already declared there). Appended to the file so the existing
        # trap restores the original on exit.
        for pkg in "${PLATFORM_TORCH_PACKAGES[@]}"; do
            if grep -qE "^[[:space:]]*\"${pkg}\\b" "$PYPROJECT_FILE"; then
                continue
            fi
            sed -i "/^dependencies = \\[\$/a\\    \"${pkg}\"," "$PYPROJECT_FILE"
        done

        {
            echo ""
            echo "[[tool.uv.index]]"
            echo "name = \"pytorch-platform\""
            echo "url = \"${PLATFORM_TORCH_INDEX}\""
            echo "explicit = true"
            echo ""
            echo "[tool.uv.sources]"
            for pkg in "${PLATFORM_TORCH_PACKAGES[@]}"; do
                echo "${pkg} = { index = \"pytorch-platform\" }"
            done
        } >> "$PYPROJECT_FILE"
        echo "[install.sh] Routed ${PLATFORM_TORCH_PACKAGES[*]} through index ${PLATFORM_TORCH_INDEX} (explicit; promoted to direct deps as needed)"
    fi

    echo "[install.sh] Original pyproject.toml will be restored on exit."
}

install_uv() {
    # Ensure uv is installed
    if ! command -v uv &> /dev/null; then
        echo "uv command not found. Installing uv..."
        # Check if pip is available
        if ! command -v pip &> /dev/null; then
            echo "pip command not found. Please install pip first." >&2
            exit 1
        fi
        pip_failed=0
        pip install uv || pip_failed=1
        if [ $pip_failed -eq 1 ]; then
            echo "Cannot install uv via pip. Installing uv using installer script..."
            if ! command -v wget &> /dev/null; then
                echo "wget command not found. Please install wget first." >&2
                exit 1
            fi
            
            # If uv already exists in ~/.local/bin, use it
            if [ -f ~/.local/bin/uv ]; then
                echo "uv already exists in ~/.local/bin. Using it..."
            else
                wget -qO- https://astral.sh/uv/install.sh | sh
            fi
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
}

setup_mirror() {
    if [ "$USE_MIRRORS" -eq 1 ]; then
        export USE_MIRRORS
        export GITHUB_PREFIX="${GITHUB_PREFIX:-https://gh-proxy.com/}"
        export UV_PYTHON_INSTALL_MIRROR=${GITHUB_PREFIX}https://github.com/astral-sh/python-build-standalone/releases/download
        export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
        export HF_ENDPOINT=https://hf-mirror.com
        git config --global url."${GITHUB_PREFIX}github.com/".insteadOf "https://github.com/"
        trap 'unset_mirror' EXIT INT TERM HUP
    fi
}

unset_mirror() {
    if [ "$USE_MIRRORS" -eq 1 ]; then
        unset UV_PYTHON_INSTALL_MIRROR
        unset UV_DEFAULT_INDEX
        unset HF_ENDPOINT
        git config --global --unset url."${GITHUB_PREFIX}github.com/".insteadOf "https://github.com/" || true
        unset GITHUB_PREFIX
    fi
}

create_and_sync_venv() {
    local required_python_mm
    required_python_mm="$(echo "$PYTHON_VERSION" | awk -F. '{print $1"."$2}')"
    local venv_args=()
    if [ "$PLATFORM_SYSTEM_SITE_PACKAGES" -eq 1 ]; then
        venv_args+=("--system-site-packages")
    fi

    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "Found existing venv at $VENV_DIR; validating Python version compatibility..."
        # shellcheck disable=SC1090
        source "$VENV_DIR/bin/activate"

        local active_python_mm
        active_python_mm="$(python - <<'EOF'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)"

        if [ "$active_python_mm" != "$required_python_mm" ]; then
            echo "Venv Python version mismatch: required ${required_python_mm}.x (from PYTHON_VERSION=${PYTHON_VERSION}), found ${active_python_mm}.x. Recreating venv..." >&2
            deactivate || true
            rm -rf "$VENV_DIR"

            # Create new venv
            install_uv
            uv venv "$VENV_DIR" --python "$PYTHON_VERSION" "${venv_args[@]}"
            # shellcheck disable=SC1090
            source "$VENV_DIR/bin/activate"
        elif [ "$PLATFORM_SYSTEM_SITE_PACKAGES" -eq 1 ] \
            && ! grep -qi '^include-system-site-packages = true$' "$VENV_DIR/pyvenv.cfg"; then
            echo "Venv at $VENV_DIR does not expose system site-packages, which platform=${PLATFORM} needs; recreating..." >&2
            deactivate || true
            rm -rf "$VENV_DIR"

            install_uv
            uv venv "$VENV_DIR" --python "$PYTHON_VERSION" "${venv_args[@]}"
            # shellcheck disable=SC1090
            source "$VENV_DIR/bin/activate"
        else
            echo "Reusing existing venv at $VENV_DIR"
            install_uv
        fi
    else
        # Create new venv
        install_uv
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION" "${venv_args[@]}"
        # shellcheck disable=SC1090
        source "$VENV_DIR/bin/activate"
    fi
    if [ -n "$PLATFORM_VENV_HOOK" ]; then
        "$PLATFORM_VENV_HOOK"
    fi
    uv sync --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
}

install_flash_attn() {
    local flash_ver="2.7.4.post1"

    if [ "$DISABLE_FLASH_ATTN" -eq 1 ]; then
        echo "[install.sh] --no-flash-attn was specified; skipping flash-attn install."
        return 0
    fi
    if [ "$PLATFORM_FLASH_ATTN_INSTALL" -ne 1 ]; then
        echo "[install.sh] flash-attn is unsupported on platform=${PLATFORM}; skipping install."
        return 0
    fi

    local torch_ge_28
    if torch_ge_28=$(python - <<'EOF' 2>/dev/null
import re
import torch

version = torch.__version__.split("+", 1)[0]
match = re.match(r"^(\d+)\.(\d+)", version)
if match is None:
    print("0")
else:
    major, minor = (int(part) for part in match.groups())
    print("1" if (major, minor) >= (2, 8) else "0")
EOF
    ); then
        if [ "$torch_ge_28" = "1" ]; then
            flash_ver="2.8.3"
        fi
    fi

    local prebuilt_flash_versions=("$flash_ver")

    if [ "$PLATFORM_FLASH_ATTN_PREBUILT" -ne 1 ]; then
        echo "[install.sh] Building flash-attn==${flash_ver} from source on platform=${PLATFORM}..."
        uv pip uninstall flash-attn || true
        FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install "flash-attn==${flash_ver}" --no-build-isolation
        return 0
    fi
    # Detect Python tags
    local py_major py_minor
    py_major=$(python - <<'EOF'
import sys
print(sys.version_info.major)
EOF
)
    py_minor=$(python - <<'EOF'
import sys
print(sys.version_info.minor)
EOF
)
    local py_tag="cp${py_major}${py_minor}"   # e.g. cp311
    local abi_tag="${py_tag}"                 # we assume cpXY-cpXY ABI, adjust if needed

    # Detect torch version (major.minor) and strip dots, e.g. 2.6.0 -> 26
    local torch_mm
    torch_mm=$(python - <<'EOF'
import torch
v = torch.__version__.split("+")[0]
parts = v.split(".")
print(f"{parts[0]}.{parts[1]}")
EOF
)

    # Detect CUDA major, e.g. 12 from 12.4
    local cuda_mm cuda_major
    cuda_mm=$(detect_cuda_major_minor) || {
        echo "[install.sh] Could not detect CUDA version; falling back to source build." >&2
        FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install "flash-attn==${flash_ver}" --no-build-isolation
        return 0
    }
    cuda_major="${cuda_mm%% *}"

    local cu_tag="cu${cuda_major}"            # e.g. cu12
    local torch_tag="torch${torch_mm}"        # e.g. torch2.6

    # Match flash-attn wheel ABI to the currently installed torch build.
    local platform_tag="linux_x86_64"
    local cxx_abi
    cxx_abi=$(python - <<'EOF'
import torch

print("cxx11abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abiFALSE")
EOF
)

    uv pip uninstall flash-attn || true
    local prebuilt_ver base_url wheel_name flash_attn_release_hosts host
    flash_attn_release_hosts=(
        "https://github.com/Dao-AILab/flash-attention/releases/download"
        "https://github.com/RLinf/flash-attention/releases/download"
    )
    for prebuilt_ver in "${prebuilt_flash_versions[@]}"; do
        wheel_name="flash_attn-${prebuilt_ver}+${cu_tag}${torch_tag}${cxx_abi}-${py_tag}-${abi_tag}-${platform_tag}.whl"
        for host in "${flash_attn_release_hosts[@]}"; do
            base_url="${GITHUB_PREFIX}${host}/v${prebuilt_ver}"
            echo "[install.sh] Installing flash-attn prebuilt wheel ${wheel_name} from ${host}..."
            if uv pip install "${base_url}/${wheel_name}"; then
                return 0
            fi
            echo "[install.sh] flash-attn prebuilt wheel v${prebuilt_ver} was unavailable or failed to install from ${host}."
        done
    done
    echo "Flash attn installation via prebuilt wheels failed. Attempting to install from source..."
    FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install "flash-attn==${flash_ver}" --no-build-isolation
}

install_apex() {
    if [ "$DISABLE_APEX" -eq 1 ]; then
        echo "[install.sh] --no-apex was specified; skipping apex install."
        return 0
    fi
    if [ "$PLATFORM" != "nvidia" ]; then
        echo "[install.sh] Skipping apex install on platform=${PLATFORM} (CUDA-only)."
        return 0
    fi
    # Example URL: https://github.com/RLinf/apex/releases/download/25.09/apex-0.1+torch2.6-cp311-cp311-linux_x86_64.whl
    local base_url="${GITHUB_PREFIX}https://github.com/RLinf/apex/releases/download/25.09"

    local py_major py_minor
    py_major=$(python - <<'EOF'
import sys
print(sys.version_info.major)
EOF
)
    py_minor=$(python - <<'EOF'
import sys
print(sys.version_info.minor)
EOF
)

# Detect torch version (major.minor) and strip dots, e.g. 2.6.0 -> 26
    local torch_mm
    torch_mm=$(python - <<'EOF'
import torch
v = torch.__version__.split("+")[0]
parts = v.split(".")
print(f"{parts[0]}.{parts[1]}")
EOF
)
    local torch_tag="torch${torch_mm}"        # e.g. torch2.6
    local py_tag="cp${py_major}${py_minor}"   # e.g. cp311
    local abi_tag="${py_tag}"                 # we assume cpXY-cpXY ABI, adjust if needed
    local platform_tag="linux_x86_64"
    # Newer wheels embed the CUDA major (e.g. +cu13torch2.11); prefer that, then
    # fall back to the legacy torch-only name, then a source build.
    local torch_cu
    torch_cu=$(python - <<'EOF'
import torch
v = (torch.version.cuda or "").split(".")
print("cu" + v[0] if v and v[0] else "")
EOF
)
    local wheel_cu=""
    [ -n "$torch_cu" ] && wheel_cu="apex-0.1+${torch_cu}${torch_tag}-${py_tag}-${abi_tag}-${platform_tag}.whl"
    local wheel_legacy="apex-0.1+${torch_tag}-${py_tag}-${abi_tag}-${platform_tag}.whl"

    uv pip uninstall apex || true
    export NUM_THREADS=$(nproc)
    export NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-"--threads ${NUM_THREADS}"}
    export APEX_PARALLEL_BUILD=${APEX_PARALLEL_BUILD:-${NUM_THREADS}}
    if [ -n "$wheel_cu" ] && uv pip install "${base_url}/${wheel_cu}"; then
        :
    elif uv pip install "${base_url}/${wheel_legacy}"; then
        :
    else
        echo "Apex installation via wheel failed. Attempting to install from source..."
        APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install git+${GITHUB_PREFIX}https://github.com/RLinf/apex.git --no-build-isolation
    fi
}

clone_or_reuse_repo() {
    # Usage: clone_or_reuse_repo ENV_VAR_NAME DEFAULT_DIR GIT_URL [GIT_CLONE_ARGS...]
    # - If ENV_VAR_NAME is set, use it as the checkout location: reuse it when it
    #   already exists (no pull), otherwise clone GIT_URL into it. This lets a single
    #   path be shared across multiple venvs/models — clone once, reuse everywhere
    #   (e.g. set GR00T_PATH so every GR00T venv reuses one Isaac-GR00T clone).
    # - Otherwise, clone GIT_URL (with optional GIT_CLONE_ARGS) into DEFAULT_DIR if it doesn't exist.
    # If env var is not set and the directory already exists as a git repo, check if it is intact and re-clone it if not.
    # The resolved directory path is printed to stdout.
    local env_var_name="$1"
    local default_dir="$2"
    local git_url="$3"
    shift 3

    # Read the value of the environment variable safely under `set -u`.
    local env_value
    env_value="$(printenv "$env_var_name" 2>/dev/null || true)"

    local target_dir
    if [ -n "$env_value" ]; then
        target_dir="$env_value"
        if [ ! -d "$target_dir" ]; then
            echo "$env_var_name=$target_dir does not exist yet; cloning $git_url into it..." >&2
            git clone "$@" "$git_url" "$target_dir" >&2
        else
            echo "Reusing existing checkout at $env_var_name=$target_dir." >&2
            local want_ref="" prev="" arg current_ref
            for arg in "$@"; do
                [ "$prev" = "-b" ] && want_ref="$arg"
                prev="$arg"
            done
            if [ -n "$want_ref" ] && [ -d "$target_dir/.git" ]; then
                current_ref=$(git -C "$target_dir" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
                if [ -n "$current_ref" ] && [ "$current_ref" != "$want_ref" ]; then
                    echo "[install.sh] WARNING: $env_var_name=$target_dir is on '${current_ref}', but this install asked for '${want_ref}'. Unset $env_var_name to have install.sh clone the right one." >&2
                fi
            fi
        fi
    else
        target_dir="$default_dir"
        if [ ! -d "$target_dir" ]; then
            git clone "$@" "$git_url" "$target_dir" >&2
        elif [ -d "$target_dir/.git" ]; then
            echo "Checking git repo $target_dir..." >&2
            local git_intact=1
            git -C "$target_dir" status --porcelain >/dev/null 2>&1 || git_intact=0
            if [ $git_intact -eq 1 ]; then
                echo "Git repo $target_dir is intact." >&2
            else
                echo "Git repo $target_dir is corrupted. Re-cloning..." >&2
                rm -rf "$target_dir"
                git clone "$@" "$git_url" "$target_dir" >&2
            fi
        fi
    fi

    printf '%s\n' "$(realpath "$target_dir")"
}

#=======================EMBODIED INSTALLERS=======================
assert_transformers_version() {
    local expected="$1"
    python - "$expected" <<'EOF'
from importlib.metadata import version
import sys

expected = sys.argv[1]
actual = version("transformers")
if actual != expected:
    raise SystemExit(f"Expected transformers=={expected}, found {actual}.")
EOF
}

install_qwen3_vl_sglang_deps() {
    local missing_tags=()
    [ -z "$TORCH_VERSION" ] && missing_tags+=("--torch 2.8.0")
    [ -z "$SGLANG_VERSION" ] && missing_tags+=("--sglang 0.5.4")
    [ -z "$TRANSFORMERS_VERSION" ] && missing_tags+=("--transformers 4.57.1")
    if [ ${#missing_tags[@]} -ne 0 ]; then
        echo "[install.sh] Qwen3-VL SGLang reward serving requires explicit install tags: ${missing_tags[*]}" >&2
        exit 1
    fi

    uv sync --extra agentic --inexact --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
    install_engine_requirements "$(agentic_requirements_file sglang "$SGLANG_VERSION")"
    uv pip install "transformers==${TRANSFORMERS_VERSION}"
    python - "$TORCH_VERSION" "$SGLANG_VERSION" "$TRANSFORMERS_VERSION" <<'EOF'
from importlib.metadata import version
import sys

from packaging.version import Version
import sglang_router
import torch

expected_torch, expected_sglang, expected_transformers = sys.argv[1:4]
actual_torch = torch.__version__.split("+", 1)[0]
if actual_torch != expected_torch:
    raise SystemExit(f"Expected torch=={expected_torch}, found {torch.__version__}.")
actual_sglang = Version(version("sglang"))
expected_sglang_version = Version(expected_sglang)
if actual_sglang != expected_sglang_version:
    raise SystemExit(f"Expected sglang=={expected_sglang_version}, found {actual_sglang}.")
actual_transformers = version("transformers")
if actual_transformers != expected_transformers:
    raise SystemExit(
        f"Expected transformers=={expected_transformers}, found {actual_transformers}."
    )
version("sglang-router")
assert sglang_router is not None
EOF
}

install_common_embodied_deps() {
    uv sync --extra embodied --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
    if [ -n "$PLATFORM_COMMON_REQ_EXCLUDE_RE" ]; then
        grep -Ev "$PLATFORM_COMMON_REQ_EXCLUDE_RE" "$SCRIPT_DIR/embodied/envs/common.txt" \
            | uv pip install -r -
    else
        uv pip install -r $SCRIPT_DIR/embodied/envs/common.txt
    fi
    if [ "$NO_ROOT" -eq 0 ]; then
        bash $SCRIPT_DIR/sys_deps.sh "$PLATFORM"
    fi
    if [ ${#PLATFORM_VENV_EXPORTS[@]} -gt 0 ]; then
        printf '%s\n' "${PLATFORM_VENV_EXPORTS[@]}" >> "$VENV_DIR/bin/activate"
    fi
}

is_aarch64_platform() {
    case "$(uname -m)" in
        aarch64|arm64)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

maybe_build_decord_from_source() {
    is_aarch64_platform || return 0

    local installed_version
    installed_version=$(python - <<'EOF'
try:
    import importlib.metadata as metadata
    print(metadata.version("decord"))
except Exception:
    pass
EOF
)
    if [ "$installed_version" = "0.6.0" ]; then
        echo "[install.sh] decord ${installed_version} already installed; skipping source build."
        return 0
    fi

    # The build needs cmake + a C/C++ toolchain from sys_deps.sh, which is
    # skipped under --no-root. Fail early with an actionable message instead of
    # a cryptic mid-build error.
    local tool
    for tool in cmake make cc; do
        if ! command -v "$tool" &>/dev/null; then
            echo "[install.sh] '$tool' not found, required to build decord from source on $(uname -m)." >&2
            echo "[install.sh] Install the build toolchain (run sys_deps.sh, i.e. drop --no-root) or set DECORD_PATH to a pre-built decord checkout." >&2
            return 1
        fi
    done

    echo "[install.sh] Building decord==0.6.0 from source on $(uname -m)..."
    local decord_path
    decord_path=$(clone_or_reuse_repo DECORD_PATH "$VENV_DIR/decord" https://github.com/dmlc/decord.git -b v0.6.0 --recurse-submodules)

    mkdir -p "$decord_path/build"
    (
        cd "$decord_path/build"
        cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
        make -j"$(nproc)"
    )
    uv pip install "$decord_path/python" --no-build-isolation
}

install_openvla_model() {
    case "$ENV_NAME" in
        maniskill_libero|libero)
            create_and_sync_venv
            install_common_embodied_deps
            install_${ENV_NAME}_env
            ;;
        frankasim)
            create_and_sync_venv
            install_common_embodied_deps
            install_frankasim_env
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for OpenVLA model." >&2
            exit 1
            ;;
    esac
    uv pip install git+${GITHUB_PREFIX}https://github.com/openvla/openvla.git --no-build-isolation
    install_flash_attn
    uv pip uninstall pynvml || true
}

install_openvla_oft_model() {
    case "$ENV_NAME" in
        behavior)
            PYTHON_VERSION="3.10"
            create_and_sync_venv
            install_common_embodied_deps
            uv pip install git+${GITHUB_PREFIX}https://github.com/moojink/openvla-oft.git  --no-build-isolation
            install_behavior_env
            pushd ~ >/dev/null
            install_flash_attn
            popd >/dev/null
            ;;
        maniskill_libero|libero)
            create_and_sync_venv
            install_common_embodied_deps
            install_${ENV_NAME}_env
            install_flash_attn
            uv pip install git+${GITHUB_PREFIX}https://github.com/moojink/openvla-oft.git  --no-build-isolation
            ;;
        metaworld)
            create_and_sync_venv
            install_common_embodied_deps
            install_flash_attn
            install_metaworld_env
            uv pip install git+${GITHUB_PREFIX}https://github.com/moojink/openvla-oft.git  --no-build-isolation
            ;;
        calvin)
            create_and_sync_venv
            install_common_embodied_deps
            install_flash_attn
            install_calvin_env
            uv pip install git+${GITHUB_PREFIX}https://github.com/moojink/openvla-oft.git  --no-build-isolation
            ;;
        robotwin)
            create_and_sync_venv
            install_common_embodied_deps
            install_flash_attn
            uv pip install git+${GITHUB_PREFIX}https://github.com/RLinf/openvla-oft.git@RLinf/v0.1  --no-build-isolation
            install_robotwin_env
            ;;
        opensora)
            create_and_sync_venv
            install_common_embodied_deps
            install_maniskill_libero_env
            install_opensora_world_model
            install_flash_attn
            uv pip install git+${GITHUB_PREFIX}https://github.com/moojink/openvla-oft.git
            ;;
        wan)
            create_and_sync_venv
            install_common_embodied_deps
            install_maniskill_libero_env
            install_wan_world_model
            install_flash_attn
            uv pip install git+${GITHUB_PREFIX}https://github.com/moojink/openvla-oft.git
            ;;
        liberopro)
            create_and_sync_venv
            install_common_embodied_deps
            install_liberopro_env
            install_flash_attn
            uv pip install git+${GITHUB_PREFIX}https://github.com/moojink/openvla-oft.git  --no-build-isolation
            ;;
        liberoplus)
            create_and_sync_venv
            install_common_embodied_deps
            install_liberoplus_env
            install_flash_attn
            uv pip install git+${GITHUB_PREFIX}https://github.com/moojink/openvla-oft.git  --no-build-isolation
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for OpenVLA-OFT model." >&2
            exit 1
            ;;
    esac
    uv pip uninstall pynvml || true
}

install_openpi_model() {
    case "$ENV_NAME" in
        behavior)
            PYTHON_VERSION="3.10"
            create_and_sync_venv
            install_common_embodied_deps
            uv pip install "rlinf-openpi==0.1.1"
            install_behavior_env
            uv pip install protobuf==6.33.0
            pushd ~ >/dev/null
            install_flash_attn
            popd >/dev/null
            ;;
        maniskill_libero|libero)
            create_and_sync_venv
            install_common_embodied_deps
            install_${ENV_NAME}_env
            uv pip install "rlinf-openpi==0.1.1"
            install_flash_attn
            ;;
        metaworld)
            create_and_sync_venv
            install_common_embodied_deps
            uv pip install "rlinf-openpi==0.1.1"
            install_flash_attn
            install_metaworld_env
            ;;
        calvin)
            create_and_sync_venv
            install_common_embodied_deps
            uv pip install "rlinf-openpi==0.1.1"
            install_flash_attn
            install_calvin_env
            ;;
        robocasa)
            create_and_sync_venv
            install_common_embodied_deps
            uv pip install "rlinf-openpi==0.1.1"
            install_flash_attn
            install_robocasa_env
            ;;
        robocasa365)
            create_and_sync_venv
            install_common_embodied_deps
            uv pip install git+${GITHUB_PREFIX}https://github.com/RLinf/openpi
            install_flash_attn
            install_robocasa365_env
            ;;
        robotwin)
            create_and_sync_venv
            install_common_embodied_deps
            uv pip install "rlinf-openpi==0.1.1"
            install_flash_attn
            install_robotwin_env
            ;;
        isaaclab)
            create_and_sync_venv
            install_common_embodied_deps
            uv pip install "rlinf-openpi==0.1.1"
            install_isaaclab_env
            # Torch is modified in Isaac Lab, install flash-attn afterwards
            install_flash_attn
            uv pip install numpydantic==1.7.0 pydantic==2.11.7 numpy==1.26.0
            ;;
        roboverse)
            create_and_sync_venv
            install_common_embodied_deps
            uv pip install "rlinf-openpi==0.1.1"
            install_flash_attn
            install_roboverse_env
            ;;
        franka-franky)
            create_and_sync_venv
            install_common_embodied_deps
            uv sync --extra franka --inexact --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
            if [ "$NO_ROOT" -eq 0 ]; then
                bash $SCRIPT_DIR/embodied/franky_install.sh
            fi
            install_franka_franky_env
            uv pip install "rlinf-openpi==0.1.1"
            install_flash_attn
            ;;
        polaris)
            create_and_sync_venv
            install_common_embodied_deps
            install_polaris_env
            uv pip install "rlinf-openpi==0.1.1"
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for OpenPI model." >&2
            exit 1
            ;;
    esac

    # Enforce RLinf-compatible runtime pins to avoid known breakages.
    # openpi/orbax require jax.experimental.layout.DeviceLocalLayout (removed in jax>=0.7.0).
    uv pip install -r "$SCRIPT_DIR/embodied/models/openpi.txt"

    # Replace transformers models with OpenPI's modified versions
    local py_major_minor
    py_major_minor=$(python - <<'EOF'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)
    cp -r "$VENV_DIR/lib/python${py_major_minor}/site-packages/openpi/models_pytorch/transformers_replace/"* \
        "$VENV_DIR/lib/python${py_major_minor}/site-packages/transformers/"
    
    bash $SCRIPT_DIR/embodied/download_assets.sh --assets openpi
    # rlinf-openpi pulls rlinf-transformer-openpi (a transformers 4.53 fork) into
    # the same package dir as the stock transformers, so two distributions claim
    # `transformers` with conflicting tokenizers bounds. The fork's files win at
    # runtime, so re-assert its bound; otherwise a later resolve drifts tokenizers
    # past 0.22 and transformers refuses to import.
    uv pip install "tokenizers>=0.21,<0.22"
    uv pip uninstall pynvml || true
}

install_molmoact2_model() {
    case "$ENV_NAME" in
        maniskill_libero|libero)
            create_and_sync_venv
            install_common_embodied_deps
            install_${ENV_NAME}_env
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for MolmoAct2 model." >&2
            exit 1
            ;;
    esac

    # RLinf's LeRobot fork carries the MolmoAct2 inference branch on top of
    # huggingface/lerobot, with the Python 3.11 backports and the NumPy 1.x /
    # transformers pins the LIBERO stack needs (branch RLinf/molmoact2-hf-inference).
    local molmoact2_lerobot_path
    molmoact2_lerobot_path=$(clone_or_reuse_repo MOLMOACT2_LEROBOT_PATH "$VENV_DIR/lerobot" https://github.com/RLinf/lerobot.git -b "${MOLMOACT2_LEROBOT_REF:-RLinf/molmoact2-hf-inference}" --depth 1)

    uv pip install "$molmoact2_lerobot_path"

    uv pip uninstall pynvml || true
}

install_starvla_model() {
    case "$ENV_NAME" in
        maniskill_libero|libero)
            create_and_sync_venv
            install_common_embodied_deps
            install_${ENV_NAME}_env
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for StarVLA model." >&2
            exit 1
            ;;
    esac

    local starvla_path
    starvla_path=$(clone_or_reuse_repo STARVLA_PATH "$VENV_DIR/starVLA" https://github.com/starVLA/starVLA.git -b "${STARVLA_GIT_REF:-starVLA-1.2}" --depth 1)

    # Prefer upstream StarVLA requirements first when available.
    if [ -f "$starvla_path/requirements.txt" ]; then
        uv pip install -r "$starvla_path/requirements.txt"
    fi

    # Enforce RLinf-compatible runtime pins to avoid known breakages.
    uv pip install -r "$SCRIPT_DIR/embodied/models/starvla.txt"
    uv pip install -e "$starvla_path" --no-deps

    # Some StarVLA revisions call logger.log() on an overwatch logger that only
    # provides warning/info/error. Keep this patch guarded and optional.
    local framework_init="$starvla_path/starVLA/model/framework/__init__.py"
    if [ "${STARVLA_SKIP_LOGGER_PATCH:-0}" != "1" ] && [ -f "$framework_init" ]; then
        if grep "logger\\.log\\(" "$framework_init" >/dev/null 2>&1; then
            sed -i 's/logger\.log(/logger.warning(/g' "$framework_init"
        fi
    fi

    install_flash_attn
    uv pip uninstall pynvml || true
}

install_evo1_model() {
    # Evo-1 = InternVL3-1B + flow-matching action head (branch: evo1-flash).
    case "$ENV_NAME" in
        maniskill_libero|libero)
            create_and_sync_venv
            install_common_embodied_deps
            install_${ENV_NAME}_env
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for Evo-1 model." >&2
            exit 1
            ;;
    esac

    local evo1_path
    evo1_path=$(clone_or_reuse_repo EVO1_PATH "$VENV_DIR/Evo-1" https://github.com/RLinf/Evo-1.git -b "${EVO1_GIT_REF:-evo1-flash}" --depth 1)

    uv pip install -r "$SCRIPT_DIR/embodied/models/evo1.txt"

    # The RLinf fork carries packaging metadata, so 'import scripts.Evo1' and
    # 'import config' resolve from a normal install. Editable, because Evo-1
    # resolves its dataset cache relative to __file__.
    uv pip install -e "$evo1_path"

    install_flash_attn
    uv pip uninstall pynvml || true
}

install_gr00t_model() {
    create_and_sync_venv
    install_common_embodied_deps

    local gr00t_path
    gr00t_path=$(clone_or_reuse_repo GR00T_PATH "$VENV_DIR/gr00t" https://github.com/NVIDIA/Isaac-GR00T.git -b n1.5-release)
    uv pip install -e "$gr00t_path" --no-deps
    maybe_build_decord_from_source
    uv pip install -r "$SCRIPT_DIR/embodied/models/gr00t.txt"
    case "$ENV_NAME" in
        maniskill_libero|libero)
            install_${ENV_NAME}_env
            install_flash_attn
            uv pip install -r "$SCRIPT_DIR/embodied/models/gr00t.txt"
            ;;
        isaaclab)
            install_isaaclab_env
            # Torch is modified in Isaac Lab, install flash-attn afterwards
            install_flash_attn
            uv pip install numpydantic==1.7.0 pydantic==2.11.7 numpy==1.26.0
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for Gr00t model." >&2
            exit 1
            ;;
    esac
    if [ "$PLATFORM" = "ascend" ]; then
        echo "[install.sh] Applying Ascend GR00T compatibility pins"
        uv pip install -r "$SCRIPT_DIR/embodied/models/ascend/gr00t.txt"
    fi
    uv pip uninstall pynvml || true
}

install_gr00t_n1d6_model() {
    create_and_sync_venv
    install_common_embodied_deps

    local gr00t_path
    gr00t_path=$(clone_or_reuse_repo GR00T_PATH "$VENV_DIR/gr00t" "https://github.com/RLinf/Isaac-GR00T.git" -b n1.6.1-release)
    uv pip install -e "$gr00t_path" --no-deps
    uv pip install -r "$SCRIPT_DIR/embodied/models/gr00t_n1d6.txt"

    case "$ENV_NAME" in
        maniskill_libero)
            install_maniskill_libero_env
            install_flash_attn
            ;;
        *)
            echo "Environment '$ENV_NAME' is not yet validated for Gr00t 1.6." >&2
            exit 1
            ;;
    esac

    uv pip uninstall pynvml || true
}

install_gr00t_n1d7_model() {
    create_and_sync_venv
    install_common_embodied_deps

    local gr00t_path
    gr00t_path=$(clone_or_reuse_repo GR00T_PATH "$VENV_DIR/gr00t" "https://github.com/NVIDIA/Isaac-GR00T.git" -b n1.7-release)
    uv pip install -e "$gr00t_path" --no-deps
    uv pip install -r "$SCRIPT_DIR/embodied/models/gr00t_n1d7.txt"

    case "$ENV_NAME" in
        maniskill_libero)
            install_maniskill_libero_env
            install_flash_attn
            ;;
        *)
            echo "Environment '$ENV_NAME' is not yet validated for Gr00t N1.7." >&2
            exit 1
            ;;
    esac

    uv pip uninstall pynvml || true
}

install_dexbotic_model() {
    case "$ENV_NAME" in
        maniskill_libero|libero)
            create_and_sync_venv
            install_common_embodied_deps

            local dexbotic_path
            dexbotic_path=$(clone_or_reuse_repo DEXBOTIC_PATH "$VENV_DIR/dexbotic" https://github.com/dexmal/dexbotic.git -b 0.2.0)
            uv pip install -e "$dexbotic_path"

            install_${ENV_NAME}_env
            uv pip install transformers==4.53.2
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for Dexbotic model." >&2
            exit 1
            ;;
    esac
    uv pip uninstall pynvml || true
}

install_lingbot_vla_model() {
    create_and_sync_venv
    install_common_embodied_deps
    local lingbotvla_dir
    lingbotvla_dir=$(clone_or_reuse_repo LINGBOT_PATH "$VENV_DIR/lingbot-vla" ${GITHUB_PREFIX}https://github.com/RLinf/lingbot-vla.git --recurse-submodules)
    uv pip install -e $lingbotvla_dir
    uv pip install -r $lingbotvla_dir/requirements.txt
    uv pip install -e $lingbotvla_dir/lingbotvla/models/vla/vision_models/lingbot-depth/ --no-deps
    uv pip install -e $lingbotvla_dir/lingbotvla/models/vla/vision_models/MoGe --no-deps

    install_lerobot
    env -u UV_TORCH_BACKEND uv pip install -r $SCRIPT_DIR/embodied/models/lingbotvla.txt

    case "$ENV_NAME" in
        robotwin)
            install_robotwin_env
            install_flash_attn
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for Lingbot-VLA model." >&2
            exit 1
            ;;
    esac
    uv pip uninstall pynvml || true
}

install_abot_m0_model() {
    create_and_sync_venv
    install_common_embodied_deps

    local abot_path
    local vggt_path
    abot_path=$(clone_or_reuse_repo ABOT_PATH "$VENV_DIR/abot" https://github.com/RLinf/ABot-Manipulation.git)
    vggt_path=$(clone_or_reuse_repo VGGT_PATH "$VENV_DIR/vggt" https://github.com/RLinf/vggt.git)

    uv pip install -e "$vggt_path"

    uv pip install -e "$abot_path" --no-deps

    uv pip install -r $SCRIPT_DIR/embodied/models/abot.txt

    install_flash_attn

    case "$ENV_NAME" in
        maniskill_libero)
            install_maniskill_libero_env
            ;;
        robocasa)
            install_robocasa_env
            ;;
        robotwin)
            install_robotwin_env
            ;;
        liberoplus)
            install_liberoplus_env
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for ABot-M0 model." >&2
            exit 1
            ;;
    esac

    # Keep ABot-M0 runtime PEFT on the expected version after all installs.
    uv pip install peft==0.18.1

    uv pip uninstall pynvml || true
}

install_dreamzero_deps() {
    local dreamzero_path
    dreamzero_path=$(clone_or_reuse_repo DREAMZERO_PATH "$VENV_DIR/dreamzero" https://github.com/dreamzero0/dreamzero.git)
    if [ -z "${DREAMZERO_PATH:-}" ]; then
        git -C "$dreamzero_path" checkout "${DREAMZERO_GIT_REF:-ab790c198fbce33503358efbbd4187ce9a89adf3}" >&2
    fi

    maybe_build_decord_from_source
    uv pip install -r "$SCRIPT_DIR/embodied/models/dreamzero.txt"
    python -m pip install -e "$dreamzero_path" --no-deps --ignore-requires-python
}

install_dreamzero_model() {
    case "$ENV_NAME" in
        behavior)
            # BEHAVIOR/OmniGibson currently requires Python 3.10 and installs
            # its own Torch 2.5.1 stack inside install_behavior_env.
            PYTHON_VERSION="3.10"
            create_and_sync_venv
            install_common_embodied_deps
            install_behavior_env
            install_dreamzero_deps
            pushd ~ >/dev/null
            install_flash_attn
            popd >/dev/null
            ;;
        maniskill_libero|libero)
            create_and_sync_venv
            install_common_embodied_deps
            install_${ENV_NAME}_env
            install_dreamzero_deps
            install_flash_attn
            ;;
        "")
            create_and_sync_venv
            install_common_embodied_deps
            install_dreamzero_deps
            install_flash_attn
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for DreamZero model." >&2
            exit 1
            ;;
    esac
}

install_qwen3_vl_model() {
    create_and_sync_venv
    install_common_embodied_deps

    case "$ENV_NAME" in
        maniskill_libero|libero)
            install_${ENV_NAME}_env
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for Qwen3-VL model." >&2
            exit 1
            ;;
    esac

    install_qwen3_vl_sglang_deps

    install_flash_attn
}

install_lerobot() {
    env -u UV_TORCH_BACKEND uv pip install \
        "git+${GITHUB_PREFIX}https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}"
}

install_franka_realworld_env() {
    uv sync --extra franka --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
    install_lerobot
    if [ "$SKIP_ROS" -ne 1 ]; then
        if [ "$NO_ROOT" -eq 0 ]; then
            bash $SCRIPT_DIR/embodied/ros_install.sh
        fi
        install_franka_env
    fi
}

install_env_only() {
    if [ "$ENV_NAME" = "d4rl" ]; then
        PYTHON_VERSION="3.10"
    fi
    create_and_sync_venv
    SKIP_ROS=${SKIP_ROS:-0}
    case "$ENV_NAME" in
        d4rl)
            install_d4rl_env
            ;;
        dummy)
            install_dummy_env
            ;;
        franka)
            install_franka_realworld_env
            ;;
        franka-dexhand)
            install_franka_realworld_env
            install_franka_dexhand_deps
            ;;
        franka-franky)
            uv sync --extra franka --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
            if [ "$NO_ROOT" -eq 0 ]; then
                bash $SCRIPT_DIR/embodied/franky_install.sh
            fi
            install_franka_franky_env
            ;;
        xsquare_turtle2)
            uv sync --extra xsquare_turtle2 --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
            install_xsquare_turtle2_env
            ;;
        habitat)
            install_common_embodied_deps
            install_habitat_env
            ;;
        genesis)
            install_common_embodied_deps
            install_genesis_env
            ;;
        embodichain)
            install_common_embodied_deps
            install_embodichain_env
            ;;
        gim_arm)
            uv sync --extra gim_arm --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
            ;;
        dosw1)
            install_dosw1_env
            ;;
        polaris)
            install_polaris_env
            ;;
        libero|maniskill_libero)
            install_common_embodied_deps
            install_${ENV_NAME}_env
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for env-only installation." >&2
            exit 1
            ;;
    esac
}

#=======================ENV INSTALLERS=======================

install_dummy_env() {
    uv sync --extra embodied --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
}

# LIBERO and its forks cache absolute paths in ~/.libero, ~/.liberopro and
# ~/.liberoplus on first import and never refresh them, so a config left by an
# earlier install points at assets this venv does not have.
# LIBERO resolves its data paths with os.path.realpath(__file__), so under
# UV_LINK_MODE=symlink (what CI uses) they point into the shared uv cache
# archive, while libero-download-assets writes into site-packages. Reinstall the
# package with copied files so both agree.
materialize_package_files() {
    local dist="$1" ver
    ver=$(uv pip show "$dist" 2>/dev/null | sed -n 's/^Version: //p')
    [ -n "$ver" ] || return 0
    UV_LINK_MODE=copy uv pip install --force-reinstall --no-deps "${dist}==${ver}"
}

reset_libero_config() {
    python - <<'EOF'
import importlib

for package in ("libero.libero", "liberopro.liberopro", "liberoplus.liberoplus"):
    try:
        module = importlib.import_module(package)
    except ImportError:
        continue
    module.set_libero_default_path()
    print(f"[install.sh] Reset {package} config paths to the installed package")
EOF
}

retry_cmd() {
    local max=5 delay=15 attempt=1
    until "$@"; do
        if [ "$attempt" -ge "$max" ]; then
            echo "[install.sh] '$*' failed after ${max} attempts" >&2
            return 1
        fi
        local wait=$((delay + RANDOM % 10))
        echo "[install.sh] '$*' failed (attempt ${attempt}/${max}); retrying in ${wait}s" >&2
        sleep "$wait"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
}

install_libero_env() {
    uv pip install rlinf-libero
    materialize_package_files rlinf-libero
    retry_cmd libero-download-assets --skip-existing
    reset_libero_config
}

install_maniskill_libero_env() {
    install_libero_env
    # The largest git fetch in the install; truncates on slow links.
    retry_cmd uv pip install git+${GITHUB_PREFIX}https://github.com/haosulab/ManiSkill.git@v3.0.0b22

    bash $SCRIPT_DIR/embodied/download_assets.sh --assets maniskill
}

install_d4rl_env() {
    # Install base embodied dependencies first (gym/gymnasium/transformers stack).
    uv sync --extra embodied --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD

    uv pip install "cython<3.0"
    uv pip install "gym==0.23.1"
    uv pip install "d4rl @ git+${GITHUB_PREFIX}https://github.com/Dps799/D4RL@master"

    # Install MuJoCo 2.1.0 native library (mujoco-py only provides Python bindings).
    local mujoco_root="${MUJOCO_PATH:-$HOME/.mujoco}"
    local mujoco_dir="$mujoco_root/mujoco210"
    if [ -f "$mujoco_dir/bin/libmujoco210.so" ]; then
        echo "[install_d4rl_env] MuJoCo 2.1.0 already installed at $mujoco_dir, skipping download."
    else
        echo "[install_d4rl_env] Downloading and extracting MuJoCo 2.1.0..."
        mkdir -p "$mujoco_root"
        local tmpdir archive url extracted
        tmpdir=$(mktemp -d)
        archive="$tmpdir/mujoco210.tar.gz"
        if [ -n "$GITHUB_PREFIX" ]; then
            url="${GITHUB_PREFIX}github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz"
        else
            url="https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz"
        fi
        echo "[install_d4rl_env] URL: $url"
        download_ok=0
        if command -v wget &>/dev/null; then
            wget --progress=bar:force --timeout=120 --tries=3 -O "$archive" "$url" && download_ok=1
        elif command -v curl &>/dev/null; then
            curl -fSL --connect-timeout 120 --max-time 600 --retry 3 -o "$archive" "$url" && download_ok=1
        else
            echo "Neither wget nor curl found. Please install one to download MuJoCo." >&2
            rm -rf "$tmpdir"
            exit 1
        fi
        if [ "$download_ok" -ne 1 ]; then
            echo "[install_d4rl_env] Download failed. Try without --use-mirror, or download manually:" >&2
            echo "  $url" >&2
            rm -rf "$tmpdir"
            exit 1
        fi
        tar -xzf "$archive" -C "$tmpdir"
        extracted=$(find "$tmpdir" -mindepth 1 -maxdepth 1 -type d | head -1)
        if [ -n "$extracted" ] && [ -d "$extracted" ]; then
            mv "$extracted" "$mujoco_dir"
        else
            echo "[install_d4rl_env] Unexpected tarball layout. Expected a single top-level directory." >&2
            ls -la "$tmpdir" >&2
            rm -rf "$tmpdir"
            exit 1
        fi
        rm -rf "$tmpdir"
        echo "[install_d4rl_env] MuJoCo 2.1.0 installed at $mujoco_dir"
    fi
    if ! grep -q "mujoco210/bin" "$VENV_DIR/bin/activate" 2>/dev/null; then
        echo "export LD_LIBRARY_PATH=\"${mujoco_dir}/bin:\$LD_LIBRARY_PATH\"" >> "$VENV_DIR/bin/activate"
    fi

    uv pip install "mujoco-py==2.1.2.14"
    uv pip install "tqdm"
}

install_liberopro_env() {
    uv pip install rlinf-libero rlinf-liberopro
    materialize_package_files rlinf-libero
    materialize_package_files rlinf-liberopro
    retry_cmd libero-download-assets --skip-existing
    retry_cmd liberopro-download-assets --skip-existing
    reset_libero_config
}

install_liberoplus_env() {
    uv pip install rlinf-libero "rlinf-liberoplus>=0.1.3"
    materialize_package_files rlinf-libero
    materialize_package_files rlinf-liberoplus
    retry_cmd libero-download-assets --skip-existing
    export LIBERO_PLUS_ASSETS_REPO="${LIBERO_PLUS_ASSETS_REPO:-RLinf/LIBERO-plus-assets}"
    retry_cmd liberoplus-download-assets --skip-existing
    reset_libero_config
}

install_behavior_env() {
    # Use BEHAVIOR_PATH as the checkout location if set (shared, cloned on first use);
    # otherwise clone into the venv.
    local behavior_dir
    behavior_dir=$(clone_or_reuse_repo BEHAVIOR_PATH "$VENV_DIR/BEHAVIOR-1K" https://github.com/RLinf/BEHAVIOR-1K.git -b RLinf/v3.7.2 --depth 1)

    pushd "$behavior_dir" >/dev/null
    UV_LINK_MODE=hardlink ./setup.sh --omnigibson --bddl --joylo --confirm-no-conda --accept-nvidia-eula --use-uv
    # OmniGibson's eval deps need another commit of lerobot, which is in conflict with which rlinf needs.
    # We actually does not use OmniGibson's lerobot deps, so just install other deps in OmniGibson's eval deps. 
    uv pip install "dm_tree>=0.1.9" "hydra-core>=1.3.2" "websockets>=15.0.1" "msgpack>=1.1.0" "gspread>=6.2.1" "open3d>=0.19.0" av "numpy<2"
    popd >/dev/null
    uv pip uninstall flash-attn || true
    uv pip install ml_dtypes==0.5.3 protobuf==3.20.3
    uv pip install click==8.2.1
    uv pip install llvmlite==0.47.0 numba==0.65.1
    # Re-assert the pin after OmniGibson's setup, which otherwise leaves a torch
    # tree mixing two versions (inductor then fails in _pad_mm_init).
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
}

install_metaworld_env() {
    uv pip install metaworld==3.0.0
}

install_calvin_env() {
    local calvin_dir
    calvin_dir=$(clone_or_reuse_repo CALVIN_PATH "$VENV_DIR/calvin" https://github.com/mees/calvin.git --recurse-submodules)

    uv pip install wheel cmake==3.18.4.post1 setuptools==57.5.0 wheel==0.45.1
    # NOTE: Use a fork version of pyfasthash that fixes install on Python 3.11
    uv pip install git+${GITHUB_PREFIX}https://github.com/RLinf/pyfasthash.git --no-build-isolation
    uv pip install -e ${calvin_dir}/calvin_env/tacto
    uv pip install -e ${calvin_dir}/calvin_env
    uv pip install -e ${calvin_dir}/calvin_models
    uv pip install --upgrade hydra-core==1.3.2
}

install_polaris_env() {
    local polaris_dir
    polaris_dir=$(clone_or_reuse_repo POLARIS_PATH "$VENV_DIR/polaris" https://github.com/RLinf/polaris.git --recurse-submodules)
    export OMNI_KIT_ACCEPT_EULA=YES
    if ! grep -q '^export OMNI_KIT_ACCEPT_EULA=' "$VENV_DIR/bin/activate" 2>/dev/null; then
        echo "export OMNI_KIT_ACCEPT_EULA=YES" >> "$VENV_DIR/bin/activate"
    fi

    uv pip install "setuptools<82"
    uv pip install "flatdict==4.0.1" --no-build-isolation
    uv pip install sympy==1.13.3
    uv pip install -e "$polaris_dir"
    
    python - <<'EOF'
import isaacsim
EOF
}

install_isaaclab_env() {
    local isaaclab_dir
    isaaclab_dir=$(clone_or_reuse_repo ISAAC_LAB_PATH "$VENV_DIR/isaaclab" https://github.com/RLinf/IsaacLab)

    pushd ~ >/dev/null
    uv pip install "flatdict==4.0.1" --no-build-isolation
    uv pip install "cuda-toolkit[nvcc]==12.8.0"

    # Force CMake < 4 for egl-probe / robomimic native build compatibility
    uv pip uninstall -y cmake || true
    uv pip install "cmake<4"

    $isaaclab_dir/isaaclab.sh --install
    popd >/dev/null
}

install_robocasa_env() {
    local robocasa_dir
    robocasa_dir=$(clone_or_reuse_repo ROBOCASA_PATH "$VENV_DIR/robocasa" https://github.com/RLinf/robocasa.git)
    
    uv pip install -e "$robocasa_dir"
    uv pip install protobuf==6.33.0
    python -m robocasa.scripts.setup_macros
}

install_robocasa365_env() {
    local robocasa_dir
    local assets_path
    local macros_private_path

    robocasa_dir=$(clone_or_reuse_repo ROBOCASA_PATH "$VENV_DIR/robocasa" https://github.com/robocasa/robocasa.git -b main)
    assets_path="$robocasa_dir/robocasa/models/assets"
    macros_private_path="$robocasa_dir/robocasa/macros_private.py"

    if [[ -n "${ROBOCASA_ASSETS_PATH:-}" ]]; then
        mkdir -p "$ROBOCASA_ASSETS_PATH"

        if [[ -d "$assets_path" && ! -L "$assets_path" ]]; then
            echo "[install_robocasa365_env] Copying RoboCasa assets from $assets_path to $ROBOCASA_ASSETS_PATH" >&2
            cp -an "$assets_path/." "$ROBOCASA_ASSETS_PATH/"
        fi

        rm -rf "$assets_path"
    fi

    if [[ -z "${ROBOCASA_PATH:-}" && -d "$robocasa_dir/.git" ]]; then
        git -C "$robocasa_dir" fetch origin main >&2
        git -C "$robocasa_dir" checkout main >&2 || git -C "$robocasa_dir" checkout -B main origin/main >&2
        git -C "$robocasa_dir" pull --ff-only origin main >&2
    fi

    uv pip install -e "$robocasa_dir"
    uv pip install --no-deps "lerobot @ git+${GITHUB_PREFIX}https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
    uv pip install --no-deps "robosuite @ git+${GITHUB_PREFIX}https://github.com/ARISE-Initiative/robosuite.git@master"
    uv pip install --no-deps mujoco==3.3.1
    uv pip install protobuf==6.33.0

    if [[ -n "${ROBOCASA_ASSETS_PATH:-}" ]]; then
        rm -rf "$assets_path"
        ln -s "$ROBOCASA_ASSETS_PATH" "$assets_path"

        echo "[install_robocasa365_env] Linked $assets_path -> $ROBOCASA_ASSETS_PATH" >&2
    fi

    if [[ -f "$macros_private_path" ]]; then
        echo "[install_robocasa365_env] Reusing existing $macros_private_path" >&2
    else
        python -m robocasa.scripts.setup_macros
    fi
}

install_franka_env() {
    # Install serl_franka_controller
    # Check if ROS_CATKIN_PATH is set or serl_franka_controllers is already built
    set +euo pipefail
    source /opt/ros/noetic/setup.bash
    set -euo pipefail
    ROS_CATKIN_PATH=$(realpath "$VENV_DIR/franka_catkin_ws")
    LIBFRANKA_VERSION=${LIBFRANKA_VERSION:-0.15.0}
    FRANKA_ROS_VERSION=${FRANKA_ROS_VERSION:-0.10.0}

    mkdir -p "$ROS_CATKIN_PATH/src"

    # Clone necessary repositories
    pushd "$ROS_CATKIN_PATH/src"
    if [ ! -d "$ROS_CATKIN_PATH/src/serl_franka_controllers" ]; then
        git clone https://github.com/rail-berkeley/serl_franka_controllers
    fi
    if [ ! -d "$ROS_CATKIN_PATH/libfranka" ]; then
        git clone -b "${LIBFRANKA_VERSION}" --recurse-submodules https://github.com/frankaemika/libfranka $ROS_CATKIN_PATH/libfranka
    fi
    if [ ! -d "$ROS_CATKIN_PATH/src/franka_ros" ]; then
        # Use a fork version that fixes compile issues with newer libfranka using C++17
        git clone -b "${FRANKA_ROS_VERSION}" --recurse-submodules https://github.com/RLinf/franka_ros
    fi
    popd >/dev/null

    # Build
    pushd "$ROS_CATKIN_PATH"
    # libfranka first
    if [ ! -f "$ROS_CATKIN_PATH/libfranka/build/libfranka.so" ]; then
        mkdir -p "$ROS_CATKIN_PATH/libfranka/build"
        pushd "$ROS_CATKIN_PATH/libfranka/build" >/dev/null
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_PREFIX_PATH=/opt/openrobots/lib/cmake -DBUILD_TESTS=OFF ..
        make -j$(nproc)
        popd >/dev/null
    fi
    export LD_LIBRARY_PATH=$ROS_CATKIN_PATH/libfranka/build:/opt/openrobots/lib:$LD_LIBRARY_PATH
    export CMAKE_PREFIX_PATH=$ROS_CATKIN_PATH/libfranka/build:$CMAKE_PREFIX_PATH

    # Then franka_ros
    catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DFranka_DIR:PATH=$ROS_CATKIN_PATH/libfranka/build

    # Finally serl_franka_controllers
    catkin_make -DCMAKE_CXX_STANDARD=17 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 --pkg serl_franka_controllers
    popd >/dev/null

    echo "export LD_LIBRARY_PATH=$ROS_CATKIN_PATH/libfranka/build:/opt/openrobots/lib:\$LD_LIBRARY_PATH" >> "$VENV_DIR/bin/activate"
    echo "export CMAKE_PREFIX_PATH=$ROS_CATKIN_PATH/libfranka/build:\$CMAKE_PREFIX_PATH" >> "$VENV_DIR/bin/activate"
    echo "source /opt/ros/noetic/setup.bash" >> "$VENV_DIR/bin/activate"
    echo "source $ROS_CATKIN_PATH/devel/setup.bash" >> "$VENV_DIR/bin/activate"
}

install_franka_franky_env() {
    # Prebuilt franky-control wheel (libfranka bundled), published per
    # libfranka version by the Brunch-Life/franky fork.  LIBFRANKA_VERSION
    # must match your Franka firmware (compatibility matrix:
    # https://frankarobotics.github.io/docs/compatibility.html); defaults
    # to 0.19.0.  Override FRANKY_WHEEL (URL / local path / PyPI spec) when
    # the host cannot reach github.com.
    local LIBFRANKA_VERSION="${LIBFRANKA_VERSION:-0.19.0}"
    local PYTAG
    PYTAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    local FRANKY_WHEEL="${FRANKY_WHEEL:-${GITHUB_PREFIX}https://github.com/Brunch-Life/franky/releases/download/wheels-libfranka-${LIBFRANKA_VERSION}/franky_control-1.1.3-${PYTAG}-${PYTAG}-manylinux_2_28_x86_64.whl}"
    echo "Installing franky-control (libfranka $LIBFRANKA_VERSION): $FRANKY_WHEEL"
    # --no-deps keeps the franka extra's pins (e.g. numpy<2); letting pip
    # re-resolve them breaks Ray pickling across nodes.
    uv pip install --reinstall-package franky-control --no-deps "$FRANKY_WHEEL"
    install_lerobot
}

install_franka_dexhand_deps() {
    uv pip install "RLinf-dexterous-hands[glove]"
}

install_xsquare_turtle2_env() {
    install_lerobot
    uv pip install git+${GITHUB_PREFIX}https://github.com/RLinf/xsquare_turtle_basics.git
}

install_robotwin_env() {
    # Set TORCH_CUDA_ARCH_LIST based on the CUDA version
    local cuda_mm cuda_major cuda_minor
    cuda_mm=$(detect_cuda_major_minor) || {
        echo "Could not detect CUDA version. Cannot build robotwin environment." >&2
        exit 1
    }
    cuda_major="${cuda_mm%% *}"
    cuda_minor="${cuda_mm##* }"
    if [ "$cuda_major" -gt 12 ] || { [ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -ge 8 ]; }; then
        # Include Blackwell support for CUDA 12.8+
        export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0;10.0"
    else
        export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
    fi

    uv pip install mplib==0.2.1 gymnasium==0.29.1 av open3d zarr openai

    uv pip install git+${GITHUB_PREFIX}https://github.com/facebookresearch/pytorch3d.git@v0.7.9  --no-build-isolation
    uv pip install warp-lang==1.11.1
    uv pip install git+${GITHUB_PREFIX}https://github.com/NVlabs/curobo.git  --no-build-isolation

    # patch sapien and mplib for robotwin
    SAPIEN_LOCATION=$(uv pip show sapien | grep 'Location' | awk '{print $2}')/sapien
    # Adjust some code in wrapper/urdf_loader.py
    URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
    # ----------- before -----------
    # 667         with open(urdf_file, "r") as f:
    # 668             urdf_string = f.read()
    # 669 
    # 670         if srdf_file is None:
    # 671             srdf_file = urdf_file[:-4] + "srdf"
    # 672         if os.path.isfile(srdf_file):
    # 673             with open(srdf_file, "r") as f:
    # 674                 self.ignore_pairs = self.parse_srdf(f.read())
    # ----------- after  -----------
    # 667         with open(urdf_file, "r", encoding="utf-8") as f:
    # 668             urdf_string = f.read()
    # 669 
    # 670         if srdf_file is None:
    # 671             srdf_file = urdf_file[:-4] + ".srdf"
    # 672         if os.path.isfile(srdf_file):
    # 673             with open(srdf_file, "r", encoding="utf-8") as f:
    # 674                 self.ignore_pairs = self.parse_srdf(f.read())
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER

    MPLIB_LOCATION=$(uv pip show mplib | grep 'Location' | awk '{print $2}')/mplib
    # Adjust some code in planner.py
    # ----------- before -----------
    # 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
    # 808                 return {"status": "screw plan failed"}
    # ----------- after  ----------- 
    # 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
    # 808                 return {"status": "screw plan failed"}
    PLANNER=$MPLIB_LOCATION/planner.py
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER
}

install_frankasim_env() {
    local serldir
    serldir=$(clone_or_reuse_repo SERL_PATH "$VENV_DIR/serl" https://github.com/RLinf/serl.git -b RLinf/franka-sim)
    uv pip install -e "$serldir/franka_sim"
    uv pip install -r "$serldir/franka_sim/requirements.txt"
}

install_embodichain_env() {
    # >=0.2.4 relocates official task envs to embodichain_tasks and moves
    # build_env into embodichain.lab.gym.utils.registration.
    uv pip install "embodichain>=0.2.4" --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
}

install_dosw1_env() {
    # Reuse the standard embodied extra so dosw1 picks up the same
    # transformers/imageio/gymnasium dependency set as other embodied envs.
    uv sync --extra embodied --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
    # The default patch_syncer uses nvcomp_lz4. Keep DOSW1 lightweight by
    # installing only this shared compression runtime instead of the full
    # common simulator dependency set.
    uv pip install nvidia-nvcomp-cu12
    uv pip install evdev opencv-python

    # Install DOSW1 SDK. The wheel / airbot_api source are pre-deployed on the
    # DOS-W1 robot under ~/dos_w1/airbot by default; on a generic server they
    # are usually absent. Users may override the paths via env vars:
    #   DOSW1_SDK_WHEEL  - path to airbot_py-*.whl
    #   DOSW1_API_PATH   - path to the airbot_api source tree
    # If the paths are missing, we skip the SDK install with a warning so the
    # rest of the env still gets set up (e.g. for server-side training runs
    # that talk to the robot over gRPC and do not need the local SDK).
    local dosw1_sdk_wheel="${DOSW1_SDK_WHEEL:-$HOME/dos_w1/airbot/5.1.6/airbot_py-5.1.6-py3-none-any.whl}"
    local dosw1_api_path="${DOSW1_API_PATH:-$HOME/dos_w1/airbot/airbot_api}"

    if [ -f "$dosw1_sdk_wheel" ]; then
        uv pip install "$dosw1_sdk_wheel"
    else
        echo "[dosw1] WARNING: DOSW1 SDK wheel not found at '$dosw1_sdk_wheel'." >&2
        echo "[dosw1] WARNING: Skipping 'airbot_py' install. Set DOSW1_SDK_WHEEL to the wheel path if you need the local SDK." >&2
    fi

    if [ -d "$dosw1_api_path" ]; then
        uv pip install -e "$dosw1_api_path"
    else
        echo "[dosw1] WARNING: DOSW1 airbot_api source not found at '$dosw1_api_path'." >&2
        echo "[dosw1] WARNING: Skipping 'airbot_api' install. Set DOSW1_API_PATH to the source directory if you need the local SDK." >&2
    fi

    local repo_root
    repo_root="$(dirname "$SCRIPT_DIR")"
    uv pip install -e "$repo_root" --no-deps
}

install_habitat_env() {
    local habitat_sim_dir
    habitat_sim_dir=$(clone_or_reuse_repo HABITAT_SIM_PATH "$VENV_DIR/habitat" https://github.com/facebookresearch/habitat-sim.git -b v0.3.3 --recurse-submodules)
    if [ -d "$habitat_sim_dir/build" ]; then
        rm -rf $habitat_sim_dir/build
    fi
    export CMAKE_POLICY_VERSION_MINIMUM=3.5
    uv pip install "$habitat_sim_dir" --config-settings="--build-option=--headless" --config-settings="--build-option=--with-bullet"
    uv pip install $habitat_sim_dir/build/deps/magnum-bindings/src/python/

    local habitat_lab_dir
    # Use a fork version of habitat-lab that fixes Python 3.11 compatibility issues
    habitat_lab_dir=$(clone_or_reuse_repo HABITAT_LAB_PATH "$VENV_DIR/habitat-lab" https://github.com/RLinf/habitat-lab.git -b v0.3.3 --recurse-submodules)
    uv pip install -e $habitat_lab_dir/habitat-lab
    uv pip install -e $habitat_lab_dir/habitat-baselines
}

install_genesis_env() {
    echo "Installing Genesis environment dependencies..."
    uv pip install "transformers==4.57.6"
    uv pip install "cuda-python==12.9.6"
    uv pip install "genesis-world==0.4.5"
    uv pip install "pyglet==2.1.14"
    uv pip install "matplotlib==3.10.8"

    uv pip install "torch==2.8.0"
    uv pip install "torchvision==0.23.0"
    uv pip install "torchaudio==2.8.0"
    uv pip install "torchcodec==0.6"
}

install_opensora_world_model() {
    # Clone opensora repository
    local opensora_dir
    opensora_dir=$(clone_or_reuse_repo OPENSORA_PATH "$VENV_DIR/opensora" ${GITHUB_PREFIX}https://github.com/RLinf/opensora.git)
    
    uv pip install -e "$opensora_dir"
    
    # xformers embeds the torch version in its local label; 0.0.35 ships
    # wheels for torch 2.11. Install it against the resolved torch build.
    uv pip install "xformers==0.0.35"

    # Install remaining opensora dependencies (xformers handled above).
    uv pip install -r $SCRIPT_DIR/embodied/models/opensora.txt
    install_tensornvme
    echo "export LD_LIBRARY_PATH=~/.tensornvme/lib:\$LD_LIBRARY_PATH" >> "$VENV_DIR/bin/activate"
    install_apex
}

install_tensornvme() {
    local url="git+${GITHUB_PREFIX}https://github.com/fangqi-Zhu/TensorNVMe.git"
    uv pip install "$url" --no-build-isolation

    local tnvme_env=(env "LD_LIBRARY_PATH=$HOME/.tensornvme/lib:${LD_LIBRARY_PATH:-}")
    if ! "${tnvme_env[@]}" python -c "import tensornvme._C" >/dev/null 2>&1; then
        echo "[install.sh] tensornvme does not load against this torch; rebuilding."
        uv cache clean tensornvme || true
        uv pip install "$url" --no-build-isolation --reinstall-package tensornvme
        "${tnvme_env[@]}" python -c "import tensornvme._C" >/dev/null 2>&1 \
            || echo "[install.sh] WARNING: tensornvme still does not import; expected without a GPU, otherwise check the torch ABI."
    fi
}

install_wan_world_model() {
    local wan_dir
    wan_dir=$(clone_or_reuse_repo WAN_PATH "$VENV_DIR/wan" https://github.com/RLinf/diffsynth-studio.git)
    uv pip install -e "$wan_dir"
    uv pip install -r $SCRIPT_DIR/embodied/models/wan.txt
}

install_roboverse_env() {
    local roboverse_dir
    roboverse_dir=$(clone_or_reuse_repo ROBOVERSE_PATH "$VENV_DIR/roboverse" https://github.com/tiny-xie/roboverse.git)
    uv pip install -e "${roboverse_dir}[mujoco]"
    uv pip install git+${GITHUB_PREFIX}https://github.com/facebookresearch/pytorch3d.git@v0.7.9 --no-build-isolation
    uv pip install -e "${roboverse_dir}[sapien3]"
    uv pip install -e "${roboverse_dir}[genesis]"
    
    local pyroki_dir
    pyroki_dir=$(clone_or_reuse_repo PYROKI_PATH "$roboverse_dir/pyroki" https://github.com/chungmin99/pyroki.git)
    uv pip install -e "$pyroki_dir"
    uv pip install "numpy==1.26.4" --force-reinstall
    uv pip install "mujoco==3.3.7" "dm-control==1.0.34" --force-reinstall
}

#=======================AGENTIC INSTALLER=======================

install_te_2_17() {
    local torch_cu_major te_extra
    torch_cu_major=$(python - <<'EOF'
import torch

cuda = torch.version.cuda or ""
print(cuda.split(".")[0] if cuda else "")
EOF
)
    if [ -z "$torch_cu_major" ]; then
        echo "[install.sh] ERROR: torch has no CUDA build; cannot install transformer-engine." >&2
        exit 1
    fi
    te_extra="core_cu${torch_cu_major}"
    if [ "$torch_cu_major" != "12" ]; then
        echo "[install.sh] Installing TE 2.17.0 (${te_extra})..."
        NVTE_PYTORCH_FORCE_BUILD=TRUE uv pip install --no-build-isolation \
            "transformer-engine[pytorch,${te_extra}]==2.17.0"
        echo "[install.sh] TE 2.17.0 installed."
        return 0
    fi

    echo "[install.sh] Installing TE 2.17.0 (${te_extra}, source build with nvcc)..."
    uv pip install --no-build-isolation "transformer-engine[${te_extra}]==2.17.0"
    uv pip install einops onnx onnxscript packaging pydantic nvdlfw-inspect
    NVTE_PYTORCH_FORCE_BUILD=TRUE uv pip install --no-build-isolation --no-deps \
        --no-binary transformer-engine-torch "transformer-engine-torch==2.17.0"
    if uv pip show transformer-engine-cu13 >/dev/null 2>&1; then
        echo "[install.sh] ERROR: transformer-engine-cu13 present on a CUDA 12 torch; its core would shadow the cu12 one." >&2
        exit 1
    fi
    echo "[install.sh] TE 2.17.0 installed."
}

install_mbridge() {
    # megatron-bridge 0.4.2 requires python >= 3.12, so RLinf publishes a
    # py3.10/3.11 fork. --no-deps avoids re-resolving torch and nemo-toolkit.
    echo "[install.sh] Installing rlinf-megatron-bridge 0.4.2 (PyPI wheel)..."
    uv pip install --no-deps --extra-index-url https://pypi.org/simple "rlinf-megatron-bridge==0.4.2"
    echo "[install.sh] rlinf-megatron-bridge 0.4.2 installed (import: megatron.bridge)."
}

# FA4 backward is sm90+ only; on sm<9 drop it so TE falls back to FA2.
uninstall_fa4_conditional() {
    local gpu_cc
    gpu_cc=$(python -c "import torch;print(torch.cuda.get_device_capability(0)[0])" 2>/dev/null || true)
    if [ -z "$gpu_cc" ]; then
        echo "[install.sh] WARNING: Could not detect GPU compute capability; keeping FA4."
        return 0
    fi
    if [ "$gpu_cc" -lt 9 ]; then
        echo "[install.sh] GPU sm${gpu_cc} < sm90: FA4 backward unsupported, uninstalling flash-attn-4 → TE will use FA2."
        uv pip uninstall flash-attn-4 || true
    else
        echo "[install.sh] GPU sm${gpu_cc} >= sm90: FA4 usable, keeping flash-attn-4."
    fi
}

# TE 2.17's .so files carry no RPATH, so the venv's NVIDIA libs must precede a
# stale system libnccl.
setup_nccl_env() {
    local nvlib
    nvlib="$(python -c 'import nvidia; print(nvidia.__path__[0])' 2>/dev/null || true)"
    if [ -z "$nvlib" ]; then
        echo "[install.sh] WARNING: nvidia package not found in venv; skipping NCCL env setup."
        return 0
    fi
    if grep -q "^# RLinf NCCL fix$" "$VENV_DIR/bin/activate" 2>/dev/null; then
        return 0
    fi
    cat >> "$VENV_DIR/bin/activate" <<EOF

# RLinf NCCL fix
export LD_LIBRARY_PATH="\$(ls -d ${nvlib}/*/lib 2>/dev/null | tr '\n' ':')\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
    echo "[install.sh] NCCL LD_LIBRARY_PATH export added to $VENV_DIR/bin/activate."
}

install_agentic() {
    local engine engine_ver torch211_stack=0
    engine=$(effective_engine)
    engine_ver=$(effective_engine_version)
    if engine_needs_torch211; then
        torch211_stack=1
        echo "[install.sh] ${engine} ${engine_ver} runs on torch 2.11: using TE 2.17, mcore 0.17, megatron-bridge."
    fi

    local engine_req
    engine_req=$(agentic_requirements_file "$engine" "$engine_ver")

    uv sync --extra agentic --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
    install_engine_requirements "$engine_req"
    echo "[install.sh] Installed engine: $(basename "$engine_req")"
    uv pip check || echo "[install.sh] WARNING: dependency conflicts reported above"
    if [ "$NO_ROOT" -eq 0 ]; then
        bash $SCRIPT_DIR/sys_deps.sh "$PLATFORM"
    fi

    # Megatron-LM
    # Use MEGATRON_PATH as the checkout location if set (shared, cloned on first use);
    # otherwise clone into the venv.
    local megatron_branch="core_r0.13.0"
    [ "$torch211_stack" -eq 1 ] && megatron_branch="core_r0.17.0"
    local megatron_dir
    megatron_dir=$(clone_or_reuse_repo MEGATRON_PATH "$VENV_DIR/Megatron-LM" https://github.com/NVIDIA/Megatron-LM.git -b "$megatron_branch")

    echo "export PYTHONPATH=$(realpath "$megatron_dir"):\$PYTHONPATH" >> "$VENV_DIR/bin/activate"

    # If TEST_BUILD is 1, skip the heavy transformer-engine build (megatron.txt
    # pins TE 2.1.0; the torch 2.11 stack builds TE 2.17 instead).
    if [ "$TEST_BUILD" -ne 1 ]; then
        if [ "$torch211_stack" -eq 1 ]; then
            if [ -f /etc/profile.d/cuda.sh ]; then
                source /etc/profile.d/cuda.sh
            fi
            install_te_2_17
        else
            uv pip install -r $SCRIPT_DIR/agentic/megatron.txt --no-build-isolation
        fi
    fi

    if [ "$torch211_stack" -eq 1 ]; then
        install_mbridge
        uninstall_fa4_conditional
        setup_nccl_env
    fi

    # --transformers / --xgrammar (and the version derived from --sglang) win
    # over the engine's own pins.
    [ -n "$TRANSFORMERS_VERSION" ] && uv pip install "transformers==${TRANSFORMERS_VERSION}"
    [ -n "$XGRAMMAR_VERSION" ] && uv pip install "xgrammar==${XGRAMMAR_VERSION}"

    install_apex
    install_flash_attn
    uv pip uninstall pynvml || true
}

#=======================DOCUMENTATION INSTALLER=======================

install_docs() {
    uv sync --extra agentic --active "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
    install_engine_requirements "$(agentic_requirements_file vllm "$(agentic_latest_version vllm)")"
    install_engine_requirements "$(agentic_requirements_file sglang "$(agentic_latest_version sglang)")"
    uv sync --extra embodied --active --inexact "${PLATFORM_UV_SYNC_ARGS[@]}" $NO_INSTALL_RLINF_CMD
    uv pip install -r $SCRIPT_DIR/docs/requirements.txt
    uv pip uninstall pynvml || true
}

main() {
    parse_args "$@"
    validate_python_version
    apply_env_default_torch
    apply_agentic_torch_default
    configure_platform
    setup_mirror
    apply_torch_override

    case "$TARGET" in
        embodied)
            # validate --model
            if [ -n "$MODEL" ]; then
                if [[ ! " ${SUPPORTED_MODELS[*]} " =~ " $MODEL " ]]; then
                    echo "Unknown embodied model: $MODEL. Supported models: ${SUPPORTED_MODELS[*]}" >&2
                    exit 1
                fi
            fi
            # check --env is set and supported
            if [ -n "$ENV_NAME" ]; then
                if [[ ! " ${SUPPORTED_ENVS[*]} " =~ " $ENV_NAME " ]]; then
                    echo "Unknown environment: $ENV_NAME. Supported environments: ${SUPPORTED_ENVS[*]}" >&2
                    exit 1
                fi
            elif [ "$MODEL" != "dreamzero" ]; then
                echo "--env must be specified when target=embodied." >&2
                exit 1
            fi

            case "$MODEL" in
                openvla)
                    install_openvla_model
                    ;;
                openvla-oft)
                    install_openvla_oft_model
                    ;;
                openpi)
                    install_openpi_model
                    ;;
                molmoact2)
                    install_molmoact2_model
                    ;;
                starvla)
                    install_starvla_model
                    ;;
                gr00t)
                    install_gr00t_model
                    ;;
                gr00t_n1d6)
                    install_gr00t_n1d6_model
                    ;;
                gr00t_n1d7)
                    install_gr00t_n1d7_model
                    ;;
                dexbotic)
                    install_dexbotic_model
                    ;;
                lingbotvla)                  
                    install_lingbot_vla_model 
                    ;;
                abot_m0)
                    install_abot_m0_model
                    ;;
                dreamzero)
                    install_dreamzero_model
                    ;;
                qwen3_vl)
                    install_qwen3_vl_model
                    ;;
                evo1)
                    install_evo1_model
                    ;;
                "")
                    install_env_only
                    ;;
            esac
            ;;
        agentic)
            create_and_sync_venv
            install_agentic
            ;;
        docs)
            create_and_sync_venv
            install_docs
            ;;
        *)
			echo "Unknown target: $TARGET" >&2
			echo "Supported targets: ${SUPPORTED_TARGETS[*]}" >&2
            exit 1
            ;;
    esac

    install_platform_extras
}

main "$@"
