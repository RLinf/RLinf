#! /bin/bash

set -euo pipefail

TARGET=""

MODEL=""
ENV_NAME=""
VENV_DIR=".venv"
PYTHON_VERSION="3.11.14"
TEST_BUILD=${TEST_BUILD:-0}
# Absolute path to this script (resolves symlinks)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

SUPPORTED_TARGETS=("embodied" "reason")
SUPPORTED_MODELS=("openvla" "openvla-oft" "openpi")
SUPPORTED_ENVS=("behavior" "maniskill_libero" "metaworld")

print_help() {
        cat <<EOF
Usage: bash install.sh <target> [options]

Targets:
    embodied               Install embodied model and envs (default).
    reason                 Install reasoning stack (Megatron etc.).

Options (for target=embodied):
    --model <name>         Embodied model to install: ${SUPPORTED_MODELS[*]}.
    --env <name>           Single environment to install: ${SUPPORTED_ENVS[*]}.

Common options:
    -h, --help             Show this help message and exit.
    --venv <dir>           Virtual environment directory name (default: .venv).
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

create_and_sync_venv() {
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    UV_TORCH_BACKEND=auto uv sync --active
}

install_prebuilt_flash_attn() {
    # Base release info – adjust when bumping flash-attn
    local flash_ver="2.7.4.post1"
    local base_url="https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_ver}"

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
    local cuda_major
    cuda_major=$(python - <<'EOF'
import torch
from packaging.version import Version
v = Version(torch.version.cuda)
print(v.base_version.split(".")[0])
EOF
)

    local cu_tag="cu${cuda_major}"            # e.g. cu12
    local torch_tag="torch${torch_mm}"        # e.g. torch2.6

    # We currently assume cxx11 abi FALSE and linux x86_64
    local platform_tag="linux_x86_64"
    local cxx_abi="cxx11abiFALSE"

    local wheel_name="flash_attn-${flash_ver}+${cu_tag}${torch_tag}${cxx_abi}-${py_tag}-${abi_tag}-${platform_tag}.whl"
    uv pip uninstall flash-attn || true
    uv pip install "${base_url}/${wheel_name}"
}

install_prebuilt_apex() {
    # Example URL: https://github.com/RLinf/apex/releases/download/25.09/apex-0.1-cp311-cp311-linux_x86_64.whl
    local base_url="https://github.com/RLinf/apex/releases/download/25.09"
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
    local platform_tag="linux_x86_64"
    local wheel_name="apex-0.1-${py_tag}-${abi_tag}-${platform_tag}.whl"
        
    uv pip uninstall apex || true
    uv pip install "${base_url}/${wheel_name}" || (echo "Apex wheel is not available for Python ${py_major}.${py_minor}, please install apex manually. See https://github.com/NVIDIA/apex" >&2; exit 1)
}

#=======================EMBODIED INSTALLERS=======================

install_common_embodied_deps() {
    uv sync --extra embodied --active
    bash $SCRIPT_DIR/embodied/sys_deps.sh
    {
        echo "export PYTHONPATH=$(pwd)/$VENV_DIR/libero:\$PYTHONPATH"
        echo "export NVIDIA_DRIVER_CAPABILITIES=all"
        echo "export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json"
        echo "export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json"
    } >> "$VENV_DIR/bin/activate"
}

install_openvla_model() {
    case "$ENV_NAME" in
        "")
            ;;
        maniskill_libero)
            create_and_sync_venv
            install_common_embodied_deps
            install_maniskill_libero_env
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for OpenVLA model." >&2
            exit 1
            ;;
    esac
    UV_TORCH_BACKEND=auto uv pip install -r $SCRIPT_DIR/embodied/models/openvla.txt --no-build-isolation
    install_prebuilt_flash_attn
    uv pip uninstall pynvml || true
}

install_openvla_oft_model() {
    case "$ENV_NAME" in
        "")
            ;;
        behavior)
            PYTHON_VERSION="3.10"
            create_and_sync_venv
            install_common_embodied_deps
            UV_TORCH_BACKEND=auto uv pip install -r $SCRIPT_DIR/embodied/models/openvla_oft.txt --no-build-isolation
            install_behavior_env
            ;;
        maniskill_libero)
            create_and_sync_venv
            install_common_embodied_deps
            install_maniskill_libero_env
            install_prebuilt_flash_attn
            UV_TORCH_BACKEND=auto uv pip install -r $SCRIPT_DIR/embodied/models/openvla_oft.txt --no-build-isolation
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
        "")
            ;;
        maniskill_libero)
            create_and_sync_venv
            install_common_embodied_deps
            install_maniskill_libero_env
            UV_TORCH_BACKEND=auto GIT_LFS_SKIP_SMUDGE=1 uv pip install -r $SCRIPT_DIR/embodied/models/openpi.txt
            install_prebuilt_flash_attn
            ;;
        metaworld)
            create_and_sync_venv
            install_common_embodied_deps
            UV_TORCH_BACKEND=auto GIT_LFS_SKIP_SMUDGE=1 uv pip install -r $SCRIPT_DIR/embodied/models/openpi.txt
            install_prebuilt_flash_attn
            install_metaworld_env
            ;;
        *)
            echo "Environment '$ENV_NAME' is not supported for OpenPI model." >&2
            exit 1
            ;;
    esac

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
    uv pip uninstall pynvml || true
}

#=======================ENV INSTALLERS=======================

install_maniskill_libero_env() {
    if [ ! -d "$VENV_DIR/libero" ]; then
        git clone https://github.com/RLinf/LIBERO.git "$VENV_DIR/libero"
    fi
    uv pip install -e "$VENV_DIR/libero"
    uv pip install -r $SCRIPT_DIR/embodied/envs/maniskill.txt

    # Maniskill assets
    bash $SCRIPT_DIR/embodied/download_assets.sh --assets maniskill
}

install_behavior_env() {
    if [ ! -d "$VENV_DIR/BEHAVIOR-1K" ]; then
        git clone -b RLinf/v3.7.1 --depth 1 https://github.com/RLinf/BEHAVIOR-1K.git "$VENV_DIR/BEHAVIOR-1K"
    fi
    pushd "$VENV_DIR/BEHAVIOR-1K" >/dev/null
    UV_LINK_MODE=hardlink ./setup.sh --omnigibson --bddl --joylo --confirm-no-conda --accept-nvidia-eula --use-uv
    popd >/dev/null
    uv pip uninstall flash-attn || true
    uv pip install ml_dtypes==0.5.3 protobuf==3.20.3
    uv pip install click==8.2.1
    pushd ~ >/dev/null
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
    install_prebuilt_flash_attn
    popd >/dev/null
}

install_metaworld_env() {
    uv pip install -r $SCRIPT_DIR/embodied/envs/metaworld.txt
}

#=======================REASONING INSTALLER=======================

install_reason() {
    uv sync --extra sglang-vllm --active

    # Megatron-LM
    if [ ! -d "$VENV_DIR/Megatron-LM" ]; then
        git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.13.0 "$VENV_DIR/Megatron-LM"
    fi
    echo "export PYTHONPATH=$(pwd)/$VENV_DIR/Megatron-LM:\$PYTHONPATH" >> "$VENV_DIR/bin/activate"

    # If TEST_BUILD is 1, skip installing megatron.txt
    if [ "$TEST_BUILD" -ne 1 ]; then
        uv pip install -r $SCRIPT_DIR/reason/megatron.txt --no-build-isolation
    fi

    install_prebuilt_apex
    install_prebuilt_flash_attn
    uv pip uninstall pynvml || true
}

main() {
    parse_args "$@"

    case "$TARGET" in
        embodied)
            if [ -z "$MODEL" ]; then
                echo "--model is required when target=embodied. Supported models: ${SUPPORTED_MODELS[*]}" >&2
                exit 1
            fi
            # validate model
            if [[ ! " ${SUPPORTED_MODELS[*]} " =~ " $MODEL " ]]; then
                echo "Unknown embodied model: $MODEL. Supported models: ${SUPPORTED_MODELS[*]}" >&2
                exit 1
            fi
            # check --env is set and supported
            if [ -n "$ENV_NAME" ]; then
                if [[ ! " ${SUPPORTED_ENVS[*]} " =~ " $ENV_NAME " ]]; then
                    echo "Unknown environment: $ENV_NAME. Supported environments: ${SUPPORTED_ENVS[*]}" >&2
                    exit 1
                fi
            else
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
            esac
            ;;
        reason)
            create_and_sync_venv
            install_reason
            ;;
        *)
			echo "Unknown target: $TARGET" >&2
			echo "Supported targets: ${SUPPORTED_TARGETS[*]}" >&2
            exit 1
            ;;
    esac
}

main "$@"
