#! /bin/bash

TARGET="${1:-"openvla"}"
EMBODIED_TARGET=("openvla" "openvla-oft" "openpi")

# Get the remaining args
while [ "$#" -gt 0 ]; do
    case "$2" in
        --enable-behavior)
            ENABLE_BEHAVIOR="true"
            shift
            ;;
        --test-build)
            TEST_BUILD="true"
            shift
            ;;
        *)
            break
            ;;
    esac
done

PYTHON_VERSION="3.11.10"
if [ "$ENABLE_BEHAVIOR" = "true" ]; then
    PYTHON_VERSION="3.10"
fi

# Behavior check
if [ "$ENABLE_BEHAVIOR" = "true" ] && [[ "$TARGET" != "openvla-oft" ]]; then
    echo "--enable-behavior can only be used with the openvla-oft target."
    exit 1
fi

# Common dependencies
uv venv --python=$PYTHON_VERSION
source .venv/bin/activate
UV_TORCH_BACKEND=auto uv sync

if [[ " ${EMBODIED_TARGET[*]} " == *" $TARGET "* ]]; then
    uv sync --extra embodied
    uv pip uninstall pynvml
    bash requirements/install_embodied_deps.sh # Must be run after the above command
    mkdir -p /opt && git clone https://github.com/RLinf/LIBERO.git /opt/libero
    echo "export PYTHONPATH=/opt/libero:$PYTHONPATH" >> .venv/bin/activate
    echo "export NVIDIA_DRIVER_CAPABILITIES=all" >> .venv/bin/activate
    echo "export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json" >> .venv/bin/activate
    echo "export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json" >> .venv/bin/activate
fi

if [ "$TARGET" = "openvla" ]; then
    UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla.txt --no-build-isolation
elif [ "$TARGET" = "openvla-oft" ]; then
    UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla_oft.txt --no-build-isolation
    if [ "$ENABLE_BEHAVIOR" = "true" ]; then
        git clone -b RLinf/v3.7.1 --depth 1 https://github.com/RLinf/BEHAVIOR-1K.git /opt/BEHAVIOR-1K
        cd /opt/BEHAVIOR-1K && ./setup.sh --omnigibson --bddl --joylo --confirm-no-conda --accept-nvidia-eula && cd -
        uv pip uninstall flash-attn
        uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
        uv pip install ml_dtypes==0.5.3 protobuf==3.20.3
        pip install click==8.2.1
        cd && uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && cd -
    fi
elif [ "$TARGET" = "openpi" ]; then
    UV_TORCH_BACKEND=auto GIT_LFS_SKIP_SMUDGE=1 uv pip install -r requirements/openpi.txt
    if [ "$ENABLE_BEHAVIOR" = "true" ]; then
        git clone -b RLinf/v3.7.1 --depth 1 https://github.com/RLinf/BEHAVIOR-1K.git /opt/BEHAVIOR-1K
        cd /opt/BEHAVIOR-1K && ./setup.sh --omnigibson --bddl --joylo --confirm-no-conda --accept-nvidia-eula && cd -
        uv pip uninstall flash-attn
        uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
        uv pip install ml_dtypes==0.5.3 protobuf==3.20.3
        pip install click==8.2.1
        cd && uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && cd -
    fi
    cp -r .venv/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
    export TOKENIZER_DIR=~/.cache/openpi/big_vision/ && mkdir -p $TOKENIZER_DIR && gsutil -m cp -r gs://big_vision/paligemma_tokenizer.model $TOKENIZER_DIR
elif [ "$TARGET" = "reason" ]; then
    uv sync --extra sglang-vllm
    uv pip uninstall pynvml
    mkdir -p /opt && git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.13.0 /opt/Megatron-LM
    if [ "$TEST_BUILD" != "true" ]; then
        APEX_CPP_EXT=1 APEX_CUDA_EXT=1 NVCC_APPEND_FLAGS="--threads 24" APEX_PARALLEL_BUILD=24 uv pip install -r requirements/megatron.txt --no-build-isolation
    fi
    echo "export PYTHONPATH=/opt/Megatron-LM:$PYTHONPATH" >> .venv/bin/activate
else
    echo "Unknown target: $TARGET. Supported targets are: openvla, openvla-oft, openpi, reason."
    exit 1
fi