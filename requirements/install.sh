#! /bin/bash

TARGET="${1:-"openvla"}"
EMBODIED_TARGET=("openvla" "openvla-oft" "openpi")

# Common dependencies
uv venv
source .venv/bin/activate
UV_TORCH_BACKEND=auto uv sync

if [[ " ${EMBODIED_TARGET[*]} " == *" $TARGET "* ]]; then
    uv sync --extra embodied
    uv pip uninstall pynvml
    bash requirements/install_embodied_deps.sh # Must be run after the above command
    mkdir -p /opt && git clone https://github.com/RLinf/LIBERO.git /opt/libero
    echo "export PYTHONPATH=/opt/libero:$PYTHONPATH" >> .venv/bin/activate
fi

if [ "$TARGET" = "openvla" ]; then
    UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla.txt --no-build-isolation
elif [ "$TARGET" = "openvla-oft" ]; then
    UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla-oft.txt --no-build-isolation
elif [ "$TARGET" = "openpi" ]; then
    UV_TORCH_BACKEND=auto GIT_LFS_SKIP_SMUDGE=1 uv pip install -r requirements/openpi.txt
    cp -r .venv/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
    export TOKENIZER_DIR=/root/.cache/openpi/big_vision/ && mkdir -p $TOKENIZER_DIR && gsutil -m cp -r gs://big_vision/paligemma_tokenizer.model $TOKENIZER_DIR
elif [ "$TARGET" = "reason" ]; then
    uv sync --extra sglang-vllm
    uv pip uninstall pynvml
    mkdir -p /opt && git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.13.0 /opt/Megatron-LM
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 NVCC_APPEND_FLAGS="--threads 24" APEX_PARALLEL_BUILD=24 uv pip install -r requirements/megatron.txt --no-build-isolation
    echo "export PYTHONPATH=/opt/Megatron-LM:$PYTHONPATH" >> .venv/bin/activate
else
    echo "Unknown target: $TARGET. Supported targets are: openvla, openvla-oft, openpi, reason."
    exit 1
fi