ARG BUILD_TARGET=reason

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image

ARG NO_MIRROR
SHELL ["/bin/bash", "-c"]
ENV PATH=/opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive
RUN if [ -z "$NO_MIRROR" ]; then sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list; fi
RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim libibverbs-dev openssh-server sudo runit runit-systemd tmux \
    build-essential python3-dev cmake pkg-config iproute2 pciutils python3 python3-pip \
    wget unzip curl

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN python3 -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --upgrade pip setuptools wheel uv

ENV HF_HOME=/opt/.cache/huggingface
RUN mkdir -p $HF_HOME

# UV setup
ENV UV_DEFAULT_INDEX=${NO_MIRROR:+"https://pypi.org/simple/"}
ENV UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-"https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"}
ENV UV_PATH=/opt/venv
RUN mkdir $UV_PATH
WORKDIR $UV_PATH
COPY project $UV_PATH/pyproject.toml
ENV UV_LINK_MODE=symlink
ENV UV_CACHE_DIR=$UV_PATH/.cache
ENV UV_PYTHON_INSTALL_DIR=${UV_PATH}/.python

# switch_env utility
RUN cat <<EOF > /usr/local/bin/switch_env
#!/bin/bash
if [ -z "\$1" ]; then
    echo "Usage: switch_env <env_name>"
    exit 1
fi
if [ ! -d "${UV_PATH}/\$1" ]; then
    echo "Environment \$1 does not exist in ${UV_PATH}."
    exit 1
fi
source ${UV_PATH}/\$1/bin/activate
EOF
RUN chmod +x /usr/local/bin/switch_env

ENV PYTHON_VERSION=3.11
ENV FLASH_ATTENTION_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

FROM base-image AS reason-image

# Install Megatron-LM
RUN git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.13.0 /opt/Megatron-LM
ENV PYTHONPATH=/opt/Megatron-LM:$PYTHONPATH

# Install reasoning env
RUN uv venv reason --python=${PYTHON_VERSION}
RUN source switch_env reason && UV_TORCH_BACKEND=auto uv sync --active
RUN source switch_env reason && uv sync --extra sglang-vllm --active
RUN source switch_env reason && uv pip install https://github.com/RLinf/apex/releases/download/25.09/apex-0.1-cp311-cp311-linux_x86_64.whl
RUN source switch_env reason && uv pip install transformer_engine[pytorch]==2.1.0 --no-build-isolation
RUN source switch_env reason && uv pip install ${FLASH_ATTENTION_WHEEL}
RUN source switch_env reason && uv pip uninstall pynvml

# Set default env
RUN echo "source ${UV_PATH}/reason/bin/activate" >> ~/.bashrc

FROM base-image AS embodied-common-image

# Embodied NVIDIA_DRIVER_CAPABILITIES
ENV NVIDIA_DRIVER_CAPABILITIES="all"

# Embodied system dependencies
RUN apt-get install -y --no-install-recommends \
    mesa-utils \
    libosmesa6-dev \
    freeglut3-dev \
    libglew-dev \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# link_assets utility
RUN cat <<EOF > /usr/local/bin/link_assets
#!/bin/bash
if [ -d /opt/.maniskill ]; then
    ln -s /opt/.maniskill ~/.maniskill
fi
if [ -d /opt/.sapien ]; then
    ln -s /opt/.sapien ~/.sapien
fi
mkdir -p ~/.cache
if [ -d /opt/.openpi ]; then
    ln -s /opt/.openpi ~/.cache/openpi
fi
EOF
RUN chmod +x /usr/local/bin/link_assets

# download_assets utility
RUN cat <<EOF > /usr/local/bin/download_assets
#!/bin/bash

# Download assets for ManiSkill
export MS_ASSET_DIR=/opt/.maniskill
python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
python -m mani_skill.utils.download_asset widowx250s -y

# Download assets for SAPIEN
export PHYSX_VERSION=105.1-physx-5.3.1.patch0
export PHYSX_DIR=/opt/.sapien/physx/\$PHYSX_VERSION
mkdir -p \$PHYSX_DIR
wget -O \$PHYSX_DIR/linux-so.zip https://github.com/sapien-sim/physx-precompiled/releases/download/\$PHYSX_VERSION/linux-so.zip
unzip \$PHYSX_DIR/linux-so.zip -d \$PHYSX_DIR && rm \$PHYSX_DIR/linux-so.zip

# Download assets for OpenPI
export TOKENIZER_DIR=/opt/.openpi/big_vision/
mkdir -p \$TOKENIZER_DIR && gsutil -m cp -r gs://big_vision/paligemma_tokenizer.model \$TOKENIZER_DIR
EOF
RUN chmod +x /usr/local/bin/download_assets

# Install LIBERO
ENV LIBERO_REPO=RLinf/LIBERO
ENV LIBERO_BRANCH=master
# For invalidating docker build cache when the branch updates
ADD https://api.github.com/repos/${LIBERO_REPO}/git/refs/heads/${LIBERO_BRANCH} /tmp/.libero_version.json
RUN git clone -b ${LIBERO_BRANCH} https://github.com/${LIBERO_REPO}.git /opt/libero
ENV PYTHONPATH=/opt/libero:$PYTHONPATH

FROM embodied-common-image AS embodied-image

# Install openvla env
RUN uv venv openvla --python=${PYTHON_VERSION}
RUN source switch_env openvla && UV_TORCH_BACKEND=auto uv sync --active
RUN source switch_env openvla && uv sync --extra embodied --active
RUN source switch_env openvla && uv pip install git+https://github.com/openvla/openvla.git --no-build-isolation
RUN source switch_env openvla && uv pip install $FLASH_ATTENTION_WHEEL --no-build-isolation
RUN source switch_env openvla && uv pip uninstall pynvml

# Install openvla-oft env
RUN uv venv openvla-oft --python=${PYTHON_VERSION}
RUN source switch_env openvla-oft && UV_TORCH_BACKEND=auto uv sync --active
RUN source switch_env openvla-oft && uv sync --extra embodied --active
RUN source switch_env openvla-oft && uv pip install git+https://github.com/moojink/openvla-oft.git --no-build-isolation
RUN source switch_env openvla-oft && uv pip install $FLASH_ATTENTION_WHEEL --no-build-isolation
RUN source switch_env openvla-oft && uv pip uninstall pynvml

# Install openpi env
RUN uv venv openpi --python=${PYTHON_VERSION}
RUN source switch_env openpi && UV_TORCH_BACKEND=auto uv sync --active
RUN source switch_env openpi && uv sync --extra embodied --active
RUN source switch_env openpi && GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/RLinf/openpi
RUN source switch_env openpi && uv pip install metaworld==3.0.0
RUN source switch_env openpi && uv pip uninstall pynvml
RUN cp -r ${UV_PATH}/openpi/lib/python${PYTHON_VERSION}/site-packages/openpi/models_pytorch/transformers_replace/* ${UV_PATH}/openpi/lib/python${PYTHON_VERSION}/site-packages/transformers/

RUN source switch_env openvla && download_assets
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openvla/bin/activate" >> ~/.bashrc

FROM embodied-common-image AS embodied-behavior-image

ENV PYTHON_VERSION=3.10
ENV FLASH_ATTENTION_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

ENV BEHAVIOR_REPO=RLinf/BEHAVIOR-1K
ENV BEHAVIOR_BRANCH=RLinf/v3.7.1
# For invalidating docker build cache when the branch updates
ADD https://api.github.com/repos/${BEHAVIOR_REPO}/git/refs/heads/${BEHAVIOR_BRANCH} /tmp/.behavior_version.json
RUN git clone -b ${BEHAVIOR_BRANCH} --depth 1 https://github.com/${BEHAVIOR_REPO}.git /opt/BEHAVIOR-1K

# Install openvla-oft env
RUN uv venv openvla-oft --python=${PYTHON_VERSION}
RUN source switch_env openvla-oft && UV_TORCH_BACKEND=auto uv sync --active
RUN source switch_env openvla-oft && uv sync --extra embodied --active
RUN source switch_env openvla-oft && cd /opt/BEHAVIOR-1K && UV_LINK_MODE=hardlink ./setup.sh --omnigibson --bddl --joylo --confirm-no-conda --accept-nvidia-eula --use-uv && cd -
RUN source switch_env openvla-oft && uv pip install git+https://github.com/moojink/openvla-oft.git --no-build-isolation
RUN source switch_env openvla-oft && uv pip install $FLASH_ATTENTION_WHEEL --no-build-isolation
RUN source switch_env openvla-oft && uv pip install ml_dtypes==0.5.3 protobuf==3.20.3
RUN source switch_env openvla-oft && cd && uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && cd -
RUN source switch_env openvla-oft && uv pip uninstall pynvml
# omnigibson downloads assets to /tmp during initialization
RUN source switch_env openvla-oft && python -c "import omnigibson as og; cfg={}; og.Environment(cfg); og.shutdown()" || exit 0

RUN source switch_env openvla-oft && download_assets
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

FROM ${BUILD_TARGET}-image AS final-image

# Clean up
RUN uv cache prune
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]