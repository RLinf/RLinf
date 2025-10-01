FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG BUILD_TARGET=reason

SHELL ["/bin/bash", "-c"]
ENV PATH=/opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim libibverbs-dev openssh-server sudo runit runit-systemd tmux \
    build-essential python3-dev cmake pkg-config iproute2 pciutils python3 python3-pip

# Embodied dependencies
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

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN python3 -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --upgrade pip setuptools wheel uv

ENV HF_HOME=/opt/.cache/huggingface
RUN mkdir -p $HF_HOME

# UV index
RUN mkdir -p /etc/uv
RUN cat <<EOF > /etc/uv/uv.toml
[[index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
default = true
EOF

# UV setup
ENV UV_PATH=/opt/venv
RUN mkdir $UV_PATH
WORKDIR $UV_PATH
COPY project $UV_PATH/pyproject.toml
ENV UV_LINK_MODE=symlink
ENV UV_CACHE_DIR=$UV_PATH/.cache

# Install Megatron-LM
RUN git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.13.0 /opt/Megatron-LM
ENV PYTHONPATH=/opt/Megatron-LM:$PYTHONPATH

# Install reasoning env
RUN if [ "$BUILD_TARGET" = "reason" ]; then \
        uv venv reason && source ${UV_PATH}/reason/bin/activate && \
        UV_TORCH_BACKEND=auto uv sync --active && \
        uv sync --extra sglang-vllm --active && \
        APEX_CPP_EXT=1 APEX_CUDA_EXT=1 NVCC_APPEND_FLAGS="--threads 24" APEX_PARALLEL_BUILD=24 uv pip install git+https://github.com/NVIDIA/apex.git --no-build-isolation && \
        uv pip install transformer_engine[pytorch]==2.1.0 --no-build-isolation && \
        uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl && \
        uv pip uninstall pynvml; \
    fi

# Install openvla env
RUN if [ "$BUILD_TARGET" = "embodied" ]; then \
        uv venv openvla && source ${UV_PATH}/openvla/bin/activate && \
        UV_TORCH_BACKEND=auto uv sync --active && \
        uv sync --extra embodied --active && \
        uv pip install git+https://github.com/openvla/openvla.git --no-build-isolation && \
        uv pip install flash-attn==2.5.5 --no-build-isolation && \
        uv pip uninstall pynvml; \
    fi

# Install openvla-oft env
RUN if [ "$BUILD_TARGET" = "embodied" ]; then \
        uv venv openvla-oft && source ${UV_PATH}/openvla-oft/bin/activate && \
        UV_TORCH_BACKEND=auto uv sync --active && \
        uv sync --extra embodied --active && \
        uv pip install git+https://github.com/moojink/openvla-oft.git --no-build-isolation && \
        uv pip install flash-attn==2.5.5 --no-build-isolation && \
        uv pip uninstall pynvml; \
    fi

# Install pi0 env
RUN if [ "$BUILD_TARGET" = "embodied" ]; then \
        uv venv pi0 && source ${UV_PATH}/pi0/bin/activate && \
        UV_TORCH_BACKEND=auto uv sync --active && \
        uv sync --extra embodied --active && \
        uv pip install lerobot>=0.3.3 && \
        uv pip install flash-attn==2.5.5 --no-build-isolation && \
        uv pip uninstall pynvml; \
    fi

# Install maniskill assets
RUN if [ "$BUILD_TARGET" = "embodied" ]; then \
        source ${UV_PATH}/openvla/bin/activate && \
        python -m mani_skill.utils.download_asset bridge_v2_real2sim -y && \
        python -m mani_skill.utils.download_asset widowx250s -y; \
    fi

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

# Set default environment
RUN if [ "$BUILD_TARGET" = "reason" ]; then \
        echo "source ${UV_PATH}/reason/bin/activate" >> ~/.bashrc; \
    elif [ "$BUILD_TARGET" = "embodied" ]; then \
        echo "source ${UV_PATH}/openvla/bin/activate" >> ~/.bashrc; \
    fi

# Clean up
RUN uv clean prune
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]