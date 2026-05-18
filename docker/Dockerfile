ARG BUILD_TARGET=embodied
ARG TARGETARCH
# Hardware platform for the install: nvidia (default), amd (ROCm), or ascend (CANN).
ARG PLATFORM=nvidia
# Per-platform runtime versions. Override these for common CUDA/ROCm/CANN bumps, or
# override the full base-image args below for a different image family/Ubuntu.
ARG CUDA_VER=12.4.1
ARG ROCM_VER=6.4
ARG ROCM_ARCHS=gfx90a;gfx942
ARG CANN_VER=9.0.0-910b
ARG UBUNTU_VER=22.04
ARG NVIDIA_BASE_IMAGE=nvidia/cuda:${CUDA_VER}-cudnn-devel-ubuntu${UBUNTU_VER}
ARG AMD_BASE_IMAGE=rocm/dev-ubuntu-${UBUNTU_VER}:${ROCM_VER}-complete
ARG ASCEND_BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${CANN_VER}-ubuntu${UBUNTU_VER}-py3.11


##################################################################################################
# Platform base resolution (selects between NVIDIA, AMD/ROCm, and Ascend/CANN)
##################################################################################################
FROM ${NVIDIA_BASE_IMAGE} AS base-image-platform-nvidia
FROM ${AMD_BASE_IMAGE} AS base-image-platform-amd
ARG ROCM_ARCHS
ENV AMDGPU_TARGETS=${ROCM_ARCHS}
ENV GPU_ARCHS=${ROCM_ARCHS}
ENV HCC_AMDGPU_TARGET=${ROCM_ARCHS}
ENV PYTORCH_ROCM_ARCH=${ROCM_ARCHS}
ENV AMD_VISIBLE_DEVICES=all
FROM ${ASCEND_BASE_IMAGE} AS base-image-platform-ascend
ENV ASCEND_VISIBLE_DEVICES=all

##################################################################################################
# Base images for different build targets — all derive from the platform base
##################################################################################################
FROM base-image-platform-${PLATFORM} AS base-image-reason
FROM base-image-platform-${PLATFORM} AS base-image-embodied-maniskill_libero
FROM base-image-platform-${PLATFORM} AS base-image-embodied-behavior-openvlaoft
FROM base-image-platform-${PLATFORM} AS base-image-embodied-behavior-openpi
FROM base-image-platform-${PLATFORM} AS base-image-embodied-metaworld
FROM base-image-platform-${PLATFORM} AS base-image-embodied-calvin
FROM base-image-platform-${PLATFORM} AS base-image-embodied-robocasa
FROM base-image-platform-${PLATFORM} AS base-image-embodied-isaaclab
FROM base-image-platform-${PLATFORM} AS base-image-embodied-robotwin
FROM base-image-platform-${PLATFORM} AS base-image-embodied-opensora
FROM base-image-platform-${PLATFORM} AS base-image-embodied-frankasim
FROM base-image-platform-${PLATFORM} AS base-image-embodied-wan
FROM base-image-platform-${PLATFORM} AS base-image-embodied-libero
FROM base-image-platform-${PLATFORM} AS base-image-embodied-liberopro
FROM base-image-platform-${PLATFORM} AS base-image-embodied-liberoplus
FROM base-image-platform-${PLATFORM} AS base-image-embodied-embodichain
FROM base-image-platform-${PLATFORM} AS base-image-embodied-roboverse


# Franka depends on ros-noetic (Ubuntu 20.04) and is GPU-agnostic, so it
# bypasses the platform switch and always uses a plain Ubuntu base.
FROM ubuntu:20.04 AS base-image-embodied-franka


##################################################################################################
# Select base image according to BUILD_TARGET
##################################################################################################
FROM base-image-$BUILD_TARGET AS base-image
ARG PLATFORM
ENV RLINF_PLATFORM=${PLATFORM}


##################################################################################################
# System-level setup (apt, mirrors, basic tools)
##################################################################################################
ARG TARGETARCH
ARG NO_MIRROR
SHELL ["/bin/bash", "-c"]
ENV NO_MIRROR=${NO_MIRROR}
ENV DEBIAN_FRONTEND=noninteractive
RUN if [ "$NO_MIRROR" != "1" ]; then \
        if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
            if [ "$TARGETARCH" = "arm64" ]; then \
                sed -i 's@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources; \
            else \
                sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources; \
            fi; \
        elif [ "$TARGETARCH" = "arm64" ]; then \
            sed -i \
                -e 's@//ports.ubuntu.com/\? @//ports.ubuntu.com/ubuntu-ports @g' \
                -e 's@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g' \
                /etc/apt/sources.list; \
        else \
            sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list; \
        fi; \
    fi
RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim libibverbs-dev openssh-server sudo runit runit-systemd tmux \
    build-essential python3-dev cmake pkg-config iproute2 pciutils python3 python3-pip \
    wget unzip curl


##################################################################################################
# Python / pip / uv setup
##################################################################################################
RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
RUN python3 -m pip install -i https://mirrors.ustc.edu.cn/pypi/simple --upgrade pip setuptools wheel uv

ENV HF_HOME=/opt/.cache/huggingface
RUN mkdir -p $HF_HOME

ENV UV_DEFAULT_INDEX=${NO_MIRROR:+"https://pypi.org/simple/"}
ENV UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-"https://mirrors.ustc.edu.cn/pypi/simple"}
ENV INSTALL_MIRROR_OPTION=${NO_MIRROR:+" "}
ENV INSTALL_MIRROR_OPTION=${INSTALL_MIRROR_OPTION:-"--use-mirror"}

ENV UV_PATH=/opt/venv
RUN mkdir $UV_PATH
WORKDIR $UV_PATH

ENV UV_LINK_MODE=hardlink
ENV UV_CACHE_DIR=$UV_PATH/.cache
ENV UV_PYTHON_INSTALL_DIR=${UV_PATH}/.python


##################################################################################################
# Project requirements
##################################################################################################
COPY ./pyproject.toml $UV_PATH/pyproject.toml
COPY ./requirements $UV_PATH/requirements


##################################################################################################
# Utilities: switch_env
##################################################################################################
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


##################################################################################################
# Reason image
##################################################################################################
FROM base-image AS reason-image

RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} agentic --venv reason

# Set default env
RUN echo "source ${UV_PATH}/reason/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied common base (shared by all embodied environments)
##################################################################################################
FROM base-image AS embodied-common-image

# Embodied NVIDIA_DRIVER_CAPABILITIES
ENV NVIDIA_DRIVER_CAPABILITIES="all"


##################################################################################################
# Utilities: asset linking & downloading
##################################################################################################
RUN cat <<EOF > /usr/local/bin/link_assets
#!/bin/bash
if [ -d /opt/assets/.maniskill ]; then
    if [ -d ~/.maniskill ]; then
        rm -rf ~/.maniskill
    fi
    ln -s /opt/assets/.maniskill ~/.maniskill
fi
if [ -d /opt/assets/.sapien ]; then
    if [ -d ~/.sapien ]; then
        rm -rf ~/.sapien
    fi
    ln -s /opt/assets/.sapien ~/.sapien
fi
mkdir -p ~/.cache
if [ -d /opt/assets/.cache/openpi ]; then
    if [ -d ~/.cache/openpi ]; then
        rm -rf ~/.cache/openpi
    fi
    ln -s /opt/assets/.cache/openpi ~/.cache/openpi
fi
EOF
RUN chmod +x /usr/local/bin/link_assets

RUN cp $UV_PATH/requirements/embodied/download_assets.sh /usr/local/bin/download_assets
RUN chmod +x /usr/local/bin/download_assets


##################################################################################################
# Embodied: ManiSkill / Libero
##################################################################################################
FROM embodied-common-image AS embodied-maniskill_libero-image

# Install openvla env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla --model openvla --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv gr00t --model gr00t --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv dexbotic --model dexbotic --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv starvla --model starvla --env maniskill_libero

RUN source switch_env openvla && download_assets --dir /opt/assets --assets maniskill,openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openvla/bin/activate" >> ~/.bashrc

FROM embodied-common-image AS embodied-behavior-openvlaoft-image

# Install openvla-oft env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env behavior

# omnigibson downloads assets to /tmp during initialization
RUN source switch_env openvla-oft && python -c "import omnigibson as og; cfg={}; og.Environment(cfg); og.shutdown()" || exit 0

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

FROM embodied-common-image AS embodied-behavior-openpi-image

# Install openpi env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env behavior

# omnigibson downloads assets to /tmp during initialization
RUN source switch_env openpi && python -c "import omnigibson as og; cfg={}; og.Environment(cfg); og.shutdown()" || exit 0

RUN source switch_env openpi && download_assets --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: MetaWorld
##################################################################################################
FROM embodied-common-image AS embodied-metaworld-image

# Install openvla-oft and openpi envs
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env metaworld && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env metaworld

RUN source switch_env openpi && download_assets --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: CALVIN
##################################################################################################
FROM embodied-common-image AS embodied-calvin-image

# Install openvla-oft and openpi envs
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env calvin && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env calvin

RUN source switch_env openpi && download_assets --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: RoboCasa
##################################################################################################
FROM embodied-common-image AS embodied-robocasa-image

# Install openpi env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env robocasa

RUN source switch_env openpi && download_assets --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: IsaacLab
##################################################################################################
FROM embodied-common-image AS embodied-isaaclab-image

# Install gr00t env, and openpi env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv gr00t --model gr00t --env isaaclab && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env isaaclab

# Set default env
RUN echo "source ${UV_PATH}/gr00t/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: Franka (Ubuntu 20.04, ROS Noetic)
##################################################################################################
FROM embodied-common-image AS embodied-franka-image

# Install multiple Franka environments with different libfranka versions
# franka uses ubuntu:20.04 with no GPU stack regardless of build-arg platform,
# so always invoke install.sh with --platform nvidia to skip ROCm detection.
RUN for v in 0.10.0 0.13.3 0.14.1 0.15.0 0.18.0; do \
            LIBFRANKA_VERSION="$v" FRANKA_ROS_VERSION=0.10.0 bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform nvidia embodied --venv "franka-$v" --env franka; \
        done

# Set default env (recommended libfranka 0.15.0)
RUN echo "source ${UV_PATH}/franka-0.15.0/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: RoboTwin
##################################################################################################
FROM embodied-common-image AS embodied-robotwin-image

# Install openvla-oft, openpi, and lingbotvla envs
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env robotwin && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env robotwin && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv lingbotvla --model lingbotvla --env robotwin

RUN source switch_env openpi && download_assets --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: OpenSora
##################################################################################################
FROM embodied-common-image AS embodied-opensora-image

# Install openvla-oft env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env opensora

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: WAN
##################################################################################################
FROM embodied-common-image AS embodied-wan-image

# Install openvla-oft env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env wan

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: FrankaSim
##################################################################################################
FROM embodied-common-image AS embodied-frankasim-image

# Install openvla env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla --model openvla --env frankasim

# Set default env
RUN echo "source ${UV_PATH}/openvla/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: EmbodiChain
##################################################################################################
FROM embodied-common-image AS embodied-embodichain-image

RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv embodichain --env embodichain

RUN echo "source ${UV_PATH}/embodichain/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: LIBERO
##################################################################################################
FROM embodied-common-image AS embodied-libero-image

# Install openvla env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla --model openvla --env libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env libero

# Set default env
RUN echo "source ${UV_PATH}/openvla/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: LIBERO-Pro
##################################################################################################
FROM embodied-common-image AS embodied-liberopro-image

# Install openvla-oft env for LIBERO-Pro
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env liberopro

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: LIBERO-Plus
##################################################################################################
FROM embodied-common-image AS embodied-liberoplus-image

# Install openvla-oft env for LIBERO-Plus
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env liberoplus

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: RoboVerse
##################################################################################################
FROM embodied-common-image AS embodied-roboverse-image

# Install openpi env for RoboVerse
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env roboverse

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc

##################################################################################################
# Final image selection
##################################################################################################
FROM ${BUILD_TARGET}-image AS final-image


##################################################################################################
# Cleanup & runtime defaults
##################################################################################################
RUN uv cache prune
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]