ARG BUILD_TARGET=reason

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image-reason
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image-embodied-maniskill_libero
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image-embodied-behavior
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image-embodied-metaworld
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image-embodied-calvin
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image-embodied-robocasa
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image-embodied-isaaclab
# Franka depends on ros-noetic, which is not available on Ubuntu 22.04
FROM ubuntu:20.04 AS base-image-embodied-franka

FROM base-image-$BUILD_TARGET AS base-image

ARG NO_MIRROR
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN if [ -z "$NO_MIRROR" ]; then sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list; fi
RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim libibverbs-dev openssh-server sudo runit runit-systemd tmux \
    build-essential python3-dev cmake pkg-config iproute2 pciutils python3 python3-pip \
    wget unzip curl

RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
RUN python3 -m pip install -i https://mirrors.ustc.edu.cn/pypi/simple --upgrade pip setuptools wheel uv

ENV HF_HOME=/opt/.cache/huggingface
RUN mkdir -p $HF_HOME

# UV setup
ENV UV_DEFAULT_INDEX=${NO_MIRROR:+"https://pypi.org/simple/"}
ENV UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-"https://mirrors.ustc.edu.cn/pypi/simple"}
ENV UV_PATH=/opt/venv
RUN mkdir $UV_PATH
WORKDIR $UV_PATH
ENV UV_LINK_MODE=symlink
ENV UV_CACHE_DIR=$UV_PATH/.cache
ENV UV_PYTHON_INSTALL_DIR=${UV_PATH}/.python

# Requirements
COPY ./pyproject.toml $UV_PATH/pyproject.toml
COPY ./requirements $UV_PATH/requirements

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

FROM base-image AS reason-image

RUN bash requirements/install.sh reason --venv reason

# Set default env
RUN echo "source ${UV_PATH}/reason/bin/activate" >> ~/.bashrc

FROM base-image AS embodied-common-image

# Embodied NVIDIA_DRIVER_CAPABILITIES
ENV NVIDIA_DRIVER_CAPABILITIES="all"

# link_assets utility
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

FROM embodied-common-image AS embodied-maniskill_libero-image

# Install openvla env
RUN bash requirements/install.sh embodied --venv openvla --model openvla --env maniskill_libero

# Install openvla-oft env
RUN bash requirements/install.sh embodied --venv openvla-oft --model openvla-oft --env maniskill_libero

# Install openpi env
RUN bash requirements/install.sh embodied --venv openpi --model openpi --env maniskill_libero

# Install gr00t env
RUN bash requirements/install.sh embodied --venv gr00t --model gr00t --env maniskill_libero

RUN source switch_env openvla && download_assets --dir /opt/assets --assets maniskill,openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openvla/bin/activate" >> ~/.bashrc

FROM embodied-common-image AS embodied-behavior-image

# Install openvla-oft env
RUN bash requirements/install.sh embodied --venv openvla-oft --model openvla-oft --env behavior
# omnigibson downloads assets to /tmp during initialization
RUN source switch_env openvla-oft && python -c "import omnigibson as og; cfg={}; og.Environment(cfg); og.shutdown()" || exit 0

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

FROM embodied-common-image AS embodied-metaworld-image

# Install openpi env
RUN bash requirements/install.sh embodied --venv openpi --model openpi --env metaworld

RUN source switch_env openpi && download_assets --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc

FROM embodied-common-image AS embodied-calvin-image

# Install openpi env
RUN bash requirements/install.sh embodied --venv openpi --model openpi --env calvin

RUN source switch_env openpi && download_assets --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc

FROM embodied-common-image AS embodied-robocasa-image

# Install openpi env
RUN bash requirements/install.sh embodied --venv openpi --model openpi --env robocasa

RUN source switch_env openpi && download_assets --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc

FROM embodied-common-image AS embodied-isaaclab-image

# Install gr00t env
RUN bash requirements/install.sh embodied --venv gr00t --model gr00t --env isaaclab

# Set default env
RUN echo "source ${UV_PATH}/gr00t/bin/activate" >> ~/.bashrc

FROM embodied-common-image AS embodied-franka-image

# Install multiple Franka environments with different libfranka versions
RUN for v in 0.10.0 0.13.3 0.14.1 0.15.0 0.18.0; do \
            LIBFRANKA_VERSION="$v" FRANKA_ROS_VERSION=0.10.0 bash requirements/install.sh embodied --venv "franka-$v" --env franka; \
        done

# Set default env (recommended libfranka 0.15.0)
RUN echo "source ${UV_PATH}/franka-0.15.0/bin/activate" >> ~/.bashrc

FROM ${BUILD_TARGET}-image AS final-image

# Clean up
RUN uv cache prune
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]