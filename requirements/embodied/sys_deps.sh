#!/bin/bash

# Embodied dependencies

# Check if apt is available
if ! command -v apt-get &> /dev/null; then
    echo "apt-get could not be found. This script is intended for Debian-based systems."
    exit 1
fi

# Check for sudo privileges
if ! sudo -n true 2>/dev/null; then
    # Check if already running as root
    if [ "$EUID" -eq 0 ]; then
        apt-get update -y
        apt-get install -y --no-install-recommends sudo
    else
        echo "This script requires sudo privileges. Please run as a user with sudo access."
        exit 1
    fi
fi

sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    wget \
    unzip \
    curl \
    lsb-release \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libibverbs-dev \
    ncurses-term \
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
    libgomp1 || {
        ubuntu_ver=""
        if command -v lsb_release >/dev/null 2>&1; then
            ubuntu_ver=$(lsb_release -rs || true)
        elif [ -f /etc/os-release ]; then
            ubuntu_ver=$(grep '^VERSION_ID=' /etc/os-release | cut -d= -f2 | tr -d '"')
        fi

        if [ -n "$ubuntu_ver" ]; then
            # Check if version is higher than 22.04
            if [ "$(printf '%s\n' "22.04" "$ubuntu_ver" | sort -V | head -n1)" = "22.04" ] && [ "$ubuntu_ver" != "22.04" ]; then
                echo "apt-get install failed and your Ubuntu version ($ubuntu_ver) is higher than 22.04. This script currently supports Ubuntu 22.04 or lower; please use 22.04/below or install dependencies manually." >&2
            else
                echo "apt-get install failed on Ubuntu $ubuntu_ver. Please check your apt sources or install dependencies manually." >&2
            fi
        else
            echo "apt-get install failed and the Ubuntu version could not be detected. Please ensure you are using Ubuntu version 22.04/below or install dependencies manually." >&2
        fi
        exit 1
    }

# Install rendering runtime configuration files if not exist
sudo mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d
if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi
if [ ! -f /usr/share/glvnd/egl_vendor.d/50_mesa.json ]; then
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_mesa.so.0"\n    }\n}\n' | sudo tee /usr/share/glvnd/egl_vendor.d/50_mesa.json
fi
if [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194"\n    }\n}\n' | sudo tee /etc/vulkan/icd.d/nvidia_icd.json
fi
if [ ! -f /etc/vulkan/implicit_layer.d/nvidia_layers.json ]; then
    printf '{\n    "file_format_version" : "1.0.0",\n    "layer": {\n        "name": "VK_LAYER_NV_optimus",\n        "type": "INSTANCE",\n        "library_path": "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194",\n        "implementation_version" : "1",\n        "description" : "NVIDIA Optimus layer",\n        "functions": {\n            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",\n            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"\n        },\n        "enable_environment": {\n            "__NV_PRIME_RENDER_OFFLOAD": "1"\n        },\n        "disable_environment": {\n            "DISABLE_LAYER_NV_OPTIMUS_1": ""\n        }\n    }\n}\n' | sudo tee /etc/vulkan/implicit_layer.d/nvidia_layers.json
fi


