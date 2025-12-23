#!/bin/bash

# Embodied dependencies
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    wget \
    unzip \
    curl \
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
    libgomp1

# Install rendering runtime configuration files if not exist
sudo mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d
if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    sudo printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json || true
fi
if [ ! -f /usr/share/glvnd/egl_vendor.d/50_mesa.json ]; then
    sudo printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_mesa.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/50_mesa.json || true
fi
if [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    sudo printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194"\n    }\n}\n' > /etc/vulkan/icd.d/nvidia_icd.json || true
fi
if [ ! -f /etc/vulkan/implicit_layer.d/nvidia_layers.json ]; then
    sudo printf '{\n    "file_format_version" : "1.0.0",\n    "layer": {\n        "name": "VK_LAYER_NV_optimus",\n        "type": "INSTANCE",\n        "library_path": "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194",\n        "implementation_version" : "1",\n        "description" : "NVIDIA Optimus layer",\n        "functions": {\n            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",\n            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"\n        },\n        "enable_environment": {\n            "__NV_PRIME_RENDER_OFFLOAD": "1"\n        },\n        "disable_environment": {\n            "DISABLE_LAYER_NV_OPTIMUS_1": ""\n        }\n    }\n}\n' > /etc/vulkan/implicit_layer.d/nvidia_layers.json || true
fi


