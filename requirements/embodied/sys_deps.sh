#!/bin/bash

# Embodied dependencies
# Supports Debian/Ubuntu (apt), RHEL/CentOS/Fedora/AlmaLinux/RockyLinux (dnf/yum), and Arch Linux (pacman)

# Detect package manager and OS
detect_pkg_manager() {
    if command -v apt-get &> /dev/null; then
        echo "apt"
    elif command -v dnf &> /dev/null; then
        echo "dnf"
    elif command -v yum &> /dev/null; then
        echo "yum"
    elif command -v pacman &> /dev/null; then
        echo "pacman"
    else
        echo "unknown"
    fi
}

PKG_MANAGER=$(detect_pkg_manager)

if [ "$PKG_MANAGER" = "unknown" ]; then
    echo "No supported package manager found (apt, dnf, yum, or pacman)."
    echo "Please install dependencies manually."
    exit 1
fi

install_sudo() {
    local cmd_prefix=()

    if [ "$EUID" -ne 0 ]; then
        if ! command -v su >/dev/null 2>&1; then
            echo "sudo is not installed and 'su' is unavailable to install it. Please install sudo or run as root."
            exit 1
        fi
        cmd_prefix=(su -c)
    fi

    run_with_su() {
        local cmd="$*"
        if [ ${#cmd_prefix[@]} -eq 0 ]; then
            eval "$cmd"
        else
            "${cmd_prefix[@]}" "$cmd"
        fi
    }

    case "$PKG_MANAGER" in
        apt)
            run_with_su "apt-get update -y && apt-get install -y --no-install-recommends sudo"
            ;;
        dnf)
            run_with_su "dnf -y update && dnf install -y sudo"
            ;;
        yum)
            run_with_su "yum -y update && yum install -y sudo"
            ;;
        pacman)
            run_with_su "pacman -Sy --noconfirm sudo"
            ;;
    esac
}

# Privilege and sudo availability checks
if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo not found; installing with root privileges..."
    install_sudo
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "This script requires sudo to be installed. Please install sudo or run as root."
    exit 1
fi

# Verify sudo works non-interactively after ensuring it exists
if ! sudo -n true 2>/dev/null; then
    echo "This script requires sudo privileges. Please run as a user with sudo access."
    exit 1
fi

# Install packages based on package manager
install_deps_apt() {
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends libgl1-mesa-glx || sudo apt-get install -y --no-install-recommends libglx-mesa0
    sudo apt-get install -y --no-install-recommends \
        wget \
        unzip \
        curl \
        cmake \
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
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        libaio-dev \
        libgomp1 || {
            echo "apt-get install failed. Please check your repositories or install dependencies manually." >&2
            exit 1
        }
}

install_deps_dnf() {
    # DNF package names for RHEL/CentOS/Fedora/AlmaLinux
    sudo dnf install -y epel-release 2>/dev/null || true  # Enable EPEL for extra packages
    # Enable CRB (CodeReady Builder) repository for additional packages
    sudo dnf config-manager --set-enabled crb 2>/dev/null || \
        sudo dnf config-manager --set-enabled powertools 2>/dev/null || true
    sudo dnf install -y --allowerasing \
        wget \
        unzip \
        curl \
        cmake \
        ffmpeg-free-devel \
        libibverbs-devel \
        ncurses \
        mesa-demos \
        mesa-libOSMesa \
        freeglut-devel \
        glew-devel \
        mesa-libEGL \
        mesa-libGLES \
        libglvnd-devel \
        glfw-devel \
        mesa-libGL-devel \
        glib2 \
        libSM \
        libXext \
        libXrender-devel \
        libXrandr-devel \
        libXinerama-devel \
        libXcursor-devel \
        libXi-devel \
        libaio-devel \
        libgomp || {
            echo "dnf install failed. Please check your repositories or install dependencies manually." >&2
            exit 1
        }
}

install_deps_yum() {
    # YUM package names (similar to DNF)
    sudo yum install -y epel-release 2>/dev/null || true  # Enable EPEL for extra packages
    sudo yum install -y \
        wget \
        unzip \
        curl \
        cmake \
        ffmpeg-devel \
        libibverbs-devel \
        ncurses \
        mesa-demos \
        mesa-libOSMesa \
        freeglut-devel \
        glew-devel \
        mesa-libEGL \
        mesa-libGLES \
        libglvnd-devel \
        glfw-devel \
        mesa-libGL-devel \
        glib2 \
        libSM \
        libXext \
        libXrender-devel \
        libXrandr-devel \
        libXinerama-devel \
        libXcursor-devel \
        libXi-devel \
        libaio-devel \
        libgomp || {
            echo "yum install failed. Please check your repositories or install dependencies manually." >&2
            exit 1
        }
}

install_deps_pacman() {
    # Pacman package names for Arch Linux
    sudo pacman -Sy --noconfirm \
        wget \
        unzip \
        curl \
        lsb-release \
        cmake \
        ffmpeg \
        rdma-core \
        ncurses \
        mesa-utils \
        mesa \
        freeglut \
        glew \
        libglvnd \
        glfw \
        glib2 \
        libsm \
        libxext \
        libxrender \
        libxrandr \
        libxinerama \
        libxcursor \
        libxi \
        libaio \
        gcc || {
            echo "pacman install failed. Please check your repositories or install dependencies manually." >&2
            exit 1
        }
}

# Run installation based on detected package manager
case "$PKG_MANAGER" in
    apt)
        echo "Detected Debian/Ubuntu system (apt)"
        install_deps_apt
        ;;
    dnf)
        echo "Detected RHEL/CentOS/Fedora/AlmaLinux/RockyLinux system (dnf)"
        install_deps_dnf
        ;;
    yum)
        echo "Detected RHEL/CentOS system (yum)"
        install_deps_yum
        ;;
    pacman)
        echo "Detected Arch Linux system (pacman)"
        install_deps_pacman
        ;;
esac

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


