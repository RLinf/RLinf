#!/bin/bash

# Configure ROS apt source for the current Ubuntu version using USTC mirror.

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
    curl \
    lsb-release \
    gnupg \
    cmake \
    build-essential

# Detect Ubuntu codename (e.g., focal, jammy)
ubuntu_codename=""
if command -v lsb_release >/dev/null 2>&1; then
    ubuntu_codename=$(lsb_release -cs || true)
elif [ -f /etc/os-release ]; then
    ubuntu_codename=$(grep '^UBUNTU_CODENAME=' /etc/os-release | cut -d= -f2)
fi

if [ -z "$ubuntu_codename" ]; then
    echo "Failed to detect Ubuntu codename. Cannot configure ROS apt source automatically." >&2
    exit 1
fi

ros_mirror="http://mirrors.ustc.edu.cn/ros/ubuntu"
test_url="${ros_mirror}/dists/${ubuntu_codename}/"

# Check whether the ROS mirror provides packages for this Ubuntu codename
if ! curl -fsSL --head "$test_url" >/dev/null 2>&1; then
    echo "ROS Noetic mirror $ros_mirror does not appear to provide packages for Ubuntu codename '$ubuntu_codename'." >&2
    echo "Tested URL: $test_url" >&2
    echo "Please make sure you are running Ubuntu version 20.04 or below." >&2
    exit 1
fi

source_line="deb ${ros_mirror} ${ubuntu_codename} main"

# Check if the source already exists anywhere under /etc/apt
if sudo grep -Rqs -- "$source_line" /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null; then
    echo "ROS source already present in /etc/apt, skipping addition: $source_line"
else
    echo "$source_line" | sudo tee /etc/apt/sources.list.d/ros-latest.list >/dev/null
    echo "Added ROS source: $source_line"
fi

# Add ROS GPG key
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Install ROS Noetic packages
sudo apt update -y
sudo apt install -y --no-install-recommends ros-noetic-ros-base || {
    echo "Failed to install ROS Noetic packages. Please check your apt sources or install manually." >&2
    exit 1
}

# Install libfranka and franka_ros dependencies
# Ensure Ubuntu is 20.04 (Focal) for libfranka compatibility
if [ "$ubuntu_codename" != "focal" ]; then
    echo "libfranka is officially supported only on Ubuntu 20.04 (Focal)." >&2
    exit 1
fi

# libfranka dependencies
sudo apt-get install -y libpoco-dev libeigen3-dev libfmt-dev
sudo apt-get install -y lsb-release curl
sudo mkdir -p /etc/apt/keyrings
curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | sudo tee /etc/apt/keyrings/robotpkg.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $ubuntu_codename robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list
sudo apt-get update
sudo apt-get install -y robotpkg-pinocchio

# franka_ros dependencies
sudo apt-get install -y --no-install-recommends \
    ros-noetic-boost-sml \
    ros-noetic-ros-control \
    ros-noetic-eigen-conversions \
    ros-noetic-gazebo-dev \
    ros-noetic-gazebo-ros-control \
    ros-noetic-urdfdom-py \
    ros-noetic-tf-conversions \
    ros-noetic-kdl-parser

