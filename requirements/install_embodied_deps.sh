#!/bin/bash

# Embodied dependencies
apt-get update -y
apt-get install -y --no-install-recommends \
    wget \
    unzip \
    curl \
    libibverbs-dev \
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

python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
python -m mani_skill.utils.download_asset widowx250s -y

PHYSX_VERSION=105.1-physx-5.3.1.patch0
PHYSX_DIR=~/.sapien/physx/$PHYSX_VERSION
mkdir -p $PHYSX_DIR && wget -O $PHYSX_DIR/linux-so.zip https://github.com/sapien-sim/physx-precompiled/releases/download/$PHYSX_VERSION/linux-so.zip && unzip $PHYSX_DIR/linux-so.zip -d $PHYSX_DIR && rm $PHYSX_DIR/linux-so.zip


