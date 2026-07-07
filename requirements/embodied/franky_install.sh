#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.


set -eo pipefail

if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

echo "=== franky system dependencies ==="

$SUDO apt-get update

# Core RT/diagnostic tools.
$SUDO apt-get install -y \
    rt-tests \
    ethtool \
    iputils-ping

# Build-from-source fallback deps.  Franky-control on PyPI usually
# ships a manylinux wheel, but if your Python/libfranka combination
# falls outside the wheel matrix pip will fall back to a source build
# and will need these.
$SUDO apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    libpoco-dev \
    libfmt-dev \
    git

# robotpkg apt source — provides pinocchio.  Without it the
# `apt-cache show robotpkg-py3*-pinocchio` lookup below finds nothing
# and pinocchio is silently skipped (i.e. it never installs).
$SUDO apt-get install -y lsb-release curl

# Detect Ubuntu codename (e.g., focal, jammy)
ubuntu_codename=""
if command -v lsb_release >/dev/null 2>&1; then
    ubuntu_codename=$(lsb_release -cs || true)
elif [ -f /etc/os-release ]; then
    ubuntu_codename=$(grep '^UBUNTU_CODENAME=' /etc/os-release | cut -d= -f2)
fi

if [ -n "$ubuntu_codename" ]; then
    $SUDO mkdir -p /etc/apt/keyrings
    curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
        | $SUDO tee /etc/apt/keyrings/robotpkg.asc >/dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub ${ubuntu_codename} robotpkg" \
        | $SUDO tee /etc/apt/sources.list.d/robotpkg.list >/dev/null
    $SUDO apt-get update
else
    echo "Could not detect Ubuntu codename; skipping robotpkg source (pinocchio may be unavailable)." >&2
fi

if apt-cache show robotpkg-py3*-pinocchio >/dev/null 2>&1; then
    $SUDO apt-get install -y robotpkg-py3*-pinocchio || true
fi

# Old pybind11 purge — harmless if already absent.
if dpkg -l pybind11-dev >/dev/null 2>&1; then
    echo "Removing Ubuntu apt pybind11-dev (2.4.3 is broken on Python 3.11)"
    $SUDO apt-get purge -y pybind11-dev
fi

echo ""
echo "franky system deps installed. See the dual-Franka guide for the"
echo "per-boot real-time tuning and verification steps (governor, sysctl,"
echo "NIC coalescing, cyclictest / ping baseline)."
