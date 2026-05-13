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

if apt-cache show robotpkg-py3*-pinocchio >/dev/null 2>&1; then
    $SUDO apt-get install -y robotpkg-py3*-pinocchio || true
fi

# Old pybind11 purge — harmless if already absent.
if dpkg -l pybind11-dev >/dev/null 2>&1; then
    echo "Removing Ubuntu apt pybind11-dev (2.4.3 is broken on Python 3.11)"
    $SUDO apt-get purge -y pybind11-dev
fi

echo ""
cat <<'EOF'
================================================================
 franky system deps installed.

 Per-boot RT tuning — NOT automated, run manually whenever you
 reboot the workstation that talks to the Franka:

   # CPU governor → performance (biggest single contributor to
   # 1 kHz cycle jitter — P-state transitions cost 100-400 µs)
   sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
       echo performance > "$g"
   done'

   # Unthrottle RT scheduling budget (default 950000 / 1000000 us
   # throttles SCHED_FIFO threads to 95% CPU which breaks tight loops)
   sudo sysctl -w kernel.sched_rt_runtime_us=-1

   # Close NIC interrupt coalescing on the Franka link so ICMP/TCP
   # responses don't get batched with up to 1 ms of extra jitter
   sudo ethtool -C eno1 rx-usecs 0 tx-usecs 0 2>/dev/null || true

 Persist the sysctl across reboots (optional):

   echo 'kernel.sched_rt_runtime_us = -1' | sudo tee /etc/sysctl.d/99-franka-rt.conf

 One-time limits — check these are set (should already be via
 /etc/security/limits.d/99-realtime.conf on this host):

   ulimit -r       # expected: 99 or unlimited
   ulimit -l       # expected: unlimited

 Diagnostic / baseline:

   uname -a | grep -o PREEMPT_RT           # must print PREEMPT_RT
   sudo cyclictest -p 80 -t 4 -i 1000 -l 300000 -m
       # max latency < 150 µs is healthy; > 500 µs is broken
   sudo ping -c 1000 -i 0.001 172.16.0.2 | tail -3
       # Franka direct link: avg < 0.5 ms, max < 2 ms
================================================================
EOF
