Requirements
============

Match your hardware to the workload you want to run. You can start on a single 24 GB GPU and scale the same recipes to multi-node clusters without code changes.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/hardware_tiers.svg
   :align: center
   :width: 100%

   RLinf hardware tiers from a single GPU to multi-node clusters.

GPU Tiers by Workload
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - Tier
     - Workloads
     - Example Starting Point
   * - 1 GPU (24 GB+)
     - MLP / CNN simulation RL, VLA evaluation, LoRA SFT, real-robot data collection, and deployment
     - :doc:`Quick Start (MLP) <vla>`, :doc:`Evaluation Quick Tour <../evaluations/get_started/quick_tour>`
   * - 1–2 GPUs (40–80 GB)
     - Single-workstation OpenVLA / OpenPI RL quick start, collocated PPO / GRPO
     - :doc:`Quick Start (VLA) <vla>`
   * - 4–8 GPUs (80 GB)
     - Full-parameter VLA SFT and RL benchmarks (LIBERO, ManiSkill, RoboTwin), 7B–14B reasoning RL
     - :doc:`Embodied Training <../examples/index>`
   * - Multi-node (8×H100+)
     - Large-scale disaggregated and hybrid runs, Megatron + SGLang / vLLM, world models
     - :doc:`Multi-Node Training <../guides/multi_node>`

RLinf also supports AMD ROCm, Huawei Ascend CANN, Moore Threads MUSA, Intel XPU, Kunlunxin XPU, MetaX MACA, and Biren SUPA. See :ref:`the hardware support matrix <model-hardware-support>` for verified model and environment combinations.

Software
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Supported / Tested
   * - Operating System
     - Linux (tested on Ubuntu 22.04)
   * - Python
     - 3.10–3.12 (managed automatically by ``requirements/install.sh``)
   * - NVIDIA Driver & CUDA
     - Driver 535+ with graphics / EGL support for simulation rendering; CUDA 12.4+
   * - Containers (optional)
     - Docker 26+ with NVIDIA Container Toolkit (images on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__)

Large-Scale Reference Cluster
-----------------------------

Large-scale multi-node benchmarks and full reproductions run on this cluster profile:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Reference Configuration
   * - GPU
     - 8×H100 per node
   * - CPU
     - 192 cores per node
   * - Memory
     - 1.8 TB per node
   * - Network
     - NVLink + RoCE / IB (3.2 Tbps)
   * - Storage
     - 1 TB local SSD (single-node) / 10 TB shared NAS (multi-node)
