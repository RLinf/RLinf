Installation
============

Install RLinf into a local ``uv`` virtual environment with ``requirements/install.sh``, or pull a prebuilt image from Docker Hub. Each virtual environment holds one ``--model`` + ``--env`` (or ``agentic --engine``) stack; pass ``--venv <dir>`` when installing additional stacks so you do not overwrite ``.venv``. Every recipe page also includes its exact setup command; use this page as the global installer and hardware reference.

Setup and Verify
----------------

Choose your target stack below and run the commands:

.. tab-set::

   .. tab-item:: Embodied (Sim)
      :sync: embodied

      Swap ``--model`` and ``--env`` from the table below to match your model and simulator (omit ``--model`` for lightweight MLP / CNN policies):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate
         python -c "import torch, rlinf; print(rlinf.__file__, 'CUDA:', torch.cuda.is_available())" && ray --version

      Expected output:

      .. code-block:: text

         /path/to/RLinf/rlinf/__init__.py CUDA: True
         ray, version 2.x.x

   .. tab-item:: Real-World Robot
      :sync: realworld

      Install the robot driver stack on the controller node (no GPU needed) and the model stack on the GPU node (swap ``franka`` for ``arx5``, ``yam``, ``xsquare_turtle2``, ``so101``, ``piper``, ``lekiwi``, ``dosw1``, or ``gim_arm``; for Robbyant R2 over RFrame/ROS 2, see :doc:`../examples/embodied/r2`, and see :doc:`../examples/real_world_index` for all robots):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf

         # Robot controller node (driver + sensors)
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

         # GPU / training node (or single workstation with both GPU and robot)
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

   .. tab-item:: Agentic / Reasoning
      :sync: agentic

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate
         python -c "import torch, rlinf; print(rlinf.__file__, 'CUDA:', torch.cuda.is_available())" && ray --version

      Expected output:

      .. code-block:: text

         /path/to/RLinf/rlinf/__init__.py CUDA: True
         ray, version 2.x.x

   .. tab-item:: Prebuilt Docker
      :sync: docker

      Pick an image tag from Docker Hub for your model and simulator, then launch a GPU container with graphics/EGL enabled:

      .. code-block:: bash

         docker run -it --rm --gpus all --shm-size=16g --network=host \
           -e NVIDIA_DRIVER_CAPABILITIES=all \
           -v "$(pwd)":/workspace/RLinf \
           rlinf/rlinf:<tag> bash

      .. grid:: 1 1 1 1
         :gutter: 2

         .. grid-item-card:: Docker Hub — rlinf/rlinf
            :link: https://hub.docker.com/r/rlinf/rlinf

            Browse prebuilt images, tag lists, container launch commands, and regional mirrors on Docker Hub.

   .. tab-item:: Docs
      :sync: docs

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh docs --venv .docs-venv

.. tip::

   **Mainland China mirror:** append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

Supported Models, Environments, and Platforms
---------------------------------------------

Pass any of the values below to ``--model``, ``--env``, ``--engine``, or ``--platform``:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Option
     - Supported Values
   * - ``--model`` (:doc:`VLA <../examples/vla_wam_index>`)
     - ``openvla``, ``openvla-oft``, ``openpi``, ``psi0``, ``pi0_fast``, ``gr00t``, ``gr00t_n1d6``, ``gr00t_n1d7``, ``dexbotic``, ``starvla``, ``lingbotvla``, ``lingbotvlav2``, ``abot_m0``, ``molmoact2``, ``evo1``
   * - ``--model`` (:doc:`WAM / VLM / Gen <../examples/vla_wam_index>`)
     - ``dreamzero``, ``fastwam``, ``cosmos3``, ``qwen3_vl``, ``diffusion``, ``sglang``
   * - ``--env`` (:doc:`Simulators <../examples/simulators_index>`)
     - ``maniskill_libero``, ``libero``, ``liberopro``, ``liberoplus``, ``behavior``, ``metaworld``, ``calvin``, ``isaaclab``, ``robocasa``, ``robocasa365``, ``frankasim``, ``robotwin``, ``robodojo``, ``roboverse``, ``habitat``, ``genesis``, ``geniesim``, ``embodichain``, ``d4rl``, ``mujoco_warp``, ``polaris``, ``simple``, ``robolab``
   * - ``--env`` (:doc:`World Models <../examples/world_model_index>`)
     - ``opensora``, ``wan``
   * - ``--env`` (:doc:`Real Robots <../examples/real_world_index>`)
     - ``franka``, ``franka-ros``, ``xsquare_turtle2``, ``dosw1``, ``gim_arm``, ``so101``, ``lekiwi``, ``piper``, ``arx5``, ``yam``, ``dummy`` (Robbyant R2 uses RFrame and ROS 2 rather than an ``install.sh --env`` target; see :doc:`../examples/embodied/r2`)
   * - ``--engine``
     - ``sglang`` (default for ``agentic``), ``vllm`` (``agentic``), ``phyai`` (``embodied`` with ``--model openpi --env libero``)
   * - ``--platform``
     - ``nvidia`` (default), ``amd``, ``ascend``, ``musa``, ``kunlun``, ``biren``, ``metax`` (Intel XPU uses a PyTorch XPU environment rather than a ``--platform`` flag; see :ref:`the hardware matrix <model-hardware-support>`)

.. dropdown:: More installer flags (optional)

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - Flag
        - Purpose
      * - ``--use-mirror``
        - Use mainland China mirrors for faster package and GitHub downloads (also export ``HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``).
      * - ``--venv <dir>``
        - Choose the virtual environment directory (default ``.venv``).
      * - ``--python <version>``
        - Override the Python version when needed (for example ``3.10`` for ``behavior`` or ``d4rl``).
      * - ``--no-root``
        - Skip ``sudo`` system package installs on hosts where system libraries are already present.
      * - ``--no-flash-attn`` / ``--no-apex``
        - Skip ``flash-attn`` or ``apex`` builds when not needed or on hosts without a matching CUDA toolchain.

   Run ``bash requirements/install.sh --help`` for all version-override flags.

.. _hardware-and-software-requirements:

Hardware and Software Requirements
----------------------------------

Match your hardware to the workload you want to run. You can start on a single GPU and scale the same recipes to multi-node clusters without code changes.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/hardware_tiers.svg
   :align: center
   :width: 100%

   RLinf hardware tiers from a single GPU to multi-node clusters.

.. list-table:: GPU Tiers by Workload
   :header-rows: 1
   :widths: 22 44 34

   * - Tier
     - Workloads
     - Example Starting Point
   * - 1 GPU (8–24 GB+)
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

.. list-table:: Software Prerequisites
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

.. dropdown:: Large-Scale Reference Cluster (8×H100+)

   Multi-node benchmarks and full reproductions typically run on this cluster profile:

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
