Installation & Hardware
=======================

.. important::

   **Want to run a specific task? You do not need to read this page.** Every model, simulator, real-robot, and Agent & LLM recipe includes its own complete install and launch commands — go straight to that recipe, or start with :doc:`Quickstart <vla>` for your first single-GPU run. Use this page only as a reference for ``install.sh`` options, prebuilt Docker images, regional mirrors, and GPU/cluster requirements.

Install RLinf into a local ``uv`` virtual environment with ``requirements/install.sh``, or pull a prebuilt image from Docker Hub. Each virtual environment holds one model-and-environment stack; pass ``--venv <dir>`` to keep multiple stacks side by side.

Setup
-----

Clone the repository first:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git && cd RLinf

Then pick your target stack below:

.. tab-set::

   .. tab-item:: Embodied (Sim)
      :sync: embodied

      Swap ``--model`` and ``--env`` from the table below to match your model and simulator (omit ``--model`` for lightweight MLP / CNN policies):

      .. code-block:: bash

         bash requirements/install.sh embodied --model openpi --env maniskill_libero
         source .venv/bin/activate

   .. tab-item:: Real-World Robot
      :sync: realworld

      Install the robot driver stack on the controller node (no GPU needed) and the model stack on the GPU node. Swap ``franka`` for your robot target from the table below (see :doc:`../examples/real_world_index` for all supported robots, including Robbyant R2 over ROS 2):

      .. code-block:: bash

         # Robot controller node (driver + sensors)
         bash requirements/install.sh embodied --env franka
         source .venv/bin/activate

         # GPU / training node (or single workstation with both GPU and robot)
         bash requirements/install.sh embodied --model openpi --env franka
         source .venv/bin/activate

   .. tab-item:: Agent & LLM
      :sync: agentic

      .. code-block:: bash

         bash requirements/install.sh agentic --engine sglang
         source .venv/bin/activate

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
     - ``abot_m0``, ``dexbotic``, ``evo1``, ``gr00t``, ``gr00t_n1d6``, ``gr00t_n1d7``, ``lingbotvla``, ``lingbotvlav2``, ``molmoact2``, ``openpi``, ``openvla``, ``openvla-oft``, ``pi0_fast``, ``psi0``, ``starvla``
   * - ``--model`` (:doc:`WAM / VLM / Gen <../examples/vla_wam_index>`)
     - ``cosmos3``, ``diffusion``, ``dreamzero``, ``fastwam``, ``qwen3_vl``, ``sglang``
   * - ``--env`` (:doc:`Simulators <../examples/simulators_index>`)
     - ``behavior``, ``calvin``, ``d4rl``, ``embodichain``, ``frankasim``, ``genesis``, ``geniesim``, ``habitat``, ``isaaclab``, ``libero``, ``liberoplus``, ``liberopro``, ``maniskill_libero``, ``metaworld``, ``mujoco_warp``, ``polaris``, ``robocasa``, ``robocasa365``, ``robodojo``, ``robolab``, ``robotwin``, ``roboverse``, ``simple``
   * - ``--env`` (:doc:`World Models <../examples/world_model_index>`)
     - ``opensora``, ``wan``
   * - ``--env`` (:doc:`Real Robots <../examples/real_world_index>`)
     - ``arx5``, ``dosw1``, ``dummy``, ``franka``, ``franka-ros``, ``gim_arm``, ``lekiwi``, ``piper``, ``so101``, ``xsquare_turtle2``, ``yam`` (Robbyant R2 uses RFrame and ROS 2 rather than an ``install.sh --env`` target; see :doc:`../examples/embodied/r2`)
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

Match your hardware to the workload you want to run; the same recipes scale from a single GPU to multi-node clusters.

.. list-table:: GPU Tiers by Workload
   :header-rows: 1
   :widths: 22 44 34

   * - Tier
     - Workloads
     - Example Starting Point
   * - 1 GPU (4–24 GB)
     - MLP / CNN simulation RL (~4 GB), VLA evaluation (24 GB+), LoRA SFT, real-robot data collection, and deployment
     - :doc:`Quickstart (Embodied AI) <vla>`, :doc:`Evaluation Quick Tour <../evaluations/get_started/quick_tour>`
   * - 1–2 GPUs (40–80 GB)
     - Single-workstation OpenVLA / OpenPI RL, collocated PPO / GRPO, 1.5B Agent & LLM GRPO
     - :doc:`Quickstart (Agent & LLM) <vla>`, :doc:`Embodied AI <../examples/index>`
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
