Installation
============

Install RLinf into a local ``uv`` virtual environment with ``requirements/install.sh``, or use a prebuilt image from Docker Hub.

Setup and Verify
----------------

Choose your target below and run the setup commands:

.. tab-set::

   .. tab-item:: Embodied (Sim)
      :sync: embodied

      Example for the ManiSkill3 Quick Start (swap ``--model`` and ``--env`` from the table below for other setups):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate
         python -c "import rlinf; print(rlinf.__file__)" && ray --version

   .. tab-item:: Real-World Robot
      :sync: realworld

      Install the robot driver stack on the controller node (no GPU needed) and the model stack on the GPU node (swap ``franka`` for ``arx5``, ``yam``, ``xsquare_turtle2``, ``so101``, ``piper``, ``lekiwi``, ``dosw1``, or ``gim_arm``; see :doc:`../examples/real_world_index`):

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
         python -c "import rlinf; print(rlinf.__file__)" && ray --version

   .. tab-item:: Prebuilt Docker
      :sync: docker

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
   * - ``--model`` (VLA)
     - ``openvla``, ``openvla-oft``, ``openpi``, ``psi0``, ``pi0_fast``, ``gr00t``, ``gr00t_n1d6``, ``gr00t_n1d7``, ``dexbotic``, ``starvla``, ``lingbotvla``, ``lingbotvlav2``, ``abot_m0``, ``molmoact2``, ``evo1``
   * - ``--model`` (WAM / VLM / Gen)
     - ``dreamzero``, ``fastwam``, ``cosmos3``, ``qwen3_vl``, ``diffusion``, ``sglang``
   * - ``--env`` (Simulators)
     - ``maniskill_libero``, ``libero``, ``liberopro``, ``liberoplus``, ``behavior``, ``metaworld``, ``calvin``, ``isaaclab``, ``robocasa``, ``robocasa365``, ``frankasim``, ``robotwin``, ``robodojo``, ``roboverse``, ``habitat``, ``genesis``, ``geniesim``, ``embodichain``, ``d4rl``, ``mujoco_warp``, ``polaris``, ``simple``, ``robolab``
   * - ``--env`` (World Models)
     - ``opensora``, ``wan``
   * - ``--env`` (Real Robots)
     - ``franka``, ``franka-ros``, ``xsquare_turtle2``, ``dosw1``, ``gim_arm``, ``so101``, ``lekiwi``, ``piper``, ``arx5``, ``yam``, ``dummy``
   * - ``--engine``
     - ``sglang`` (default for ``agentic``), ``vllm`` (``agentic``), ``phyai`` (``embodied`` with ``--model openpi --env libero``)
   * - ``--platform``
     - ``nvidia`` (default), ``amd``, ``ascend``, ``musa``, ``kunlun``, ``biren``, ``metax`` (see :ref:`the hardware matrix <model-hardware-support>`)

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

Troubleshooting and Next Steps
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - EGL or Vulkan rendering error
     - Check graphics driver and EGL setup in :ref:`Rendering Issues <faq-rendering>`.
   * - Stale Ray cluster or NCCL error
     - Reset local Ray with ``ray stop`` (:ref:`CUDA and Memory Issues <faq-cuda>`).

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: VLA Quick Start
      :link: vla
      :link-type: doc

      Train OpenVLA or OpenVLA-OFT on ManiSkill3 in four steps.

   .. grid-item-card:: Scale to a Cluster
      :link: ../guides/launch-scale/index
      :link-type: doc

      Configure placement and multi-node Ray clusters.

   .. grid-item-card:: Cheat Sheet
      :link: cheat_sheet
      :link-type: doc

      Shortest commands for install, training, and evaluation.
