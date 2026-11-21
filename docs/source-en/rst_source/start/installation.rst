Installation
============

Install RLinf into a local uv virtual environment with one ``requirements/install.sh`` command, or pull a prebuilt image from Docker Hub.

Install Locally
---------------

Clone the repository, then run the command for your target:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git && cd RLinf

.. tab-set::

   .. tab-item:: Embodied

      .. code-block:: bash

         bash requirements/install.sh embodied --model <model> --env <env> && source .venv/bin/activate

   .. tab-item:: Agentic / Reasoning

      .. code-block:: bash

         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate

   .. tab-item:: Docs

      .. code-block:: bash

         bash requirements/install.sh docs --venv .docs-venv

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

Common Options
--------------

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

Run ``bash requirements/install.sh --help`` for all advanced version-pinning flags.

Prebuilt Docker Images
----------------------

.. grid:: 1 1 1 1
   :gutter: 2

   .. grid-item-card:: Docker Hub — rlinf/rlinf
      :link: https://hub.docker.com/r/rlinf/rlinf

      Browse prebuilt images, tag lists, container launch commands, and regional mirrors on Docker Hub.

Verify
------

Confirm that RLinf and Ray are available in the active environment:

.. code-block:: bash

   python -c "import rlinf; print(rlinf.__file__)" && ray --version

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - EGL or Vulkan rendering error
     - Check graphics driver and EGL setup in :ref:`Rendering Issues <faq-rendering>`.
   * - Stale Ray cluster or NCCL error
     - Reset local Ray with ``ray stop`` (:ref:`CUDA and Memory Issues <faq-cuda>`).

Next Steps
----------

- :doc:`Run the VLA Quick Start <vla>`
- :doc:`Scale beyond one machine <../guides/launch-scale/index>`
- :doc:`Open the Cheat Sheet <cheat_sheet>`
