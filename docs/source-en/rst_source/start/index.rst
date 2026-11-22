Get Started
===========

Go from a fresh machine to your first RLinf run, then pick the workflow that matches your hardware and task.

Core Setup
----------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 1. Installation & Requirements
      :link: installation
      :link-type: doc

      Hardware tiers, one-line ``install.sh`` setup, and prebuilt Docker Hub images.

   .. grid-item-card:: 2. Quick Start
      :link: vla
      :link-type: doc

      Run a 1-GPU MLP quick test (no downloads) or OpenVLA / OpenVLA-OFT PPO on ManiSkill3.

   .. grid-item-card:: 3. Cheat Sheet
      :link: cheat_sheet
      :link-type: doc

      Shortest commands once you know the workflow.

Pick Your First Run
-------------------

Start with the entry point closest to your goal:

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Goal
     - Typical Hardware
     - Recommended Starting Point
   * - Verify setup in ~5 minutes
     - 1 GPU (24 GB+)
     - :doc:`LIBERO Evaluation Quick Tour <../evaluations/get_started/quick_tour>`
   * - Lightweight RL (no downloads)
     - 1 GPU (8–24 GB+)
     - :doc:`Quick Start — MLP Tab <vla>`
   * - VLA RL fine-tuning in simulation
     - 1–8 GPUs (40–80 GB)
     - :doc:`Quick Start — OpenVLA / OpenVLA-OFT <vla>`
   * - Real-world robot setup and RL
     - Controller PC + GPU node
     - :doc:`Real-World Robots <../examples/real_world_index>`
   * - LLM math reasoning RL
     - 4–8 GPUs (80 GB)
     - :doc:`Math Reasoning GRPO <../examples/agentic/math_reasoning/reasoning>`

Find Your Way Around the Repo
-----------------------------

Everything you touch day to day lives in six top-level directories, organized by task:

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Training Scripts & Configs (``examples/``)
      :link: ../reference/configuration
      :link-type: doc

      Launch scripts and Hydra YAMLs grouped by task: embodied RL, SFT, offline RL, reasoning, agents, and diffusion.

   .. grid-item-card:: Standalone Evaluation (``evaluations/``)
      :link: ../evaluations/index
      :link-type: doc

      Unified ``run_eval.sh`` launcher and ready-to-run benchmark configs (LIBERO, RoboTwin, BEHAVIOR, ManiSkill, real robots).

   .. grid-item-card:: Helper Utilities (``toolkits/``)

      Standalone tools for LeRobot datasets, real-robot hardware checks, replay buffer inspection, auto-placement, and checkpoint conversion.

   .. grid-item-card:: Core Framework (``rlinf/``)
      :link: ../concepts/overview
      :link-type: doc

      Distributed runtime, workers, runners, models, simulator and real-robot environments, robotics abstractions, and algorithm registries.

   .. grid-item-card:: Installers & Images (``requirements/`` & ``docker/``)
      :link: installation
      :link-type: doc

      One-line ``install.sh`` environment builder and Dockerfiles for NVIDIA, AMD, Ascend, and MUSA.

   .. grid-item-card:: Tests & Extensions (``tests/`` & ``docs/``)
      :link: ../extending/index
      :link-type: doc

      Unit tests, end-to-end configs, fake vendor SDKs in ``robot_mocks/``, and English/Chinese docs.

.. dropdown:: Inside ``toolkits/`` — What Tools Are Included?

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Directory
        - Purpose
      * - ``toolkits/lerobot/``
        - Compute ``norm_stats.json``, merge LeRobot datasets, and export episode MP4 previews.
      * - ``toolkits/realworld_check/``
        - Interactive hardware checks for arms, grippers, cameras, GELLO, and PICO before running real-robot training.
      * - ``toolkits/replay_buffer/``
        - Inspect, visualize, merge, or split saved trajectory replay buffers.
      * - ``toolkits/auto_placement/``
        - Search optimal collocated vs disaggregated GPU layouts from measured iteration timings.
      * - ``toolkits/cosmos3/``, ``dual_franka/``, ``standalone_eval_scripts/``
        - Model- and robot-specific checkpoint converters, dataset cleaners, and standalone eval wrappers.

What's Next
-----------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: Embodied Training
      :link: ../examples/index
      :link-type: doc

      Browse simulator, VLA model, SFT, RL, and world model recipes.

   .. grid-item-card:: Concepts
      :link: ../concepts/index
      :link-type: doc

      Understand execution, scheduling, and the robotics interface.

   .. grid-item-card:: Guides
      :link: ../guides/index
      :link-type: doc

      Configure launches, logging, checkpoints, and clusters.

   .. grid-item-card:: FAQ
      :link: ../resources/faq
      :link-type: doc

      Fixes for rendering, OOM, network, and Ray issues.
