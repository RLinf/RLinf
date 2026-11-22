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
