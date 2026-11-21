Get Started
===========

Go from a fresh machine to your first RLinf run, then pick the workflow that matches your hardware and task.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/onboarding_flow.svg
   :align: center
   :width: 100%

   The four-step path from installation to monitoring and scaling.

Core Setup
----------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: 1. Requirements
      :link: requirements
      :link-type: doc

      GPU tiers from a single 24 GB card to multi-node clusters.

   .. grid-item-card:: 2. Install
      :link: installation
      :link-type: doc

      One-line ``install.sh`` setup or prebuilt Docker Hub images.

   .. grid-item-card:: 3. VLA Quick Start
      :link: vla
      :link-type: doc

      Train OpenVLA or OpenVLA-OFT on ManiSkill3 with PPO.

   .. grid-item-card:: 4. Cheat Sheet
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
   * - Lightweight RL (no large checkpoint)
     - 1 GPU (24 GB+)
     - :doc:`MLP Policy RL <../examples/embodied/mlp>`
   * - VLA RL fine-tuning in simulation
     - 1–8 GPUs (40–80 GB)
     - :doc:`VLA Quick Start (ManiSkill3) <vla>`
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

   .. grid-item-card:: Examples
      :link: ../examples/index
      :link-type: doc

      Browse embodied, SFT, world model, and agentic recipes.

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
