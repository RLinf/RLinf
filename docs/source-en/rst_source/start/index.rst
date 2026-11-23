Get Started
===========

Go from a fresh machine to your first RLinf run, then pick the workflow that matches your hardware and task.

Core Setup
----------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: 1. Installation
      :link: installation
      :link-type: doc

      Hardware requirements, local ``install.sh`` setup, and prebuilt Docker Hub images.

   .. grid-item-card:: 2. Quick Start
      :link: vla
      :link-type: doc

      Run MLP PPO on ManiSkill3, or start OpenVLA / OpenVLA-OFT training.

   .. grid-item-card:: 3. RLinf Walkthrough
      :link: walkthrough
      :link-type: doc

      Visual tour of RLinf's architecture, training loop, hardware scaling, and repo layout.

   .. grid-item-card:: 4. Cheat Sheet
      :link: cheat_sheet
      :link-type: doc

      Common commands once you know the workflow.

Pick Your First Run
-------------------

Start with the entry point closest to your goal:

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Goal
     - Typical Hardware
     - Recommended Starting Point
   * - Lightweight RL
     - 1 GPU, 4 GB+
     - :doc:`Quick Start — MLP Tab <vla>`
   * - Run VLA evaluation
     - 1 GPU, 24 GB+
     - :doc:`LIBERO Evaluation Quick Tour <../evaluations/get_started/quick_tour>`
   * - VLA RL fine-tuning in simulation
     - 1–8 GPUs, 40–80 GB
     - :doc:`Quick Start — OpenVLA / OpenVLA-OFT <vla>`
   * - Real-world robot setup and RL
     - Controller PC + GPU node
     - :doc:`Real-World Robots <../examples/real_world_index>`
   * - Agentic, reasoning, or diffusion RL
     - 1 GPU to multi-node
     - :doc:`Agent & LLM <../examples/agentic/index>`

What's Next
-----------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: RLinf Walkthrough
      :link: walkthrough
      :link-type: doc

      Visual map of how runs execute and where configs, scripts, and tools live.

   .. grid-item-card:: Embodied Training
      :link: ../examples/index
      :link-type: doc

      Browse simulator, VLA model, SFT, RL, and world model recipes.

   .. grid-item-card:: Guides
      :link: ../guides/index
      :link-type: doc

      Configure launches, checkpoints, logging, and multi-node clusters.

   .. grid-item-card:: FAQ
      :link: ../resources/faq
      :link-type: doc

      Fixes for rendering, OOM, network, and Ray issues.
