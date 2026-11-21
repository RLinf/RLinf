.. _home:

Welcome to RLinf
================

.. raw:: html

   <div class="rlinf-hero">
     <img class="rlinf-hero-logo" src="_static/svg/logo_white.svg" alt="RLinf logo" />
     <h1 class="rlinf-hero-title">Welcome to RLinf</h1>
     <p class="rlinf-hero-subtitle">Scalable RL Post-Training for Foundation Models and Embodied Agents</p>
   </div>

RLinf is an open-source infrastructure for reinforcement learning post-training of embodied policies and foundation models.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Get Started
      :link: rst_source/start/index
      :link-type: doc
      :text-align: center

   .. grid-item-card:: Install
      :link: rst_source/start/installation
      :link-type: doc
      :text-align: center

   .. grid-item-card:: Examples
      :link: rst_source/examples/index
      :link-type: doc
      :text-align: center

   .. grid-item-card:: Evaluation
      :link: rst_source/evaluations/index
      :link-type: doc
      :text-align: center

Your First Run in Four Steps
----------------------------

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/onboarding_flow.svg
   :align: center
   :width: 100%

   From install to your first run in four steps.

Choose Your Path
----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 5-Minute First Run (Evaluate VLA)
      :link: rst_source/evaluations/get_started/quick_tour
      :link-type: doc

      Verify your setup by evaluating OpenPI π₀.₅ on LIBERO Spatial in about 5 minutes.

   .. grid-item-card:: Embodied Simulation RL
      :link: rst_source/start/vla
      :link-type: doc

      Train OpenVLA or OpenVLA-OFT on ManiSkill3 with PPO, then branch to LIBERO, RoboTwin, and more.

   .. grid-item-card:: Real-World Robots
      :link: rst_source/examples/real_world_index
      :link-type: doc

      Run hardware checks, teleoperation, data collection, SFT, and online RL on Franka, SO-101, ARX X5, R2, and more.

   .. grid-item-card:: Agentic / Reasoning RL
      :link: rst_source/examples/agentic/math_reasoning/reasoning
      :link-type: doc

      Start with Qwen2.5 math reasoning GRPO, or browse multi-agent and tool-use workflows.

   .. grid-item-card:: Bring Your Own
      :link: rst_source/extending/index
      :link-type: doc

      Add a model, environment, robot, or algorithm and plug it into RLinf.

   .. grid-item-card:: Scale to a Cluster
      :link: rst_source/guides/launch-scale/index
      :link-type: doc

      Configure collocated, disaggregated, and hybrid placement across GPUs and nodes.

Why RLinf
---------

.. list-table::
   :header-rows: 1

   * - Strength
     - What it gives you
   * - Fast
     - Hybrid fine-grained pipelining delivers 120%+ higher throughput than comparable frameworks, plus automatic online scaling.
   * - Flexible
     - Switch FSDP + Hugging Face for prototyping or Megatron + SGLang for large-scale training, with no code changes.
   * - Proven
     - Built-in PPO, GRPO, DAPO, SAC, and Reinforce++ with verified recipes for embodied and reasoning tasks.

.. toctree::
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Get Started <rst_source/start/index>
  Examples <rst_source/examples/index>
  Evaluation <rst_source/evaluations/index>
  Guides <rst_source/guides/index>
  Extending <rst_source/extending/index>
  Concepts <rst_source/concepts/index>
  Reference <rst_source/reference/index>
  Resources <rst_source/resources/index>
