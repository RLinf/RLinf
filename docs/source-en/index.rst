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
  :caption: Get Started
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  Overview <rst_source/start/index>
  Requirements <rst_source/start/requirements>
  Installation <rst_source/start/installation>
  VLA Quick Start <rst_source/start/vla>
  Cheat Sheet <rst_source/start/cheat_sheet>
  FAQ <rst_source/resources/faq>

.. toctree::
  :caption: Embodied Training
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Example Gallery <rst_source/examples/index>
  Simulators <rst_source/examples/simulators_index>
  Embodied Models <rst_source/examples/vla_wam_index>
  SFT & Imitation <rst_source/examples/sft_index>
  RL Recipes <rst_source/examples/methods_index>
  World Models <rst_source/examples/world_model_index>

.. toctree::
  :caption: Real-World Robots
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Overview <rst_source/examples/real_world_index>
  Single-Arm Franka <rst_source/examples/embodied/franka_index>
  Dual-Arm Franka <rst_source/examples/embodied/dual_franka_index>
  YAM <rst_source/examples/embodied/yam_index>
  ARX X5 <rst_source/examples/embodied/arx5>
  GimArm <rst_source/examples/embodied/gim_arm>
  AgileX Piper <rst_source/examples/embodied/piper>
  XSquare Turtle2 <rst_source/examples/embodied/xsquare_turtle2>
  Dexmal DOS-W1 <rst_source/examples/embodied/dosw1>
  Robbyant R2 <rst_source/examples/embodied/r2>
  SO-101 <rst_source/examples/embodied/so101>
  LeKiwi <rst_source/examples/embodied/lekiwi>

.. toctree::
  :caption: Evaluation
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Overview <rst_source/evaluations/index>
  Architecture & Layout <rst_source/evaluations/get_started/overview>
  Installation <rst_source/evaluations/get_started/installation>
  Quick Tour <rst_source/evaluations/get_started/quick_tour>
  Benchmarks <rst_source/evaluations/guides/index>
  Eval Reference <rst_source/evaluations/reference/index>

.. toctree::
  :caption: Agentic & Generative
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Agentic & Reasoning <rst_source/examples/agentic/index>
  Image & Video Generation <rst_source/examples/video_models>

.. toctree::
  :caption: Guides
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  Guides Overview <rst_source/guides/index>
  Configure <rst_source/guides/configure/index>
  Data & Checkpoints <rst_source/guides/data-checkpoints/index>
  Launch & Scale <rst_source/guides/launch-scale/index>
  Performance & Parallelism <rst_source/guides/performance/index>
  Rollout Engines <rst_source/guides/rollout-engines/index>

.. toctree::
  :caption: Extending
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  Extending Hub <rst_source/extending/index>
  Extension Registries <rst_source/extending/overview>
  New Model (FSDP) <rst_source/extending/new_model_fsdp>
  New Model (Megatron) <rst_source/extending/new_model_megatron>
  SGLang Embodied Model <rst_source/extending/sglang_embodied_model>
  New SFT Model <rst_source/extending/new_model_sft>
  New Simulator Env <rst_source/extending/new_env>
  New Real-World Robot <rst_source/extending/new_robot>
  New Real-World Task <rst_source/extending/new_task>
  Reward Model <rst_source/extending/reward_model>
  Advanced Integrations <rst_source/extending/advanced-integrations/index>

.. toctree::
  :caption: Concepts & Reference
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Concepts <rst_source/concepts/index>
  Algorithms <rst_source/reference/algorithms/index>
  Config & Metrics <rst_source/reference/training/index>
  Python API <rst_source/reference/api/index>

.. toctree::
  :caption: Resources
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  Why RLinf <rst_source/resources/why_rlinf>
  v0.4 Release Notes <rst_source/resources/release_v0.4>
  v0.3 Release Notes <rst_source/resources/release_v0.3>
  v0.1 & v0.2 Release Notes <rst_source/resources/release_v0.1_v0.2>
  Blog <rst_source/resources/blog/index>
  Publications <rst_source/resources/publications/index>
  Contributor Policy <rst_source/resources/maintainer_policy>
