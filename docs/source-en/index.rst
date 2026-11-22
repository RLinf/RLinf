.. _home:

Welcome to RLinf
================

.. raw:: html

   <div class="rlinf-hero">
     <img class="rlinf-hero-logo" src="_static/svg/logo_white.svg" alt="RLinf logo" />
     <h1 class="rlinf-hero-title">Welcome to RLinf</h1>
     <p class="rlinf-hero-subtitle">RL Post-Training Infrastructure for Embodied Policies and Foundation Models</p>
   </div>

Train, fine-tune, and evaluate vision-language-action (VLA) policies, real-world robots, and reasoning agents on one GPU or across multi-node clusters.

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
     - Hybrid fine-grained pipelining delivers :doc:`120%+ higher throughput <rst_source/resources/why_rlinf>` than comparable frameworks, plus automatic online scaling.
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
  Installation <rst_source/start/installation>
  Quick Start <rst_source/start/vla>
  Cheat Sheet <rst_source/start/cheat_sheet>
  FAQ <rst_source/resources/faq>

.. toctree::
  :caption: Embodied Training
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Overview <rst_source/examples/index>
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
  Robbyant R2 <rst_source/examples/embodied/r2>
  Dexmal DOS-W1 <rst_source/examples/embodied/dosw1>
  XSquare Turtle2 <rst_source/examples/embodied/xsquare_turtle2>
  SO-101 <rst_source/examples/embodied/so101>
  LeKiwi <rst_source/examples/embodied/lekiwi>

.. toctree::
  :caption: Evaluation
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Overview <rst_source/evaluations/index>
  Quick Tour <rst_source/evaluations/get_started/quick_tour>
  Benchmarks <rst_source/evaluations/guides/index>
  Eval Reference <rst_source/evaluations/reference/index>

.. toctree::
  :caption: Agentic & Generative
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  Overview <rst_source/examples/agentic/index>
  Math Reasoning <rst_source/examples/agentic/math_reasoning/index>
  Search-R1 <rst_source/examples/agentic/searchr1/index>
  WideSeek-R1 <rst_source/examples/agentic/wideseek_r1/index>
  rStar2 <rst_source/examples/agentic/rstar2>
  AgentLightning <rst_source/examples/agentic/agentlightning_calc_x>
  Coding Online RL <rst_source/examples/agentic/coding_online_rl>
  Qwen3-VL GRPO <rst_source/examples/agentic/qwen3_vl_geo3k>
  Image & Video Generation <rst_source/examples/video_models>

.. toctree::
  :caption: Guides
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  Overview <rst_source/guides/index>
  Envs & Logging <rst_source/guides/configure/index>
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

  Overview <rst_source/extending/index>
  FSDP Model <rst_source/extending/new_model_fsdp>
  SFT Model <rst_source/extending/new_model_sft>
  SGLang Embodied <rst_source/extending/sglang_embodied_model>
  Simulator Env <rst_source/extending/new_env>
  Real-World Robot <rst_source/extending/new_robot>
  Real-World Task <rst_source/extending/new_task>
  Reward Model <rst_source/extending/reward_model>
  Algorithms & Rule Rewards <rst_source/extending/new_algorithm>
  Agent & Tool Calling <rst_source/extending/new_agent>
  Megatron-Bridge <rst_source/extending/mbridge>
  Accelerator Backend <rst_source/extending/new_accelerator>
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
  v0.1–v0.2 Release Notes <rst_source/resources/release_v0.1_v0.2>
  Blog <rst_source/resources/blog/index>
  Publications <rst_source/resources/publications/index>
  Contributor Policy <rst_source/resources/maintainer_policy>
