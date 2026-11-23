.. _home:

Welcome to RLinf
================

.. raw:: html

   <div class="rlinf-hero">
     <img class="rlinf-hero-logo" src="_static/svg/logo_white.svg" alt="RLinf logo" />
     <h1 class="rlinf-hero-title">Welcome to RLinf</h1>
     <p class="rlinf-hero-subtitle">Reinforcement Learning Infrastructure for Embodied, World-Model, and Agentic AI</p>
   </div>

**RLinf** is a flexible, scalable open-source reinforcement learning infrastructure built for Embodied and Agentic AI. The **inf** stands for both **Infrastructure**, highlighting its role as a backbone for next-generation post-training, and **Infinite**, reflecting open-ended learning and continuous generalization.

One framework unifies **VLA and world-model simulation**, **real-world robot online RL**, and **multi-turn reasoning and tool-use agents**, scaling from a single GPU to multi-node heterogeneous clusters.

Quick Try
---------

Launch PPO training on ManiSkill3 ``PickCube-v1``:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git && cd RLinf
   bash requirements/install.sh embodied --env maniskill_libero && source .venv/bin/activate
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

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

   .. grid-item-card:: Quick Training
      :link: rst_source/start/vla
      :link-type: doc

      Start with MLP PPO on ManiSkill3, or scale up to OpenVLA / OpenVLA-OFT.

   .. grid-item-card:: VLA Evaluation
      :link: rst_source/evaluations/get_started/quick_tour
      :link-type: doc

      Evaluate OpenPI π₀.₅ on LIBERO Spatial with automatic Hugging Face checkpoint caching.

   .. grid-item-card:: Real-World Robots
      :link: rst_source/examples/real_world_index
      :link-type: doc

      Run hardware checks, teleoperation, data collection, SFT, and online RL on Franka, SO-101, ARX X5, R2, and more.

   .. grid-item-card:: Agent & LLM
      :link: rst_source/examples/agentic/index
      :link-type: doc

      Pick a single-node or multi-node track across math reasoning, tool-use agents, multimodal RL, and diffusion.

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
  RLinf Walkthrough <rst_source/start/walkthrough>
  Cheat Sheet <rst_source/start/cheat_sheet>
  FAQ <rst_source/resources/faq>

.. toctree::
  :caption: Embodied AI
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
  Evaluation <rst_source/evaluations/index>

.. toctree::
  :caption: Real-World Robots
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Overview <rst_source/examples/real_world_index>
  Workflows <rst_source/examples/real_world_workflows_index>
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
  :caption: Agent & LLM
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
  Configure & Run <rst_source/guides/configure/index>
  Scale to a Cluster <rst_source/guides/launch-scale/index>
  Profile & Speed Up <rst_source/guides/performance/index>
  Rollout Backends <rst_source/guides/rollout-engines/index>

.. toctree::
  :caption: Extending
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  Overview <rst_source/extending/index>
  Add a Model <rst_source/extending/models/index>
  Add an Env or Robot <rst_source/extending/envs-robots/index>
  Add an Algorithm <rst_source/extending/algorithms-agents/index>
  Add an Agent <rst_source/extending/new_agent>
  Add a Hardware Backend <rst_source/extending/new_accelerator>

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
  Release Notes <rst_source/resources/releases_index>
  Blog <rst_source/resources/blog/index>
  Publications <rst_source/resources/publications/index>
  Contributor Policy <rst_source/resources/maintainer_policy>
