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

Workflow
--------

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/onboarding_flow.svg
   :align: center
   :width: 100%

   Workflow from installation and stack selection to training and cluster scaling.

Choose Your Path
----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Quickstart
      :link: rst_source/start/vla
      :link-type: doc

      Clone RLinf, install one environment, and run your first single-GPU Embodied AI or Agent & LLM training job.

   .. grid-item-card:: Tour of RLinf
      :link: rst_source/start/walkthrough
      :link-type: doc

      See how training runs end to end, how jobs scale across GPUs, and where configs, models, and tools live.

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

   .. grid-item-card:: Extending RLinf
      :link: rst_source/extending/index
      :link-type: doc

      Add a custom model, environment, robot, algorithm, reward, or hardware backend.

Why RLinf
---------

.. list-table::
   :header-rows: 1

   * - Strength
     - What it gives you
   * - Fast
     - Hybrid fine-grained pipelining delivers :doc:`120%+ higher throughput <rst_source/resources/why_rlinf>` than comparable frameworks, plus automatic online scaling.
   * - Flexible
     - Switch between FSDP + Hugging Face and Megatron + SGLang through config.
   * - Proven
     - Built-in PPO, GRPO, DAPO, SAC, and Reinforce++ with verified recipes for embodied and reasoning tasks.

.. toctree::
  :caption: Get Started
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  Quickstart <rst_source/start/vla>
  Tour of RLinf <rst_source/start/walkthrough>
  Installation & Hardware <rst_source/start/installation>
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
  AgileX Piper <rst_source/examples/embodied/piper>
  ARX X5 <rst_source/examples/embodied/arx5>
  Dexmal DOS-W1 <rst_source/examples/embodied/dosw1>
  Franka (Dual-Arm) <rst_source/examples/embodied/dual_franka_index>
  Franka (Single-Arm) <rst_source/examples/embodied/franka_index>
  GimArm <rst_source/examples/embodied/gim_arm>
  LeKiwi <rst_source/examples/embodied/lekiwi>
  Robbyant R2 <rst_source/examples/embodied/r2>
  SO-101 <rst_source/examples/embodied/so101>
  XSquare Turtle2 <rst_source/examples/embodied/xsquare_turtle2>
  YAM <rst_source/examples/embodied/yam_index>

.. toctree::
  :caption: Agent & LLM
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  Overview <rst_source/examples/agentic/index>
  Reasoning <rst_source/examples/agentic/math_reasoning/index>
  Agentic RL <rst_source/examples/agentic/agents/index>
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
  Add an Algorithm <rst_source/extending/new_algorithm>
  Add a Reward <rst_source/extending/rewards/index>
  Add an Agent <rst_source/extending/agents/index>
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
