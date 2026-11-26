.. _home:

欢迎使用 RLinf
==============

.. raw:: html

   <div class="rlinf-hero">
     <img class="rlinf-hero-logo" src="_static/svg/logo_white.svg" alt="RLinf logo" />
     <h1 class="rlinf-hero-title">欢迎使用 RLinf</h1>
     <p class="rlinf-hero-subtitle">为具身智能、世界模型与智能体而生的强化学习基础设施</p>
   </div>

**RLinf** 是一个灵活、可扩展的开源强化学习后训练框架。名称中的 **inf** 既代表 **Infrastructure**，为下一代智能体与机器人训练提供坚实底座；也代表 **Infinite**，寓意开放式学习与持续泛化的无限可能。

同一套框架打通 **VLA 与世界模型仿真训练**、**真机机器人在线 RL** 与 **多轮推理 / 工具调用智能体**，支持从单卡开发扩展到多机异构集群。

工作流程
--------

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/onboarding_flow_zh.svg
   :align: center
   :width: 100%

   从安装环境、选择场景到启动训练与扩展集群的流程。

选择你的路径
------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 快速开始
      :link: rst_source/start/vla
      :link-type: doc

      克隆仓库、装好单套环境，在单卡上跑通第一个具身智能（ManiSkill3）或 Agent / LLM（GRPO）训练任务。

   .. grid-item-card:: 框架导览
      :link: rst_source/start/walkthrough
      :link-type: doc

      通过图示了解训练闭环、单卡到多卡的扩展方式，以及配置、模型与工具的目录位置。

   .. grid-item-card:: VLA 评测
      :link: rst_source/evaluations/get_started/quick_tour
      :link-type: doc

      在 LIBERO Spatial 上评测 OpenPI π₀.₅，自动拉取 Hugging Face 权重。

   .. grid-item-card:: 真机机器人
      :link: rst_source/examples/real_world_index
      :link-type: doc

      在 Franka、SO-101、ARX X5、R2 等真机上运行硬件测试、遥操作与数据采集、SFT 与在线 RL。

   .. grid-item-card:: Agent 与 LLM
      :link: rst_source/examples/agentic/index
      :link-type: doc

      按场景选择数学推理、多轮工具调用、多 Agent 与图像/视频生成示例，支持单机到多节点扩展。

   .. grid-item-card:: 自定义扩展
      :link: rst_source/extending/index
      :link-type: doc

      接入自定义模型、Env、机器人、算法、Reward 或硬件后端。

为什么选择 RLinf
----------------

.. list-table::
   :header-rows: 1

   * - 优势
     - 你将获得
   * - 快
     - 结合细粒度流水线的混合式执行架构，吞吐率比同类框架\ :doc:`高 120% 以上 <rst_source/resources/why_rlinf>`\ ，并支持自动在线扩缩。
   * - 灵活
     - 通过配置在 FSDP + Hugging Face 与 Megatron + SGLang 之间切换。
   * - 可靠
     - 内建 PPO、GRPO、DAPO、SAC 与 Reinforce++，并提供具身与推理任务的复现配置。

.. toctree::
  :caption: 入门
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  快速开始 <rst_source/start/vla>
  框架导览 <rst_source/start/walkthrough>
  安装与硬件参考 <rst_source/start/installation>
  常见问题 <rst_source/resources/faq>

.. toctree::
  :caption: 具身智能
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  总览 <rst_source/examples/index>
  模拟器 <rst_source/examples/simulators_index>
  具身模型 <rst_source/examples/vla_wam_index>
  SFT 与模仿学习 <rst_source/examples/sft_index>
  RL 算法示例 <rst_source/examples/methods_index>
  世界模型 <rst_source/examples/world_model_index>
  模型评测 <rst_source/evaluations/index>

.. toctree::
  :caption: 真机机器人
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  总览 <rst_source/examples/real_world_index>
  通用流程 <rst_source/examples/real_world_workflows_index>
  AgileX Piper <rst_source/examples/embodied/piper>
  ARX X5 <rst_source/examples/embodied/arx5>
  Dexmal DOS-W1 <rst_source/examples/embodied/dosw1>
  Franka（双臂） <rst_source/examples/embodied/dual_franka_index>
  Franka（单臂） <rst_source/examples/embodied/franka_index>
  GimArm <rst_source/examples/embodied/gim_arm>
  LeKiwi <rst_source/examples/embodied/lekiwi>
  Robbyant R2 <rst_source/examples/embodied/r2>
  SO-101 <rst_source/examples/embodied/so101>
  XSquare Turtle2 <rst_source/examples/embodied/xsquare_turtle2>
  YAM <rst_source/examples/embodied/yam_index>

.. toctree::
  :caption: Agent 与 LLM
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  总览 <rst_source/examples/agentic/index>
  推理训练 <rst_source/examples/agentic/math_reasoning/index>
  Agent 训练 <rst_source/examples/agentic/agents/index>
  图像与视频生成 <rst_source/examples/video_models>

.. toctree::
  :caption: 使用指南
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  总览 <rst_source/guides/index>
  配置与运行 <rst_source/guides/configure/index>
  扩展到多机集群 <rst_source/guides/launch-scale/index>
  性能诊断与提速 <rst_source/guides/performance/index>
  Rollout 后端与服务 <rst_source/guides/rollout-engines/index>

.. toctree::
  :caption: 扩展开发
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  总览 <rst_source/extending/index>
  新增模型 <rst_source/extending/models/index>
  新增 Env 与真机 <rst_source/extending/envs-robots/index>
  新增算法 <rst_source/extending/new_algorithm>
  新增 Reward <rst_source/extending/rewards/index>
  新增 Agent <rst_source/extending/agents/index>
  新增硬件后端 <rst_source/extending/new_accelerator>

.. toctree::
  :caption: 原理与参考
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  核心概念 <rst_source/concepts/index>
  算法参考 <rst_source/reference/algorithms/index>
  配置与指标 <rst_source/reference/training/index>
  Python API <rst_source/reference/api/index>

.. toctree::
  :caption: 社区与资源
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  为什么选择 RLinf <rst_source/resources/why_rlinf>
  发布说明 <rst_source/resources/releases_index>
  博客 <rst_source/resources/blog/index>
  论文 <rst_source/resources/publications/index>
  贡献者政策 <rst_source/resources/maintainer_policy>
