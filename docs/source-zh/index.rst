.. _home:

欢迎使用 RLinf
==============

.. raw:: html

   <div class="rlinf-hero">
     <img class="rlinf-hero-logo" src="_static/svg/logo_white.svg" alt="RLinf logo" />
     <h1 class="rlinf-hero-title">欢迎使用 RLinf</h1>
     <p class="rlinf-hero-subtitle">面向基础模型与具身智能体的可扩展强化学习后训练框架</p>
   </div>

RLinf 是面向具身 policy 与基础模型强化学习后训练的开源基础设施。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 快速开始
      :link: rst_source/start/index
      :link-type: doc
      :text-align: center

   .. grid-item-card:: 安装
      :link: rst_source/start/installation
      :link-type: doc
      :text-align: center

   .. grid-item-card:: 具身训练
      :link: rst_source/examples/index
      :link-type: doc
      :text-align: center

   .. grid-item-card:: 评测
      :link: rst_source/evaluations/index
      :link-type: doc
      :text-align: center

四步上手
--------

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/onboarding_flow_zh.svg
   :align: center
   :width: 100%

   从安装到跑通第一个任务的四步流程。

选择你的路径
------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 5 分钟首次跑通（VLA 评测）
      :link: rst_source/evaluations/get_started/quick_tour
      :link-type: doc

      在 LIBERO Spatial 上评测 OpenPI π₀.₅，约 5 分钟验证完整环境。

   .. grid-item-card:: 具身仿真 RL
      :link: rst_source/start/vla
      :link-type: doc

      在 ManiSkill3 上用 PPO 训练 OpenVLA 或 OpenVLA-OFT，再扩展到 LIBERO、RoboTwin 等基准。

   .. grid-item-card:: 真机机器人
      :link: rst_source/examples/real_world_index
      :link-type: doc

      在 Franka、SO-101、ARX X5、R2 等真机上运行硬件测试、遥操数据采集、SFT 与在线 RL。

   .. grid-item-card:: 智能体 / 推理 RL
      :link: rst_source/examples/agentic/math_reasoning/reasoning
      :link-type: doc

      从 Qwen2.5 数学推理 GRPO 入手，或继续浏览多智能体与工具调用示例。

   .. grid-item-card:: 自定义扩展
      :link: rst_source/extending/index
      :link-type: doc

      接入自定义模型、环境、机器人或算法。

   .. grid-item-card:: 扩展到集群
      :link: rst_source/guides/launch-scale/index
      :link-type: doc

      配置跨多 GPU 与多节点的共享式、分离式与混合式部署。

为什么选择 RLinf
----------------

.. list-table::
   :header-rows: 1

   * - 优势
     - 你将获得
   * - 快
     - 结合细粒度流水线的混合式执行架构，吞吐率比同类框架高 120% 以上，并支持自动在线扩缩。
   * - 灵活
     - 在 FSDP + Hugging Face（快速原型）与 Megatron + SGLang（大规模训练）之间切换，无需修改代码。
   * - 可靠
     - 内建 PPO、GRPO、DAPO、SAC 与 Reinforce++，并提供具身与推理任务的复现配置。

.. toctree::
  :caption: 快速开始
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  总览 <rst_source/start/index>
  环境要求 <rst_source/start/requirements>
  安装说明 <rst_source/start/installation>
  VLA 快速上手 <rst_source/start/vla>
  命令速查表 <rst_source/start/cheat_sheet>
  常见问题 <rst_source/resources/faq>

.. toctree::
  :caption: 具身训练
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  具身总览 <rst_source/examples/index>
  模拟器 <rst_source/examples/simulators_index>
  具身模型 <rst_source/examples/vla_wam_index>
  SFT 与模仿学习 <rst_source/examples/sft_index>
  RL 算法示例 <rst_source/examples/methods_index>
  世界模型 <rst_source/examples/world_model_index>

.. toctree::
  :caption: 真机机器人
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  真机总览 <rst_source/examples/real_world_index>
  单臂 Franka <rst_source/examples/embodied/franka_index>
  双臂 Franka <rst_source/examples/embodied/dual_franka_index>
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
  :caption: 统一评测
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  评测总览 <rst_source/evaluations/index>
  架构与目录 <rst_source/evaluations/get_started/overview>
  评测安装 <rst_source/evaluations/get_started/installation>
  快速上手 <rst_source/evaluations/get_started/quick_tour>
  评测基准 <rst_source/evaluations/guides/index>
  评测参考 <rst_source/evaluations/reference/index>

.. toctree::
  :caption: 智能体与生成
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  智能体与推理 <rst_source/examples/agentic/index>
  图像与视频生成 <rst_source/examples/video_models>

.. toctree::
  :caption: 使用指南
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  指南总览 <rst_source/guides/index>
  配置 <rst_source/guides/configure/index>
  数据与 Checkpoint <rst_source/guides/data-checkpoints/index>
  启动与扩展 <rst_source/guides/launch-scale/index>
  性能与并行 <rst_source/guides/performance/index>
  Rollout 引擎 <rst_source/guides/rollout-engines/index>

.. toctree::
  :caption: 扩展开发
  :maxdepth: 2
  :includehidden:
  :titlesonly:
  :hidden:

  扩展总览 <rst_source/extending/index>
  扩展架构 <rst_source/extending/overview>
  FSDP 模型 <rst_source/extending/new_model_fsdp>
  Megatron 模型 <rst_source/extending/new_model_megatron>
  SGLang 具身模型 <rst_source/extending/sglang_embodied_model>
  SFT 模型 <rst_source/extending/new_model_sft>
  仿真环境 <rst_source/extending/new_env>
  真机机器人 <rst_source/extending/new_robot>
  真机任务 <rst_source/extending/new_task>
  奖励模型 <rst_source/extending/reward_model>
  高级集成 <rst_source/extending/advanced-integrations/index>

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
  v0.4 发布说明 <rst_source/resources/release_v0.4>
  v0.3 发布说明 <rst_source/resources/release_v0.3>
  v0.1–v0.2 发布说明 <rst_source/resources/release_v0.1_v0.2>
  博客 <rst_source/resources/blog/index>
  论文 <rst_source/resources/publications/index>
  贡献者政策 <rst_source/resources/maintainer_policy>
