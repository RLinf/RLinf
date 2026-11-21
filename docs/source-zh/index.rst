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

   .. grid-item-card:: 示例
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

      在 Franka、SO-101、ARX X5、R2 等真机上运行硬件测试、遥操采数、SFT 与在线 RL。

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
  :maxdepth: 3
  :includehidden:
  :titlesonly:
  :hidden:

  快速开始 <rst_source/start/index>
  示例 <rst_source/examples/index>
  评测 <rst_source/evaluations/index>
  指南 <rst_source/guides/index>
  概念 <rst_source/concepts/index>
  参考 <rst_source/reference/index>
  扩展 <rst_source/extending/index>
  资源 <rst_source/resources/index>
