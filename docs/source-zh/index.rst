.. _home:

欢迎使用 RLinf
==============

.. raw:: html

   <div class="rlinf-hero">
     <img class="rlinf-hero-logo" src="_static/svg/logo_white.svg" alt="RLinf logo" />
     <h1 class="rlinf-hero-title">欢迎使用 RLinf</h1>
     <p class="rlinf-hero-subtitle">面向基础模型与具身智能体的可扩展强化学习后训练框架</p>
   </div>

RLinf 是一个灵活且可扩展的开源基础架构，专为通过强化学习对基础模型进行后训练而设计。名称中的 "inf" 既代表 Infrastructure（基础架构）——新一代训练的强大支撑，也代表 Infinite（无限）——象征开放式学习与持续泛化。

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

选择你的路径
------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 🤖 具身智能 RL
      :link: rst_source/start/vla
      :link-type: doc

      使用 PPO 或 GRPO，在 LIBERO、ManiSkill、RoboTwin 等环境上微调 VLA 模型。

   .. grid-item-card:: 🧠 智能体 / 推理 RL
      :link: rst_source/examples/agentic/index
      :link-type: doc

      浏览 Qwen / DeepSeek 模型的智能体与推理配方。

   .. grid-item-card:: 🧩 自定义扩展
      :link: rst_source/extending/index
      :link-type: doc

      添加新的模型、环境或算法，并将其接入 RLinf。

   .. grid-item-card:: 🚀 扩展到集群
      :link: rst_source/guides/launch-scale/index
      :link-type: doc

      跨多 GPU 与多节点的共享式、分离式与混合式部署。

为什么选择 RLinf
----------------

.. list-table::
   :header-rows: 1

   * - 优势
     - 你将获得
   * - 快
     - 结合细粒度流水线的混合式相比同类框架实现 120%+ 的吞吐率提升，并支持自动在线扩缩。
   * - 灵活
     - 在 FSDP + Hugging Face（快速原型）与 Megatron + SGLang（大规模训练）之间切换，无需修改代码。
   * - 可靠
     - 内建 PPO、GRPO、DAPO、Reinforce++，并为具身与推理任务提供 SOTA 配方。

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
