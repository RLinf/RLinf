快速开始
========

从零开始跑通第一个 RLinf 任务，再按硬件和目标选择对应的工作流。

基础准备
--------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: 1. 安装与环境
      :link: installation
      :link-type: doc

      硬件档位、一行 ``install.sh`` 本地安装，或前往 Docker Hub 拉取镜像。

   .. grid-item-card:: 2. 训练快速开始
      :link: vla
      :link-type: doc

      在 ManiSkill3 上跑通 MLP PPO，或启动 OpenVLA / OpenVLA-OFT 训练。

   .. grid-item-card:: 3. 快速熟悉 RLinf
      :link: walkthrough
      :link-type: doc

      用直观图示串起 RLinf 整体结构、训练闭环、多卡扩展方式与仓库目录。

   .. grid-item-card:: 4. 速查表
      :link: cheat_sheet
      :link-type: doc

      熟悉流程后直接复制最常用命令。

选择你的第一个任务
------------------

按目标直接进入对应的入口页面：

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - 目标
     - 典型硬件
     - 推荐起点
   * - 轻量 RL
     - 单卡 4 GB+
     - :doc:`训练快速开始 — MLP <vla>`
   * - 约 5 分钟体验 VLA 评测
     - 单卡 24 GB+
     - :doc:`LIBERO 评测快速体验 <../evaluations/get_started/quick_tour>`
   * - 仿真环境 VLA 强化学习
     - 1–8 卡，40–80 GB
     - :doc:`训练快速开始 — OpenVLA / OpenVLA-OFT <vla>`
   * - 真机联调、数据采集与训练
     - 控制机 + GPU 节点
     - :doc:`真实机器人强化学习 <../examples/real_world_index>`
   * - 智能体、推理与扩散模型 RL
     - 单卡至多节点
     - :doc:`智能体与生成 <../examples/agentic/index>`

接下来
------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: 快速熟悉 RLinf
      :link: walkthrough
      :link-type: doc

      一页看懂训练闭环、部署模式与配置文件、工具箱的存放位置。

   .. grid-item-card:: 具身训练
      :link: ../examples/index
      :link-type: doc

      浏览模拟器、具身模型、SFT、RL 算法与世界模型配方。

   .. grid-item-card:: 使用指南
      :link: ../guides/index
      :link-type: doc

      配置启动、checkpoint、日志与多机集群。

   .. grid-item-card:: 常见问题
      :link: ../resources/faq
      :link-type: doc

      排查渲染、显存、网络与 Ray 问题。
