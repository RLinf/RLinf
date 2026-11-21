快速开始
========

从零开始跑通第一个 RLinf 任务，再按硬件和目标选择对应的工作流。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/onboarding_flow_zh.svg
   :align: center
   :width: 100%

   从安装到监控与扩展的四步上手路径。

基础准备
--------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: 1. 环境要求
      :link: requirements
      :link-type: doc

      从单张 24 GB 显卡到多机集群的硬件档位说明。

   .. grid-item-card:: 2. 安装
      :link: installation
      :link-type: doc

      一行 ``install.sh`` 本地安装，或前往 Docker Hub 拉取镜像。

   .. grid-item-card:: 3. VLA 快速上手
      :link: vla
      :link-type: doc

      在 ManiSkill3 上用 PPO 训练 OpenVLA 或 OpenVLA-OFT。

   .. grid-item-card:: 4. 速查表
      :link: cheat_sheet
      :link-type: doc

      熟悉流程后直接复制最常用命令。

选择你的第一个任务
------------------

按目标直接进入对应的上手页面：

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - 目标
     - 典型硬件
     - 推荐起点
   * - 约 5 分钟验证环境
     - 单卡（24 GB 起）
     - :doc:`LIBERO 评测快速体验 <../evaluations/get_started/quick_tour>`
   * - 轻量 RL（无需下载大模型）
     - 单卡（24 GB 起）
     - :doc:`MLP policy 强化学习 <../examples/embodied/mlp>`
   * - 仿真环境 VLA 强化学习
     - 1–8 卡（40–80 GB）
     - :doc:`VLA 快速上手（ManiSkill3） <vla>`
   * - 真机联调、数据采集与训练
     - 控制机 + GPU 节点
     - :doc:`真实机器人强化学习 <../examples/real_world_index>`
   * - 大模型数学推理 RL
     - 4–8 卡（80 GB）
     - :doc:`数学推理 GRPO <../examples/agentic/math_reasoning/reasoning>`

接下来
------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: 示例库
      :link: ../examples/index
      :link-type: doc

      浏览具身、SFT、世界模型与智能体示例。

   .. grid-item-card:: 概念
      :link: ../concepts/index
      :link-type: doc

      理解执行流、调度策略与机器人接口。

   .. grid-item-card:: 指南
      :link: ../guides/index
      :link-type: doc

      配置启动、日志、检查点与多机集群。

   .. grid-item-card:: 常见问题
      :link: ../resources/faq
      :link-type: doc

      排查渲染、显存、网络与 Ray 问题。
