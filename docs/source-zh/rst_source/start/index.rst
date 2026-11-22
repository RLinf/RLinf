快速开始
========

从零开始跑通第一个 RLinf 任务，再按硬件和目标选择对应的工作流。

基础准备
--------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 1. 安装与环境
      :link: installation
      :link-type: doc

      硬件档位、一行 ``install.sh`` 本地安装，或前往 Docker Hub 拉取镜像。

   .. grid-item-card:: 2. 训练快速开始
      :link: vla
      :link-type: doc

      单卡零下载跑通 MLP，或在 ManiSkill3 上用 PPO 训练 OpenVLA / OpenVLA-OFT。

   .. grid-item-card:: 3. 速查表
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
   * - 约 5 分钟验证环境
     - 单卡（24 GB 起）
     - :doc:`LIBERO 评测快速体验 <../evaluations/get_started/quick_tour>`
   * - 轻量 RL（零权重下载）
     - 单卡（8–24 GB 起）
     - :doc:`训练快速开始（MLP） <vla>`
   * - 仿真环境 VLA 强化学习
     - 1–8 卡（40–80 GB）
     - :doc:`训练快速开始（OpenVLA / OpenVLA-OFT） <vla>`
   * - 真机联调、数据采集与训练
     - 控制机 + GPU 节点
     - :doc:`真实机器人强化学习 <../examples/real_world_index>`
   * - 大模型数学推理 RL
     - 4–8 卡（80 GB）
     - :doc:`数学推理 GRPO <../examples/agentic/math_reasoning/reasoning>`

快速熟悉 RLinf 仓库
-------------------

日常会用到的配置文件、启动脚本和工具都在六个核心目录里，按目标查找即可：

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: 训练入口与配置（``examples/``）
      :link: ../reference/configuration
      :link-type: doc

      按任务分目录存放启动脚本与 Hydra YAML 配置：具身 RL、SFT、离线 RL、推理、智能体与扩散模型。

   .. grid-item-card:: 独立评测（``evaluations/``）
      :link: ../evaluations/index
      :link-type: doc

      统一评测入口 ``run_eval.sh`` 与各基准评测配置（LIBERO、RoboTwin、BEHAVIOR、ManiSkill、真机等）。

   .. grid-item-card:: 实用工具箱（``toolkits/``）

      按需调用的辅助脚本：LeRobot 数据处理、真机硬件自检、Replay Buffer 可视化、自动 Placement 与权重转换。

   .. grid-item-card:: 核心框架（``rlinf/``）
      :link: ../concepts/overview
      :link-type: doc

      分布式调度与 worker、runner 循环、模型、仿真与真机环境、机器人硬件抽象及算法注册表。

   .. grid-item-card:: 安装脚本与镜像（``requirements/`` 与 ``docker/``）
      :link: installation
      :link-type: doc

      一键环境构建脚本 ``install.sh`` 及各硬件平台 Dockerfile。

   .. grid-item-card:: 测试与扩展（``tests/`` 与 ``docs/``）
      :link: ../extending/index
      :link-type: doc

      单元测试、端到端配置、真机假驱动（``robot_mocks/``）及中英文文档源码。

.. dropdown:: 查看 ``toolkits/`` 常用工具速查

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - 子目录
        - 用途
      * - ``toolkits/lerobot/``
        - 计算 ``norm_stats.json``、合并多个 LeRobot 数据集、导出 MP4 视频预览。
      * - ``toolkits/realworld_check/``
        - 真机训练前的交互式硬件自检（机械臂、夹爪、相机、GELLO、PICO）。
      * - ``toolkits/replay_buffer/``
        - 查看、可视化、合并或拆分已保存的轨迹 Replay Buffer。
      * - ``toolkits/auto_placement/``
        - 根据实测单步耗时自动搜索最优共置/分离 GPU 放置方案。
      * - ``toolkits/cosmos3/``、``dual_franka/``、``standalone_eval_scripts/``
        - 特定模型与机型的权重转换、数据清理与独立评测脚本。

接下来
------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: 具身训练
      :link: ../examples/index
      :link-type: doc

      浏览模拟器、具身模型、SFT、RL 算法与世界模型配方。

   .. grid-item-card:: 概念
      :link: ../concepts/index
      :link-type: doc

      理解执行流、调度策略与机器人接口。

   .. grid-item-card:: 指南
      :link: ../guides/index
      :link-type: doc

      配置启动、日志、checkpoint 与多机集群。

   .. grid-item-card:: 常见问题
      :link: ../resources/faq
      :link-type: doc

      排查渲染、显存、网络与 Ray 问题。
