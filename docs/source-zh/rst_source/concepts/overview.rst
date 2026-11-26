架构总览
========

本页用三张图建立 RLinf 的整体心智模型：一次训练步的数据闭环、三种组件部署模式，以及衔接真机与仿真各阶段的落盘数据格式。

核心训练闭环
------------

每次 RLinf 运行都由单个入口脚本启动：创建基于 Ray 的 ``Cluster``，将 ``Env``、``Rollout``、可选的 ``Reward`` 与 ``Actor`` worker 组放置到硬件上，再由 ``Runner`` 驱动采样、优势计算、策略更新与权重同步闭环。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/core_dataflow_zh.svg
   :align: center
   :width: 100%
   :alt: RLinf 核心训练闭环：环境、Rollout、奖励与缓冲区、Actor 训练及权重同步

   RLinf 核心训练闭环与权重同步。

.. list-table::
   :header-rows: 1
   :widths: 20 26 54

   * - 组件
     - 配置段
     - 职责
   * - 环境 / Agent
     - ``env.*``、``agentloop.*``
     - 驱动仿真器、真机机器人或工具调用沙箱，产出观测、奖励与结束信号。
   * - Rollout
     - ``rollout.*``
     - 通过 HuggingFace、SGLang 或 vLLM 执行策略推理，产出动作、回复与 logprobs。
   * - 奖励与缓冲区
     - ``algorithm.*``、``reward.*``
     - 执行规则或模型打分，计算 GAE / GRPO 优势，或提供离策略 Replay Buffer 采样批次。
   * - Actor
     - ``actor.*``
     - 使用 FSDP 或 Megatron 更新策略与价值权重，将新权重同步回 Rollout 并保存 checkpoint。

深入执行层细节见 :doc:`execution_flow`、:doc:`worker` 与 :doc:`channel`。

组件部署模式
------------

Collocated、Disaggregated 与 Hybrid 对应同一套 worker 的三种硬件放置方式，通过 YAML 中的 ``cluster.component_placement`` 即可切换。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/deployment_modes_zh.svg
   :align: center
   :width: 100%
   :alt: RLinf 三种组件部署模式对比：Collocated、Disaggregated 与 Hybrid 异构部署

   Collocated、Disaggregated 与 Hybrid 异构部署对比。

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - 模式
     - 适用卡数 / 拓扑
     - 适用场景
   * - Collocated 共置
     - 1–8 卡
     - 快速上手与单机训练默认模式，所有组件共享 GPU 并配合 offload 运行。
   * - Disaggregated 分离
     - 8+ 卡
     - 各组件独占 GPU，消除 offload 开销，适合流水线并行与异步 RL。
   * - Hybrid 混合 / 异构
     - 多节点或真机 + GPU
     - 将共享 GPU 组件与独立 CPU 节点或真机控制主机灵活组合。

放置语法与多机拓扑详见 :doc:`execution_modes`、:doc:`placement` 与 :doc:`../guides/multi_node`。

数据格式流转
------------

具身工作流通过 ``save_dir``\ （默认为 ``${runner.logger.log_path}``）下的三种落盘格式，串联遥操作数据采集、仿真导出、SFT、在线 RLPD 与奖励模型训练。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/data_formats_zh.svg
   :align: center
   :width: 100%
   :alt: RLinf 数据格式流转图：采集来源、三种存储格式与下游消费阶段

   数据采集来源、存储格式与下游训练阶段。

.. list-table::
   :header-rows: 1
   :widths: 22 32 46

   * - 格式
     - 落盘目录
     - 下游消费阶段
   * - ``lerobot_dataset``
     - ``{save_dir}/lerobot_dataset/``
     - OpenPI / VLA SFT、HG-DAgger 与离线 RL（RECAP、STEAM）。
   * - ``replay_buffer``
     - ``{save_dir}/demo_buffer/``
     - 在线 RLPD / SAC（通过 ``algorithm.demo_buffer.load_path`` 加载）。
   * - ``pickle_episode``
     - ``{save_dir}/pickle_episode/``
     - ResNet 二分类奖励模型与 VLM 趋势奖励模型预处理。

下一步
------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 训练配置地图
      :link: ../reference/configuration
      :link-type: doc

      对照带注释的最小 YAML 与 Hydra 预设结构。

   .. grid-item-card:: 数据采集指南
      :link: ../guides/data_collection
      :link-type: doc

      了解导出格式配置、流式分片与数据集合并。

   .. grid-item-card:: 第一个训练任务
      :link: ../start/vla
      :link-type: doc

      运行单卡具身智能或 Agent / LLM 训练示例。
