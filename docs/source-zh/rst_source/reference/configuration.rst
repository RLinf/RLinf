训练配置
========

RLinf 示例使用 Hydra YAML 配置。这个页面作为共享配置参考；示例页面应链接到这里，
不要重复大段配置字段表。

配置位置
--------

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - 工作负载
     - 配置位置
     - 启动入口
   * - 具身 RL
     - ``examples/embodiment/config/*.yaml``
     - ``bash examples/embodiment/run_embodiment.sh <config_name>``
   * - 推理 RL
     - ``examples/reasoning/config/**.yaml``
     - ``bash examples/reasoning/run_main_grpo_math.sh <config_name>``
   * - 智能体工作流
     - ``examples/agent/**/config/*.yaml``
     - 对应 ``examples/agent/<recipe>/`` 目录下的启动脚本
   * - SFT
     - ``examples/sft/**`` 和各 recipe 的配置目录
     - 对应 recipe 的 ``run_*_sft.sh`` 启动脚本
   * - 评测
     - ``evaluations/<benchmark>/*.yaml``
     - ``bash evaluations/run_eval.sh <benchmark> <config_name>``

常用配置段
----------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 配置段
     - 用途
   * - ``cluster``
     - 节点数量、节点组，以及 actor、rollout、env、reward 或 agent worker 的组件放置。
   * - ``actor``
     - 训练后端、模型路径、优化器、batch size、offload、checkpoint 与 loss 设置。
   * - ``rollout``
     - 推理引擎、采样参数、模型路径和 rollout batch 设置。
   * - ``env``
     - 训练 / 评测环境类型、任务选择、资产路径、视频设置和 episode 控制。
   * - ``runner``
     - 任务类型、日志、checkpoint 间隔、验证间隔和断点续训行为。
   * - ``algorithm``
     - PPO、GRPO、SAC、IQL 或 DAgger 等算法的 advantage、loss 与专用设置。
   * - ``data``
     - 数据集路径、prompt / answer 字段、预处理、训练 / 验证划分和 SFT 数据选项。

修改 Recipe
-----------

1. 从 ``examples/`` 或 ``evaluations/`` 下的命名配置开始。
2. 设置本地路径，例如 ``rollout.model.model_path``、``actor.model.model_path``、
   数据集路径和环境资产路径。
3. 将硬件相关放置保留在 ``cluster`` 中。多节点运行时，设置 ``cluster.num_nodes``，
   并在每个节点启动 Ray 后再启动 recipe。
4. 将日志和 checkpoint 放在 ``runner.logger.log_path`` 下，便于统一管理 TensorBoard、
   视频和 checkpoint。

继续阅读
--------

- :doc:`基础配置 <../guides/basic_config>`
- :doc:`具身配置 <../guides/embodiment_config>`
- :doc:`智能体配置 <../guides/agentic_config>`
- :doc:`Placement <../concepts/placement>`
- :doc:`执行模式 <../concepts/execution_modes>`
- :doc:`训练指标 <metrics>`
- :doc:`评测配置 <../evaluations/reference/configuration>`
