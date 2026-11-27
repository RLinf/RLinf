RoboLab 训练与评估
==================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robolab.png
   :align: center
   :width: 90%
   :alt: RoboLab manipulation benchmark

   RoboLab (`NVlabs/RoboLab <https://github.com/NVlabs/RoboLab>`__).

RoboLab 是基于 Isaac Sim 的机器人操作 benchmark，包含 120 个任务，覆盖程序性、关系性和视觉能力，以及三个难度等级。RLinf 支持使用 OpenPI π₀.₅ policy 在 DROID joint-position 任务上开展 PPO 训练和评估。

概览
----

使用两路 RGB 观测、机器人状态和自然语言指令训练或评估 policy。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅ / OpenPI

   .. grid-item-card:: 算法
      :text-align: center

      PPO 与评估

   .. grid-item-card:: 任务
      :text-align: center

      120 个操作任务

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA GPU / Isaac Sim

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 分类
     - 内容
   * - Difficulty
     - simple / moderate / complex
   * - Procedural
     - affordance / reorientation / stacking
   * - Relational
     - conjunction / counting / spatial
   * - Visual
     - color / semantics / size

任务元数据包含名称、指令、难度、能力属性、时长和源码路径。按照 :doc:`评估指南 <../../evaluations/guides/robolab>` 中的导出命令查询当前安装版本的全部任务。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 格式
   * - Observation
     - 两路 224×224 RGB 图像；7 维关节位置与 1 维夹爪状态。
   * - Action
     - 8 维动作：7 维关节位置目标与 1 维夹爪目标，动作块长度为 15。
   * - Reward
     - 成功回合的终止奖励为 1，其余为 0。
   * - Prompt
     - 任务元数据中的自然语言指令。

运行
----

按照 :doc:`评估指南 <../../evaluations/guides/robolab>` 安装环境、下载 ``RLinf-Pi05-RoboLab-droid-jointpos`` checkpoint、查看任务列表并运行评估。

在 ``BananaInBowlTask`` 上启动 PPO 训练时，先在 ``examples/embodiment/config/robolab_banana_in_bowl_ppo_openpi_pi05.yaml`` 中将 ``actor.model.model_path`` 设为本地 checkpoint 目录，再在 RLinf 仓库根目录运行：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh robolab_banana_in_bowl_ppo_openpi_pi05

日志与 TensorBoard 事件写入 ``logs/<timestamp>-robolab_banana_in_bowl_ppo_openpi_pi05/`` 。

训练模式支持按环境自动重置。

.. warning::

   保持 ``env.train.enable_offload: false`` 。RoboLab 不支持在 rollout 与训练之间挂起并恢复运行中的 Isaac Sim 环境。

可视化与结果
------------

PPO 训练期间在终端或 TensorBoard（``tensorboard --logdir <log_path>``\ ）中查看 ``env/success_once`` 和 ``env/return`` ，验证与评估期间查看 ``eval/success_once`` 和 ``eval/score`` 。训练指标定义见 :doc:`训练指标 <../../reference/metrics>` 。下面列出同一组 120 个任务、每任务 10 回合的基准评估结果；评估指标与日志说明见 :doc:`评估指南 <../../evaluations/guides/robolab>` 。


.. list-table::
   :header-rows: 1

   * - 评测
     - 成功率
     - 归一化得分
   * - RoboLab paper π₀.₅
     - 28.00%
     - 0.4300
   * - Native RoboLab + JAX
     - 27.83%
     - 0.4198
   * - Native RoboLab + Torch (compiled)
     - 26.50%
     - 0.4134
   * - RLinf + Torch (aligned, eager)
     - 27.00%
     - 0.4228
