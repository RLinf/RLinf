RoboLab 评估
============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robolab.png
   :align: center
   :width: 90%
   :alt: RoboLab manipulation benchmark

   RoboLab (`NVlabs/RoboLab <https://github.com/NVlabs/RoboLab>`__).

RoboLab 是基于 Isaac Sim 的机器人操作 benchmark，包含 120 个任务，覆盖程序性、关系性和视觉能力，以及三个难度等级。RLinf 支持使用 OpenPI π₀.₅ policy 评估其中的 DROID joint-position 任务。

概览
----

使用两路 RGB 观测、机器人状态和自然语言指令完成评估。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅ / OpenPI

   .. grid-item-card:: 算法
      :text-align: center

      仅评估

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

运行评估
--------

按照 :doc:`评估指南 <../../evaluations/guides/robolab>` 安装环境、下载 checkpoint、配置任务并运行评估。

可视化与结果
------------

在终端和 TensorBoard 中查看 ``eval/success_once`` 和 ``eval/score``。下面列出同一组 120 个任务、每任务 10 回合的结果；详细指标和日志说明见 :doc:`评估指南 <../../evaluations/guides/robolab>`。


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
