RoboLab 评估
============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robolab.png
   :align: center
   :width: 90%
   :alt: RoboLab manipulation benchmark

   RoboLab (`NVlabs/RoboLab <https://github.com/NVlabs/RoboLab>`__).

使用 RLinf 的 π₀.₅ policy 评测 RoboLab 的 120 个 DROID joint-position 操作任务。此示例覆盖环境安装、checkpoint 下载、任务查询及成功率和归一化得分统计，执行流程由 RLinf 的环境接口和评估 runner 调度。

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

安装环境
--------

通过 ``bash requirements/install.sh embodied --model openpi --env robolab`` 安装单个 uv 环境。硬件要求、安装检查限制和已有源码复用方式见 :doc:`评估指南 <../../evaluations/guides/robolab>`。

下载模型
--------

使用 ``RLinf/RLinf-Pi05-RoboLab-droid-jointpos`` checkpoint，并将 ``rollout.model.model_path`` 指向下载目录。下载命令和文件结构见 :doc:`评估指南 <../../evaluations/guides/robolab>`。

运行评估
--------

配置入口为 ``evaluations/robolab/pi05_droid_jointpos.yaml``。使用 ``bash evaluations/run_eval.sh robolab pi05_droid_jointpos`` 启动；先按 :doc:`评估指南 <../../evaluations/guides/robolab>` 设置模型目录和任务步数，再选择单任务或完整 benchmark。

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

本次 RLinf 评测使用 8 张 A800，每卡 15 个任务；每任务 10 个并行环境，policy 每批处理 1 条观测。原生 Torch 启用了编译，RLinf 使用 eager 执行。实验未逐回合对齐随机数流，汇总结果接近不代表逐步动作完全相同。
