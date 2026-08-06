MolmoAct2 评测
==============

通过 RLinf 的统一 LIBERO 入口评测官方 MolmoAct2-LIBERO checkpoint。当前接入仅支持评测。

安装
----

在仓库根目录安装 MolmoAct2 和 LIBERO：

.. code-block:: bash

   bash requirements/install.sh embodied --model molmoact2 --env libero
   source .venv/bin/activate

本命令会：

1. 安装 LIBERO 环境和 RLinf embodied 依赖。
2. 安装 `RLinf/lerobot <https://github.com/RLinf/lerobot/tree/RLinf/molmoact2-hf-inference>`__：RLinf 的 LeRobot fork，其 ``RLinf/molmoact2-hf-inference`` 分支提供 MolmoAct2 policy，并固定了 LIBERO 依赖栈所需的版本。

下载模型
--------

下载官方 `allenai/MolmoAct2-LIBERO <https://huggingface.co/allenai/MolmoAct2-LIBERO>`__ checkpoint：

.. code-block:: bash

   hf download allenai/MolmoAct2-LIBERO \
     --local-dir /path/to/model/MolmoAct2-LIBERO

运行
----

启动 ``libero_10_molmoact2_eval`` 配置，并覆盖其中的模型占位路径：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_10_molmoact2_eval \
     rollout.model.model_path=/path/to/model/MolmoAct2-LIBERO

本命令会：

1. 通过 MolmoAct2 model adapter 加载官方 checkpoint。
2. 使用 ``evaluations/libero/libero_10_molmoact2_eval.yaml`` 中的评测设置运行 LIBERO-Long suite。
3. 将终端输出和 ``eval/success_once`` 写入带时间戳的评测日志。

.. warning::

   默认配置会覆盖完整 LIBERO-Long suite，可能需要数小时。如果只需 smoke test，请通过 ``env.eval`` Hydra 覆盖项缩小评测规模。

其他 Task Suite
---------------

每个 LIBERO task suite 都有对应的配置文件。每个配置使用 20 个并行环境、每个环境 25 个 episode，即完整的 500 条轨迹；step 预算为 ``max_steps_per_rollout_epoch = max_episode_steps x 25``。

.. list-table::
   :header-rows: 1
   :widths: 22 38 20 20

   * - Suite
     - 配置
     - ``max_episode_steps``
     - 轨迹数
   * - Spatial
     - ``libero_spatial_molmoact2_eval``
     - 240
     - 500
   * - Object
     - ``libero_object_molmoact2_eval``
     - 240
     - 500
   * - Goal
     - ``libero_goal_molmoact2_eval``
     - 320
     - 500
   * - Long
     - ``libero_10_molmoact2_eval``
     - 520
     - 500

以 LIBERO-Spatial 为例：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_molmoact2_eval \
     rollout.model.model_path=/path/to/model/MolmoAct2-LIBERO

输入与推理设置
--------------

RLinf 将 ``main_images`` 映射为 MolmoAct2-LIBERO 所需的 agent view，将 ``wrist_images`` 映射为 wrist view，并原样传入 ``states`` 与 ``task_descriptions``；这四个键都是必需的。模型 preset 已在 ``molmoact2`` 配置块中设置连续动作推理、``norm_tag: libero`` 和 ``num_steps: 10``，无需在命令行中重复设置。

MolmoAct2 在上游以 fp32 加载权重，因此 ``rollout.model.precision`` 不会生效。它还会为每个 batch 索引维护独立的动作队列，因此请保持 ``rollout.pipeline_stage_num: 1``。

查看结果
--------

终端会输出 ``eval/success_once``。日志写入：

.. code-block:: text

   logs/<timestamp>-libero_10_molmoact2_eval/eval_embodiment.log

评测协议见 :doc:`LIBERO 评测 <../../evaluations/guides/libero>`，指标解释见 :doc:`评测结果 <../../evaluations/reference/results>`。
