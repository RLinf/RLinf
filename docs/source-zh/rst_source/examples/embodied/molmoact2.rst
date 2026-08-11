MolmoAct2 评测
==============

`MolmoAct2 <https://github.com/allenai/molmoact2>`__ 是 AllenAI 开源的
vision-language-action 模型，通过 LeRobot policy 提供服务，并使用 flow-matching
action expert 预测连续动作。RLinf 通过其 LIBERO 入口运行官方 MolmoAct2-LIBERO
checkpoint。当前接入**仅支持评测**：上游 policy 只提供推理，没有训练路径。

概览
----

在四个 LIBERO suite 上评测官方 MolmoAct2-LIBERO checkpoint。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO

   .. grid-item-card:: 算法
      :text-align: center

      仅评测

   .. grid-item-card:: 任务
      :text-align: center

      Spatial · Object · Goal · Long

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 1–8 GPU

| **你将完成：** 安装 → 下载 checkpoint → 启动 → 观察 ``eval/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · MolmoAct2-LIBERO checkpoint（见下文步骤）。

任务
~~~~

每个 suite 都有对应的配置文件。每个配置使用 20 个并行环境、每个环境 25 个
episode，即完整的 500 条轨迹；step 预算为 ``max_episode_steps × 25``。

.. list-table::
   :header-rows: 1
   :widths: 18 40 20 22

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

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - 观测
     - 两路相机视角：``main_images`` 映射为 agent view，``wrist_images`` 映射为
       wrist view；此外还有 ``states`` 中的 8 维机器人状态。
   * - 动作
     - action expert 输出的连续 7 自由度动作（6 自由度末端位姿增量 + 夹爪）。
       每个 rollout step 执行一个动作。
   * - 奖励
     - LIBERO 任务成功与否。
   * - 提示词
     - 来自 ``task_descriptions`` 的自然语言任务指令。

安装
----

.. include:: _setup_common.rst

**选项 1：Docker 镜像** — 镜像标签 ``agentic-rlinf0.4-libero``：

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-libero
      # 国内镜像：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.4-libero

   # 在容器内切换到 MolmoAct2 虚拟环境：
   source switch_env molmoact2

**选项 2：自定义环境** — 安装 ``--model molmoact2 --env libero`` 依赖组合：

.. code-block:: bash

   # 国内用户可以添加 --use-mirror 加速下载。
   bash requirements/install.sh embodied --model molmoact2 --env libero
   source .venv/bin/activate

安装前设置 ``MOLMOACT2_LEROBOT_PATH``，可以复用已有的
`RLinf/lerobot <https://github.com/RLinf/lerobot/tree/RLinf/molmoact2-hf-inference>`__ checkout。

下载模型
--------

下载官方 `allenai/MolmoAct2-LIBERO <https://huggingface.co/allenai/MolmoAct2-LIBERO>`__ checkpoint：

.. code-block:: bash

   hf download allenai/MolmoAct2-LIBERO \
     --local-dir /path/to/model/MolmoAct2-LIBERO

然后将评测配置中的 ``rollout.model.model_path`` 指向该目录。该配置没有 ``actor``
部分，因此不需要同步第二个路径。

运行
----

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_10_molmoact2_eval \
     rollout.model.model_path=/path/to/model/MolmoAct2-LIBERO

本命令会：

1. 通过 MolmoAct2 model adapter 加载官方 checkpoint。
2. 使用 ``evaluations/libero/libero_10_molmoact2_eval.yaml`` 中的设置运行 LIBERO-Long suite。
3. 将终端输出和 ``eval/success_once`` 写入带时间戳的日志。

评测其他 suite 时，替换为上方任务表中的对应配置即可。

.. warning::

   完整 suite 包含 500 条轨迹，可能需要数小时。如果只需 smoke test，请调小
   ``env.eval.max_steps_per_rollout_epoch``：每个环境跑一个 episode 即
   ``max_episode_steps``。

.. admonition:: 进一步配置
   :class: note

   - 推理设置（``num_steps``、``norm_tag``、动作模式）→ ``examples/embodiment/config/model/molmoact2.yaml`` 中的 ``molmoact2`` 配置块。
   - ``rollout.model.precision`` 不会生效：MolmoAct2 在上游以 fp32 加载权重。
   - 请保持 ``rollout.pipeline_stage_num: 1``：该 policy 以 batch 索引为键维护每个环境的动作队列。
   - 请保持 ``env.eval.max_episode_steps`` 是该 policy 10 步动作队列的整数倍（240 / 320 / 520 均满足），否则新 episode 会从上一个 episode 的残留动作开始。
   - 放置与吞吐 → :doc:`放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`。

可视化与结果
------------

终端会输出 ``eval/success_once``。日志写入：

.. code-block:: text

   logs/<timestamp>-libero_10_molmoact2_eval/eval_embodiment.log

评测协议见 :doc:`LIBERO 评测 <../../evaluations/guides/libero>`，指标解释见
:doc:`评测结果 <../../evaluations/reference/results>`。
