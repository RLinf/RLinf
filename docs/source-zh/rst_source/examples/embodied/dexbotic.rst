Dexbotic 强化学习
=================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png
   :align: center
   :width: 90%

   Dexbotic 模型概览（图片来源：`Dexbotic <https://github.com/dexmal/dexbotic>`__）。

`Dexbotic <https://github.com/dexmal/dexbotic>`__ 是 Dexmal 推出的开源 VLA 工具箱。RLinf 将 Dexbotic π₀ 和 DM0 策略作为 LIBERO 动作生成模型，并用 PPO 在线强化学习微调。

概览
----------------------------------------

在 LIBERO 上用 PPO 微调 Dexbotic π₀ 或 DM0。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      LIBERO Spatial · Object · Goal · 10

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 GPU

| **你将完成：** 安装依赖 → 下载 Dexbotic 检查点 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 兼容 LIBERO 的 Dexbotic 检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

根据环境、任务族以及配置或权重工件选择对应的模型页面。

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_ppo_dexbotic_*``
     - 在 spatial 操作任务上使用 Dexbotic pi0/dm0 策略。
   * - LIBERO
     - LIBERO-Object
     - ``libero_object_ppo_dexbotic_pi0``
     - 在物体操作任务上使用 Dexbotic pi0。
   * - LIBERO
     - LIBERO-Goal / LIBERO-10
     - ``libero_goal_ppo_dexbotic_pi0`` / ``libero_10_ppo_dexbotic_pi0``
     - 覆盖目标条件和长程 LIBERO 套件。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - 为 Dexbotic 策略打包的 LIBERO 相机流与本体状态。
   * - Action
     - 选定 Dexbotic 策略后端生成的分块连续动作，包括 flow-matching / flow-SDE 设置。
   * - Reward
     - PPO 更新使用的 LIBERO 成功信号或仿真器奖励。
   * - Prompt
     - 策略 processor 消费的 LIBERO 自然语言指令。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: π₀

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model dexbotic --env maniskill_libero && source .venv/bin/activate

      **2. 下载模型**

      .. code-block:: bash

         hf download Dexmal/libero-db-pi0 --local-dir ./libero-db-pi0

      **3. 启动训练**

      ``examples/embodiment/config/`` 下内置的 π₀ 配置如下：

      .. list-table::
         :header-rows: 1
         :widths: 36 64

         * - 任务套件
           - 配置文件
         * - LIBERO Spatial
           - ``libero_spatial_ppo_dexbotic_pi0.yaml``
         * - LIBERO Object
           - ``libero_object_ppo_dexbotic_pi0.yaml``
         * - LIBERO Goal
           - ``libero_goal_ppo_dexbotic_pi0.yaml``
         * - LIBERO 10
           - ``libero_10_ppo_dexbotic_pi0.yaml``

      .. include:: _model_path.rst

      使用 ``run_embodiment.sh`` 启动 PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_dexbotic_pi0 actor.model.model_path=$PWD/libero-db-pi0

      .. dropdown:: 进阶配置（可选）

         - **参数精度**：π₀ 加载参数时使用 ``actor.model.precision`` 指定的精度。需要以 BF16 运行时，将其设为 ``bf16`` 并让 ``rollout.model`` 使用相同精度；设为 ``null`` 则保留 checkpoint 的默认精度。
         - **动作块长度**：π₀ 默认使用 ``num_action_chunks: 5``。
         - **独立评测**：使用 Dexbotic 的 LIBERO evaluator 评测训练后的检查点，或使用 RLinf 统一的 :doc:`评估 <../../evaluations/index>` 流程：

           .. code-block:: bash

              python toolkits/standalone_eval_scripts/dexbotic/libero_eval.py \
                --config_name db_pi0_libero \
                --pretrained_path ./libero-db-pi0 \
                --task_suite_name libero_spatial \
                --num_trials_per_task 50 \
                --action_chunk 5 \
                --num_steps 10

         - **组件放置与指标**：详见 :doc:`组件放置 <../../concepts/placement>`、:doc:`执行模式 <../../concepts/execution_modes>` 与 :doc:`训练指标 <../../reference/metrics>`。

   .. tab-item:: DM0

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model dexbotic --env maniskill_libero && source .venv/bin/activate

      **2. 下载模型**

      .. code-block:: bash

         hf download Dexmal/DM0-libero --local-dir ./DM0-libero

      **3. 启动训练**

      DM0 内置配置为 ``examples/embodiment/config/libero_spatial_ppo_dexbotic_dm0.yaml``。

      .. include:: _model_path.rst

      使用 ``run_embodiment.sh`` 启动 PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_dexbotic_dm0 actor.model.model_path=$PWD/DM0-libero

      .. dropdown:: 进阶配置（可选）

         - **动作块长度**：DM0 默认使用 ``num_action_chunks: 10``。
         - **独立评测**：使用 Dexbotic 的 LIBERO evaluator 评测训练后的检查点，或使用 RLinf 统一的 :doc:`评估 <../../evaluations/index>` 流程：

           .. code-block:: bash

              python toolkits/standalone_eval_scripts/dexbotic/libero_eval.py \
                --config_name dm0_libero \
                --pretrained_path ./DM0-libero \
                --task_suite_name libero_spatial \
                --num_trials_per_task 50 \
                --action_chunk 10 \
                --num_steps 10

         - **组件放置与指标**：详见 :doc:`组件放置 <../../concepts/placement>`、:doc:`执行模式 <../../concepts/execution_modes>` 与 :doc:`训练指标 <../../reference/metrics>`。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs`` 观察 ``env/success_once``；如需保存评测视频，设置 ``env.eval.video_cfg.save_video=True``，更多日志指标见 :doc:`训练指标 <../../reference/metrics>`。
