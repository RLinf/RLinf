Dexbotic模型强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png
   :align: center
   :width: 90%

   Dexbotic 模型概览（图片来源：`Dexbotic <https://github.com/dexmal/dexbotic>`__）。

`Dexbotic <https://github.com/dexmal/dexbotic>`__ 是 Dexmal 推出的开源 VLA 工具箱。
RLinf 将 Dexbotic π\ :sub:`0`\ 和 DM0 策略作为 LIBERO 动作生成模型，并使用 PPO 进行在线强化学习微调。

概览
----------------------------------------

在 LIBERO 上用 PPO 微调 Dexbotic π\ :sub:`0`\ 或 DM0。

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

安装
----------------------------------------

.. include:: _setup_common.rst

**选项 1：Docker 镜像** — 镜像标签 ``agentic-rlinf0.2-maniskill_libero``：

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 国内镜像：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

   # 在容器内切换到 Dexbotic 虚拟环境：
   source switch_env dexbotic

**选项 2：自定义环境** — 安装 ``--model dexbotic --env maniskill_libero`` 依赖组合：

.. code-block:: bash

   # 国内用户可以添加 --use-mirror 加速下载。
   bash requirements/install.sh embodied --model dexbotic --env maniskill_libero
   source .venv/bin/activate

下载模型
----------------------------------------

下载一个或两个 Dexbotic 检查点（任选一种方式）：

.. code-block:: bash

   # 方法 1：git clone
   git lfs install
   git clone https://huggingface.co/Dexmal/libero-db-pi0
   git clone https://huggingface.co/Dexmal/DM0-libero

   # 方法 2：huggingface-hub（国内用户可设置 HF_ENDPOINT=https://hf-mirror.com）
   pip install huggingface-hub
   huggingface-cli download Dexmal/libero-db-pi0 --local-dir libero-db-pi0
   huggingface-cli download Dexmal/DM0-libero --local-dir DM0-libero

.. include:: _model_path.rst

运行
----------------------------------------

每个配方都是 ``examples/embodiment/config/`` 下的一个 YAML 配置：

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - 任务套件
     - 模型
     - 配置
   * - LIBERO Spatial
     - Dexbotic π₀
     - ``libero_spatial_ppo_dexbotic_pi0.yaml``
   * - LIBERO Spatial
     - DM0
     - ``libero_spatial_ppo_dexbotic_dm0.yaml``
   * - LIBERO Object
     - Dexbotic π₀
     - ``libero_object_ppo_dexbotic_pi0.yaml``
   * - LIBERO Goal
     - Dexbotic π₀
     - ``libero_goal_ppo_dexbotic_pi0.yaml``
   * - LIBERO 10
     - Dexbotic π₀
     - ``libero_10_ppo_dexbotic_pi0.yaml``

使用 ``run_embodiment.sh`` 启动一个配置：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_dexbotic_pi0

**这个命令会：**

1. 加载 ``examples/embodiment/config/libero_spatial_ppo_dexbotic_pi0.yaml``。
2. 按 ``cluster.component_placement`` 构建 LIBERO actor、rollout 和 env worker。
3. 运行 PPO，并把日志和检查点写入 ``runner.logger.log_path``。

.. admonition:: 进一步配置
   :class: note

   - π₀ 检查点路径 → 将 ``actor.model.model_path`` 和 ``rollout.model.model_path`` 设置为 ``libero-db-pi0``。
   - DM0 检查点路径 → 在 ``libero_spatial_ppo_dexbotic_dm0.yaml`` 中将两个 model path 设置为 ``DM0-libero``。
   - Action chunks → π₀ 使用 ``num_action_chunks: 5``；DM0 使用 ``num_action_chunks: 10``。
   - 指标定义和日志后端 → :doc:`训练指标 <../../reference/metrics>`
   - 组件放置和吞吐调优 → :doc:`组件放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`

独立评测
----------------------------------------

对训练后的检查点运行 Dexbotic 的 LIBERO evaluator：

.. code-block:: bash

   python toolkits/standalone_eval_scripts/dexbotic/libero_eval.py \
      --config_name db_pi0_libero \
      --pretrained_path /path/to/checkpoint \
      --task_suite_name libero_spatial \
      --num_trials_per_task 50 \
      --action_chunk 5 \
      --num_steps 10

如需评估 DM0，切换 evaluator 配置和 action chunk：

.. code-block:: bash

   python toolkits/standalone_eval_scripts/dexbotic/libero_eval.py \
      --config_name dm0_libero \
      --pretrained_path /path/to/checkpoint \
      --task_suite_name libero_spatial \
      --num_trials_per_task 50 \
      --action_chunk 10 \
      --num_steps 10

也可以使用 RLinf 统一的 VLA 评估流程，详见 :doc:`评估 <../../evaluations/index>`。

可视化与结果
----------------------------------------

启动 TensorBoard 实时观察训练：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

最值得关注的指标是 **``env/success_once``** —— 回合成功率。每个日志指标的含义见 :doc:`训练指标 <../../reference/metrics>`。
