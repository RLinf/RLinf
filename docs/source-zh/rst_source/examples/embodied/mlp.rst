MLP 强化学习
============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/3_layer_mlp.jpg
   :align: center
   :width: 55%

   多层感知机（MLP）策略。

MLP 策略是一种轻量网络，用于从低维状态输入（关节角、末端执行器位姿、物体状态等）控制机器人。RLinf 在多个仿真器上用 PPO、SAC、GRPO 训练 MLP 策略——便于快速验证环境、训练流程与网络结构。

概览
----------------------------------------

在低维状态下，用 PPO/SAC/GRPO 在 ManiSkill、LIBERO-Spatial、FrankaSim 上训练 MLP 策略。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      ManiSkill · LIBERO · FrankaSim

   .. grid-item-card:: 算法
      :text-align: center

      PPO · SAC · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      PickCube · LIBERO-Spatial

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · GPU

| **你将完成：** 安装目标仿真器 → 选择配置 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 目标仿真器的依赖。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通过 ``defaults`` 列表选择环境（``env/<env_name>@env.train`` / ``@env.eval``）；可在 ``env.train`` / ``env.eval`` 下覆写并行环境数、回合长度与录制等参数。

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - ManiSkill3
     - PickCube
     - ``maniskill_ppo_mlp`` / ``maniskill_sac_mlp``
     - 基于低维状态的策略训练。
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_0_grpo_mlp``
     - 在 LIBERO spatial 任务上使用 MLP 策略运行 GRPO。
   * - MuJoCo / FrankaSim
     - PickCube
     - ``frankasim_ppo_mlp``
     - FrankaSim 中基于状态的 PPO 训练。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - 低维状态向量，例如机器人关节、末端执行器位姿和物体状态。
   * - Action
     - 由 ``policy_setup``、``obs_dim`` 和 ``action_dim`` 配置的连续机器人控制命令。
   * - Reward
     - 仿真器任务奖励或成功信号。
   * - Prompt
     - MLP 策略不使用 prompt；任务通过 Hydra 配置选择。

安装与运行
----------

.. include:: _setup_common.rst

无需预训练 checkpoint，MLP 策略从头训练。

.. tab-set::

   .. tab-item:: ManiSkill / LIBERO

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env maniskill_libero && source .venv/bin/activate

      **2. 启动训练**

      ``examples/embodiment/config/`` 下的默认配置：

      .. list-table::
         :header-rows: 1
         :widths: 30 34 36

         * - 配方
           - 配置文件
           - 启动命令后缀
         * - ManiSkill + PPO + MLP
           - ``maniskill_ppo_mlp.yaml``
           - ``maniskill_ppo_mlp``
         * - ManiSkill + SAC + MLP
           - ``maniskill_sac_mlp.yaml``
           - ``maniskill_sac_mlp``
         * - LIBERO-Spatial + GRPO + MLP
           - ``libero_spatial_0_grpo_mlp.yaml``
           - ``libero_spatial_0_grpo_mlp``

      .. code-block:: bash

         # ManiSkill（PPO-MLP）
         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

         # ManiSkill（SAC-MLP）
         bash examples/embodiment/run_embodiment.sh maniskill_sac_mlp

         # LIBERO-Spatial（GRPO-MLP）
         bash examples/embodiment/run_embodiment.sh libero_spatial_0_grpo_mlp

   .. tab-item:: Franka-Sim

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env frankasim && source .venv/bin/activate

      **2. 启动训练**

      ``examples/embodiment/config/`` 下的默认配置：

      .. list-table::
         :header-rows: 1
         :widths: 30 34 36

         * - 配方
           - 配置文件
           - 启动命令后缀
         * - Franka-Sim + PPO + MLP
           - ``frankasim_ppo_mlp.yaml``
           - ``frankasim_ppo_mlp``

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh frankasim_ppo_mlp

.. note::

   **SAC 注意事项。** SAC 通过 Bellman 回传与熵正则学习 Q 值（off-policy），需在配置中启用 Q 相关 head（``add_q_head: True``），并支持通过 ``entropy_tuning``\ （如 ``alpha_type: softplus``）自动调节熵以平衡探索与利用。

.. dropdown:: 进阶配置（可选）

   MLP 模型由 ``model/mlp_policy@actor.model`` 引入，可按配方覆盖以下主要默认字段：

   .. code-block:: yaml

      model_type: "mlp_policy"
      model_path: ""
      policy_setup: "panda-qpos"
      obs_dim: 42
      action_dim: 8
      num_action_chunks: 1
      hidden_dim: 256
      precision: "32"
      add_value_head: True
      is_lora: False
      lora_rank: 32

   真机多节点部署（Actor/Policy 部署在 GPU 服务器，Env/Robot 部署在控制机）见 :doc:`franka`。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 曲线，通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制，完整指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。
