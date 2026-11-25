IQL (D4RL)
==========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/d4rl.png
   :align: center
   :width: 70%

   D4RL 基准上的离线强化学习。

使用本配方在 RLinf 中运行基于 **D4RL 的 IQL（Implicit Q-Learning）离线强化学习训练**，直接用离线数据集训练策略，无需在线环境交互。

主要目标为训练一个策略，使其：

1. **仅使用离线数据**：训练阶段不与环境交互，数据来自 D4RL 数据集。
2. **遵循 IQL**：Value 用 expectile 回归、Actor 用 AWR 风格加权、双 Q 网络用 TD 目标。
3. **接入 RLinf 体系**：离线数据在 IQL Actor 内加载；EnvWorker、RolloutWorker 与 OfflineRunner 负责评测等；支持 PyTorch + FSDP。

概览
----------------------------------------

用 IQL 从 D4RL 离线数据集训练策略——无需在线环境交互。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      IQL

   .. grid-item-card:: 模型
      :text-align: center

      MLP

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      D4RL

   .. grid-item-card:: 训练
      :text-align: center

      离线

| **你将完成：** 安装（含 D4RL）→ 选择配置 → 运行 ``run_offline_rl.sh`` → 观察 ``eval/return``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · D4RL 数据集（首次运行时自动下载）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 为三类 D4RL 任务族提供 IQL 配置；各任务的观测与动作空间由 D4RL 定义。

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - 任务族
     - 示例任务
     - 配置
   * - MuJoCo 运动控制
     - ``halfcheetah-medium-v2``
     - ``d4rl_iql_mujoco.yaml``
   * - AntMaze
     - ``antmaze-large-play-v0``
     - ``d4rl_iql_antmaze.yaml``
   * - Kitchen / Adroit
     - 操作 / 灵巧手
     - ``d4rl_iql_kitchen_adroit.yaml``

IQL 原理
----------------------------------------

**核心算法组件**

1. **IQL（Implicit Q-Learning）**

   - **Value**：按目标 Q 值与当前价值估计的差值做非对称 expectile 回归，由 ``algorithm.expectile`` 控制权重。
   - **Actor**：采用 AWR 风格的优势加权极大似然，用 ``algorithm.temperature`` 缩放优势后取指数并截断到 100。
   - **Critic**\ （双 Q）：以折扣后的下一状态价值估计作为 TD 目标，由 ``algorithm.discount`` 控制折扣系数。
   - **Target**：按 ``algorithm.tau`` 对目标 Critic 做 Polyak 软更新。

2. **训练流程**

   每个 update step：Actor 从各 rank 本地的离线数据加载器取一个 batch，然后按顺序执行 IQL 更新：更新 Value → 更新 Actor → 更新 Critic → 软更新目标 Critic。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: D4RL

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env d4rl && source .venv/bin/activate

      启动脚本默认设置 ``MUJOCO_GL=egl`` 与 ``PYOPENGL_PLATFORM=egl``，便于无头运行。

      **2. 启动训练**

      RLinf 为三类 D4RL 任务族内置了默认 IQL 配置（``data.dataset_path`` 保持 ``null`` 时，首次运行自动下载对应 D4RL 数据集）：

      .. list-table::
         :header-rows: 1
         :widths: 26 74

         * - 任务族
           - 配置文件
         * - MuJoCo
           - ``examples/embodiment/config/d4rl_iql_mujoco.yaml``
         * - AntMaze
           - ``examples/embodiment/config/d4rl_iql_antmaze.yaml``
         * - Kitchen / Adroit
           - ``examples/embodiment/config/d4rl_iql_kitchen_adroit.yaml``

      在仓库根目录启动任一任务族（省略参数时默认使用 ``d4rl_iql_mujoco``；日志写入 ``logs/<timestamp>-<config_name>/``）：

      .. code-block:: bash

         # MuJoCo（默认）
         ./examples/embodiment/run_offline_rl.sh d4rl_iql_mujoco

         # AntMaze
         ./examples/embodiment/run_offline_rl.sh d4rl_iql_antmaze

         # Kitchen / Adroit
         ./examples/embodiment/run_offline_rl.sh d4rl_iql_kitchen_adroit

.. dropdown:: 进阶配置（可选）

   **切换任务与本地数据集。** 修改 ``data.task_name`` 和 ``env.eval.task_name`` 可切换 D4RL 任务（如 ``antmaze-large-play-v0``）。``data.dataset_path`` 为 ``null`` 时，RLinf 优先读取 ``~/.d4rl/datasets/<task_name>.hdf5``，不存在时由 D4RL 自动下载。若要使用本地 HDF5 文件，将 ``data.dataset_path`` 设为对应路径（观测和动作维度须与 ``data.task_name`` 一致）；标准 D4RL 文件按 ``timeouts`` 经过 ``d4rl.qlearning_dataset`` 转换，已整理好的 transition 文件（包含 ``next_observations`` 且无 ``timeouts``）则直接使用。

   **断点续训。** 将 ``runner.resume_dir`` 设为 checkpoint 目录（如 ``checkpoints/global_step_XXXXX``），再执行相同启动命令即可从该步加载权重并继续训练。

   内置配置中的主要默认超参数如下：

   .. code-block:: yaml

      runner:
        task_type: "offline"

      algorithm:
        loss_type: "offline_iql"
        batch_size: 256
        actor_lr: 3.0e-4
        value_lr: 3.0e-4
        critic_lr: 3.0e-4
        discount: 0.99
        tau: 0.005
        expectile: 0.9
        temperature: 10.0
        gamma: 0.99

      actor:
        model:
          iql_config:
            type: "actor"
            hidden_dims: [256, 256]
            dropout_rate: null
            log_std_min: -5.0
            log_std_max: 2.0

      data:
        dataset_type: "d4rl"
        task_name: "antmaze-large-play-v0"
        dataset_path: null

      env:
        task_name: "antmaze-large-play-v0"

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``eval/return`` 曲线，通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制（纯状态 D4RL 环境在 ``MUJOCO_GL=egl`` 下会自动回退到 ``env.render()``），完整指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。

IQL 相关指标：

- **训练指标**：

  - ``train/value_loss``：Value 函数 expectile 损失。
  - ``train/actor_loss``：AWR 风格 policy 损失。
  - ``train/critic_loss``：双 Q 网络 TD 损失。
  - ``train/v``：Value 函数估计（batch 均值）。
  - ``train/q1``、``train/q2``：双 Q 网络估计。
  - ``train/adv_mean``、``train/adv_std``：优势的均值与标准差。

- **时间指标**：

  - ``time/step``：每训练步墙钟时间（取数 + actor 更新）。
  - ``time/eval``：评估墙钟时间（``runner.eval_episodes`` > 0 时）。
  - ``time/actor/update_one_epoch``：每步 actor 更新时间。

- **评估指标**\ （``runner.eval_episodes`` > 0 时）：

  - ``eval/return``：评估 rollout 的平均 episode 回报。
  - ``eval/episode_len``：平均 episode 长度。
  - ``eval/num_trajectories``：评估轨迹数量。
  - ``eval/terminated_at_end``：以终止（非截断）结束的 episode 比例；仅在环境启用 ``ignore_terminations`` 时存在。
