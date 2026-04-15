基于D4RL评测平台的强化学习训练
============================================

本文说明如何在 RLinf 中运行基于 **D4RL 的 IQL（Implicit Q-Learning）离线强化学习训练**，面向直接使用离线数据集训练策略、无需在线环境交互的场景。

主要目标为训练一个策略，使其：

1. **仅使用离线数据**：训练阶段不与环境交互，数据来自 D4RL 数据集。
2. **遵循 IQL**：Value 用 expectile 回归、Actor 用 AWR 风格加权、双 Q 网络用 TD 目标。
3. **接入 RLinf 体系**：离线数据在 IQL Actor 内加载；EnvWorker、RolloutWorker 与 OfflineRunner 负责评测等；支持 PyTorch + FSDP。

环境
-----------

**D4RL（Datasets for Deep Data-Driven Reinforcement Learning）**

RLinf 使用 D4RL 基准套件，并为不同任务族提供配置：

- **MuJoCo 运动**：如 ``halfcheetah-medium-v2``、``hopper-medium-replay-v2``，连续控制、基于状态。
- **AntMaze**：如 ``antmaze-large-play-v0``，目标条件导航、稀疏奖励。
- **Kitchen / Adroit**：机械臂与灵巧手任务，高维状态与动作。

观测与动作空间由各 D4RL 任务定义。

算法
-----------

**核心算法组件**

1. **IQL（Implicit Q-Learning）**

   - **Value** :math:`V(s)`：用 expectile 回归更新，基于 :math:`Q_{\mathrm{target}}(s,a) - V(s)`，权重 :math:`w(d) = \tau \cdot \mathbb{I}(d>0) + (1-\tau) \cdot \mathbb{I}(d \le 0)`。
   - **Actor** :math:`\pi(a|s)`：AWR 风格优势加权极大似然；优势 :math:`A = Q_{\mathrm{target}}(s,a) - V(s)`，权重 :math:`w = \min(\exp(A \cdot \beta), 100)`。
   - **Critic** （双 Q）：TD 损失，目标 :math:`y = r + \gamma \cdot \mathrm{mask} \cdot V(s')`。
   - **Target**：以 :math:`\tau` 软更新目标 Critic。

2. **训练流程**

   每个 update step：Actor 从各 rank 本地的 ``DataLoader``（在 ``EmbodiedIQLFSDPPolicy.build_offline_dataloader`` 中构建）取一个 batch，然后按当前实现顺序执行 IQL：更新 Value → 更新 Actor → 更新 Critic → 软更新目标 Critic。

依赖安装
----------------------------

安装带 D4RL 的 embodied 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --env d4rl
   source .venv/bin/activate

启动脚本默认设置 ``MUJOCO_GL=egl`` 与 ``PYOPENGL_PLATFORM=egl``，便于无头运行。

运行脚本
----------------------------

**1. 配置文件**

RLinf 为不同 D4RL 任务族提供默认 IQL 配置：

- **MuJoCo**：``examples/embodiment/config/d4rl_iql_mujoco.yaml``
- **AntMaze**：``examples/embodiment/config/d4rl_iql_antmaze.yaml``
- **Kitchen / Adroit**：``examples/embodiment/config/d4rl_iql_kitchen_adroit.yaml``

**2. 关键参数配置**

**2.1 Runner 与 Algorithm**

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

**2.2 Actor（模型）**

.. code-block:: yaml

   actor:
     model:
       iql_config:
         type: "actor"
         hidden_dims: [256, 256]
         dropout_rate: null
         log_std_min: -5.0
         log_std_max: 2.0

**2.3 数据**

.. code-block:: yaml

   data:
     dataset_type: "d4rl"
     task_name: "antmaze-large-play-v0"
     dataset_path: null

**2.4 环境**

.. code-block:: yaml

   env:
     task_name: "antmaze-large-play-v0"

将 ``data.dataset_type`` 设为 ``d4rl``，并将 ``data.task_name`` 和 ``env.eval.task_name`` 设为所需的 D4RL 任务（如 ``antmaze-large-play-v0``）。

**3. 启动脚本**

- 脚本：``examples/embodiment/run_offline_rl.sh``
- 默认配置（不传参数）：``d4rl_iql_mujoco``
- 日志目录：``<repo>/logs/<timestamp>-<config_name>/``
- 实际命令：

.. code-block:: bash

   python examples/embodiment/train_offline_rl.py \
     --config-path examples/embodiment/config/ \
     --config-name <config_name> \
     runner.logger.log_path=<log_dir> runner.logger.experiment_name=<config_name>

**4. 启动命令**

在仓库根目录执行：

**MuJoCo（默认）**

::

   ./examples/embodiment/run_offline_rl.sh d4rl_iql_mujoco

**AntMaze**

::

   ./examples/embodiment/run_offline_rl.sh d4rl_iql_antmaze

**Kitchen / Adroit**

::

   ./examples/embodiment/run_offline_rl.sh d4rl_iql_kitchen_adroit

断点续训
----------------------------

将 ``runner.resume_dir`` 设为 checkpoint 目录（如 ``checkpoints/global_step_XXXXX``），再执行相同启动命令即可从该步加载权重并继续训练。

可视化与结果
----------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键跟踪指标**

- **训练指标**：

  - ``train/value_loss``: Value 函数 expectile 损失。
  - ``train/actor_loss``: AWR 风格策略损失。
  - ``train/critic_loss``: 双 Q 网络 TD 损失。
  - ``train/v``: Value 函数估计（batch 均值）。
  - ``train/q1``、``train/q2``: 双 Q 网络估计。
  - ``train/adv_mean``、``train/adv_std``: 优势的均值与标准差。

- **时间指标**：

  - ``time/step``: 每训练步墙钟时间（取数 + actor 更新）。
  - ``time/eval``: 评估墙钟时间（当 ``runner.eval_episodes`` > 0 时）。
  - ``time/actor/update_one_epoch``: 每步 actor 更新时间。

- **评估指标** （当 ``runner.eval_episodes`` > 0 时）：

  - ``eval/return``: 评估 rollout 的平均 episode 回报。
  - ``eval/episode_len``: 平均 episode 长度。
  - ``eval/num_trajectories``: 评估轨迹数量。
  - ``eval/terminated_at_end``: 以终止（非截断）结束的 episode 比例；仅当环境启用 ``ignore_terminations`` 时存在。

**3. 视频生成**

D4RL 观测为纯状态（无图像键）。当观测中没有图像字段时，录像会**回退到 ``env.render()``**，因此只需将 ``save_video`` 设为 true 即可生成评估视频。``video_base_dir`` 为可选项（默认 ``./video``），你也可以显式配置以便统一管理输出目录。需保证 MuJoCo 环境支持渲染（当 ``save_video`` 为 true 时会自动以 ``render_mode="rgb_array"`` 创建）。无头服务器请设置 ``MUJOCO_GL=egl`` 与 ``PYOPENGL_PLATFORM=egl``。

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: true
            video_base_dir: ${runner.logger.log_path}/video/eval  # 可选，默认 ./video

**4. 训练日志工具集成**

.. code-block:: yaml

   runner:
      task_type: "offline"
      logger:
         log_path: "../results"
         project_name: rlinf
         experiment_name: "d4rl_iql_mujoco"
         logger_backends: ["tensorboard"] # wandb, swanlab