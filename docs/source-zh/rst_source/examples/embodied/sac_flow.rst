SAC-Flow
========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-overview.png
   :align: center
   :width: 80%

   SAC-Flow 总览。

本示例展示 RLinf 框架使用 **SAC（Soft Actor-Critic）** 算法训练 **Flow Matching** 策略网络的完整流程。该算法结合了最大熵强化学习（SAC）与生成式流匹配模型（Flow Matching）的优势，支持在仿真环境（ManiSkill3）和真机环境（Franka）中训练。

相关论文：`SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling <https://arxiv.org/abs/2509.25756>`_

概览
----------------------------------------

用 SAC 训练流匹配（Flow Matching）策略——可在 ManiSkill 仿真或真实 Franka（插销）上运行。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      SAC · RLPD

   .. grid-item-card:: 模型
      :text-align: center

      Flow Matching 策略

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      ManiSkill · Franka

   .. grid-item-card:: 训练
      :text-align: center

      Sim & Real

| **你将完成：** 安装（仿真或真机）→ 选择配置 → 启动 → 观察 ``env/success_once``。
| **前置条件：** 仿真请参考 :doc:`安装 </rst_source/start/installation>`，真机请参考 :doc:`franka`。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 32 28 24

   * - 设置
     - 环境与任务
     - 观测
     - 动作
   * - 仿真
     - ManiSkill3 —— ``PickCube-v1``
     - 关节角 + 物体状态
     - 4 维：3D 位置 + 夹爪
   * - 真机
     - Franka Panda + RealSense —— 插销
     - RGB + 本体感知
     - 6 维末端执行器位姿

SAC-Flow 工作原理
----------------------------------------

**核心算法组件**

1.  **SAC (Soft Actor-Critic)**

    -   通过 Bellman 公式和熵正则化学习 Q 值。

    -   使用 **Flow Matching** 网络作为 Actor 策略。

    -   学习温度参数以平衡探索与利用。

2.  **Flow Matching Policy**

    -   **速度网络参数化**：将流策略的 K 步采样视为 RNN，将流策略中的速度网络替换成为循环而生的现代 Transformer 架构，解决训练稳定问题。

    -   **对数似然计算**：在每步采样中添加高斯噪声 + 配套漂移修正，保证末端动作分布不变，同时把路径密度分解为单步高斯似然的连乘，从而得到可微的动作对数似然。

3. **RLPD (Reinforcement Learning with Prior Data)**

   - SAC 的一种变体，结合离线数据和在线数据训练。

   - 为加速在真实世界的训练，SAC-Flow 也可结合 RLPD 使用预采集的离线数据作为演示缓冲区。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 仿真（ManiSkill）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env maniskill_libero && source .venv/bin/activate

      **2. 启动训练**

      在单机上运行 ``examples/embodiment/config/maniskill_sac_flow_state.yaml``：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_sac_flow_state

   .. tab-item:: 真机（Franka）

      **1. 安装环境**

      在 GPU 训练节点和 Franka 控制节点上安装环境：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export LIBFRANKA_VERSION=0.19.0
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. 启动训练**

      ``examples/embodiment/config/realworld_sac_flow_image.yaml`` 默认使用 **2 节点**\ （``cluster.num_nodes: 2``：GPU 服务器运行 Actor/Policy，Franka 控制机运行 Env/Robot）。先按 :doc:`franka` 完成集群与硬件配置，再在 head 节点启动：

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_sac_flow_image

.. dropdown:: 进阶配置（可选）

   仓库自带配置已设置好 Flow Matching 模型和 SAC 算法默认参数，仅在需要调优时修改。

   **模型参数**

   .. code-block:: yaml

      actor:
        model:
          model_type: "flow_policy"
          # 输入类型：'state'（仿真）或 'mixed'（真机，图像 + 状态）
          input_type: "state"

          # Flow Matching 相关参数
          denoising_steps: 4  # 生成动作去噪步数
          d_model: 256        # Transformer 维度
          n_head: 4           # 注意力头数
          n_layers: 2         # 层数
          use_batch_norm: False  # 是否使用批归一化
          batch_norm_momentum: 0.99  # 批归一化动量
          flow_actor_type: "JaxFlowTActor"  # JAX 风格的 "JaxFlowTActor" 或 torch 风格的 "FlowTActor"
          noise_std_head: False  # 是否使用单独的头来预测噪声标准差，否则使用固定标准差
          log_std_min_train: -5  # 训练时最小对数标准差（如果使用 noise_std_head）
          log_std_max_train: 2   # 训练时最大对数标准差（如果使用 noise_std_head）
          log_std_min_rollout: -20  # 推理时最小对数标准差（如果使用 noise_std_head）
          log_std_max_rollout: 0    # 推理时最大对数标准差（如果使用 noise_std_head）
          noise_std_train: 0.3  # 训练时固定噪声标准差（如果不使用 noise_std_head）
          noise_std_rollout: 0.02  # 推理时固定噪声标准差（如果不使用 noise_std_head）

   **算法参数**

   .. code-block:: yaml

      algorithm:
        gamma: 0.96          # 折扣因子
        tau: 0.005           # 目标网络软更新系数
        entropy_tuning:
          alpha_type: softplus # 熵系数参数化方式
          initial_alpha: 0.01  # 初始熵系数
          target_entropy: -4
          optim:
            lr: 3.0e-4     # 熵系数学习率
            lr_scheduler: torch_constant
            clip_grad: 10.0
        critic_actor_ratio: 4  # Critic 与 Actor 训练次数比例
        update_epoch: 30     # 每次交互后的训练步数

可视化与结果
----------------------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs

**2. 关键监控指标**

指标含义见 :doc:`训练指标 <../../reference/metrics>`。SAC 相关指标：

- **环境指标**：

  - ``env/episode_len``：该回合实际经历的环境步数（单位：step）
  - ``env/return``：回合总回报
  - ``env/reward``：环境的 step-level 奖励
  - ``env/success_once``：回合中至少成功一次标志（0 或 1）

- **训练指标**：

  - ``train/sac/critic_loss``：Q 函数的损失
  - ``train/critic/grad_norm``：Q 函数的梯度范数

  - ``train/sac/actor_loss``：policy 损失
  - ``train/actor/entropy``：policy 熵
  - ``train/actor/grad_norm``：policy 梯度范数

  - ``train/sac/alpha_loss``：温度参数的损失
  - ``train/sac/alpha``：温度参数的值
  - ``train/alpha/grad_norm``：温度参数的梯度范数

  - ``train/replay_buffer/size``：当前 replay buffer 大小
  - ``train/replay_buffer/max_reward``：replay buffer 中的最大奖励
  - ``train/replay_buffer/min_reward``：replay buffer 中的最小奖励
  - ``train/replay_buffer/mean_reward``：replay buffer 中的平均奖励
  - ``train/replay_buffer/std_reward``：replay buffer 中的奖励标准差
  - ``train/replay_buffer/utilization``：replay buffer 利用率

真机结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下方是 SAC-Flow 在插销任务上的加速演示视频与训练曲线。训练约 30 分钟后，机器人即可稳定完成任务。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-success-rate.png
   :align: center
   :width: 80%

   训练曲线

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-peg-insertion.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>插块插入（Peg Insertion）</em></p>
  </div>
