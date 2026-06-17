流匹配策略SAC强化学习训练
===============================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-overview.png
   :align: center
   :width: 80%

   SAC-Flow 总览。

本示例展示 RLinf 框架使用 **SAC (Soft Actor-Critic)** 算法训练 **Flow Matching** 策略网络的完整流程。
该算法结合了最大熵强化学习（SAC）与生成式流匹配模型（Flow Matching）的优势，支持在仿真环境（ManiSkill3）和真机环境（Franka）中进行训练。

相关论文：`SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling <https://arxiv.org/abs/2509.25756>`_

概览
----------------------------------------

用 SAC 训练流匹配（Flow Matching）策略——可在 ManiSkill 仿真中或真实 Franka（插销）上进行。

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

    -   **对数似然计算**：在每步采样中填加高斯噪声 + 配套漂移修正，保证末端动作分布不变，同时把路径密度分解为单步高斯似然的连乘，从而得到可微的 :math:`\log p_{\theta}(A|s)`。

3. **RLPD (Reinforcement Learning with Prior Data)**

   - SAC 的一种变体，结合离线数据和在线数据进行训练。

   - 为加速在真实世界的训练，SAC-Flow 也可结合 RLPD 使用预采集的离线数据作为演示缓冲区。

安装
----------------------------------------

对于在仿真环境运行，请参考 :doc:`../../start/installation` 进行安装。

对于在真机上运行，请参考 :doc:`franka` 进行安装和硬件配置。

运行
----------------------------------------

**1. 配置文件**

RLinf 提供了针对仿真和真机环境的默认配置文件：

-   **仿真 (ManiSkill)**: ``examples/embodiment/config/maniskill_sac_flow_state.yaml``
-   **真机 (Franka)**: ``examples/embodiment/config/realworld_sac_flow_image.yaml``

**2. 关键参数配置**

**2.1 模型参数 (Model)**

.. code:: yaml

   actor:
     model:
       model_type: "flow_policy"
       # 输入类型: 'state' (仿真) 或 'mixed' (真机, 图像+状态)
       input_type: "state"

       # Flow Matching 相关参数
       denoising_steps: 4  # 生成动作去噪步数
       d_model: 256        # Transformer 维度
       n_head: 4           # 注意力头数
       n_layers: 2         # 层数
       use_batch_norm: False  # 是否使用批归一化
       batch_norm_momentum: 0.99  # 批归一化动量
       flow_actor_type: "JaxFlowTActor"  # JAX风格的 "JaxFlowTActor" 或 torch风格的"FlowTActor"。"JaxFlowTActor" 支持以下噪声标准差设置：
       noise_std_head: False  # 是否使用单独的头来预测噪声标准差，否则使用固定标准差
       # 推理（rollout）时使用的噪声标准差可以比训练时更小，以平衡探索与利用
       log_std_min_train: -5  # 训练时最小对数标准差（如果使用 noise_std_head）
       log_std_max_train: 2   # 训练时最大对数标准差（如果使用 noise_std_head）
       log_std_min_rollout: -20  # 推理时最小对数标准差（如果使用 noise_std_head）
       log_std_max_rollout: 0    # 推理时最大对数标准差（如果使用 noise_std_head）
       noise_std_train: 0.3  # 训练时固定噪声标准差（如果不使用 noise_std_head）
       noise_std_rollout: 0.02  # 推理时固定噪声标准差（如果不使用 noise_std_head）


**2.2 算法参数 (Algorithm)**

.. code:: yaml

   algorithm:
      # SAC 超参数
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

      # 训练与交互频率
      update_epoch: 30     # 每次交互后的训练步数

**2.3 集群与硬件配置 (Cluster)**

对于真机训练，使用多节点配置，将 Actor/Policy 部署在 GPU 服务器上，将 Env/Robot 部署在控制机（NUC/工控机）上。具体配置可参考 :doc:`franka` 。


**3. 启动命令**

**仿真训练 (ManiSkill)**

在单机上启动仿真训练：

::

   bash examples/embodiment/run_embodiment.sh maniskill_sac_flow_state

**真机训练 (Franka)**

在分布式环境下启动真机训练（需在主节点运行，并配置好集群）：

::

   bash examples/embodiment/run_realworld_async.sh realworld_sac_flow_image

可视化与结果
----------------------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs

**2. 关键监控指标**

指标含义见 :doc:`训练指标 <../../reference/metrics>`。SAC 相关指标：

- **环境指标**:

  - ``env/episode_len``：该回合实际经历的环境步数（单位：step）
  - ``env/return``：回合总回报
  - ``env/reward``：环境的 step-level 奖励
  - ``env/success_once``：回合中至少成功一次标志（0或1）

- **Training Metrics**:

  - ``train/sac/critic_loss``: Q 函数的损失
  - ``train/critic/grad_norm``: Q 函数的梯度范数

  - ``train/sac/actor_loss``: 策略损失
  - ``train/actor/entropy``: 策略熵
  - ``train/actor/grad_norm``: 策略的梯度范数

  - ``train/sac/alpha_loss``: 温度参数的损失
  - ``train/sac/alpha``: 温度参数的值
  - ``train/alpha/grad_norm``: 温度参数的梯度范数

  - ``train/replay_buffer/size``: 当前重放缓冲区的大小
  - ``train/replay_buffer/max_reward``: 重放缓冲区中存储的最大奖励
  - ``train/replay_buffer/min_reward``: 重放缓冲区中存储的最小奖励
  - ``train/replay_buffer/mean_reward``: 重放缓冲区中存储的平均奖励
  - ``train/replay_buffer/std_reward``: 重放缓冲区中存储的奖励标准差
  - ``train/replay_buffer/utilization``: 重放缓冲区的利用率

真实世界结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
以下提供了SAC-Flow算法插块插入任务的演示视频（经加速处理）和训练曲线。在 30分钟 的训练时间内，机器人能够学习到一套能够持续成功完成任务的策略。

.. raw:: html

  <div style="flex: 0.8; text-align: center;">
      <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-success-rate.png" style="width: 100%;"/>
      <p><em>训练曲线</em></p>
    </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-peg-insertion.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>插块插入（Peg Insertion）</em></p>
  </div>
