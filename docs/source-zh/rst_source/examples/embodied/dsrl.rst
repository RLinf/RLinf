DSRL：基于扩散模型的潜在空间强化学习
======================================================

本文档介绍如何在 RLinf 框架中使用 **DSRL（Diffusion Steering via Reinforcement Learning）** 对预训练的 **Pi0 扩散策略** 进行强化学习微调。
DSRL 通过在潜在噪声空间中训练一个轻量级 SAC 智能体来引导冻结的 Pi0 策略，仅需约 500K 可训练参数即可实现 RL 微调。

相关论文： `Steering Your Diffusion Policy with Latent Space Reinforcement Learning <https://arxiv.org/abs/2506.15799>`_ （CoRL 2025, Wagenmaker et al.）

参考实现： `dsrl_pi0 <https://github.com/nakamotoo/dsrl_pi0>`_

核心思路：

1. **轻量级 SAC 智能体**：一个小型 SAC 智能体（约 500K 参数），配备紧凑的 CNN/MLP 编码器，处理观测并在潜在空间中生成噪声。
2. **噪声注入**：生成的噪声作为初始噪声输入到 Pi0 的扩散去噪器中，替代随机采样。
3. **冻结 VLM 主干**：预训练的 Pi0 VLM 和扩散专家模块保持冻结，保留泛化能力。
4. **噪声空间中的 SAC 训练**：SAC 智能体在噪声空间上使用环境奖励进行训练，采用 10 个 Q-head 集成的 Critic 实现稳定的价值估计。

环境
----

**LIBERO Spatial 环境**

- **环境**：LIBERO Spatial 基准
- **任务**：桌面操作任务，涉及空间推理
- **观测**：机器人本体感知（8维）+ RGB 图像
- **动作空间**：Pi0 扩散去噪器生成的连续动作（由 SAC 噪声引导）

算法
-----------------------------------------

**DSRL 流程**

1. **观测编码**：轻量级 CNN（64×64 → 64维）和状态编码器（8维 → 64维）处理观测数据。

2. **噪声生成**： ``GaussianPolicy`` （SquashedNormal 分布）为每个动作步生成 32 维噪声动作。

3. **扩散去噪**：噪声作为初始噪声注入 Pi0 的 ``sample_actions()``，冻结的扩散去噪器将噪声转换为真实动作。

4. **SAC 训练**：标准 SAC 配合自动熵调节训练噪声生成器：

   - **Actor**： ``GaussianPolicy`` ，3 层 MLP（128维隐藏层）
   - **Critic**： ``CompactMultiQHead`` — 10 个 Q 网络集成（共约 500K 参数）
   - **目标网络**：Float32 EMA 影子缓冲区，解决 bfloat16 精度问题

依赖安装
--------

DSRL 使用与 Pi0 相同的环境和模型依赖。请参考 :doc:`pi0` 获取完整的安装指南，包括 Docker 镜像配置、依赖安装和模型下载。

运行脚本
--------

**1. 配置文件**

- **DSRL 训练**： ``examples/embodiment/config/libero_spatial_dsrl_openpi.yaml``

**2. 关键参数配置**

**2.1 DSRL 模型参数**

.. code:: yaml

   actor:
     model:
       openpi:
         use_dsrl: True              # 启用 DSRL 模式
         dsrl_state_dim: 8           # 机器人本体感知维度
         dsrl_action_noise_dim: 32   # 每步噪声动作维度
         dsrl_num_q_heads: 10        # 集成 Critic 中的 Q-head 数量
         dsrl_image_latent_dim: 64   # 图像编码器输出维度
         dsrl_state_latent_dim: 64   # 状态编码器输出维度
         dsrl_hidden_dims: [128, 128, 128]  # MLP 隐藏层维度

**2.2 算法参数**

.. code:: yaml

   algorithm:
     adv_type: embodied_sac
     loss_type: embodied_sac
     gamma: 0.999             # 折扣因子
     tau: 0.005               # 目标网络软更新系数
     update_epoch: 200        # 每次交互后的训练步数
     train_actor_steps: 10    # Actor 训练延迟步数（先训练 Critic）
     entropy_tuning:
       alpha_type: softplus
       initial_alpha: 1.0
       target_entropy: -16
       optim:
         lr: 3.0e-4

**2.3 环境参数**

.. code:: yaml

   env:
     train:
       total_num_envs: 16
       use_step_penalty: True  # 使用 -1/0 奖励风格（步惩罚 + 终止奖励）
       max_episode_steps: 240
     eval:
       total_num_envs: 500
       use_step_penalty: True

**3. 启动命令**

::

   bash examples/embodiment/run_embodiment.sh libero_spatial_dsrl_openpi

可视化与结果
------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs

**2. 关键监控指标**

- **环境指标**：

  - ``env/episode_len``：该回合实际经历的环境步数
  - ``env/return``：回合总回报
  - ``env/reward``：环境的 step-level 奖励
  - ``env/success_once``：回合中至少成功一次标志（0 或 1）

- **训练指标**：

  - ``train/sac/critic_loss``：Q 函数集成的损失
  - ``train/critic/grad_norm``：Q 函数的梯度范数

  - ``train/sac/actor_loss``：策略损失（噪声空间中的 GaussianPolicy）
  - ``train/actor/entropy``：策略熵
  - ``train/actor/grad_norm``：策略的梯度范数

  - ``train/sac/alpha_loss``：温度参数的损失
  - ``train/sac/alpha``：温度参数的值

  - ``train/replay_buffer/size``：当前重放缓冲区的大小
  - ``train/replay_buffer/utilization``：重放缓冲区的利用率
