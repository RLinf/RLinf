NFT：π\ :sub:`0`\  正负样本对比强化学习
=============================================

本文档介绍如何在 RLinf 框架中使用 **NFT（Negative-aware FineTuning）** 对预训练的 **π₀ flow-matching 策略** 进行强化学习微调。
NFT 是一个无需 critic 和 likelihood 的在线 RL 框架，专为 flow-based 视觉语言动作（VLA）策略设计。它在 flow-matching 去噪轨迹上直接应用步级 DPO 风格的偏好优化，每个优化步骤仅需一次前向传播，无需辅助价值网络。

论文: `π-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs <https://arxiv.org/abs/2603.02083>`_ (Wang et al., 2026)

参考实现: `pi-StepNFT <https://github.com/wangst0181/pi-StepNFT>`_

核心思路:

1. **SDE 探索**: 通过逆时 SDE 公式在 flow-matching 采样中注入随机性，将探索空间扩展到确定性 ODE 轨迹之外。
2. **步级监督**: 不优化整个去噪链，而是针对紧邻的下一个去噪步进行噪声感知回归信号监督，提供更细粒度的学习。
3. **DPO 偏好损失**: 逻辑对比排序损失实现推拉动力学——提升成功轨迹的速度预测，同时抑制失败轨迹。
4. **轻量训练**: 每个优化步骤仅需一次前向传播，不需要 critic 网络或 likelihood 计算，比基于 PPO 的方法显著更高效。

环境
----

**LIBERO 环境**

- **环境**: LIBERO Goal 基准（也支持 Spatial、Object、Long）
- **任务**: 需要目标导向推理的桌面操作任务
- **观测**: 机器人本体感知 + RGB 图像 (224 x 224)
- **动作空间**: 7 维连续动作（3D 位置 + 3D 旋转 + 夹爪）

算法
----

**NFT 流程**

1. **SDE 采样 Rollout**: π₀ 策略使用多步 flow-matching 去噪生成动作轨迹。在每条轨迹的随机去噪步处，记录中间状态 ``x_t``、速度 ``v_t`` 和下一状态 ``x_{t+1}`` 作为 NFT traces。

2. **优势计算**: 从环境奖励（稀疏成功/失败信号）计算原始优势。

3. **DPO 偏好信号**: 优势映射为偏好标签 ``y ∈ [-1, 1]``，正优势表示偏好轨迹，负优势表示非偏好轨迹，由 ``adv_clip_max`` 裁剪。

4. **速度漂移损失**: 基于能量的 NFT 损失计算当前速度预测 ``v_θ`` 与旧速度 ``v_old`` 之间的漂移，结合 SDE 噪声尺度 ``σ_i``。速度变化 ``δv`` 由 ``max_drift`` 裁剪以稳定训练。

5. **单次前向传播**: 每个训练步骤仅需 action expert 的一次前向传播——冻结的 VLM 前缀被缓存并重用。

安装
----

NFT 使用与 π₀ 相同的环境和模型依赖。请参阅 :doc:`pi0` 获取完整的安装指南，包括 Docker 镜像设置、依赖安装和模型下载。

运行脚本
--------

**1. 配置文件**

- **NFT 训练**: ``examples/embodiment/config/libero_goal_nft_openpi.yaml``

**2. 关键参数配置**

**2.1 NFT 模型参数**

.. code:: yaml

   actor:
     model:
       add_value_head: False        # 无需 critic
       openpi:
         is_nft: True               # 启用 NFT trace 采集
         noise_level: 0.2           # SDE 噪声强度

**2.2 算法参数**

.. code:: yaml

   algorithm:
     adv_type: raw                  # 原始优势（无 GAE）
     loss_type: embodied_nft        # NFT 损失类型
     update_epoch: 4                # 每轮 rollout 的训练 epoch 数
     nft_beta: 1.0                  # 速度缩放系数
     adv_clip_max: 1.0              # DPO 信号的优势裁剪上界
     dpo_beta: 1.0                  # DPO 损失温度
     max_drift: 0.5                 # 速度漂移范数的最大裁剪值

**2.3 环境参数**

.. code:: yaml

   env:
     train:
       total_num_envs: 64
       max_episode_steps: 320
     eval:
       total_num_envs: 500

**3. 启动命令**

::

   bash examples/embodiment/run_embodiment.sh libero_goal_nft_openpi

可视化与结果
------------

**1. TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs

**2. 关键监控指标**

- **环境指标**:

  - ``env/episode_len``: 平均 episode 长度
  - ``env/success_once``: 任务成功率（每个 episode 0 或 1）

- **训练指标**:

  - ``train/actor/nft_loss``: NFT 偏好损失
  - ``train/actor/E_pos_mean``: 偏好（成功）轨迹的能量
  - ``train/actor/E_neg_mean``: 非偏好（失败）轨迹的能量
  - ``train/actor/delta_v_norm``: 速度漂移范数（裁剪前）
  - ``train/actor/grad_norm``: 梯度范数

引用
----

.. code-block:: bibtex

   @misc{wang2026pistepnft,
     title={$\pi$-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs},
     author={Wang, Siting and Wang, Xiaofeng and Zhu, Zheng and Pei, Minnan and Cui, Xinyu and Deng, Cheng and Zhao, Jian and Huang, Guan and Zhang, Haifeng and Wang, Jun},
     year={2026},
     eprint={2603.02083},
     archivePrefix={arXiv},
     primaryClass={cs.RO},
     url={https://arxiv.org/abs/2603.02083},
   }
