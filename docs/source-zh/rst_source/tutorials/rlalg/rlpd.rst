结合先验数据的强化学习 (RLPD)
====================================

1. 简介
----------

RLPD (结合先验数据的强化学习) 是一种高样本效率的强化学习算法，旨在利用离线数据集加速在线强化学习。该算法构建于 :doc:`SAC <sac>` 框架之上，RLPD引入了三个极小但关键的设计选择来稳定训练并提高样本效率：

- 对称采样：一种平衡的采样策略，即使用智能体的在线经验回放池和离线演示数据集按50/50的比例构建训练批次。

- 层归一化：在Critic网络中集成层归一化，以防止在从静态数据集学习时出现灾难性的值过估计和外推误差。

- 稳定样本高效更新(针对高UTD或异步更新)：RLPD采用大规模Critic集成 (例如10个网络) 和随机子集 (随机集成蒸馏) 来稳定高更新数据比 (UTD) 或异步更新下的训练。

RLPD 证明了标准的异策略 RL 算法无需复杂的预训练即可有效利用离线数据，已被广泛应用于真实世界强化学习中。

更多详情，请参阅原版 `RLPD <https://arxiv.org/abs/2302.02948>`_ 论文。

2. 目标函数
------------

RLPD继续使用SAC的最大熵目标。策略 :math:`\pi` 被训练为最大化预期回报和策略的熵。与SAC的核心区别在于Critic的更新。RLPD利用了一个包含 :math:`E` 个Critic网络的集成 (例如 :math:`E=10` )。
每个Critic :math:`Q_{\theta_i}` 的损失函数是在由在线数据 :math:`\mathcal{D}_{\text{online}}` 和离线数据 :math:`\mathcal{D}_{\text{offline}}` 等量组成的混合批次 :math:`\mathcal{B}` 上计算的：

.. math::

   L(\theta_i, \mathcal{B}) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{B}} \left[ \left( Q_{\theta_i}(s, a) - y \right)^2 \right]

目标 :math:`y` 使用随机集成蒸馏 (REDQ) 计算，其中选择目标Critic集成的随机子集 :math:`\mathcal{Z}` 来计算悲观估计：

.. math::

   y = r + \gamma \left( \min_{j \in \mathcal{Z}} Q_{\theta'_j}(s', a') - \alpha \log \pi_{\phi}(a'|s') \right)

其中 :math:`a' \sim \pi_{\phi}(\cdot|s')` ， :math:`\theta'` 表示目标网络参数，且 :math:`\mathcal{Z} \subset \{1, \dots, E\}` 。

Actor损失与SAC保持相似，更新策略 :math:`\pi_{\phi}` 以最大化期望Q值 (在集成或子集上平均) 和熵项：

.. math::

   L(\phi, \mathcal{B}) = \mathbb{E}_{s \sim \mathcal{B}, a \sim \pi_{\phi}} \left[ \alpha \log \pi_{\phi}(a|s) - \frac{1}{E} \sum_{i=1}^{E} Q_{\theta_i}(s, a) \right]

3. 具体设计
-----------

RLPD依靠特定的架构来处理由离线数据引起的分布偏移：

- 对称采样：RLPD从在线经验回放池 :math:`\mathcal{D}_{\text{online}}` 和离线数据集 :math:`\mathcal{D}_{\text{offline}}` 中采样不同的小批次，并将它们连接成单个训练批次。标准比例是50%在线和50%离线。这确保了智能体在适应新的在线经验的同时保留离线数据的稳定性。

- 层归一化：为了缓解分布外动作的Q值发散的问题，RLPD在Q网络的MLP第一层之后应用层归一化。这通过权重矩阵的范数隐式地限制了Q值，从而在稀疏奖励或复杂设置中稳定学习。

- 集成Q：为了提高样本效率，我们的 RLPD 执行异步更新。为了防止通常与频繁更新相关的过拟合，RLPD使用Critic集成 (例如 :math:`E=10` 或 :math:`E=20`) 并在目标计算期间对它们进行子集化，类似于REDQ。



4. 配置
----------

RLPD建立在SAC配置之上，增加了离线数据集等内容。

.. code-block:: yaml

   data: # 添加离线演示数据
      type: robot_demo
      channel:
      name: demo_data
      path: "/path/to/demo_data"

   algorithm:
      update_epoch: 30
      group_size: 1
      agg_q: mean


      backup_entropy: False # 移除熵项
      critic_subsample_size: 2 # 目标计算时采样的 Critic 数量
      eval_rollout_epoch: 1

      adv_type: embodied_sac
      loss_type: embodied_sac

      loss_agg_func: "token-mean"
       
      bootstrap_type: standard
      gamma: 0.96
      tau: 0.005

   rollout:
      group_name: "RolloutGroup"
      backend: "huggingface"
      model_dir: ""
      enable_offload: False
      pipeline_stage_num: 1

   model:
      model_path: "/path/to/model"
      precision: ${actor.model.precision}
      num_q_heads: 10 # 集成的 Q 网络数量

