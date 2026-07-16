Cross-Q
===========

1. 简介
----------

CrossQ是 :doc:`SAC <sac>` 算法的一种轻量级进化版本，旨在提高样本效率。它在标准 SAC 框架上引入了三个独特的架构更新：

- 移除目标网络：消除滞后的目标 Q 网络以加速价值学习

- 批重归一化 (Batch Renormalization, BRN)：在Critic和Actor中集成批重归一化以稳定训练，使用当前状态和未来状态的联合前向传播来校正分布统计。

- 更宽的Critic网络：扩展Critic网络层的宽度(例如扩展到2048个单元)以增强表示能力和优化速度。

通过移除目标网络，Cross-Q避免了价值传播中的延迟，而批重归一化的应用确保了以前在RL中难以实现的训练稳定性。
这种架构使CrossQ能够达到或超越REDQ和DroQ等计算昂贵方法的样本效率，同时保持UTD为1，且计算占用显著降低。
更多详情，请参阅原版 `CrossQ <https://arxiv.org/abs/1902.05605>`_ 论文。



2. 目标函数
----------------

CrossQ与SAC使用相同的最大熵RL目标。
设策略为 :math:`\pi` 。则 :math:`\pi` 的Q函数满足相同的贝尔曼方程：

.. math::

   Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P, a \sim \pi} \left[ r(s, a) + \gamma (Q^{\pi}(s', a') - \alpha \log \pi(a'|s')) \right].

这里 :math:`\gamma` 是折扣因子， :math:`\alpha` 是温度参数。

CrossQ在参数化和更新Q函数与SAC不同。具体来说，它完全移除了目标网络。因此，第i个Q函数 :math:`Q_{\phi_{i}}` 的损失是使用当前网络参数 :math:`\phi_{i}` 而非单独的目标网络定义的：

.. math::

   L(\phi_{i}, D) = \mathbb{E}{(s, a, r, s', d) \sim D} \left[ \frac{1}{2} \left( Q_{\phi_{i}}(s, a) - (r + \gamma (1 - d) \cdot \text{sg}(\min_{i} Q_{\phi_{i}}(s', a') - \alpha \log \pi_{\theta}(a'|s'))) \right)^2 \right],

其中 :math:`D` 是经验回放池， :math:`\text{sg}(\cdot)` 表示停止梯度算子 (防止梯度通过自举目标回传)， :math:`a'` 从当前策略 :math:`\pi_{\theta}` 中采样。

Actor损失函数与SAC保持一致。Actor :math:`\pi_{\theta}` 被训练为最大化期望Q值和熵：

.. math::

   L(\theta, D) = \mathbb{E}{s \sim D, a \sim \pi{\theta}} \left[ \alpha \log \pi_{\theta}(a|s) - \min_{i} Q_{\phi_i}(s, a) \right].

同样，温度系数 :math:`\alpha` 通过相同的损失函数进行更新：

.. math::

   L(\alpha, D) = - \alpha (H_{\text{targ}} - H(\pi(\cdot, d))).

3. 特殊设计
------------

CrossQ 引入了三个关键的设计：

- 移除目标网络：与SAC依赖缓慢更新的目标网络 :math:`\phi_{\text{targ}}` 来稳定学习不同，CrossQ使用当前网络 :math:`\phi` 来计算TD目标。梯度在自举项上被停止以防止发散。

- 批重归一化 (BRN)：为了在没有目标网络的情况下稳定训练，CrossQ在Critic和Actor网络中集成了批重归一化。为了解决训练样本 :math:`(s, a)` 和自举样本 :math:`(s', a')` 之间的分布不匹配问题，CrossQ 通过连接这些批次来执行联合前向传播：

.. math::

  \left[ \begin{matrix} q \ q' \end{matrix} \right] = Q_{\phi} \left( \left[ \begin{matrix} s \ s' \end{matrix} \right], \left[ \begin{matrix} a \ a' \end{matrix} \right] \right),

这确保了BRN的统计量是在当前数据和回放数据的混合上计算的 。

- 更宽的Critic网络：CrossQ扩展了Critic网络层的宽度 (例如从256个隐藏单元增加到2048个)。增加的宽度结合BRN，加速了优化并显著提高了相较于标准SAC架构的样本效率。

4. 配置
----------

CrossQ与SAC使用几乎相同的配置。参数 `q_head_type` 可用于在CrossQ和标准SAC架构之间切换。

.. code-block:: yaml

   algorithm:
      update_epoch: 32
      group_size: 1
      agg_q: min # ["min", "mean"] # 聚合多个 Q 值的选项
      adv_type: embodied_sac
      loss_type: embodied_sac
      loss_agg_func: "token-mean"
      q_head_type: "crossq" # ["crossq", "default"] 选择 CrossQ 或标准 SAC Q 头
        
      bootstrap_type: standard
      gamma: 0.8
      tau: 0.01
      target_update_freq: 1
      entropy_tuning:
         alpha_type: softplus  # ["softplus","exp","fixed_alpha"]
         initial_alpha: 0.01
         target_entropy: -4
         optim:
            lr: 3.0e-4
            lr_scheduler: torch_constant
            clip_grad: 10.0
        
      # 回放缓冲区设置
      replay_buffer:
         enable_cache: True # 启用内存缓存以减少I/O开销
         cache_size: 6000 # 内存缓存的轨迹数量
         sample_window_size: 6000 # 滑动采样窗口大小
         min_buffer_size: 2  # 开始更新策略时缓冲区数据量最小值（以Trajectory为单位）