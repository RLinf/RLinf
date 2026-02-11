软演员评论家 (SAC) 
==================================

1. 引言
---------------

SAC 是目前应用最广泛的强化学习 (RL) 算法之一。  
它包含两个核心部分：  

- Actor （策略模型）: 根据当前状态生成动作。  
- Critic （Q值模型）: 评估当前状态与所选动作的价值。  

SAC 是一种用于连续动作控制的 off-policy 深度强化学习算法。
它基于最大熵强化学习框架，用熵扩展标准的强化学习优化目标，以平衡利用与探索。
SAC 利用熵正则化的 Bellman 方程和自动温度调控，同时学习一个随机策略和多个Q函数。
因其样本效率和稳定性，SAC 在机器人控制领域有广泛应用。

更多细节请参考原始论文  
`SAC <https://arxiv.org/abs/1801.01290>`_.


1. 目标函数
----------------------

设策略为 :math:`\pi`。则 :math:`\pi` 的 Q 值函数定义为: :math:`Q^{\pi}(s, a)`。
在 SAC 里，Q 值函数满足如下松弛的 Bellman 公式： 

.. math::

   Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P, a \sim \pi} \left[
      r(s, a) + \gamma (Q^{\pi}(s', a') + H(\pi(\cdot|s')))
   \right]
   = \mathbb{E}_{s' \sim P, a \sim \pi} \left[
      r(s, a) + \gamma (Q^{\pi}(s', a') - \alpha \log \pi(a'|s'))
   \right].

这里 :math:`\gamma` 是折扣因子, :math:`H` 是策略的熵, and :math:`\alpha` 是温度参数（平衡熵和期望回报）。

因此，第 i 个 Q 值函数的损失函数 :math:`Q_{\phi_{i}}` 如下：

.. math::

   L(\phi_{i}, D) = \mathbb{E}_{(s, a, r, s', d) \sim D} \left[
      \frac{1}{2} \left(
         Q_{\phi_{i}}(s, a) - (r + \gamma (1 - d)(\min_{i} Q_{\overline{\phi_{\text{targ}, i}}}(s', a') - \alpha \log \pi_{\theta}(a'|s')))
      \right)^2
   \right],

这里 :math:`D` 是回放缓冲区, :math:`\overline{\phi_{\text{targ}, i}}` 是目标 Q 函数的参数，:math:`a'` 是从当前策略 :math:`\pi_{\theta}` 采样的动作。

策略 :math:`\pi_{\theta}` 最大化熵和期望回报的期望。因此，策略损失如下：

.. math::

   L(\theta, D) = \mathbb{E}_{s \sim D, a \sim \pi_{\theta}} \left[
      \alpha \log \pi_{\theta}(a|s) - \min_{i} Q_{\phi_i}(s, a)
   \right].

实践上，温度参数 :math:`\alpha` 是可学习的参数。其损失函数如下：

.. math::

   L(\alpha) = - \alpha (H_{\text{targ}} - H(\pi(\cdot, d))), 

这里 :math:`H_{\text{targ}}` 是一个超参数，表示熵的目标大小，通常设成动作维度的相反数。 


1. 配置
-----------------

目前，在我们的框架中，SAC 仅支持具身任务。  
算法配置如下所示：  

.. code-block:: yaml

   algorithm:
      update_epoch: 32
      group_size: 1
      agg_q: min # ["min", "mean"]. 选择如何聚合多个Q函数的值。


      adv_type: embodied_sac
      loss_type: embodied_sac
      loss_agg_func: "token-mean"
      
      bootstrap_type: standard # [standard, always]. 是否累积下一步 Q 值的判断准则。
      tau: 0.01  # 松弛更新目标Q值网络的比例
      target_update_freq: 1  # 目标Q函数的更新频率
      entropy_tuning:
         alpha_type: softplus  # ["softplus","exp","fixed_alpha"]
         initial_alpha: 0.01  # 初始温度值
         target_entropy: -4  # 目标熵（默认 -action_dim）
         optim:
            lr: 3.0e-4  # 温度参数学习率
            lr_scheduler: torch_constant
            clip_grad: 10.0
      
      # 回放缓冲区设置
      replay_buffer:
         enable_cache: True # 启用内存缓存以减少I/O开销
         cache_size: 6000 # 内存缓存的轨迹数量
         sample_window_size: 6000 # 滑动采样窗口大小
         min_buffer_size: 2  # 开始更新策略时缓冲区数据量最小值（以Trajectory为单位）