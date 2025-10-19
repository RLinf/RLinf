近端策略优化 (PPO)
==================================

1. 引言
---------------

近端策略优化 (PPO) 是目前应用最广泛的强化学习 (RL) 算法之一。  
它包含两个核心部分：  

- Actor（策略模型）：根据当前状态生成动作。  
- Critic（价值模型）：评估所选动作的价值。  

PPO 是一种稳定的策略梯度方法，它在原始策略梯度 (Policy Gradient) 的基础上改进。  
它通过限制策略更新的步长，提高了训练的稳定性和效率。  
此外，PPO 使用广义优势估计 (GAE) 来降低价值估计的方差。  

在 RLHF (Reinforcement Learning from Human Feedback) 的早期阶段，PPO 得到了广泛应用。  
然而，由于需要一个大型 Critic 模型（通常是另一种 LLM），它会带来高昂的计算成本和训练开销。  

更多细节请参考原始论文  
`PPO <https://arxiv.org/abs/1707.06347>`_ 以及它在 RLHF 中的应用  
`InstructGPT <https://arxiv.org/abs/2203.02155>`_。


2. 目标函数
----------------------

设策略为 :math:`\pi_\theta`。  
对于包含问答对 :math:`(q,a)` 的数据集 :math:`\mathcal{D}`，  
PPO 的目标函数定义如下：  

.. math::

   J_{\mathrm{PPO}}(\theta)
   = \mathbb{E}_{(q,a)\sim\mathcal{D},\, o_{\le t}\sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}
   \Big[
     \min\!\Big(
       r_t(\theta)\,\hat{A}_t,\;
       \mathrm{clip}\,\big(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big)\,\hat{A}_t
     \Big)
   \Big],

其中：  

- :math:`r_t(\theta) = \dfrac{\pi_\theta(o_t \mid q, o_{<t})}
  {\pi_{\theta_{\mathrm{old}}}(o_t \mid q, o_{<t})}`  
  表示重要性采样比率，用来比较新旧策略。  

- :math:`\varepsilon` 是裁剪范围，一个超参数，用于防止更新过大。  

- :math:`\hat{A}_t` 是时间步 :math:`t` 的优势估计。  

使用广义优势估计 (GAE) 时，优势的计算公式为：  

.. math::

   \hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
   = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l},
   \qquad
   \delta_l = R_l + \gamma V(s_{l+1}) - V(s_l),
   \quad 0 \le \gamma, \lambda \le 1.

其中：  

- :math:`\gamma` （折扣因子）和 :math:`\lambda` （GAE 参数）是超参数。  
- :math:`V(s)` 是 Critic 模型给出的价值估计。  


3. 配置
-----------------

目前，在我们的框架中，PPO 仅支持具身任务。  
算法配置如下所示：  

.. code-block:: yaml

   algorithm:

      # 核心 PPO 设置（建议不要修改）
      normalize_advantages: True
      group_size: 1
      adv_type: embodied_gae
      loss_type: embodied_ppo
      loss_agg_func: "token-mean"

      # 算法参数（通常需要调优）

      rollout_micro_batch_size: 256
      logprob_forward_micro_batch_size: 16  # 较大的 batch_size 可以提高稳定性。
                                            # 请根据算力和模型大小调整。

      entropy_bonus: 0          # 可选：鼓励探索
      clip_ratio_high: 0.2      # PPO 裁剪参数 (epsilon)
      clip_ratio_low: 0.2       # 应与 clip_ratio_high 保持一致
      value_clip: 0.2           # 稳定价值函数更新

      gamma: 0.99               # GAE 的折扣因子
      gae_lambda: 0.95          # GAE 的 Lambda 参数

      huber_delta: 10.0         # 价值训练中 Huber 损失的 Delta 参数


4. 注意事项
-----------

- 使用奖励归一化来稳定训练。  
- 监控 KL 散度以检测策略是否更新过度。  
- 对于大型 LLM，增加 batch size 可以减少方差。  
