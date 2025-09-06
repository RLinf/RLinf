REINFORCE++ 
=====================

1. 引言
---------------

**REINFORCE++** 是一种轻量级的强化学习后训练方法。  
它基于经典的 REINFORCE 算法，并借鉴了 PPO 的两个关键思想 —— *逐 token 的 KL 惩罚* 和 *优势归一化* ——  
同时 **避免了额外的策略裁剪损失**，因为裁剪可能会导致梯度估计出现偏差。  

核心设计选择包括：  

* **单响应训练** (`group_size = 1`)：每个 prompt 只采样一个答案。  
* **逐 token KL 惩罚** （默认 *k₂*）：KL 不作为额外损失项加入，而是从标量奖励中扣除。  
* **全局优势归一化**：在整个 batch 上进行归一化。  
* **REINFORCE++ 基线**：当 `group_size > 1` 时，使用每个 prompt 内的平均奖励作为基线，*然后* 再进行全局归一化。  

2. 目标函数
----------------------

设 :math:`q` 为 prompt，:math:`o_{1:T}` 为生成的 token 序列，:math:`\pi_{\theta}^{\text{RL}}` 为当前策略。  
在时间步 :math:`t` 的逐 token 优势为：  

.. math::

   A_{q,0,t} \;=\; r(o_{1:T}, q)\;-\;\beta
   \sum_{i=t}^{T} \operatorname{KL}(i) \tag{8}

其中：  

.. math::

   \operatorname{KL}(t) \;=\;
   \log\!\left(
     \frac{\pi^{\text{RL}}_{\theta_{\text{old}}}(o_t \mid q,\,o_{<t})}
          {\pi^{\text{SFT}}(o_t \mid q,\,o_{<t})}
   \right) \tag{9}

:math:`\pi^{\text{SFT}}` 是冻结的监督微调 (SFT) 参考策略，  
:math:`\beta` 控制 KL 惩罚的强度。  

为了稳定训练，优势在 **全局 batch 范围内归一化**：  

.. math::

   A^{\text{norm}}_{q,o_t} \;=\;
   \frac{
     A_{q,o_t} \;-\;
     \operatorname{mean}\ \bigl(A_{q,o_t}\,\mid\,A_{q,o_t}\in\mathcal{D}_{\text{batch}}\bigr)
   }{
     \operatorname{std}\ \bigl(A_{q,o_t}\,\mid\,A_{q,o_t}\in\mathcal{D}_{\text{batch}}\bigr)
   } \tag{10}

然后使用标准的 REINFORCE 梯度来更新策略：  
:math:`\nabla_{\theta}\,\log\pi_{\theta}(o_t\!\mid\!q,o_{<t})\,A^{\text{norm}}_{q,o_t}` 。  

3. 配置
-----------------

REINFORCE++

.. code-block:: yaml

   algorithm:
     adv_type:      "reinpp"       # 使用 REINFORCE++
     group_size:    1              # 每个 prompt 一个响应
     kl_beta:       0.0001
     normalize_advantages: False   # 全局归一化已启用

   data:
     rollout_batch_size: 8192

REINFORCE++ 基线

.. code-block:: yaml

   algorithm:
     adv_type:      "reinpp_baseline"
     group_size:    16             # 每个 prompt 多个响应
     kl_beta:       0.0001

   data:
     rollout_batch_size: 512

4. 注意事项
-----------

- REINFORCE++ 使用的是所谓的 :math:`k_1` KL。  
- GRPO 算法使用的是 :math:`k_3` 形式，它混合了 on-policy 和参考策略的概率，但该估计是有偏的。  
- 使用 :math:`k_1` KL 可以保持更新无偏，同时仍能抑制过大的策略偏移。  
