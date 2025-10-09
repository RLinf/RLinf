组相对策略优化 (GRPO)
=========================================

1. 引言
---------------

组相对策略优化 (GRPO) 是一种针对提示级相对比较设计的 PPO 变体。  

- 在 PPO 中，需要一个 Critic 模型来评估采样到的行为（即回答序列）是好还是坏。  
- 在 GRPO 中，对于同一个提示，会采样多个回答组成一个组。  
  组内回答的相对表现被用来计算每个回答的优势，  
  使得策略更新专注于相同上下文下的相对表现。  

由于 GRPO 不再依赖单独的 Critic 模型，它显著降低了计算资源需求。  
这使得它特别适合大规模 LLM 训练，因为维护一个 Critic 模型代价过高。  

更多细节请参考原始论文：  
`DeepSeek-R1 <https://arxiv.org/abs/2501.12948>`_。

2. 目标函数
----------------------

对于一个问答对 :math:`(q,a)`，行为策略  
:math:`\pi_{\theta_{\mathrm{old}}}` 会采样一个包含 :math:`G` 个回答的组  
:math:`\{o_i\}_{i=1}^{G}`，并得到对应的序列奖励 :math:`\{R_i\}_{i=1}^{G}`。  

序列 :math:`i` 中每个 token 的组相对优势定义为：  

.. math::

   \hat{A}_{i,t} = \frac{R_i - \operatorname{mean}(\{R_j\}_{j=1}^{G})}
                        {\operatorname{std}(\{R_j\}_{j=1}^{G})}.

该优势衡量一个回答相对于组均值的好坏程度，  
并通过组内奖励的方差进行归一化。  

与 PPO 类似，GRPO 采用带裁剪的代理目标函数，并可选地加上 KL 惩罚项。  
在某些实现和任务中，KL 项可能会被省略。  

.. math::

   J_{\mathrm{GRPO}}(\theta)
   = \mathbb{E}_{(q,a)\sim\mathcal{D},\,\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}
     \!\left[
       \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
         \min\!\Big(
           r_{i,t}(\theta)\,\hat{A}_{i,t},\;
           \mathrm{clip}\!\big(r_{i,t}(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big)\,\hat{A}_{i,t}
         \Big)
     \right]
     \;-\; \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\big),

其中  

.. math::

   r_{i,t}(\theta) =
   \frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}
        {\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q, o_{i,<t})}.

- :math:`r_{i,t}(\theta)` 是重要性采样比率。  
- :math:`\varepsilon` 是裁剪阈值，用来防止更新过大。  
- :math:`\beta` 控制相对于参考策略 :math:`\pi_{\mathrm{ref}}` 的 KL 惩罚强度。  


3. 配置
-----------------

我们的框架支持在 LLM 数学任务和具身任务中使用 GRPO。  
下面给出一个 LLM 数学任务的示例配置：  

.. code-block:: yaml

  algorithm:
    # 核心 GRPO 设置（建议不要修改）
    adv_type: math_grpo # 推理任务使用 math_grpo, 具身任务使用 embodied_grpo
    loss_type: math_ppo_actor # 推理任务使用 math_ppo_actor, 具身任务使用 embodied_grpo
    loss_agg_func: "token-mean"

    # 算法参数（通常需要调优）
    group_size: 16              # 每个提示采样的回答数量

    kl_beta: 0.0                # KL 惩罚系数
    kl_penalty_type: low_var_kl # 可选：low_var_kl, kl, abs, mse
    ratio_clip_eps: 0.2         # 重要性比率的裁剪范围

    calculate_entropy: False    # 可选：鼓励探索
    entropy_bonus: 0.0

    normalize_advantages: True
    early_stop_imp_ratio: 5.0   # 丢弃重要性比率极端的 minibatch，提升稳定性
    use_valid_token_scale: False # 标准 GRPO 实现。
                                 # 若为 True，则优势会除以有效 token 数（DAPO 技巧）。


4. 注意事项
-----------

- 始终为每个提示批量生成多个完成（≥ 2 个回答）。  
- 使用较高的采样温度（0.7–1.0）以鼓励生成多样化候选回答。  
- 奖励必须在同一提示组内可比，因为优势是相对计算的。  
