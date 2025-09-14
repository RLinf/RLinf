解耦裁剪与动态采样策略优化 (DAPO)
==============================================================

1. 引言
---------------

解耦裁剪与动态采样策略优化 (DAPO) 是一种最近提出的用于大规模 LLM 训练的强化学习算法。  
它在 GRPO 的基础上扩展了四项关键技术：  

1. **Clip-Higher**：非对称裁剪，设置更高的上限。  
2. **动态采样**：重新采样直到一个组中同时包含正确和错误的答案。  
3. **基于 Token 的策略梯度损失**：在 token 级别而不是序列级别计算损失。  
4. **过长奖励塑形**：对过长的回答进行惩罚。  

这些改进使得 DAPO 在长链式思维 (CoT) 推理任务中更加稳定和高效。  

更多细节请参考原始论文：  
`DAPO <https://arxiv.org/abs/2503.14476>`_  


2. 目标函数
----------------------

DAPO 最大化以下目标函数：  

.. math::

   J_{\mathrm{DAPO}}(\theta)
   = \mathbb{E}_{(q,a)\sim\mathcal{D},\,\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}
     \left[
       \frac{1}{\sum_{i=1}^{G} |o_i|}
       \sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
         \min\!\Big(
           r_{i,t}(\theta)\,\hat{A}_{i,t},\;
           \mathrm{clip}\!\big(r_{i,t}(\theta),\, 1-\varepsilon_{\mathrm{low}},\, 1+\varepsilon_{\mathrm{high}}\big)\,\hat{A}_{i,t}
         \Big)
     \right],

其中：  

- :math:`r_{i,t}(\theta)` 是重要性采样比率，  
- :math:`\varepsilon_{\mathrm{low}}, \varepsilon_{\mathrm{high}}` 定义了解耦裁剪范围，  
- :math:`\hat{A}_{i,t}` 是组相对优势。  

**动态采样**  
与其接受任意的回答组，DAPO 要求每个组中必须同时包含正确和错误的回答：  

.. math::

   0 \;<\; \big\lvert \{\, o_i \mid \mathrm{is\_equivalent}(a, o_i) \,\} \big\rvert \;<\; G.

该约束避免了平凡组（全对或全错），从而提升训练效率和稳定性。  

**Clip-Higher**  
DAPO 采用非对称裁剪策略，设置更高的上限  
(:math:`\varepsilon_{\mathrm{high}} > \varepsilon_{\mathrm{low}}`)。  
这能减少对潜在有用的探索性更新过早抑制，鼓励多样性，并缓解熵坍缩。  

**基于 Token 的损失**  
与只在 *序列级别* 计算梯度不同，DAPO 在 *token 级别* 应用策略梯度损失。  
这样能减少因回答长度不同带来的偏差，对长 CoT 强化学习任务尤其重要。  

**过长奖励塑形**  
为了稳定训练，避免过长回答带来的噪声优化，DAPO 引入了长度惩罚：  

- 当回答长度超过 :math:`\texttt{safe\_length}` 时开始惩罚；  
- 惩罚随长度线性增长，直到最大截断值；  
- 长度奖励范围在 :math:`[-1, 0]`。  

这样可以确保虚假的长回答不会主导优化过程。  


3. 配置
-----------------

目前，该框架支持在 LLM 数学任务中使用 DAPO。  

.. code-block:: yaml

  algorithm:
    # 核心 DAPO 设置（建议不要修改）
    adv_type: math_grpo # 推理任务使用 math_grpo, 具身任务使用 embodied_grpo
    loss_type: math_ppo_actor # 推理任务使用 math_ppo_actor, 具身任务使用 embodied_grpo
    loss_agg_func: "token-mean"
    use_valid_token_scale: True # 优势除以有效 token 数 → token 级损失

    # 算法参数（通常需要调优）
    group_size: 16              # 每个提示采样的回答数量
    clip_ratio_high: 0.28       # epsilon_high（非对称裁剪上限）
    clip_ratio_low: 0.20        # epsilon_low（裁剪下限）

    len_reward_penalty: 0.1     # 长度惩罚系数
    safe_length: 16384          # 安全长度阈值（超过该值 → 应用惩罚）
    max_length: 20480           # 硬截断：超过该长度的回答会被丢弃
    max_resample: 5             # 动态采样最大重采样次数

    kl_beta: 0.0                # KL 惩罚系数
    kl_penalty_type: low_var_kl # 可选：low_var_kl, kl, abs, mse

    calculate_entropy: False    # 可选：鼓励探索
    entropy_bonus: 0.0

    normalize_advantages: True
    early_stop_imp_ratio: 5.0   # 丢弃重要性比率极端的 minibatch


4. 注意事项
-----------

- 始终确保动态采样开启；平凡组会降低效果。  
- 对于长 CoT 任务，请使用基于 token 的损失（通过 ``use_valid_token_scale=True`` 启用）。  
- 小心调节 :math:`\varepsilon_{\mathrm{low}}, \varepsilon_{\mathrm{high}}`：  
  较小的间隔 → 更新更保守；较大的间隔 → 更多探索。  
- 长度塑形对于防止 LLM 出现失控的长回答至关重要。  
