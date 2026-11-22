解耦裁剪与动态采样策略优化（DAPO）
====================================

1. 引言
-------

解耦裁剪与动态采样策略优化（DAPO）在 :doc:`GRPO <grpo>` 基础上针对长链式思维（CoT）推理任务做了扩展。原始 `DAPO 论文 <https://arxiv.org/abs/2503.14476>`__ 提出了四项技术：

1. **Clip-Higher**：采用非对称裁剪上限（:math:`\varepsilon_{\mathrm{high}} > \varepsilon_{\mathrm{low}}`），避免过早抑制探索性更新并缓解熵坍缩。
2. **Token 级策略梯度损失**：按有效 token 数加权计算损失，减少变长回复带来的梯度偏差。
3. **动态采样**：过滤组内全对或全错的无效 prompt 分组。
4. **过长奖励塑形**：对过长回复施加长度相关的奖励惩罚。

在 RLinf 中，DAPO 的 **Clip-Higher**\ （``clip_ratio_low`` / ``clip_ratio_high``）与 **token 级损失缩放**\ （``use_valid_token_scale``）已直接内置在 ``grpo`` + ``actor`` 训练路径中。

2. 目标函数
-----------

DAPO 最大化如下按 token 归一化的裁剪代理目标：

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

- :math:`r_{i,t}(\theta)` 是新旧 policy 之间的重要性采样比率；
- :math:`\varepsilon_{\mathrm{low}}, \varepsilon_{\mathrm{high}}` 是解耦后的裁剪下界与上界；
- :math:`\hat{A}_{i,t}` 是组内相对优势。

3. 配置
-------

RLinf 在推理与 VQA 配方中通过 GRPO actor 配置直接启用已实现的 DAPO 参数。可直接运行的完整配置参见 :doc:`Moonlight-16B <../../examples/agentic/math_reasoning/moonlight>`\ （``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml``）、``examples/reasoning/config/math/qwen2.5-7b-grpo-megatron.yaml`` 与 ``examples/reasoning/config/vqa/qwen3-vl-2b-grpo-fsdp-geo3k.yaml``。

.. code-block:: yaml

   algorithm:
     # 核心 GRPO / DAPO 设置
     adv_type: grpo
     loss_type: actor
     loss_agg_func: "token-mean"
     use_valid_token_scale: True   # 按有效 token 数缩放，启用 token 级损失

     # 非对称裁剪（Clip-Higher）
     group_size: 16                # 每个 prompt 采样的回复数
     ratio_clip_eps: 0.2           # clip_ratio_low/high 为 null 时的默认裁剪阈值
     clip_ratio_low: 0.20          # epsilon_low（裁剪下界）
     clip_ratio_high: 0.28         # epsilon_high（非对称裁剪上界）

     # 正则化与稳定性控制
     kl_beta: 0.0                  # KL 惩罚系数
     kl_penalty_type: low_var_kl   # 可选：low_var_kl、kl、abs、mse
     calculate_entropy: False      # 是否记录 policy 熵
     entropy_bonus: 0.0
     normalize_advantages: True
     early_stop_imp_ratio: 5.0     # 跳过重要性比率极端的 minibatch

4. 注意事项
-----------

- 长 CoT 推理任务建议开启 ``use_valid_token_scale: True``，使不同长度回复按有效 token 数贡献梯度。
- 将 ``clip_ratio_high`` 设得高于 ``clip_ratio_low``\ （例如 ``0.28`` 与 ``0.20``）即可启用 Clip-Higher；上下界间隔越小更新越保守，间隔越大越鼓励探索。
- 最大回复长度通过 ``runner.seq_length`` 与 ``algorithm.sampling_params.max_new_tokens`` 控制，详见 :doc:`数学推理 <../../examples/agentic/math_reasoning/reasoning>`。
