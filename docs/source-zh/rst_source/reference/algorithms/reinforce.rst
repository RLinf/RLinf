REINFORCE++
===========

1. 引言
-------

**REINFORCE++** 是一种面向 LLM 后训练的无 critic 强化学习方法（`论文 <https://arxiv.org/abs/2501.03262>`__）。它在经典 REINFORCE 的基础上引入逐 token KL 惩罚与全局优势归一化，无需维护独立的价值模型。

核心设计包括：

- **单响应或分组采样**：标准 REINFORCE++ 可使用 ``group_size: 1``；开启 ``use_reinpp_baseline: True`` 并设置 ``group_size > 1`` 时，会在全局归一化前先减去每个 prompt 组内的平均奖励作为基线。
- **奖励内的逐 token KL 惩罚**：当 ``reinpp_kl_beta > 0`` 时，RLinf 按 ``kl_penalty_type`` 指定的估计器（``k1`` / ``kl``、``k2`` / ``mse``、``abs`` 或 ``k3`` / ``low_var_kl``）计算逐 token KL，并从奖励矩阵中扣除 ``reinpp_kl_beta * kld``。
- **全局优势归一化**：对整个 batch 中的有效 token 做均值和标准差归一化。

2. 目标函数
-----------

设 :math:`q` 为 prompt，:math:`o_{1:T}` 为生成的 token 序列，:math:`\pi_{\theta}^{\mathrm{RL}}` 为当前 policy。时间步 :math:`t` 的逐 token 优势由终局奖励减去从 :math:`t` 到 :math:`T` 的累计 KL 惩罚得到：

.. math::

   A_{q,o_t} \;=\; r(o_{1:T}, q) \;-\; \beta \sum_{i=t}^{T} \operatorname{KL}(i),

其中 :math:`\beta` 对应 ``reinpp_kl_beta``，:math:`\operatorname{KL}(t)` 按 ``kl_penalty_type`` 相对冻结参考 policy :math:`\pi^{\mathrm{ref}}` 计算。无偏的 ``k1`` 估计器形式为：

.. math::

   \operatorname{KL}_{k_1}(t) \;=\;
   \log\!\left(
     \frac{\pi^{\mathrm{RL}}_{\theta_{\mathrm{old}}}(o_t \mid q,\,o_{<t})}
          {\pi^{\mathrm{ref}}(o_t \mid q,\,o_{<t})}
   \right),

仓库中数学推理示例配置使用的 ``k2`` 估计器形式为：

.. math::

   \operatorname{KL}_{k_2}(t) \;=\; \frac{1}{2}\left(
     \log \pi^{\mathrm{RL}}_{\theta_{\mathrm{old}}}(o_t \mid q,\,o_{<t}) -
     \log \pi^{\mathrm{ref}}(o_t \mid q,\,o_{<t})
   \right)^2.

为稳定训练，RLinf 在整个 batch 的有效 token 上归一化优势：

.. math::

   A^{\mathrm{norm}}_{q,o_t} \;=\;
   \frac{
     A_{q,o_t} \;-\;
     \operatorname{mean}\!\bigl(A_{q,o_t} \mid A_{q,o_t}\in\mathcal{D}_{\mathrm{batch}}\bigr)
   }{
     \operatorname{std}\!\bigl(A_{q,o_t} \mid A_{q,o_t}\in\mathcal{D}_{\mathrm{batch}}\bigr)
   }.

3. 配置
-------

完整运行指南参见 :doc:`数学推理 <../../examples/agentic/math_reasoning/reasoning>` 与可运行配置 ``examples/reasoning/config/math/qwen2.5-1.5b-reinpp-megatron.yaml``。

REINFORCE++：

.. code-block:: yaml

   algorithm:
     adv_type: "reinpp"            # 启用 REINFORCE++ 优势计算
     loss_type: actor
     loss_agg_func: "token-mean"
     reinpp_kl_beta: 0.001         # 优势计算内的逐 token KL 惩罚系数
     kl_penalty_type: k2           # 可选：k1 (kl)、k2 (mse)、abs、k3 (low_var_kl)
     use_reinpp_baseline: False
     group_size: 16
     kl_beta: 0.0                  # actor 损失中的独立 KL 惩罚（REINFORCE++ 通常设为 0.0）

   data:
     rollout_batch_size: 512

REINFORCE++ 基线模式：

.. code-block:: yaml

   algorithm:
     adv_type: "reinpp"
     loss_type: actor
     reinpp_kl_beta: 0.001
     kl_penalty_type: k1
     use_reinpp_baseline: True     # 全局归一化前先减去组内平均奖励
     group_size: 16
     kl_beta: 0.0

   data:
     rollout_batch_size: 512

4. 注意事项
-----------

- ``kl_penalty_type`` 决定 ``rlinf/algorithms/advantages.py`` 中 ``compute_reinpp_advantages`` 调用 ``rlinf/algorithms/utils.py`` 里 ``kl_penalty`` 时采用的估计器：``k1``\ （``kl``）为无偏估计；``k2``\ （``mse``）惩罚对数概率偏移的平方，也是 ``examples/reasoning/config/math/qwen2.5-1.5b-reinpp-megatron.yaml`` 的默认选项；``k3``\ （``low_var_kl``）为 Schulman 低方差估计器。
- 使用 ``algorithm.reinpp_kl_beta`` 在奖励内施加 KL 惩罚时，应将 ``algorithm.kl_beta`` 设为 ``0.0``，避免重复计算 KL 惩罚。
