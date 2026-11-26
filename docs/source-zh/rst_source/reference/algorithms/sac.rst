Soft Actor-Critic（SAC）
========================

1. 引言
-------

Soft Actor-Critic（SAC）是一种面向连续动作控制的 off-policy 最大熵强化学习算法，包含两个核心部分：

- **Actor**\ （policy 模型）：根据当前状态输出随机动作分布。
- **Critic**\ （Q 值模型）：使用双 Q 网络与目标网络评估状态-动作对的价值。

SAC 在期望回报中加入 policy 熵正则项，并支持自动调节温度系数。更多细节参见原始 `SAC 论文 <https://arxiv.org/abs/1801.01290>`__。

2. 目标函数
-----------

设 policy 为 :math:`\pi`，其 Q 函数 :math:`Q^{\pi}(s, a)` 满足软 Bellman 方程：

.. math::

   Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P,\, a' \sim \pi} \left[
      r(s, a) + \gamma \left(Q^{\pi}(s', a') + \alpha H(\pi(\cdot \mid s'))\right)
   \right]
   = \mathbb{E}_{s' \sim P,\, a' \sim \pi} \left[
      r(s, a) + \gamma \left(Q^{\pi}(s', a') - \alpha \log \pi(a' \mid s')\right)
   \right].

其中 :math:`\gamma` 为折扣因子，:math:`H` 为 policy 熵，:math:`\alpha` 为平衡熵与奖励的温度系数。

第 :math:`i` 个 Q 网络 :math:`Q_{\phi_{i}}` 的损失为：

.. math::

   L(\phi_{i}, D) = \mathbb{E}_{(s, a, r, s', d) \sim D} \left[
      \frac{1}{2} \left(
         Q_{\phi_{i}}(s, a) - \left(r + \gamma (1 - d)\left(\min_{j} Q_{\overline{\phi}_{\mathrm{targ}, j}}(s', a') - \alpha \log \pi_{\theta}(a' \mid s')\right)\right)
      \right)^2
   \right],

其中 :math:`D` 为回放缓冲区，:math:`\overline{\phi}_{\mathrm{targ}, j}` 为目标 Q 网络参数，:math:`a' \sim \pi_{\theta}(\cdot \mid s')`。

Policy :math:`\pi_{\theta}` 的优化损失为：

.. math::

   L(\theta, D) = \mathbb{E}_{s \sim D,\, a \sim \pi_{\theta}} \left[
      \alpha \log \pi_{\theta}(a \mid s) - \min_{i} Q_{\phi_i}(s, a)
   \right].

开启自动温度调节时，可学习温度参数 :math:`\alpha` 的损失为：

.. math::

   L(\alpha, D) = \mathbb{E}_{s \sim D} \left[ -\alpha \left(H_{\mathrm{targ}} - H(\pi(\cdot \mid s))\right) \right],

其中 :math:`H_{\mathrm{targ}}` 为目标熵，通常设为动作维度的相反数（:math:`-\mathrm{action\_dim}`）。

3. 配置
-------

在 RLinf 中，SAC 用于具身连续控制任务（仿真与真机）。完整示例参见 :doc:`MLP Policy 示例 <../../examples/embodied/mlp>` 及可运行配置 ``examples/embodiment/config/maniskill_sac_mlp.yaml``、``examples/embodiment/config/maniskill_sac_mlp_async.yaml``：

.. code-block:: yaml

   algorithm:
      update_epoch: 32
      group_size: 1
      agg_q: min                     # ["min", "mean"]
      adv_type: embodied_sac
      loss_type: embodied_sac
      loss_agg_func: "token-mean"
      bootstrap_type: always         # ["standard", "always"]
      gamma: 0.8
      tau: 0.01                      # 目标 Q 网络软更新系数
      target_update_freq: 1
      entropy_tuning:
         alpha_type: softplus        # ["softplus", "exp", "fixed_alpha"]
         initial_alpha: 0.01
         target_entropy: -4          # -action_dim
         optim:
            lr: 3.0e-4
            lr_scheduler: torch_constant
            clip_grad: 10.0
      replay_buffer:
         enable_cache: True
         cache_size: 10000
         sample_window_size: 10000
         min_buffer_size: 0
