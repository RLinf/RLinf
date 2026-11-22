CrossQ
======

1. 简介
-------

CrossQ 在 :doc:`Soft Actor-Critic（SAC） <sac>` 基础上移除目标网络，以更低的计算开销提升样本效率。其核心改动包括三点：

- **移除目标网络**：省去滞后的目标 Q 网络，加快价值传播。
- **批重归一化（Batch Renormalization，BRN）**：在 critic 与 actor 中引入批重归一化，并将当前状态与下一状态拼接做联合前向传播，校正批统计量分布。
- **更宽的 Critic 网络**：扩大 critic 隐藏层宽度（例如 2048），提升表征能力与收敛速度。

更多细节参见原始 `CrossQ 论文 <https://arxiv.org/abs/1902.05605>`__。

2. 目标函数
-----------

CrossQ 与 SAC 共享最大熵 RL 目标。设 policy 为 :math:`\pi`，其软 Q 函数满足软 Bellman 方程：

.. math::

   Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P,\, a' \sim \pi} \left[ r(s, a) + \gamma \left( Q^{\pi}(s', a') - \alpha \log \pi(a' \mid s') \right) \right].

其中 :math:`\gamma` 为折扣因子，:math:`\alpha` 为熵温度系数。由于移除目标网络，第 :math:`i` 个 critic :math:`Q_{\phi_{i}}` 的损失直接使用当前参数 :math:`\phi_{i}` 并在自举目标上停止梯度：

.. math::

   L(\phi_{i}, D) = \mathbb{E}_{(s, a, r, s', d) \sim D} \left[ \frac{1}{2} \left( Q_{\phi_{i}}(s, a) - \left(r + \gamma (1 - d) \cdot \operatorname{sg}\!\left(\min_{j} Q_{\phi_{j}}(s', a') - \alpha \log \pi_{\theta}(a' \mid s')\right)\right) \right)^2 \right],

其中 :math:`D` 为回放缓冲区，:math:`\operatorname{sg}(\cdot)` 为停止梯度算子，:math:`a' \sim \pi_{\theta}(\cdot \mid s')`。

Actor :math:`\pi_{\theta}` 的损失与 SAC 相同：

.. math::

   L(\theta, D) = \mathbb{E}_{s \sim D,\, a \sim \pi_{\theta}} \left[ \alpha \log \pi_{\theta}(a \mid s) - \min_{i} Q_{\phi_i}(s, a) \right].

温度系数 :math:`\alpha` 的更新损失为：

.. math::

   L(\alpha, D) = \mathbb{E}_{s \sim D} \left[ -\alpha \left( H_{\mathrm{targ}} - H(\pi(\cdot \mid s)) \right) \right].

3. 关键设计
-----------

- **BRN 联合前向传播**：为消除当前样本 :math:`(s, a)` 与自举样本 :math:`(s', a')` 的分布差异，CrossQ 将两批输入拼接后送入同一轮 critic 前向计算：

  .. math::

     \begin{bmatrix} q \\ q' \end{bmatrix} = Q_{\phi} \left( \begin{bmatrix} s \\ s' \end{bmatrix}, \begin{bmatrix} a \\ a' \end{bmatrix} \right),

  使批重归一化统计量在混合分布上计算。
- **停止梯度自举**：TD 目标直接由当前网络 :math:`Q_{\phi}` 计算并截断梯度，无需维护 EMA 目标网络副本。
- **加宽 Critic 层**：配合 BRN 将隐藏层从 256 扩展到 2048 等更宽维度，在 UTD 为 1 时即可取得高样本效率。

4. 配置
-------

CrossQ 与 SAC 复用同一套配置，通过 ``q_head_type: "crossq"`` 切换 Q 头结构。完整示例参见 :doc:`MLP Policy 示例 <../../examples/embodied/mlp>` 与可运行配置 ``examples/embodiment/config/maniskill_crossq_mlp.yaml``：

.. code-block:: yaml

   algorithm:
      update_epoch: 32
      group_size: 1
      agg_q: min                     # ["min", "mean"]
      adv_type: embodied_sac
      loss_type: embodied_sac
      loss_agg_func: "token-mean"
      q_head_type: "crossq"          # ["crossq", "default"]
      critic_actor_ratio: 2
      bootstrap_type: always         # ["standard", "always"]
      gamma: 0.8
      tau: 0.01
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

   actor:
      model:
         add_q_head: True
         q_head_type: "crossq"
