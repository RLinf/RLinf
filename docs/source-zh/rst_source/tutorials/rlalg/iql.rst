隐式Q学习 (IQL)
==================================

1. 引言
---------------

IQL（Implicit Q-Learning）是一种面向离线强化学习（Offline RL）的经典算法。  
它在不需要对未见动作进行显式策略外推的前提下，能够从静态数据集中学习高质量策略。  

IQL 的核心思想是将价值学习和策略学习解耦：  

- Value （状态价值函数）: 使用 expectile 回归学习偏向高回报区域的状态价值。  
- Critic （Q值函数）: 通过 Bellman 目标拟合动作价值。  
- Actor （策略模型）: 通过优势加权行为克隆（Advantage-Weighted BC）更新策略。  

这种设计避免了直接对 OOD（out-of-distribution）动作进行过度估计，在 D4RL 等离线基准上表现稳定。

更多细节请参考原始论文  
`IQL <https://arxiv.org/abs/2110.06169>`_.


2. 目标函数
----------------------

设状态价值函数为 :math:`V_{\psi}(s)`，Q 值函数为 :math:`Q_{\phi}(s, a)`，策略为 :math:`\pi_{\theta}(a|s)`。  
IQL 的训练通常包含以下三个目标：

**(1) Q 函数回归**

.. math::

   L_Q(\phi) =
   \mathbb{E}_{(s, a, r, s') \sim D}
   \left[
      \left(
         Q_{\phi}(s, a) -
         \left(r + \gamma V_{\bar{\psi}}(s')\right)
      \right)^2
   \right].

**(2) Value 的 expectile 回归**

.. math::

   L_V(\psi) =
   \mathbb{E}_{(s, a) \sim D}
   \left[
      \rho_{\tau}\left(
         Q_{\bar{\phi}}(s, a) - V_{\psi}(s)
      \right)
   \right],

其中 :math:`\rho_{\tau}(u)=|\tau-\mathbb{I}(u<0)|u^2`，:math:`\tau` 是 expectile 系数（例如 0.7）。

**(3) Actor 的优势加权行为克隆**

.. math::

   L_{\pi}(\theta) =
   -\mathbb{E}_{(s, a) \sim D}
   \left[
      \exp\left(
         \frac{Q_{\bar{\phi}}(s, a)-V_{\bar{\psi}}(s)}{\beta}
      \right)
      \log \pi_{\theta}(a|s)
   \right],

其中 :math:`\beta` 对应温度系数，用于控制优势权重的尺度。


3. 配置
-----------------

在 RLinf 中，IQL 可用于离线具身任务（如 D4RL）。  
以 `d4rl_iql_mujoco.yaml` 为例，核心配置如下：

.. code-block:: yaml

   algorithm:
      loss_type: "offline_iql"
      batch_size: 256
      actor_lr: 3.0e-4
      value_lr: 3.0e-4
      critic_lr: 3.0e-4
      hidden_dims: [256, 256]
      discount: 0.99
      tau: 0.005
      expectile: 0.7
      temperature: 3.0
      gamma: 0.99
      dropout_rate: null
      opt_decay_schedule: "cosine"

   env:
      dataset_type: "d4rl"
      train:
         env_type: "d4rl"
         env_name: "halfcheetah-medium-v2"

   actor:
      worker_cls: "iql"
