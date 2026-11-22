隐式 Q 学习（IQL）
==================

1. 引言
-------

隐式 Q 学习（Implicit Q-Learning，IQL）是一种离线强化学习算法，可以在固定数据集上训练 policy，而在 Bellman 更新阶段无需查询未见过的分布外（OOD）动作。

IQL 将价值学习与 policy 提取解耦为三个部分：

- **Value**\ （:math:`V_{\psi}`）：使用 expectile 回归学习偏向高回报区域的状态价值。
- **Critic**\ （:math:`Q_{\phi}`）：使用状态价值目标通过 Bellman 回归拟合动作价值。
- **Actor**\ （:math:`\pi_{\theta}`）：使用优势加权行为克隆（Advantage-Weighted BC）更新 policy。

更多细节参见原始 `IQL 论文 <https://arxiv.org/abs/2110.06169>`__。

2. 目标函数
-----------

设状态价值函数为 :math:`V_{\psi}(s)`，Q 值函数为 :math:`Q_{\phi}(s, a)`，policy 为 :math:`\pi_{\theta}(a \mid s)`。IQL 最小化以下三个损失：

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

其中 :math:`\rho_{\tau}(u)=|\tau-\mathbb{I}(u<0)|u^2`，:math:`\tau` 为 expectile 系数（例如 ``0.7``）。

**(3) Actor 的优势加权行为克隆**

.. math::

   L_{\pi}(\theta) =
   -\mathbb{E}_{(s, a) \sim D}
   \left[
      \exp\left(
         \frac{Q_{\bar{\phi}}(s, a)-V_{\bar{\psi}}(s)}{\beta}
      \right)
      \log \pi_{\theta}(a \mid s)
   \right],

其中 :math:`\beta` 为温度系数，用于控制优势权重的尺度。

3. 配置
-------

在 RLinf 中，IQL 可直接用于 D4RL 等离线具身基准。完整运行步骤参见 :doc:`D4RL IQL 示例 <../../examples/embodied/iql_d4rl>`，可直接运行的配置文件位于 ``examples/embodiment/config/d4rl_iql_mujoco.yaml``\ （以及 ``examples/embodiment/config/d4rl_iql_antmaze.yaml`` 和 ``examples/embodiment/config/d4rl_iql_kitchen_adroit.yaml``）。

.. code-block:: yaml

   runner:
     task_type: "offline"

   algorithm:
     loss_type: "offline_iql"
     adv_type: null
     hidden_dims: [256, 256]
     gamma: 0.99
     tau: 0.005
     expectile: 0.7
     temperature: 3.0
     dropout_rate: null

   data:
     dataset_type: "d4rl"
     task_name: "halfcheetah-medium-v2"

   actor:
     micro_batch_size: 256
     global_batch_size: 256
     optim:
       lr: 3.0e-4
       lr_scheduler: torch_cosine
     critic_optim:
       lr: 3.0e-4
       lr_scheduler: torch_constant
     value_optim:
       lr: 3.0e-4
       lr_scheduler: torch_constant
