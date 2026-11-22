Implicit Q-Learning (IQL)
=========================

1. Introduction
---------------

Implicit Q-Learning (IQL) is an offline reinforcement learning algorithm that trains a policy from a fixed dataset without querying unseen out-of-distribution (OOD) actions during Bellman backups.

IQL decouples value learning from policy extraction across three networks:

- **Value (:math:`V_{\psi}`)**: learns a state value function biased toward high-return actions using expectile regression.
- **Critic (:math:`Q_{\phi}`)**: fits action values with Bellman regression against the state value function.
- **Actor (:math:`\pi_{\theta}`)**: updates the policy with advantage-weighted behavior cloning (AWBC).

For algorithm details, see the original `IQL paper <https://arxiv.org/abs/2110.06169>`__.

2. Objective Function
---------------------

Let :math:`V_{\psi}(s)` be the state value function, :math:`Q_{\phi}(s, a)` the Q-function, and :math:`\pi_{\theta}(a \mid s)` the policy. IQL minimizes three objectives:

**(1) Q-function regression**

.. math::

   L_Q(\phi) =
   \mathbb{E}_{(s, a, r, s') \sim D}
   \left[
      \left(
         Q_{\phi}(s, a) -
         \left(r + \gamma V_{\bar{\psi}}(s')\right)
      \right)^2
   \right].

**(2) Expectile regression for Value**

.. math::

   L_V(\psi) =
   \mathbb{E}_{(s, a) \sim D}
   \left[
      \rho_{\tau}\left(
         Q_{\bar{\phi}}(s, a) - V_{\psi}(s)
      \right)
   \right],

where :math:`\rho_{\tau}(u)=|\tau-\mathbb{I}(u<0)|u^2`, and :math:`\tau` is the expectile coefficient (for example, ``0.7``).

**(3) Advantage-weighted behavior cloning for Actor**

.. math::

   L_{\pi}(\theta) =
   -\mathbb{E}_{(s, a) \sim D}
   \left[
      \exp\left(
         \frac{Q_{\bar{\phi}}(s, a)-V_{\bar{\psi}}(s)}{\beta}
      \right)
      \log \pi_{\theta}(a \mid s)
   \right],

where :math:`\beta` is the temperature coefficient controlling the scale of the advantage weights.

3. Configuration
----------------

In RLinf, IQL runs on offline embodied benchmarks such as D4RL. See the :doc:`D4RL IQL recipe <../../examples/embodied/iql_d4rl>` and the runnable config ``examples/embodiment/config/d4rl_iql_mujoco.yaml`` (as well as ``examples/embodiment/config/d4rl_iql_antmaze.yaml`` and ``examples/embodiment/config/d4rl_iql_kitchen_adroit.yaml``).

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
