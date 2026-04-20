Implicit Q-Learning (IQL) Algorithm
=====================================

1. Introduction
------------------

Implicit Q-Learning (IQL) is a classic algorithm for offline reinforcement learning (Offline RL).  
It learns high-quality policies from a fixed dataset without explicitly performing policy improvement on unseen actions.  

The key idea of IQL is to decouple value learning from policy learning:

- Value (State Value Function): learns a value function biased toward high-return regions via expectile regression.
- Critic (Q-value Function): fits action values with Bellman regression.
- Actor (Policy Model): updates the policy with advantage-weighted behavior cloning (Advantage-Weighted BC).

This design avoids aggressive extrapolation on out-of-distribution (OOD) actions and is stable on offline benchmarks such as D4RL.

For more details, see the original IQL paper  
`IQL <https://arxiv.org/abs/2110.06169>`_.


2. Objective Function
----------------------

Let the state value function be :math:`V_{\psi}(s)`, the Q function be :math:`Q_{\phi}(s, a)`, and the policy be :math:`\pi_{\theta}(a|s)`.  
IQL is usually trained with the following three objectives:

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

where :math:`\rho_{\tau}(u)=|\tau-\mathbb{I}(u<0)|u^2`, and :math:`\tau` is the expectile coefficient (e.g., 0.7).

**(3) Advantage-weighted behavior cloning for Actor**

.. math::

   L_{\pi}(\theta) =
   -\mathbb{E}_{(s, a) \sim D}
   \left[
      \exp\left(
         \frac{Q_{\bar{\phi}}(s, a)-V_{\bar{\psi}}(s)}{\beta}
      \right)
      \log \pi_{\theta}(a|s)
   \right],

where :math:`\beta` is the temperature coefficient that controls the scale of advantage weights.


3. Configuration
-----------------

In RLinf, IQL can be used for offline embodied tasks (e.g., D4RL).  
Using `d4rl_iql_mujoco.yaml` as an example, the key configuration is:

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
