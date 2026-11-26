Soft Actor-Critic (SAC)
=======================

1. Introduction
---------------

Soft Actor-Critic (SAC) is an off-policy maximum-entropy reinforcement learning algorithm for continuous control. It consists of two components:

- **Actor (Policy Model)**: generates stochastic actions from the current state.
- **Critic (Q-Value Model)**: evaluates state-action pairs using clipped double Q-learning and target networks.

SAC augments expected return with a policy entropy bonus and tunes the entropy temperature automatically. For details, see the original `SAC paper <https://arxiv.org/abs/1801.01290>`__.

2. Objective Function
---------------------

Let the policy be :math:`\pi`. Its Q-function :math:`Q^{\pi}(s, a)` satisfies the soft Bellman equation:

.. math::

   Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P,\, a' \sim \pi} \left[
      r(s, a) + \gamma \left(Q^{\pi}(s', a') + \alpha H(\pi(\cdot \mid s'))\right)
   \right]
   = \mathbb{E}_{s' \sim P,\, a' \sim \pi} \left[
      r(s, a) + \gamma \left(Q^{\pi}(s', a') - \alpha \log \pi(a' \mid s')\right)
   \right].

Here :math:`\gamma` is the discount factor, :math:`H` is the policy entropy, and :math:`\alpha` is the temperature balancing entropy against reward.

The loss for the :math:`i`-th Q-function :math:`Q_{\phi_{i}}` is:

.. math::

   L(\phi_{i}, D) = \mathbb{E}_{(s, a, r, s', d) \sim D} \left[
      \frac{1}{2} \left(
         Q_{\phi_{i}}(s, a) - \left(r + \gamma (1 - d)\left(\min_{j} Q_{\overline{\phi}_{\mathrm{targ}, j}}(s', a') - \alpha \log \pi_{\theta}(a' \mid s')\right)\right)
      \right)^2
   \right],

where :math:`D` is the replay buffer, :math:`\overline{\phi}_{\mathrm{targ}, j}` are the target Q-network parameters, and :math:`a' \sim \pi_{\theta}(\cdot \mid s')`.

The policy :math:`\pi_{\theta}` minimizes:

.. math::

   L(\theta, D) = \mathbb{E}_{s \sim D,\, a \sim \pi_{\theta}} \left[
      \alpha \log \pi_{\theta}(a \mid s) - \min_{i} Q_{\phi_i}(s, a)
   \right].

With automatic entropy tuning, the learnable temperature :math:`\alpha` is updated by minimizing:

.. math::

   L(\alpha, D) = \mathbb{E}_{s \sim D} \left[ -\alpha \left(H_{\mathrm{targ}} - H(\pi(\cdot \mid s))\right) \right],

where :math:`H_{\mathrm{targ}}` is the target entropy (typically set to :math:`-\mathrm{action\_dim}`).

3. Configuration
----------------

In RLinf, SAC runs on embodied continuous-control tasks (simulation and real-world). See the :doc:`MLP Policy recipe <../../examples/embodied/mlp>` and runnable configs ``examples/embodiment/config/maniskill_sac_mlp.yaml`` and ``examples/embodiment/config/maniskill_sac_mlp_async.yaml``:

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
      tau: 0.01                      # Soft target update coefficient
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
