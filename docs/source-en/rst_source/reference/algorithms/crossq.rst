CrossQ
======

1. Introduction
---------------

CrossQ builds on :doc:`Soft Actor-Critic (SAC) <sac>` for higher sample efficiency without target networks. It introduces three architectural changes:

- **Removal of Target Networks**: eliminates lagging target Q-networks to accelerate value propagation.
- **Batch Renormalization (BRN)**: applies Batch Renormalization in the critic and actor, using a joint forward pass over current and next states to align batch statistics.
- **Wider Critic Networks**: increases critic layer width (for example, to 2048 units) to boost representation capacity and optimization speed.

For details, see the original `CrossQ paper <https://arxiv.org/abs/1902.05605>`__.

2. Objective Function
---------------------

CrossQ optimizes the same maximum-entropy objective as SAC. Let the policy be :math:`\pi`. Its soft Q-function satisfies the soft Bellman equation:

.. math::

   Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P,\, a' \sim \pi} \left[ r(s, a) + \gamma \left( Q^{\pi}(s', a') - \alpha \log \pi(a' \mid s') \right) \right].

Here :math:`\gamma` is the discount factor and :math:`\alpha` is the entropy temperature. Without target networks, the loss for the :math:`i`-th critic :math:`Q_{\phi_{i}}` uses the current parameters :math:`\phi_{i}` with a stop-gradient on the bootstrap target:

.. math::

   L(\phi_{i}, D) = \mathbb{E}_{(s, a, r, s', d) \sim D} \left[ \frac{1}{2} \left( Q_{\phi_{i}}(s, a) - \left(r + \gamma (1 - d) \cdot \operatorname{sg}\!\left(\min_{j} Q_{\phi_{j}}(s', a') - \alpha \log \pi_{\theta}(a' \mid s')\right)\right) \right)^2 \right],

where :math:`D` is the replay buffer, :math:`\operatorname{sg}(\cdot)` is the stop-gradient operator, and :math:`a' \sim \pi_{\theta}(\cdot \mid s')`.

The actor :math:`\pi_{\theta}` minimizes the same maximum-entropy policy loss as SAC:

.. math::

   L(\theta, D) = \mathbb{E}_{s \sim D,\, a \sim \pi_{\theta}} \left[ \alpha \log \pi_{\theta}(a \mid s) - \min_{i} Q_{\phi_i}(s, a) \right].

The learnable temperature :math:`\alpha` is updated with:

.. math::

   L(\alpha, D) = \mathbb{E}_{s \sim D} \left[ -\alpha \left( H_{\mathrm{targ}} - H(\pi(\cdot \mid s)) \right) \right].

3. Specific Designs
-------------------

- **Joint Forward Pass for BRN**: to resolve distribution mismatch between current state-action pairs :math:`(s, a)` and bootstrapped pairs :math:`(s', a')`, CrossQ concatenates both batches into a single critic forward pass:

  .. math::

     \begin{bmatrix} q \\ q' \end{bmatrix} = Q_{\phi} \left( \begin{bmatrix} s \\ s' \end{bmatrix}, \begin{bmatrix} a \\ a' \end{bmatrix} \right),

  computing Batch Renormalization statistics over the mixture of current and next transitions.
- **Stop-Gradient Bootstrap**: target values come directly from :math:`Q_{\phi}` with gradients detached instead of an EMA target copy.
- **Wider Critic Layers**: expanding hidden widths (for example, from 256 to 2048) with BRN improves sample efficiency at a UTD ratio of 1.

4. Configuration
----------------

CrossQ shares the SAC configuration and switches the critic head via ``q_head_type: "crossq"``. See the :doc:`MLP Policy recipe <../../examples/embodied/mlp>` and runnable config ``examples/embodiment/config/maniskill_crossq_mlp.yaml``:

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
