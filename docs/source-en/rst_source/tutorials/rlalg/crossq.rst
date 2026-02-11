Cross-Q
==================================

1. Introduction
-----------------

CrossQ is a lightweight evolution of the :doc:`Soft Actor-Critic (SAC) <sac>` algorithm designed for greater sample efficiency. It introduces three distinct architectural updates upon the standard SAC framework:

- Removal of Target Networks: Eliminates the lagging target Q-networks to accelerate value learning.

- Batch Renormalization (BRN): Integrates Batch Renormalization into the critic and actor to stabilize training, using a joint forward pass of current and future states to correct distribution statistics.

- Wider Critic Networks: Expands the width of critic network layers (e.g., to 2048 units) to enhance representation power and optimization speed.

By removing target networks, Cross-Q avoids the artificial delay in value propagation, while the careful application of Batch Renormalization ensures training stability that was previously difficult to achieve in RL. This architecture allows CrossQ to match or surpass the sample efficiency of computationally expensive methods like REDQ and DroQ, all while maintaining a standard UTD ratio of 1 and a significantly lower computational footprint.

For more details, see the original `CrossQ <https://arxiv.org/abs/1902.05605>`_ paper.

2. Objective Function
------------------------

CrossQ shares the same maximum entropy RL objective as SAC. Let the policy be :math:`\pi`. Then the Q function for :math:`\pi` satisfies the same soft Bellman equation found in SAC:

.. math::

   Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P, a \sim \pi} \left[ r(s, a) + \gamma (Q^{\pi}(s', a') - \alpha \log \pi(a'|s')) \right].

Here :math:`\gamma` is the discount factor and :math:`\alpha` is the temperature parameter. CrossQ diverges from SAC in how it parameterizes and updates the Q-function to estimate this value. Specifically, it removes the target networks entirely.

Therefore, the loss for the i-th Q-function :math:`Q_{\phi_{i}}` is defined using the current network parameters :math:`\phi_{i}` rather than a separate target network:

.. math::

   L(\phi_{i}, D) = \mathbb{E}{(s, a, r, s', d) \sim D} \left[ \frac{1}{2} \left( Q_{\phi_{i}}(s, a) - (r + \gamma (1 - d) \cdot \text{sg}(\min_{i} Q_{\phi_{i}}(s', a') - \alpha \log \pi_{\theta}(a'|s'))) \right)^2 \right],

where :math:`D` is the replay buffer, :math:`\text{sg}(\cdot)` denotes the stop-gradient operator (preventing gradients from flowing back through the bootstrapping target), and :math:`a'` is sampled from the current policy :math:`\pi_{\theta}`.

The actor loss remains identical to SAC. The actor :math:`\pi_{\theta}` is trained to maximize the expected Q value and entropy:

.. math::

   L(\theta, D) = \mathbb{E}{s \sim D, a \sim \pi{\theta}} \left[ \alpha \log \pi_{\theta}(a|s) - \min_{i} Q_{\phi_i}(s, a) \right].   

Similarly, the temperature coefficient :math:`\alpha` is learned via the same loss function:

.. math::

   L(\alpha, D) = - \alpha (H_{\text{targ}} - H(\pi(\cdot, d))).

3. Specific Designs
----------------------

CrossQ introduces three key design choices:

- Removal of Target Networks: Unlike SAC, which relies on slowly updating target networks :math:`\phi_{\text{targ}}` to stabilize learning, CrossQ uses the current network :math:`\phi` for calculating the TD target. Gradients are explicitly stopped on the target bootstrapping term to prevent divergence.

- Batch Renormalization (BRN): To stabilize training without target networks, CrossQ integrates Batch Renormalization into the critic and actor networks. To resolve the distribution mismatch between training samples :math:`(s, a)` and bootstrapping samples :math:`(s', a')`, CrossQ performs a joint forward pass by concatenating these batches:

.. math::

  \left[ \begin{matrix} q \\ q' \end{matrix} \right] = Q_{\phi} \left( \left[ \begin{matrix} s \\ s' \end{matrix} \right], \left[ \begin{matrix} a \\ a' \end{matrix} \right] \right),
 
This ensures the BN statistics are computed over a mixture of current and replay data.

- Wider Critic Networks: CrossQ expands the width of the critic network layers (e.g., from 256 to 2048 hidden units). This increased width, combined with BRN, accelerates optimization and significantly improves sample efficiency compared to standard SAC architectures.

4. Configuration
--------------------

CrossQ shares a nearly identical configuration with SAC. A single parameter, `q_head_type`, can be used to toggle between the CrossQ and standard SAC architectures.

.. code-block:: yaml

   algorithm:
      update_epoch: 32
      group_size: 1
      agg_q: min # ["min", "mean"]. Option to aggregate multiple Q-values.


      adv_type: embodied_sac
      loss_type: embodied_sac
      loss_agg_func: "token-mean"
      q_head_type: "crossq" # ["crossq", "default"]. Choose CrossQ or standard SAC Q-head.
      
      bootstrap_type: standard # [standard, always]. Bootstrap Q-values according to terminations and truncations. "standard" only bootstraps when truncations, while "always" bootstraps when truncations or terminations.
      gamma: 0.8 # Discount factor.
      tau: 0.01  # Soft update coefficient for target networks
      target_update_freq: 1  # Frequency of target network updates
      entropy_tuning:
         alpha_type: softplus  # ["softplus","exp","fixed_alpha"]
         initial_alpha: 0.01  # Initial temperature value
         target_entropy: -4  # Target entropy (-action_dim)
         optim:
            lr: 3.0e-4  # Learning rate for temperature parameter
            lr_scheduler: torch_constant
            clip_grad: 10.0
      
      # Replay buffer settings
      replay_buffer:
         enable_cache: True # Enable memory cache to reduce I/O overhead
         cache_size: 6000  # number of trajectories cached in memory
         sample_window_size: 6000  # number of latest trajectories to sample from for replay buffer
         min_buffer_size: 2  # Minimum buffer size before training starts (in number of trajectories)