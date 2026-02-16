Soft Actor-Critic (SAC) Algorithm
==================================

1. Introduction
---------------

Soft Actor-Critic (SAC) is one of the most widely used reinforcement learning (RL) algorithms.  
It consists of two key components:

- Actor (Policy Model): generates actions based on the current state.
- Critic (Q-value Model): evaluates the value of current observations and the chosen actions.

Soft Actor-Critic (SAC) is an off-policy deep reinforcement learning algorithm for continuous control. 
It is based on the maximum entropy reinforcement learning framework, 
which augments the standard reward objective with an entropy term to encourage exploration and robustness. 
SAC simultaneously learns a stochastic policy and two Q-functions, 
using entropy-regularized Bellman backups and automatic temperature tuning. 
Due to its sample efficiency and stability, 
SAC has been widely applied in robotics and continuous control benchmarks. 

For more details, see the original SAC paper 
`SAC <https://arxiv.org/abs/1801.01290>`_.


2. Objective Function
----------------------

Let the policy be :math:`\pi`. Then the Q function for :math:`\pi` is defined as: :math:`Q^{\pi}(s, a)`. In SAC, the Q function satisfies the following soft Bellman equation:

.. math::

   Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P, a \sim \pi} \left[
      r(s, a) + \gamma (Q^{\pi}(s', a') + H(\pi(\cdot|s')))
   \right]
   = \mathbb{E}_{s' \sim P, a \sim \pi} \left[
      r(s, a) + \gamma (Q^{\pi}(s', a') - \alpha \log \pi(a'|s'))
   \right].

Here :math:`\gamma` is the discount factor, :math:`H` is the entropy of the policy, and :math:`\alpha` is the temperature parameter that determines the relative importance of the entropy term against the reward.

Therefore, the loss for the i-th Q-function :math:`Q_{\phi_{i}}` is as follows:

.. math::

   L(\phi_{i}, D) = \mathbb{E}_{(s, a, r, s', d) \sim D} \left[
      \frac{1}{2} \left(
         Q_{\phi_{i}}(s, a) - (r + \gamma (1 - d)(\min_{i} Q_{\overline{\phi_{\text{targ}, i}}}(s', a') - \alpha \log \pi_{\theta}(a'|s')))
      \right)^2
   \right],

where :math:`D` is the replay buffer, :math:`\overline{\phi_{\text{targ}, i}}` are the parameters of the target Q-network, and :math:`a'` is sampled from the current policy :math:`\pi_{\theta}`.

The policy :math:`\pi_{\theta}` is to maximize the expected Q value and entropy. Therefore, the policy loss is defined as follows:

.. math::

   L(\theta, D) = \mathbb{E}_{s \sim D, a \sim \pi_{\theta}} \left[
      \alpha \log \pi_{\theta}(a|s) - \min_{i} Q_{\phi_i}(s, a)
   \right].

In practice, the temperature coefficient :math:`\alpha` is learnable. Then the alpha loss is defined as follows:

.. math::

   L(\alpha, D) = - \alpha (H_{\text{targ}} - H(\pi(\cdot, d))), 

where :math:`H_{\text{targ}}` is a hyperparameter representing the target value for entropy. It is typically set to negative action dimension. 


3. Configuration
-----------------

Currently, SAC is supported only for embodied tasks in our framework.  
The algorithm configuration is defined as follows:

.. code-block:: yaml

   algorithm:
      update_epoch: 32
      group_size: 1
      agg_q: min # ["min", "mean"]. Option to aggregate multiple Q-values.


      adv_type: embodied_sac
      loss_type: embodied_sac
      loss_agg_func: "token-mean"
      
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