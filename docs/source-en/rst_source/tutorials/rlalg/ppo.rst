Proximal Policy Optimization (PPO)
==================================

1. Introduction
---------------

Proximal Policy Optimization (PPO) is one of the most widely used reinforcement learning (RL) algorithms.  
It consists of two key components:

- Actor (Policy Model): generates actions based on the current state.
- Critic (Value Model): evaluates the value of the chosen actions.

PPO is a stable policy-gradient method that improves upon vanilla Policy Gradient.  
It achieves this by constraining the step size of policy updates, thereby enhancing training stability and efficiency.  
In addition, PPO employs Generalized Advantage Estimation (GAE) to reduce the variance of the value estimates.  

PPO was extensively applied in the early stages of RLHF (Reinforcement Learning from Human Feedback).  
However, due to the need for a large critic model (often another LLM), it can incur high computational costs and large training overhead.  

For more details, see the original PPO paper 
`PPO <https://arxiv.org/abs/1707.06347>`_ and its application in RLHF
`InstructGPT <https://arxiv.org/abs/2203.02155>`_.


2. Objective Function
----------------------

Let the policy be :math:`\pi_\theta`.  
For a dataset :math:`\mathcal{D}` containing question-answer pairs :math:`(q,a)`,  
the PPO objective is defined as:

.. math::

   J_{\mathrm{PPO}}(\theta)
   = \mathbb{E}_{(q,a)\sim\mathcal{D},\, o_{\le t}\sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}
   \Big[
     \min\!\Big(
       r_t(\theta)\,\hat{A}_t,\;
       \mathrm{clip}\,\big(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big)\,\hat{A}_t
     \Big)
   \Big],

where

- :math:`r_t(\theta) = \dfrac{\pi_\theta(o_t \mid q, o_{<t})}
  {\pi_{\theta_{\mathrm{old}}}(o_t \mid q, o_{<t})}`  
  is the importance sampling ratio, comparing the new policy with the old policy.

- :math:`\varepsilon` is the clipping range, a hyperparameter that prevents overly large updates.

- :math:`\hat{A}_t` is the advantage estimate at time step :math:`t`.

Using Generalized Advantage Estimation (GAE), the advantage is computed as:

.. math::

   \hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
   = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l},
   \qquad
   \delta_l = R_l + \gamma V(s_{l+1}) - V(s_l),
   \quad 0 \le \gamma, \lambda \le 1.

Here,

- :math:`\gamma` (discount factor) and :math:`\lambda` (GAE parameter) are hyperparameters.  
- :math:`V(s)` is the value estimate from the critic model.

3. Configuration
-----------------

Currently, PPO is supported only for embodied tasks in our framework.  
The algorithm configuration is defined as follows:

.. code-block:: yaml

   algorithm:

      # Core PPO settings (recommended not to change)
      normalize_advantages: True
      group_size: 1
      adv_type: embodied_gae
      loss_type: embodied_ppo
      loss_agg_func: "token-mean"

      # Algorithm parameters (typically require tuning)

      rollout_micro_batch_size: 256
      logprob_forward_micro_batch_size: 16  # Larger batch_size improves stability.
                                            # Adjust according to compute resources and model size.

      entropy_bonus: 0          # Optional: encourage exploration
      clip_ratio_high: 0.2      # PPO clipping parameter (epsilon)
      clip_ratio_low: 0.2       # Should match clip_ratio_high
      value_clip: 0.2           # Stabilizes value function updates

      gamma: 0.99               # Discount factor for GAE
      gae_lambda: 0.95          # Lambda parameter for GAE

      huber_delta: 10.0         # Delta parameter for Huber loss in value training


4. Notes
---------

- Use reward normalization to stabilize training.  
- Monitor KL divergence to detect policy over-updates.  
- For large LLMs, increase batch size to reduce variance.