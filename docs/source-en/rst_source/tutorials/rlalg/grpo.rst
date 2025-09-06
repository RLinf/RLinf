Group Relative Policy Optimization (GRPO)
=========================================

1. Introduction
---------------

Group Relative Policy Optimization (GRPO) is a variant of PPO designed for prompt-level relative comparison.  

- In PPO, a critic model is required to evaluate whether a sampled behavior (i.e., a response sequence) is good or bad.  
- In GRPO, for the same prompt, multiple responses are sampled to form a group.  
  The relative performance of responses within the group is used to compute each response's advantage,  
  making policy updates focus on relative performance under the same context.  

Because GRPO no longer depends on a separate critic model, it significantly reduces computational resource requirements.  
This makes it particularly suitable for large-scale LLM training, where maintaining a critic model would be prohibitively expensive.

For more details, see the original GRPO paper `DeepSeek-R1 <https://arxiv.org/abs/2501.12948>`_.

2. Objective Function
----------------------

For a question–answer pair :math:`(q,a)`, a behavior policy  
:math:`\pi_{\theta_{\mathrm{old}}}` samples a group of :math:`G` responses  
:math:`\{o_i\}_{i=1}^{G}` with corresponding sequence rewards :math:`\{R_i\}_{i=1}^{G}`.  

The group-relative advantage for every token of sequence :math:`i` is defined as:

.. math::

   \hat{A}_{i,t} = \frac{R_i - \operatorname{mean}(\{R_j\}_{j=1}^{G})}
                        {\operatorname{std}(\{R_j\}_{j=1}^{G})}.

This advantage measures how much better (or worse) a response is compared to the group average,  
normalized by the group's reward variance.

Similar to PPO, GRPO adopts a clipped surrogate objective, optionally with a KL penalty term.  
In some implementations and tasks, the KL term may be omitted.

.. math::

   J_{\mathrm{GRPO}}(\theta)
   = \mathbb{E}_{(q,a)\sim\mathcal{D},\,\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}
     \!\left[
       \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
         \min\!\Big(
           r_{i,t}(\theta)\,\hat{A}_{i,t},\;
           \mathrm{clip}\!\big(r_{i,t}(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big)\,\hat{A}_{i,t}
         \Big)
     \right]
     \;-\; \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\big),

where

.. math::

   r_{i,t}(\theta) =
   \frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}
        {\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q, o_{i,<t})}.

- :math:`r_{i,t}(\theta)` is the importance sampling ratio.  
- :math:`\varepsilon` is the clipping threshold that prevents overly large updates.  
- :math:`\beta` controls the strength of the KL penalty against a reference policy :math:`\pi_{\mathrm{ref}}`.  


3. Configuration
-----------------

Our framework supports GRPO for both LLM math tasks and embodied tasks.  
An example configuration for LLM math tasks is shown below:

.. code-block:: yaml

  algorithm:
    # Core GRPO settings (recommended not to change)
    adv_type: grpo
    loss_type: ppo
    loss_agg_func: "token-mean"

    # Algorithm parameters (typically require tuning)
    group_size: 16              # Number of responses sampled per prompt

    kl_beta: 0.0                # KL penalty coefficient
    kl_penalty_type: low_var_kl # Options: low_var_kl, kl, abs, mse
    ratio_clip_eps: 0.2         # Clipping range for importance ratio

    calculate_entropy: False    # Optional: encourage exploration
    entropy_bonus: 0.0

    normalize_advantages: True
    early_stop_imp_ratio: 5.0   # Drop minibatches with extreme importance ratios for stability
    use_valid_token_scale: False # Standard GRPO implementation.
                                 # If True, advantages are divided by valid token count (DAPO trick).


4. Notes
---------

- Always batch prompts with multiple completions (≥ 2 per prompt).  
- Use higher sampling temperature (0.7–1.0) to encourage diverse candidate responses.  
- Rewards must be comparable within the same prompt group, since the advantage is computed relatively.  
