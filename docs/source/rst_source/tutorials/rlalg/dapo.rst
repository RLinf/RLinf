Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)
==============================================================

1. Introduction
---------------

Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) is a
recent reinforcement learning algorithm for large-scale LLM training.  
It extends GRPO with four key techniques:  

1. **Clip-Higher**: asymmetric clipping with a higher upper bound.  
2. **Dynamic Sampling**: resampling until a group contains both correct and incorrect answers.  
3. **Token-Level Policy Gradient Loss**: loss computed at the token level instead of the sequence level.  
4. **Overlong Reward Shaping**: penalizing excessively long responses.  

These improvements make DAPO more stable and efficient, especially for
long chain-of-thought (CoT) reasoning tasks.  

For further details, see the original paper:  
`DAPO <https://arxiv.org/abs/2503.14476>`_.


2. Objective Function
----------------------

DAPO maximizes the following objective:

.. math::

   J_{\mathrm{DAPO}}(\theta)
   = \mathbb{E}_{(q,a)\sim\mathcal{D},\,\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}
     \left[
       \frac{1}{\sum_{i=1}^{G} |o_i|}
       \sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
         \min\!\Big(
           r_{i,t}(\theta)\,\hat{A}_{i,t},\;
           \mathrm{clip}\!\big(r_{i,t}(\theta),\, 1-\varepsilon_{\mathrm{low}},\, 1+\varepsilon_{\mathrm{high}}\big)\,\hat{A}_{i,t}
         \Big)
     \right],

where

- :math:`r_{i,t}(\theta)` is the importance sampling ratio,  
- :math:`\varepsilon_{\mathrm{low}}, \varepsilon_{\mathrm{high}}` define the decoupled clipping range,  
- and :math:`\hat{A}_{i,t}` is the group-relative advantage.  

**Dynamic Sampling.**  
Instead of accepting arbitrary groups of responses, DAPO enforces that each group contains both correct and incorrect answers.  
This is expressed as:

.. math::

   0 \;<\; \big\lvert \{\, o_i \mid \mathrm{is\_equivalent}(a, o_i) \,\} \big\rvert \;<\; G.

This constraint prevents trivial groups (all correct or all incorrect),  
improving training efficiency and stability.  

**Clip-Higher.**  
DAPO adopts an asymmetric clipping strategy with a larger upper bound  
(:math:`\varepsilon_{\mathrm{high}} > \varepsilon_{\mathrm{low}}`).  
This reduces premature suppression of potentially useful exploratory updates,  
encourages diversity, and mitigates entropy collapse.  

**Token-Level Loss.**  
Instead of computing gradients only at the *sequence level*,  
DAPO applies policy gradient loss at the *token level*.  
This reduces bias caused by variable response lengths,  
which is crucial in long-CoT RL training.  

**Overlong Reward Shaping.**  
To stabilize training and avoid noisy optimization from excessively long responses,  
DAPO introduces a length penalty.  
- Responses longer than :math:`\texttt{safe\_length}` are penalized.  
- The penalty grows linearly with length, up to a maximum cutoff.  
- The length reward lies within :math:`[-1, 0]`.  

This ensures that spurious long answers do not dominate optimization.


3. Configuration
-----------------

Currently, the framework supports DAPO for LLM math tasks.  

.. code-block:: yaml

  algorithm:
    # Core DAPO settings (recommended not to change)
    adv_type: grpo
    loss_type: ppo
    loss_agg_func: "token-mean"
    use_valid_token_scale: True # Divide advantage by valid token count → token-level loss

    # Algorithm parameters (typically require tuning)
    group_size: 16              # Number of responses sampled per prompt
    clip_ratio_high: 0.28       # epsilon_high (asymmetric clipping upper bound)
    clip_ratio_low: 0.20        # epsilon_low (clipping lower bound)

    len_reward_penalty: 0.1     # Length penalty coefficient
    safe_length: 16384          # Safe length threshold (beyond this → penalty applied)
    max_length: 20480           # Hard cutoff: responses longer than this are discarded
    max_resample: 5             # Maximum resampling attempts for dynamic sampling

    kl_beta: 0.0                # KL penalty coefficient
    kl_penalty_type: low_var_kl # Options: low_var_kl, kl, abs, mse

    calculate_entropy: False    # Optional: encourage exploration
    entropy_bonus: 0.0

    normalize_advantages: True
    early_stop_imp_ratio: 5.0   # Drop minibatches with extreme importance ratios


4. Notes
---------

- Always ensure dynamic sampling is enabled; trivial groups reduce effectiveness.  
- Use token-level loss (enabled via ``use_valid_token_scale=True``) for long-CoT tasks.  
- Tune :math:`\varepsilon_{\mathrm{low}}, \varepsilon_{\mathrm{high}}` carefully:  
  smaller gaps → more conservative updates, larger gaps → more exploration.  
- Length shaping is critical for preventing runaway long responses in LLMs.  
