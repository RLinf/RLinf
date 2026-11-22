Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)
==============================================================

1. Introduction
---------------

Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) extends :doc:`GRPO <grpo>` for long chain-of-thought (CoT) reasoning tasks. The original `DAPO paper <https://arxiv.org/abs/2503.14476>`__ proposes four techniques:

1. **Clip-Higher**: asymmetric clipping with a higher upper bound (:math:`\varepsilon_{\mathrm{high}} > \varepsilon_{\mathrm{low}}`) to avoid suppressing exploratory updates and mitigate entropy collapse.
2. **Token-Level Policy Gradient Loss**: weighting policy gradient updates across valid tokens rather than averaging equally over variable-length sequences.
3. **Dynamic Sampling**: filtering trivial prompt groups whose completions are all correct or all incorrect.
4. **Overlong Reward Shaping**: applying length-aware shaping to excessively long completions.

In RLinf, DAPO's **Clip-Higher** (``clip_ratio_low`` / ``clip_ratio_high``) and **token-level loss scaling** (``use_valid_token_scale``) are built directly into the ``grpo`` + ``actor`` training path.

2. Objective Function
---------------------

DAPO maximizes the token-normalized clipped surrogate objective:

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

- :math:`r_{i,t}(\theta)` is the importance sampling ratio between the updated policy and the behavior policy,
- :math:`\varepsilon_{\mathrm{low}}, \varepsilon_{\mathrm{high}}` define the decoupled lower and upper clipping bounds, and
- :math:`\hat{A}_{i,t}` is the group-relative advantage.

3. Configuration
----------------

RLinf exposes the implemented DAPO controls on top of the GRPO actor configuration in reasoning and VQA recipes. See :doc:`Moonlight-16B <../../examples/agentic/math_reasoning/moonlight>` (``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml``), ``examples/reasoning/config/math/qwen2.5-7b-grpo-megatron.yaml``, and ``examples/reasoning/config/vqa/qwen3-vl-2b-grpo-fsdp-geo3k.yaml`` for runnable configs.

.. code-block:: yaml

   algorithm:
     # Core GRPO / DAPO settings
     adv_type: grpo
     loss_type: actor
     loss_agg_func: "token-mean"
     use_valid_token_scale: True   # Token-level loss scaling across valid tokens

     # Decoupled asymmetric clipping (Clip-Higher)
     group_size: 16                # Number of responses sampled per prompt
     ratio_clip_eps: 0.2           # Fallback clip bound when clip_ratio_low/high is null
     clip_ratio_low: 0.20          # epsilon_low (lower clipping bound)
     clip_ratio_high: 0.28         # epsilon_high (asymmetric upper clipping bound)

     # Regularization and stability
     kl_beta: 0.0                  # KL penalty coefficient
     kl_penalty_type: low_var_kl   # Options: low_var_kl, kl, abs, mse
     calculate_entropy: False      # Log policy entropy
     entropy_bonus: 0.0
     normalize_advantages: True
     early_stop_imp_ratio: 5.0     # Drop minibatches with extreme importance ratios

4. Notes
--------

- Enable ``use_valid_token_scale: True`` for long-CoT workloads so long and short responses contribute proportionally by valid token count.
- Set ``clip_ratio_high`` above ``clip_ratio_low`` (for example, ``0.28`` vs. ``0.20``) to enable Clip-Higher; narrower gaps give more conservative updates, while wider gaps allow more exploration.
- Configure maximum response length through ``runner.seq_length`` and ``algorithm.sampling_params.max_new_tokens`` (see :doc:`Math Reasoning <../../examples/agentic/math_reasoning/reasoning>`).
