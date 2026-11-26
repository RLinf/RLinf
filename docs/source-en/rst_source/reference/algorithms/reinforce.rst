REINFORCE++
===========

1. Introduction
---------------

**REINFORCE++** is a critic-free reinforcement learning method for LLM post-training (`paper <https://arxiv.org/abs/2501.03262>`__). It starts from classical REINFORCE and adds per-token KL penalties and global advantage normalization without a separate value model.

Core design choices include:

- **Single-response or grouped rollouts**: vanilla REINFORCE++ uses ``group_size: 1``; REINFORCE++ with baseline sets ``use_reinpp_baseline: True`` with ``group_size > 1`` to subtract each prompt group's mean reward before global normalization.
- **Per-token KL penalty in the reward**: when ``reinpp_kl_beta > 0``, RLinf subtracts ``reinpp_kl_beta * kld`` from the per-token reward matrix using the estimator selected by ``kl_penalty_type`` (``k1`` / ``kl``, ``k2`` / ``mse``, ``abs``, or ``k3`` / ``low_var_kl``).
- **Global advantage normalization**: advantages are normalized across all valid tokens in the batch.

2. Objective Function
---------------------

Let :math:`q` be the prompt, :math:`o_{1:T}` the generated tokens, and :math:`\pi_{\theta}^{\mathrm{RL}}` the policy. The per-token advantage at step :math:`t` accumulates the terminal reward and per-token KL penalties from :math:`t` to :math:`T`:

.. math::

   A_{q,o_t} \;=\; r(o_{1:T}, q) \;-\; \beta \sum_{i=t}^{T} \operatorname{KL}(i),

where :math:`\beta` is ``reinpp_kl_beta`` and :math:`\operatorname{KL}(t)` is computed against the frozen reference policy :math:`\pi^{\mathrm{ref}}` according to ``kl_penalty_type``. For the unbiased ``k1`` estimator:

.. math::

   \operatorname{KL}_{k_1}(t) \;=\;
   \log\!\left(
     \frac{\pi^{\mathrm{RL}}_{\theta_{\mathrm{old}}}(o_t \mid q,\,o_{<t})}
          {\pi^{\mathrm{ref}}(o_t \mid q,\,o_{<t})}
   \right),

and for the ``k2`` estimator used in RLinf's shipped math config:

.. math::

   \operatorname{KL}_{k_2}(t) \;=\; \frac{1}{2}\left(
     \log \pi^{\mathrm{RL}}_{\theta_{\mathrm{old}}}(o_t \mid q,\,o_{<t}) -
     \log \pi^{\mathrm{ref}}(o_t \mid q,\,o_{<t})
   \right)^2.

To stabilize updates, RLinf normalizes advantages across all valid tokens in the batch:

.. math::

   A^{\mathrm{norm}}_{q,o_t} \;=\;
   \frac{
     A_{q,o_t} \;-\;
     \operatorname{mean}\!\bigl(A_{q,o_t} \mid A_{q,o_t}\in\mathcal{D}_{\mathrm{batch}}\bigr)
   }{
     \operatorname{std}\!\bigl(A_{q,o_t} \mid A_{q,o_t}\in\mathcal{D}_{\mathrm{batch}}\bigr)
   }.

3. Configuration
----------------

See :doc:`Math Reasoning <../../examples/agentic/math_reasoning/reasoning>` and the runnable config ``examples/reasoning/config/math/qwen2.5-1.5b-reinpp-megatron.yaml``.

REINFORCE++:

.. code-block:: yaml

   algorithm:
     adv_type: "reinpp"            # Use REINFORCE++ advantage
     loss_type: actor
     loss_agg_func: "token-mean"
     reinpp_kl_beta: 0.001         # Per-token KL penalty applied inside advantage computation
     kl_penalty_type: k2           # Options: k1 (kl), k2 (mse), abs, k3 (low_var_kl)
     use_reinpp_baseline: False
     group_size: 16
     kl_beta: 0.0                  # Separate actor-loss KL penalty (usually 0.0 for REINFORCE++)

   data:
     rollout_batch_size: 512

REINFORCE++ with group baseline:

.. code-block:: yaml

   algorithm:
     adv_type: "reinpp"
     loss_type: actor
     reinpp_kl_beta: 0.001
     kl_penalty_type: k1
     use_reinpp_baseline: True     # Subtract group-mean reward before global normalization
     group_size: 16
     kl_beta: 0.0

   data:
     rollout_batch_size: 512

4. Notes
--------

- ``kl_penalty_type`` controls which KL estimator ``compute_reinpp_advantages`` applies in ``rlinf/algorithms/advantages.py`` (dispatched to ``kl_penalty`` in ``rlinf/algorithms/utils.py``): ``k1`` (or ``kl``) is unbiased, ``k2`` (or ``mse``) penalizes squared log-probability drift and is used in ``examples/reasoning/config/math/qwen2.5-1.5b-reinpp-megatron.yaml``, and ``k3`` (or ``low_var_kl``) uses Schulman's low-variance estimator.
- Keep ``algorithm.kl_beta: 0.0`` when applying the KL penalty inside the reward via ``algorithm.reinpp_kl_beta`` so KL is not double-counted.
