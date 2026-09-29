Group Sequence Policy Optimization (GSPO)
=========================================

Use GSPO to train a reasoning model with response-level importance ratios and
clipping. This page explains the objective, how to configure the existing
reasoning runner, and which batch settings preserve equal response weights.

Objective
---------

GSPO compares the likelihood of each complete response under the current and
sampling policies. It averages the token log-probability differences before
exponentiating, so the ratio is normalized by response length:

.. math::

   s_i = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
       (\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
       -\log\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t}))\right).

The existing GRPO advantage estimator normalizes rewards within each prompt's
group. With the resulting response advantage :math:`\hat A_i`, the minimized
loss is

.. math::

   L = -\frac{1}{B}\sum_{i=1}^{B}\min\left(
       s_i\hat A_i,
       \operatorname{clip}(s_i,1-\epsilon_{\mathrm{low}},
       1+\epsilon_{\mathrm{high}})\hat A_i\right).

Padding is excluded from both the ratio and the advantage reduction. Every
response has the same weight regardless of length. An empty response contributes
zero policy loss and gradient and retains its place in the batch denominator.
The ratio, clipping fraction, and approximate KL diagnostics are averaged over
nonempty responses.

See equations 5--7 of the `GSPO paper <https://arxiv.org/abs/2507.18071>`_.
This implementation uses the original sequence objective with a shared advantage
per response, rather than the paper's GSPO-token variant.

Configuration
-------------

Start from ``examples/reasoning/config/math/qwen2.5-1.5b-gspo-fsdp.yaml``.
It inherits the GRPO FSDP example's dataset and placement settings, with a
smaller starting budget: 16 prompts, four responses per prompt, and 2,048 total
tokens (at most 512 prompt tokens). Increase the context budget for tasks that
need longer reasoning; truncation can change both rewards and training quality.
Set ``actor.model.model_path`` to your local model checkpoint; the rollout and
tokenizer paths follow it automatically. Also set
``data.train_data_paths`` and ``data.val_data_paths`` to your datasets.
Follow :doc:`../../examples/agentic/math_reasoning/reasoning_ppo` for the
reasoning environment and data preparation.

The GSPO settings select the new loss while retaining group-relative advantages:

.. code-block:: yaml

   runner:
     task_type: reasoning
     enable_dynamic_batch_size: False
   algorithm:
     adv_type: grpo
     loss_type: gspo
     loss_agg_func: seq-mean-token-mean
     group_size: 4
     normalize_advantages: False
     use_valid_token_scale: False
     importance_sampling_fix: False
     clip_ratio_low: 0.0003
     clip_ratio_high: 0.0004
     clip_ratio_c: null

The asymmetric clipping values follow the paper's experiment and should be tuned
for your task. Keep ``group_size`` greater than one. The GRPO estimator already
normalizes within groups; an additional batch-wide advantage normalization would
change the objective. Sequence-mean aggregation also determines how the actor
reduces optional entropy and reference-KL terms.

After editing the paths and placement, launch the existing reasoning runner:

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-gspo-fsdp

This runs rollout, group advantage calculation, and GSPO actor updates. The
example retains recomputation of old log probabilities on the actor. Track
``actor/ratio`` and ``actor/clip_fraction`` together with task rewards; these
diagnostics now describe complete responses.

Tuning the Configuration
------------------------

* ``algorithm.group_size`` controls responses per prompt and must exceed one.
  ``data.rollout_batch_size`` counts prompts, not responses.
* ``algorithm.n_minibatches`` controls optimizer minibatches per rollout.
  The runner derives the actor global batch size from
  ``rollout_batch_size * group_size / n_minibatches``. Keep this division exact.
* ``algorithm.training_batch_size_per_gpu`` controls the actual fixed
  microbatch size. The runner copies it to ``actor.micro_batch_size``; changing
  only the latter does not change the reasoning microbatch size. The derived
  global batch must be divisible by the microbatch size times actor world size.
* ``algorithm.clip_ratio_low`` and ``clip_ratio_high`` may differ. Setting
  either to ``null`` uses ``ratio_clip_eps`` for that side. Values must be finite
  and nonnegative, with the lower clipping width below one.
* ``algorithm.recompute_logprobs: True`` uses actor-recomputed old log
  probabilities. ``False`` uses rollout-provided log probabilities, exposing
  any numerical difference between the rollout and actor engines.
* ``algorithm.kl_beta`` and ``entropy_bonus`` retain the existing actor
  regularizers. They are additional terms, separate from the sequence policy
  objective. The example keeps fp32 actor parameters and optimizer state, with
  bf16 forward computation and rollout, and fp32 gradient reduction.

The current reasoning training loop does not perform held-out evaluation merely
because ``runner.val_check_interval`` is set. Evaluate exported checkpoints
separately; training reward is not held-out accuracy.

Supported Settings
------------------

Use fixed-size microbatches so gradient accumulation preserves the sequence
mean. Configuration validation rejects dynamic microbatches, token-count loss
scaling, token-wise importance-sampling correction, PPO dual clipping, and
log-ratio clamping for GSPO. The loss consumes complete response rows through
the reasoning actor interface; embodied action chunks are outside this scope.

The optional entropy and reference-KL terms use the existing actor reductions.
If enabled, they require nonempty response masks. The default example sets both
coefficients to zero.
