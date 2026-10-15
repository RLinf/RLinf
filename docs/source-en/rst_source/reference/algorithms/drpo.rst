DRPO
====

Use Decoupled Reward Policy Optimization (DRPO) to train a reasoning model with
separate signals for answer correctness and response length. This page explains
the objective, the supported FSDP configuration, and how to check the resulting
accuracy–length trade-off.

Objective
---------

For each prompt, sample a group of responses and score correctness with a binary
verifier. Shorter correct responses receive more positive weight; incorrect
responses receive negative weight according to their current likelihood.
Length affects only the relative weights of correct responses.

Let :math:`s_i` be the mean current log probability over response tokens,
:math:`L_i` the response length, and :math:`C` the configured response capacity
(``runner.seq_length - data.max_prompt_length``). Within each prompt, define
the correct and incorrect sets as :math:`P` and :math:`N`:

.. math::

   w_i = \frac{\exp(-L_i/(C\lambda))}{\sum_{j\in P}\exp(-L_j/(C\lambda))},
   \qquad
   \ell_q = -\sum_{i\in P} w_i s_i
      + \tau\log\left(\frac{1}{|N|}\sum_{i\in N}\exp(s_i/\tau)\right).

Following the `authors' implementation <https://github.com/Optimization-AI/DRPO>`_,
groups without both classes contribute zero, while the average still includes
all prompt groups. Empty responses are excluded from both classes. Padding
does not contribute to sequence scores or KL.

The old-policy KL estimate :math:`K` is token-weighted across all actor ranks
in the synchronized micro-batch. The loss adds
:math:`\frac{\beta}{2}[K-\delta]_+^2`. Its gradient matches the official code's
detached :math:`\beta[K-\delta]_+` multiplier; the reported loss is the actual
penalized objective rather than that code's gradient surrogate. Gradient
accumulation averages these micro-batch objectives, including their individual
KL penalties. Changing the micro-batch size can therefore change the constraint
estimate, even with the same global batch size.

Configure and Run
-----------------

Start with ``examples/reasoning/config/math/qwen2.5-1.5b-drpo-fsdp.yaml`` after
following :doc:`../../examples/agentic/math_reasoning/reasoning` for the
reasoning environment, model and data preparation. Set the model/tokenizer
paths and training/validation data paths in the example, then run:

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-drpo-fsdp

The shared reasoning entry point samples responses, verifies answers, and
updates the actor with ``algorithm.loss_type: drpo``. The example uses SGLang,
collocated FSDP actors, and math-verifier rewards of 0 and 1. The ``raw``
advantage setting satisfies the shared runner interface; DRPO computes its
objective from rewards and current log probabilities directly.

Keep complete prompt groups together: ``training_batch_size_per_gpu`` must
be a multiple of ``group_size``, and each rank's rollout batch must divide
evenly into the configured mini-batches and micro-batches. Set
``shuffle_rollout``, ``normalize_advantages``, ``runner.enable_dynamic_batch_size``,
``actor.enable_dp_load_balance`` and ``actor.model.variable_seq_lengths`` to
false. The data loader can still shuffle prompts between training batches.
This implementation supports collocated reasoning FSDP training; these checks
prevent response-level shuffling or balancing from silently changing the groups.

.. list-table:: DRPO Parameters
   :header-rows: 1
   :widths: 35 15 50

   * - Key under ``algorithm``
     - Default
     - Meaning
   * - ``drpo_lambda``
     - 0.1
     - Positive length-weight temperature. Smaller values favor shorter correct answers; a large finite value approaches uniform positive weights (DisCO).
   * - ``drpo_tau``
     - 10.0
     - Positive negative-score temperature. Smaller values emphasize more likely incorrect responses.
   * - ``drpo_beta``
     - 1000.0
     - Nonnegative old-policy KL penalty coefficient. Zero disables this penalty.
   * - ``drpo_delta``
     - 0.0001
     - Nonnegative KL threshold before the penalty becomes active.
   * - ``drpo_kl_type``
     - ``kl``
     - ``kl`` uses mean old minus current log probability, as in the official training script; ``low_var_kl`` uses ``exp(current-old)-1-(current-old)``. The sampled ``kl`` estimate can be negative.

Check Training
--------------

Track ``actor/ppo_kl``, ``actor/drpo_constraint`` and
``actor/drpo_mixed_group_fraction`` alongside reward and response length.
If no prompt group has both correct and incorrect answers, the discriminative
term is zero. Use a model, task and sampling setup that produce mixed groups
before interpreting a zero gradient as an implementation failure.

Evaluate held-out accuracy and mean generated tokens together, using the same
model initialization, prompts, decoding settings and seed for the length-aware
run and a large-``drpo_lambda`` control. Report wall time and generated tokens;
a shorter response alone does not demonstrate a better quality–cost trade-off.
The example is a runnable integration configuration, not a reproduction of the
paper's 8k-context training or its reported gains.

Reference: `DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization
<https://arxiv.org/abs/2510.04474>`_, Eq. (7).
