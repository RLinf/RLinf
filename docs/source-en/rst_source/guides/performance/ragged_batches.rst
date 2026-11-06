Reduce Agent Batch Memory and Communication
===========================================

Keep agent trajectories at their real lengths until the loss needs a rectangular
microbatch. This guide shows how to enable that path for the dynamic Megatron
agent actor, compares its ten training stages with the dense path, and explains
how to reproduce the validation results with Search-R1.

Enable Real-Length Batches
--------------------------

Short answers otherwise occupy the configured maximum sequence length during
storage and DP redistribution. A ragged batch stores one tensor per sequence,
so sequences of 101, 201, and 301 tokens occupy 603 token positions before
model-parallel alignment. DP means data parallelism: model replicas train on
different portions of the rollout data.

Start from :doc:`the GLM-4.7-Flash recipe
</rst_source/examples/agentic/searchr1/glm4.7_flash>` and set these fields:

.. code-block:: yaml

   actor:
     enable_ragged_batch: True
     enable_dp_load_balance: True
   agentloop:
     is_dynamic_rollout_batch: True
   runner:
     enable_dynamic_batch_size: True
   algorithm:
     adv_type: grpo_dynamic
     advantage_mode: trajectory
     normalize_advantages: False

``actor.enable_ragged_batch`` selects the new data path in ``MAMegatronActor``;
an omitted or false value selects the dense path. The Search-R1 GLM example
enables it explicitly. ``actor.enable_dp_load_balance`` independently controls
whether question groups move between DP replicas. With balancing disabled,
each replica keeps its received groups and still uses real-length batches.

Use this path with the Megatron dynamic agent actor, dynamic microbatch sizing,
``grpo_dynamic``, disabled advantage normalization, and pipeline parallel size 1.
The GPU validation uses collocated actor/rollout placement, tensor parallel size
4, DP size 2, and context parallel size 1. CPU packing tests also cover context
parallel size 2; distributed multi-node and separate inference-worker layouts
need their own validation. These results describe this actor path, rather than
every task or training backend in RLinf.

Follow the Ten Training Stages
------------------------------

The tables below use the example's maximum sequence length of 8192. A turn is
one model response; a trajectory contains all turns and tool results for one
answer attempt. One question group contains the four answer attempts selected
by ``algorithm.group_size: 4``. Rows refer to turns until compatible turns of a
trajectory are merged. The configured maximum remains an upper bound in both
paths.

With ``actor.enable_ragged_batch: False``:

.. list-table:: Dense Path
   :header-rows: 1
   :widths: 6 26 68

   * - Step
     - Owner
     - Data and Operation
   * - 1
     - ``Searchr1AgentLoopWorker``
     - Generate, search, and score; return real-length lists in ``DynamicRolloutResult`` through the Channel, the queue between workers.
   * - 2
     - ``MAMegatronActor.get_batch``
     - Receive results and broadcast them to model-parallel peers.
   * - 3
     - ``DynamicRolloutResult.to_actor_batch``
     - Pad token fields such as input IDs, masks, and available rollout probabilities to [N, 8192]. Merge received batches.
   * - 4
     - ``MAMegatronActor.run_inference``
     - When enabled, recompute old probabilities: remove padding, pack aligned tokens, run the model, and restore [B, 8192] probabilities. Send results to the training stage.
   * - 5
     - ``MAMegatronActor.run_training``
     - Convert and merge batches, compute group advantages and loss scales, and merge compatible turns. Token fields retain width 8192.
   * - 6
     - ``RolloutDataBalance.from_rollout_batches_dynamic``
     - When balancing is enabled, collect CPU fields with all_gather_object on the DP process group, concatenate global data, and select local rows by length. NCCL stages the serialized objects on the GPU.
   * - 7
     - ``MAMegatronActor`` and batch iterators
     - Split optimizer minibatches and token-budget microbatches. Stored token fields retain width 8192.
   * - 8
     - ``MAMegatronActor.get_forward_step_func``
     - Remove padding and pack aligned tokens for the training forward pass; restore current probabilities to rectangular rows.
   * - 9
     - ``policy_loss``
     - Compare current and old probabilities with the advantages, scales, and response mask at matching positions.
   * - 10
     - Megatron forward/backward schedule and optimizer
     - Backpropagate through the forward computation, synchronize gradients, and update weights.

With ``actor.enable_ragged_batch: True``:

.. list-table:: Ragged Path
   :header-rows: 1
   :widths: 6 26 68

   * - Step
     - Owner
     - Data and Operation
   * - 1
     - ``Searchr1AgentLoopWorker``
     - Produce the same real-length question-group results and rewards.
   * - 2
     - ``MAMegatronActor.get_ragged_batches``
     - Data-I/O ranks receive raw groups. If balancing is enabled, redistribute them before broadcasting to model-parallel peers; otherwise retain the received assignment.
   * - 3
     - ``balance_rollout_groups`` and ``DynamicRolloutResult.to_ragged_actor_batch``
     - Gather group lengths, assign complete groups, exchange byte counts and variable-size remote payloads with all_to_all_single, then broadcast. Construct CPU tensor lists at real lengths. Local groups stay local.
   * - 4
     - ``MAMegatronActor.run_inference``
     - When enabled, pack real-length inputs and recompute old probabilities. Return real-length probability rows through the local Channel. Training reuses the DP assignment.
   * - 5
     - ``MAMegatronActor.run_training``
     - Compute advantages and loss scales from complete question groups; merge compatible turns as real-length rows and preserve source-DP normalization.
   * - 6
     - ``MAMegatronActor._pad_ragged_rows``
     - Add zero-loss placeholder rows only when needed for batch-count divisibility. DP redistribution has already occurred; token fields remain variable length.
   * - 7
     - Batch iterators and ``split_dynamic_batch_size``
     - Split minibatches and token-budget microbatches using real lengths and the model-parallel alignment requirement.
   * - 8
     - ``preprocess_ragged_seqs`` and ``MAMegatronActor.get_forward_step_func``
     - Align and pack current microbatch tokens, run the model, restore its output rows, and pad only the loss inputs to [B, 8192] with ``materialize_ragged_batch``.
   * - 9
     - ``policy_loss``
     - Use the existing rectangular loss interface and masking semantics. The ragged path disables the optional removal of entirely masked rows to preserve its normalization.
   * - 10
     - Megatron forward/backward schedule and optimizer
     - Use the native gradient synchronization and parameter-update path. Microbatch composition and floating-point reduction order can differ.

With recomputation enabled, step 2 runs when inference first receives the
rollout results. Training reads the local results and keeps that assignment.
With recomputation disabled, training receives the original results and performs
step 2 there. If both DP balancing is disabled and batches are ragged, neither
entry point redistributes the question groups.

Keep Question Groups Complete
-----------------------------

Early redistribution precedes advantage computation, so it moves a complete
question group. For rewards [1, 0, 0, 0], the question's mean reward is 0.25.
Splitting those attempts across replicas and independently using means of 0.5
and 0 changes GRPO, which compares answers to the same question. The group also
retains every turn belonging to those attempts. After advantages and group-based
loss scales are computed, minibatches can split the rows.

The loss uses dense masks to select the training positions. A mask controls
contribution to the loss; it does not by itself skip every arithmetic operation
on padding. Packing removes most sequence padding before the model forward
pass. Tensor/context parallelism can still require short alignment suffixes,
and loss tensors are materialized at the configured width for one microbatch.

Validate the Change
-------------------

Use the checked-in reproduction instructions in
`tests/e2e_tests/agent/ragged_batch/README.md
<https://github.com/RLinf/RLinf/blob/main/tests/e2e_tests/agent/ragged_batch/README.md>`_
to prepare paired configurations, run CPU contract checks, and replay the
recorded communication benchmark. The accompanying ``REPORT.md`` separates
communication measurements, fixed-trajectory correctness, and long-run results.

For an end-to-end comparison, hold the input rows, seed, model, retrieval
service, and algorithm settings constant and toggle only the ragged flag.
Independent sampled runs can still generate different answers. Compare frozen
rollout results separately when testing field preservation and numerical
agreement. Record memory during DP exchange separately from whole-device and
model-forward peaks.
