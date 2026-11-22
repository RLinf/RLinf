Training Pipeline Mode
======================

Overlap embodied rollout with FSDP actor training so GPUs do not sit idle waiting for an entire rollout epoch to finish. This guide covers two complementary levers: splitting rollout into pipeline stages (``rollout.pipeline_stage_num > 1``) and streaming packed actor micro-batches during rollout (``runner.use_training_pipeline: True``).

When to Use Pipeline Mode
-------------------------

Enable ``rollout.pipeline_stage_num > 1`` when simulator stepping (``env``) and policy inference (``rollout``) run on separate devices or overlap on the same node, and each stage takes comparable time. Dividing parallel environments into ``2`` or more stages lets rollout run on stage ``k`` while the simulator steps stage ``k + 1``.

Add ``runner.use_training_pipeline: True`` when:

- Rollout observations contain large image tensors and sending one full epoch buffer at once creates a transfer spike.
- Environment workers and actor workers sit on different GPUs or nodes (including wide-area network links), so streaming packed micro-batches overlaps communication and actor updates with ongoing rollout.

Prerequisites
-------------

- A working embodied FSDP training config using PPO or GRPO-style actor updates (see :doc:`Embodied Training Configuration <embodiment_config>`).
- ``env.train.total_num_envs`` (and ``env.eval.total_num_envs`` if evaluation is enabled) must be divisible by ``env_world_size * rollout.pipeline_stage_num``.

Minimal Configuration
---------------------

Add these keys to an embodied FSDP training YAML:

.. code-block:: yaml

   runner:
     use_training_pipeline: True

   rollout:
     pipeline_stage_num: 2

   algorithm:
     adv_type: gae
     normalize_advantages: True
     recompute_logprobs: False

- ``rollout.pipeline_stage_num``: number of chunks each env worker splits its parallel environments into (start with ``2``; keep ``1`` for models that disallow multi-stage pipelining).
- ``runner.use_training_pipeline``: packs completed trajectories into micro-batches on env workers and streams them to actor workers while rollout is still running.
- ``algorithm.normalize_advantages``: when ``True``, normalizes advantages across the env workers feeding each actor rank before streaming micro-batches.

Compatibility and Constraints
-----------------------------

.. warning::

   Config validation rejects incompatible combinations at startup. Check the rules below before enabling ``runner.use_training_pipeline: True``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Feature / setting
     - Compatibility with training pipeline mode
   * - Algorithm and loss
     - ``algorithm.adv_type`` must be ``gae`` and ``algorithm.recompute_logprobs`` must be ``False``. Only PPO and GRPO-style embodied FSDP actor losses are supported; ``embodied_sac``, ``rlt_ac``, ``rlt_td3``, ``embodied_dagger``, ``embodied_nft``, ``embodied_refl``, and ``embodied_art`` are unsupported.
   * - Model constraints
     - Some policies (such as ``molmoact2`` and Psi0 SIMPLE) require ``rollout.pipeline_stage_num: 1``.
   * - :doc:`Heterogeneous envs <hetero_envs>`
     - ``runner.use_training_pipeline: True`` requires a single ``env.train`` configuration and cannot be combined with heterogeneous training environments.
   * - :doc:`Observation compression <obs_compression>`
     - Compatible and recommended on bandwidth-limited or cross-node setups. Enabling ``env.obs_compression.enable: True`` shrinks image payloads on the env-to-rollout path while training pipeline mode overlaps rollout-to-actor transfers.
   * - :doc:`Env decoupled mode <env_decoupled_mode>`
     - ``rollout.pipeline_stage_num > 1`` uses fixed synchronous stages for uniform simulator step times. When simulator latency varies widely across environments, use :doc:`env_decoupled_mode` instead of increasing ``pipeline_stage_num``.

Verify Throughput Gain
----------------------

Run 10–20 steps with the baseline config (``pipeline_stage_num: 1``, ``use_training_pipeline: False``) and compare against the pipelined run in TensorBoard (see :doc:`Training Metrics </rst_source/reference/metrics>`):

1. Check ``time/step``: wall-clock step time should drop compared with the baseline when rollout and actor overlap effectively.
2. Inspect ``time/env``, ``time/rollout``, and ``time/actor``: if one stage dominates (> 80% of step time), adjust GPU placement (:doc:`Placement Strategies <../concepts/placement>`) before tuning ``pipeline_stage_num`` further.
3. Confirm ``env/success_once`` and reward curves match the baseline trend to verify that micro-batch streaming did not alter training stability.
