Training Pipeline Mode
======================

Overlap environment rollout with embodied FSDP actor training by setting ``runner.use_training_pipeline: True``.

With this mode enabled, environment workers pack rollout trajectories into micro-batches and stream them to actor workers while rollout is still running. Actor workers train on incoming micro-batches without waiting for the entire rollout epoch to finish.

Use this mode when rollout observations contain large tensors or when environment workers and actor workers sit on different nodes connected over a wide-area network, where smaller packed payloads reduce transfer overhead.

Configuration
-------------

Add the setting to an embodied FSDP training YAML:

.. code-block:: yaml

   runner:
     use_training_pipeline: True

   algorithm:
     adv_type: gae
     normalize_advantages: True

Limitations
-----------

- ``algorithm.adv_type`` must be ``gae``.
- ``algorithm.normalize_advantages`` normalizes advantages across the environment workers feeding each actor rank before streaming micro-batches.
- Only PPO and GRPO-style embodied FSDP actor losses are supported; ``embodied_sac``, ``embodied_dagger``, and ``embodied_nft`` are unsupported.
