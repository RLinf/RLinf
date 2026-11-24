Multi-Env Training
==================

Train or evaluate one policy on multiple environment configurations in a single run, such as ManiSkill and LIBERO together, two LIBERO suites, or a simulator alongside a real robot.

Configure Envs
--------------

List each environment configuration under ``env.train.envs`` with a custom name. The example below trains on LIBERO-Goal and a ManiSkill task with 64 parallel envs each:

.. code-block:: yaml

   defaults:
     - env/libero_goal@env.train.envs.libero_goal
     - env/maniskill_put_on_plate_in_scene_25_main@env.train.envs.maniskill
     # model, FSDP, and weight-syncer presets as usual

   env:
     train:
       envs:
         libero_goal:
           total_num_envs: 64
           rollout_epoch: 1
           max_episode_steps: 320
           max_steps_per_rollout_epoch: 320
           group_size: 8
         maniskill:
           total_num_envs: 64
           rollout_epoch: 2
           max_episode_steps: 80
           max_steps_per_rollout_epoch: 80
           group_size: 8

Each entry under ``envs`` sets its own fields independently—``total_num_envs``, ``rollout_epoch``, ``max_episode_steps``, ``max_steps_per_rollout_epoch``, ``auto_reset``, ``ignore_terminations``, ``group_size``, rewards, and video settings. ``env.eval.envs`` uses the same format independently.

Env workers are assigned in listing order in proportion to ``total_num_envs``:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Env configuration
     - ``total_num_envs``
     - Env workers (4 total)
   * - ``libero_goal``
     - 48
     - 0, 1, 2
   * - ``maniskill``
     - 16
     - 3

Each env configuration must fill whole env workers, and its GRPO groups must divide evenly across actor workers. Use ``cluster.component_placement.env`` to place workers on specific GPUs or robot nodes (see :doc:`hetero`). Reordering ``envs`` or changing env counts shifts worker assignments, so recheck placement after either change.

Align Observations and Action Statistics
----------------------------------------

When environments emit different camera keys or image resolutions, set ``policy_obs`` to pass one observation format to the policy:

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

``keys`` selects the fields sent to the policy, and ``image_size`` resizes camera images to one height and width while video recording keeps the original frames. Putting environments in one observation format lets them share an actor training pool; requests also share a rollout forward pass when they reach the same rollout worker with the same policy fields (such as ``unnorm_key``).

For OpenVLA-OFT, set per-env ``unnorm_key`` under both ``env.train.envs`` and ``env.eval.envs`` when an environment needs different action statistics from ``actor.model.unnorm_key``:

.. code-block:: yaml

   env:
     train:
       envs:
         maniskill:
           unnorm_key: bridge_orig
     eval:
       envs:
         maniskill:
           unnorm_key: bridge_orig

Balance Training
----------------

Each step collects ``total_num_envs * rollout_epoch`` trajectories per environment. By default, every trajectory has equal loss weight: in the opening configuration, ``64 * 1 = 64`` LIBERO-Goal trajectories and ``64 * 2 = 128`` ManiSkill trajectories mean ManiSkill supplies two-thirds of the equally weighted trajectories (actual gradient contribution depends on their gradients). Adjust the mix with these options:

.. code-block:: yaml

   algorithm:
     env_weighting: trajectory  # or sample
     adv_norm_scope: env        # or all
     envs:
       maniskill:
         loss_weight: 0.5
         reward_scale: 1.0
   actor:
     mix_envs: True

- ``algorithm.env_weighting``: ``trajectory`` (default) weights every trajectory equally; ``sample`` scales each environment's trajectory weight by its nominal horizon (``max_episode_steps`` for masked runs, ``max_steps_per_rollout_epoch`` for unmasked runs).
- ``algorithm.envs.<name>``: overrides ``loss_weight`` (must be positive), ``reward_scale``, ``gamma``, ``gae_lambda``, ``rewards_lower_bound``, or ``rewards_upper_bound`` for one environment. Set ``reward_scale`` when PPO or SAC reward magnitudes differ across environments.
- ``algorithm.adv_norm_scope``: ``env`` (default) normalizes each environment separately; ``all`` normalizes across all environments together (supported for ``gae`` and ``raw`` with ``normalize_advantages: True``).
- ``actor.mix_envs``: ``True`` (default) mixes environments across optimizer steps; ``False`` puts one environment in each optimizer step (each environment's per-actor samples must fill whole global batches).

For Async PPO, ``algorithm.rollout_store_size_per_rank`` is measured in received trajectory shards per actor rank and must divide into integer quotas across environments (for example, set it to ``2`` for two environments with equal shard shares instead of the default ``1``). Off-policy replay (SAC, DAgger, RLT) allocates samples by ``total_num_envs * rollout_epoch * chunk_steps * loss_weight`` (``env_weighting`` does not change that ratio), and requires ``policy_obs`` when ``actor.mix_envs: True`` so all environments store one observation format.

Run the Example
---------------

The ``maniskill_libero_grpo_openvlaoft`` config co-trains one OpenVLA-OFT policy with GRPO on LIBERO-10 and ManiSkill.

Prerequisites:

- **Hardware**: 4 GPUs (``actor,env,rollout: 0-3``).
- **ManiSkill assets**: downloaded per :doc:`../examples/embodied/maniskill`.
- **Checkpoints**: ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` and ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora``. The config merges half of the ManiSkill LoRA into the base checkpoint (``merged_loras`` with ``scale: 0.5``) and trains a new LoRA adapter on top.

1. Install both simulators into one virtual environment:

   .. code-block:: bash

      bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

2. Download the checkpoints (set ``HF_ENDPOINT=https://hf-mirror.com`` in mainland China):

   .. code-block:: bash

      hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main --local-dir RLinf-OpenVLAOFT-ManiSkill-Base-Main
      hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir RLinf-OpenVLAOFT-ManiSkill-Base-Lora

3. In ``examples/embodiment/config/maniskill_libero_grpo_openvlaoft.yaml``, point three fields to your local paths:

   - ``actor.model.model_path``: ``RLinf-OpenVLAOFT-ManiSkill-Base-Main``
   - ``rollout.model.model_path``: ``RLinf-OpenVLAOFT-ManiSkill-Base-Main``
   - ``actor.model.merged_loras[0].path``: ``RLinf-OpenVLAOFT-ManiSkill-Base-Lora``

4. Launch training:

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh maniskill_libero_grpo_openvlaoft

5. Verify both environments in the terminal table or TensorBoard (under ``logs/<timestamp>-maniskill_libero_grpo_openvlaoft/``): look for ``env/libero_10/success_once`` (starting around 0.2) and ``env/maniskill/success_once`` (starting around 0.09) alongside the pooled ``env/success_once``. Eval and rollout metrics use the same ``<group>/<name>/<metric>`` naming, and videos save to ``<video_base_dir>/seed_<n>/<name>/``. To check per-env step latency, set ``runner.per_worker_log: True`` and read ``time/env/interact``.

Supported Scope
---------------

- Rollout backend: ``rollout.rollout_backend: huggingface``.
- Per-env ``unnorm_key``: OpenVLA-OFT only.
- NFT: requires ``policy_obs`` and ``algorithm.env_weighting: sample``, without per-env ``loss_weight``.
- Single-env only features: ``runner.use_training_pipeline``, online LeRobot collection for DAgger, DiffusionART, ``reward.use_reward_model``, ``smooth_intervene``, and ``runner.rtc``. RLT simulator replay requires all train environments to be ``maniskill_rlt``.
