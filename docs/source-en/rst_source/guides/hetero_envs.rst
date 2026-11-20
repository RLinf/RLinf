Heterogeneous Envs
==================

Train or evaluate one policy on several env configurations in the same run, such as ManiSkill and LIBERO, two LIBERO suites, or a simulator next to a real robot. RLinf calls this heterogeneous envs.

It differs from ``total_num_envs``, which runs parallel copies of one env configuration, and from :doc:`hetero`, which assigns components to machines.

How It Works
------------

Each env worker still runs one env configuration, with the same number of parallel envs on every worker, while different workers run different configurations. Each env configuration keeps its own rollout loop: how many steps an epoch runs, how many epochs a training step takes, and when episodes end.

During rollout, a rollout worker serves the env workers routed to it and drops each env worker from the batch once its epochs finish. Requests share one forward pass only when they reach the same rollout worker and match on both observation format and policy fields such as ``unnorm_key``; otherwise the rollout worker runs separate forward passes per group.

After every env configuration finishes the step, the actor computes each env configuration's advantages using its own settings, then trains the shared policy on all trajectories together.

Configure Heterogeneous Envs
----------------------------

List the env configurations under ``env.train.envs``, each under a name you choose. This config trains on LIBERO-Goal and a ManiSkill scene with 64 parallel envs each:

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
           ignore_terminations: True
           group_size: 8

Each ``defaults`` line loads an env preset under a name, here ``libero_goal`` and ``maniskill``. Each env configuration accepts all fields of ``env.train`` independently: ``total_num_envs``, ``rollout_epoch``, ``max_steps_per_rollout_epoch``, ``max_episode_steps``, ``auto_reset``, ``ignore_terminations``, and ``group_size`` apply only to its own envs. ``env.eval`` works the same way and is independent, so you can train on two env configurations and evaluate on one.

Put every other env-specific setting, such as videos or rewards, inside the env configuration it belongs to. RLinf derives mode totals such as ``env.train.total_num_envs`` automatically and stops at startup when settings conflict.

Assign Env Workers
------------------

RLinf splits env workers among the listed env configurations in listing order, proportional to ``total_num_envs``. With 4 env workers, 48 LIBERO-Goal envs, and 16 ManiSkill envs, each worker runs 16 envs:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Env configuration
     - ``total_num_envs``
     - Env workers
   * - ``libero_goal``
     - 48
     - 0, 1, 2
   * - ``maniskill``
     - 16
     - 3

Each env configuration must fill whole env workers, and each env configuration's GRPO groups must divide evenly among the actor workers; RLinf checks both at startup. Seeds, reset states, and ``override_cfgs`` split within each env configuration's own workers.

The ``env`` placement controls where these workers run. Multiple env workers can share one GPU, such as ``env: 0:0-1``. To drive a real robot, place that env configuration's workers on the robot's node (see :doc:`hetero`).

.. warning::

   Workers are assigned in listing order. Reordering ``envs`` or changing an env count moves env configurations between ranks, so recheck your placement after either change, especially with a real robot.

Balance Training Across Envs
----------------------------

Each training step collects ``total_num_envs * rollout_epoch`` trajectories per env configuration, and the slowest env configuration sets the step time. To inspect per-env timing, set ``runner.per_worker_log: True`` and read ``time/env/interact`` on each env worker.

By default, every trajectory carries the same weight in the loss regardless of horizon, so in the 64-vs-128 example above ManiSkill takes two-thirds of the gradient. Adjust the balance with these settings:

.. code-block:: yaml

   algorithm:
     env_weighting: trajectory  # or sample
     adv_norm_scope: env        # or all
     envs:
       maniskill:
         loss_weight: 0.5
   actor:
     mix_envs: True

- ``env_weighting: sample`` scales each env configuration's trajectory weight by its own nominal horizon (``max_episode_steps`` for masked runs or ``max_steps_per_rollout_epoch`` for unmasked runs), so env configurations with longer horizons get a larger share of the loss while early-terminated trajectories inside an env keep that env's per-trajectory normalization.
- ``algorithm.envs.<name>`` overrides ``loss_weight``, ``reward_scale``, ``gamma``, ``gae_lambda``, ``rewards_lower_bound``, or ``rewards_upper_bound`` for one env configuration while inheriting the rest from ``algorithm``. ``loss_weight`` must be positive and scales that env's loss contribution.
- ``adv_norm_scope: all`` normalizes advantages over all env configurations together instead of separately. It applies to batch-normalized advantage types—``gae`` and ``raw`` with ``normalize_advantages: True``—and is rejected for GRPO, which normalizes within each group.
- ``actor.mix_envs: False`` puts one env configuration in each optimizer step, in a random order shared across actor ranks. Each env configuration's samples per actor must fill whole global batches.

Async and off-policy training keep the same proportions. In Async PPO, each env configuration gets a share of ``algorithm.rollout_store_size_per_rank`` proportional to its trajectories per step, and the actor waits until every share is full before updating. Off-policy training (SAC, DAgger, RLT) keeps one replay buffer per env configuration, waits until every buffer holds ``min_buffer_size`` trajectories, and samples in proportion to each env's collected transitions times ``loss_weight`` (or draws one env per batch when ``actor.mix_envs: False``).

Rewards keep each env's scale. Because PPO and SAC share one critic across env configurations, set ``algorithm.envs.<name>.reward_scale`` when reward magnitudes differ so one env does not dominate the critic. The scale multiplies rewards before ``huber_delta`` and ``value_clip`` apply.

Handle Different Observations
-----------------------------

Different simulators often produce different observations: LIBERO sends 256x256 main and wrist views, while ManiSkill sends a single 480x640 view. Rollout groups requests by observation format (keys and shapes) and policy fields, and the actor keeps each micro-batch to one observation format, so the policy sees one shape per forward pass.

To keep different simulators in one observation format, set ``policy_obs``:

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

``keys`` selects the observations sent to the policy, and ``image_size`` resizes all camera images to one height and width. Video recording and data collection still receive the original observations. Omit ``image_size`` when cameras already match, because resizing runs every step. ``policy_obs`` does not resize robot state vectors.

Matching observation formats removes only one reason to split rollout forward passes. Requests still share a rollout forward pass only when they also land on the same rollout rank and carry the same policy fields (such as ``unnorm_key``). For example, the shipped ``maniskill_libero_grpo_openvlaoft`` config maps one env worker to each rollout worker and gives ManiSkill a different ``unnorm_key``, so its two envs run separate rollout forward passes even with ``policy_obs`` set, while ``policy_obs`` still lets them share one actor training pool.

Handle Different Action Statistics
----------------------------------

Each env worker converts actions into its own simulator's format, so the policy outputs one action format. OpenVLA-OFT also unnormalizes actions with dataset statistics selected by ``actor.model.unnorm_key``. When one env configuration needs different statistics, set ``unnorm_key`` on that env configuration for both training and evaluation:

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

ManiSkill then uses the Bridge dataset statistics while other env configurations fall back to ``actor.model.unnorm_key``. Rollout runs separate forward passes for envs with different ``unnorm_key`` values; actor training is unaffected because OpenVLA-OFT computes log-probabilities on normalized action tokens. Only OpenVLA-OFT reads per-env ``unnorm_key``; other policies fail validation at startup if it is set.

Read Per-Env Results
--------------------

Env, eval, and rollout metrics are logged both pooled (``env/success_once``, ``eval/success_once``) and per env configuration (``env/<name>/success_once``, ``eval/<name>/success_once``). Training and timing metrics (``train/actor/*``, ``time/*``) are pooled. Group-success and data-staleness metrics depend on each env's groups and rollout store, so they are logged per env only.

Videos are written to ``<video_base_dir>/seed_<n>/<name>/`` so env configurations never overwrite each other's files.

Run the Example
---------------

The ``maniskill_libero_grpo_openvlaoft`` example co-trains one OpenVLA-OFT policy with GRPO on LIBERO-10 and ManiSkill.

Prerequisites:

- **Hardware**: 4 GPUs (2 env workers for LIBERO-10 and 2 for ManiSkill, co-located with actor and rollout across GPUs 0–3).
- **ManiSkill assets**: downloaded according to :doc:`../examples/embodied/maniskill`.
- **Checkpoints**: ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` (a LIBERO-10 base policy) and ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora`` (the ManiSkill LoRA adapter trained on it). The config folds half of the ManiSkill LoRA into the base weights (``merged_loras`` with ``scale: 0.5``) so the starting policy succeeds occasionally in both envs, giving GRPO a learning signal in each, and trains a fresh LoRA adapter on top.

Install both simulators in one virtual environment:

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate

Download the base checkpoint and ManiSkill LoRA adapter:

.. code-block:: bash

   # Set HF_ENDPOINT=https://hf-mirror.com in mainland China.
   hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main --local-dir RLinf-OpenVLAOFT-ManiSkill-Base-Main
   hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir RLinf-OpenVLAOFT-ManiSkill-Base-Lora

In ``examples/embodiment/config/maniskill_libero_grpo_openvlaoft.yaml``, point three fields to your local paths:

- ``actor.model.model_path``: path to ``RLinf-OpenVLAOFT-ManiSkill-Base-Main``
- ``rollout.model.model_path``: path to ``RLinf-OpenVLAOFT-ManiSkill-Base-Main``
- ``actor.model.merged_loras[0].path``: path to ``RLinf-OpenVLAOFT-ManiSkill-Base-Lora``

Launch training:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_libero_grpo_openvlaoft

Once training starts, confirm both envs are active by checking the step table and TensorBoard for ``env/libero_10/success_once`` (starting around 0.2) and ``env/maniskill/success_once`` (starting around 0.09) alongside the pooled ``env/success_once``.

Limitations
-----------

- The policy uses one action format and does not receive an env identifier. Only OpenVLA-OFT supports per-env ``unnorm_key``.
- NFT training requires one observation format across envs (set ``policy_obs``), ``algorithm.env_weighting: sample``, and no per-env ``loss_weight``.
- Off-policy training with ``actor.mix_envs: True`` mixes envs in one batch, so all envs must store one observation format (set ``policy_obs``).
- ``runner.use_training_pipeline``, online LeRobot collection for DAgger, DiffusionART training, ``reward.use_reward_model``, ``smooth_intervene``, ``runner.rtc``, and rollout backends other than ``huggingface`` (PhyAI, SGLang, ApxInf) require a single env configuration. RLT simulator replay requires every train env configuration to be ``maniskill_rlt``.
- PPO and SAC share one critic across env configurations; use ``algorithm.envs.<name>.reward_scale`` when reward scales differ. SAC tunes one entropy temperature per env configuration, while demo-buffer samples use the mean temperature.
- In env-decoupled mode, rollout workers serve their share of the current step from a shared queue and run one forward pass per observation format and policy-field group in each batch.
