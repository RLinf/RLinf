Heterogeneous Envs
==================

This page shows you how to train or evaluate one policy on several different env
configurations in the same run, such as ManiSkill and LIBERO, two LIBERO suites,
or a simulator next to a real robot. RLinf calls this heterogeneous envs.

It is different from ``total_num_envs``, which runs many parallel copies of one
env configuration, and from :doc:`hetero`, which decides which machines the
workers run on. You'll first see how heterogeneous envs run, then configure
them, decide which env workers run which env, balance training across them,
handle different observations and action statistics, and read the results of
each env.

How It Works
------------

Normally every env worker runs the same env configuration. With heterogeneous
envs, each env worker still runs exactly one env configuration, but different
workers can run different ones.

Each env configuration also keeps its own loop: how many steps a rollout epoch
takes, how many epochs a training step runs, and when its episodes end. Rollout
collects the observations of all env workers into one batch, runs the policy
once, and sends each worker the actions for its own envs. Each worker converts
those actions for its simulator or robot and steps it. A worker whose env
configuration finishes its epochs early drops out of the batch, and the
training step ends when every env configuration has finished. The actor then
computes each env configuration's advantages with that configuration's settings
before it trains on all trajectories together.

Two things about the batch are worth knowing up front. Every env worker runs the
same number of parallel envs, which RLinf arranges for you. Observations, on the
other hand, may differ between env configurations: rollout runs the policy once
per observation format, so each forward pass sees inputs of one shape. Env
configurations that turn the policy's output into actions with different
statistics also get separate forward passes.

Configure Heterogeneous Envs
----------------------------

List the env configurations under ``env.train.envs``, each under a name you
choose. This config trains on LIBERO-Goal and a ManiSkill scene with 64 parallel
envs each:

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

Each ``defaults`` line loads an env preset under a name, here ``libero_goal`` and
``maniskill``. Each env configuration takes the same settings as a normal
``env.train``, and they may all differ between env configurations: its
``total_num_envs`` is its own number of parallel envs, and its
``rollout_epoch``, ``max_steps_per_rollout_epoch``, ``max_episode_steps``,
``auto_reset``, ``ignore_terminations``, and ``group_size`` apply to its own envs
only. Here LIBERO-Goal runs one 320-step epoch per training step, and ManiSkill
runs two 80-step epochs. ``env.eval`` works the same way and is independent, so
you can train on two env configurations and evaluate on one.

Put every other setting, such as videos or rewards, inside the env configuration
it belongs to. RLinf fills in the totals, such as ``env.train.total_num_envs``,
on its own, and stops at startup with a clear message when settings conflict.

Assign Env Workers
------------------

RLinf splits the env workers among the env configurations in the order you list
them, in proportion to their ``total_num_envs``. Say ``env`` runs on 4 workers,
with 48 LIBERO-Goal envs and 16 ManiSkill envs. Each worker then runs 16 envs:

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

Each env configuration must fill whole workers. On 2 workers, LIBERO-Goal would
need one and a half, so RLinf rejects the config; change the counts or the number
of workers.

On its workers, an env configuration behaves as if it ran alone: seeds and reset
states are split among its own workers, and an ``override_cfgs`` list has one
item per worker of that env configuration.

The actor workers split each env configuration's trajectories evenly among
themselves, so every actor worker trains on the same mix of envs. A GRPO group
must stay on one actor worker, so each env configuration's groups must divide
evenly among the actor workers; RLinf checks this at startup.

The ``env`` placement decides where these workers run. Several env workers can
share one GPU, for example ``env: 0:0-1``. To drive a real robot, place the
workers of that env configuration on the robot's node; :doc:`hetero` shows how.

.. warning::

   Workers follow the listing order. Reordering ``envs`` or changing a count
   changes which env each worker runs, so check the placement afterwards,
   especially with real robots.

Balance Training Across Envs
----------------------------

One training step holds ``total_num_envs`` times ``rollout_epoch`` trajectories
of each env configuration, where a trajectory is one env's rollout epoch. In the
example above, that is 64 LIBERO-Goal trajectories of up to 320 steps and 128
ManiSkill trajectories of 80 steps. These counts set how much each env
configuration contributes, and a step takes as long as its slowest env
configuration. To see how long each one takes, set ``runner.per_worker_log: True``
and read ``time/env/interact`` of the env workers that run it.

By default every trajectory counts the same in the loss, however long its env's
episodes are, so ManiSkill gets two thirds of the gradient in the example. These
settings change that:

.. code-block:: yaml

   algorithm:
     env_weighting: trajectory  # or sample
     adv_norm_scope: env        # or all
     envs:
       maniskill:
         loss_weight: 0.5
   actor:
     mix_envs: True

- ``env_weighting: sample`` counts every step the same instead, so an env
  configuration with longer episodes weighs more.
- ``algorithm.envs.<name>`` sets ``loss_weight``, ``reward_scale``, ``gamma``,
  ``gae_lambda``, ``rewards_lower_bound``, or ``rewards_upper_bound`` for one env
  configuration; the rest come from ``algorithm``. ``loss_weight`` multiplies the
  weight of that env configuration's steps.
- ``adv_norm_scope: all`` normalizes advantages over all env configurations
  together instead of over each one. It applies to the advantage types that
  normalize over the batch, ``gae``, ``gae_dynamic``, and ``raw``, with
  ``normalize_advantages: True``. GRPO normalizes within each group, so RLinf
  rejects ``all`` for it.
- ``actor.mix_envs: False`` gives every optimizer step data from one env
  configuration, in a random order shared by all actor workers. Each env
  configuration's samples must then fill whole global batches.

Async and off-policy training keep the same mix. In async PPO, each env
configuration gets its share of ``algorithm.rollout_store_size_per_rank``, in
proportion to the trajectories it sends each actor, and a training step waits
until every share is full, so a fast env cannot crowd out a slow one.
Off-policy training (SAC, DAgger, RLT) keeps one replay buffer per env
configuration and samples them in proportion to the transitions each collects
per step, times ``loss_weight``; with ``actor.mix_envs: False`` each batch
comes from one env configuration. Training starts once every env
configuration's buffer holds ``min_buffer_size`` trajectories.

Rewards keep each env's own scale. For PPO and SAC, whose critic serves every env
configuration, rewards of very different sizes let the larger one dominate the
critic, which gives the other env poor baselines. Set
``algorithm.envs.<name>.reward_scale`` to bring the returns to a similar size;
it multiplies that env's rewards before training sees them, and ``huber_delta``
and ``value_clip`` then act on the scaled values.

Handle Different Observations
-----------------------------

Envs from different simulators usually send different observations: LIBERO
sends a main camera and a wrist camera at 256x256, while the ManiSkill scene
sends one 480x640 camera. RLinf groups the requests of each rollout round by
observation format, meaning the observation keys and the shape of each one, and
runs one forward pass per format. The actor trains each micro-batch on a single
format too. Any policy therefore sees one shape at a time, but it must accept
every format it receives.

Separate formats cost speed: each forward pass is smaller, and a round takes one
pass per format. Set ``policy_obs`` to make the formats match, so all envs share
one forward pass again:

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

``keys`` lists the observations the policy reads; the others are not sent.
``image_size`` resizes every camera image to one height and width. Here both
envs send a 256x256 main camera image and the task text, which is what
OpenVLA-OFT reads.

Only the policy's copy changes. Videos and data collection still see each env's
original observations. Resizing costs time on every step, so
leave ``image_size`` out when the image sizes already match. ``policy_obs`` does
not resize robot states, so env configurations whose states differ in size keep
separate formats.

Handle Different Action Statistics
----------------------------------

Actions go the other way: each env worker converts its share of the policy's
actions for its own env, so the policy keeps one action format. OpenVLA-OFT
also turns its normalized output into robot actions with the statistics of one
training dataset, named by ``actor.model.unnorm_key``. When an env
configuration's robot comes from a dataset with other statistics, set
``unnorm_key`` on that env configuration, in both train and eval:

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

ManiSkill actions now use the Bridge statistics, and env configurations without
``unnorm_key`` keep ``actor.model.unnorm_key``. Rollout runs env configurations
with different keys in separate forward passes, as it does for different
observation formats. Training is unchanged, because OpenVLA-OFT computes log
probabilities on its normalized action tokens.

Only OpenVLA-OFT reads a per-env ``unnorm_key``; RLinf stops at startup when
another policy would ignore it.

Read Per-Env Results
--------------------

Env, eval, and rollout metrics are logged twice: once for all envs together and
once for each env configuration. ``env/success_once`` covers every episode, and
``env/libero_goal/success_once`` covers LIBERO-Goal only; evaluation metrics
follow the same pattern under ``eval/``. Training and timing metrics, such as
``train/actor/*`` and ``time/*``, cover all envs together. Group-success and
data-staleness metrics depend on each env's groups and rollout store, so they
are logged for each env configuration only.

Videos are saved under ``<video_base_dir>/seed_<n>/<name>/``, so env
configurations that share a video folder do not overwrite each other.

Run the Example
---------------

The training example ``maniskill_libero_grpo_openvlaoft`` trains one OpenVLA-OFT
policy with GRPO on LIBERO-10 and a ManiSkill scene, on 4 GPUs. Install both
simulators in one virtual environment:

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate

Download the ManiSkill assets as described in
:doc:`../examples/embodied/maniskill`. Then download the base checkpoint and the
ManiSkill LoRA adapter:

.. code-block:: bash

   # Set HF_ENDPOINT=https://hf-mirror.com in mainland China.
   hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main --local-dir RLinf-OpenVLAOFT-ManiSkill-Base-Main
   hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir RLinf-OpenVLAOFT-ManiSkill-Base-Lora

In ``examples/embodiment/config/maniskill_libero_grpo_openvlaoft.yaml``, set
``actor.model.model_path`` and ``rollout.model.model_path`` to the checkpoint and
the ``path`` under ``actor.model.merged_loras`` to the adapter, then launch:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_libero_grpo_openvlaoft

GRPO learns only when the policy succeeds sometimes in every env, and no
released checkpoint does that in both: LIBERO checkpoints never grasp in
ManiSkill, and the ManiSkill policy fails on LIBERO-10. The example therefore
starts from the LIBERO-10 base policy with half of the ManiSkill LoRA folded
into its weights (``merged_loras`` with ``scale: 0.5``), which succeeds sometimes
in both, and trains a new LoRA on top. Watch ``env/libero_10/success_once`` and
``env/maniskill/success_once`` rise.

To check that everything connects before a long run, the end-to-end test of the
same name runs two GRPO steps on LIBERO-Goal and ManiSkill with 8- and 16-step
episodes on 2 GPUs. Point the checkpoint and adapter paths in
``tests/e2e_tests/embodied/maniskill_libero_grpo_openvlaoft.yaml`` at your
copies first:

.. code-block:: bash

   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh maniskill_libero_grpo_openvlaoft

Limitations
-----------

- The policy has one action format and does not know which env a sample came
  from. Only OpenVLA-OFT can unnormalize actions per env configuration.
- NFT training needs every env configuration to send one observation format;
  use ``policy_obs`` to make them match. NFT also weights every valid step the
  same, so it needs ``algorithm.env_weighting: sample`` and no per-env
  ``loss_weight``.
- Off-policy training with ``actor.mix_envs: True`` mixes env configurations in
  one batch, so they must store one observation format; use ``policy_obs``.
- The training pipeline (``runner.use_training_pipeline``), online LeRobot
  collection for DAgger, DiffusionART training, reward models
  (``reward.use_reward_model``), and every rollout backend except
  ``huggingface``, such as PhyAI, SGLang, and ApxInf, need a single env
  configuration.
- PPO and SAC share one critic across env configurations; RLinf does not
  normalize rewards per env, so set ``algorithm.envs.<name>.reward_scale`` when
  reward scales differ.
  SAC tunes one entropy temperature per env configuration, except for samples
  mixed with a demo buffer, which use their mean.
- In env-decoupled mode, each rollout worker serves its share of a step's
  requests from a shared queue and runs one forward pass per observation
  format in what it receives.
- Smooth intervention (``smooth_intervene``) and real-time control evaluation
  (``runner.rtc``) work with a single env configuration only, and RLT simulator
  replay needs every train env configuration to be ``maniskill_rlt``. RLinf
  stops at startup when a config breaks these rules.
