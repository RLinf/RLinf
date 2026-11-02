Heterogeneous Envs
==================

This page shows you how to train or evaluate one policy on several different env
configurations in the same run, such as ManiSkill and LIBERO, two LIBERO suites,
or a simulator next to a real robot. RLinf calls this heterogeneous envs.

It is different from ``total_num_envs``, which runs many parallel copies of one
env configuration, and from :doc:`hetero`, which decides which machines the
workers run on. You'll first see how heterogeneous envs run, then configure
them, decide which env workers run which env, balance training across them,
handle different observations, and read the results of each env.

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
configuration finishes its epochs early simply drops out of the batch, and the
training step ends when every env configuration has finished. The actor then
computes each env configuration's advantages with that configuration's settings
before it trains on all trajectories together.

Two things about the batch are worth knowing up front. Every env worker runs the
same number of parallel envs, which RLinf arranges for you. Observations, on the
other hand, may differ between env configurations: rollout runs the policy once
per observation format, so each forward pass sees inputs of one shape.

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
configuration; ``time/env/<name>/interact`` shows how long each one takes.

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
- ``algorithm.envs.<name>`` sets ``loss_weight``, ``gamma``, ``gae_lambda``,
  ``rewards_lower_bound``, or ``rewards_upper_bound`` for one env configuration;
  the rest come from ``algorithm``. ``loss_weight`` multiplies the weight of that
  env configuration's steps.
- ``adv_norm_scope: all`` normalizes advantages over all env configurations
  together instead of over each one. It only matters for advantage types that
  normalize over the batch, such as GAE; GRPO normalizes within each group.
- ``actor.mix_envs: False`` gives every optimizer step data from one env
  configuration, in a random order shared by all actor workers. Each env
  configuration's samples must then fill whole global batches.

Rewards keep each env's own scale. For PPO and SAC, whose critic serves every env
configuration, rewards of very different sizes let the larger one dominate the
critic, which gives the other env poor baselines. Set ``reward_scale`` on an env
configuration to bring the returns to a similar size; it multiplies that env's
rewards before training sees them, and ``huber_delta`` and ``value_clip`` then
act on the scaled values.

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

Only the policy's copy changes. Reward models, videos, and data collection still
see each env's original observations. Resizing costs time on every step, so
leave ``image_size`` out when the image sizes already match. ``policy_obs`` does
not resize robot states, so env configurations whose states differ in size keep
separate formats.

Actions go the other way without extra settings: each env worker converts its
share of the policy's actions for its own env. The policy still has one action
format and one action normalization (one ``unnorm_key`` for OpenVLA-OFT), so all
env configurations must accept the same kind of action.

Read Per-Env Results
--------------------

Each metric is logged twice: once for all envs together and once for each env
configuration. ``env/success_once`` covers every episode, and
``env/libero_goal/success_once`` covers LIBERO-Goal only. Evaluation metrics
follow the same pattern under ``eval/``.

Videos are saved under ``<video_base_dir>/seed_<n>/<name>/``, so env
configurations that share a video folder do not overwrite each other.

Run the Example
---------------

The end-to-end test ``maniskill_libero_grpo_openvlaoft`` trains OpenVLA-OFT with
GRPO on LIBERO-Goal and ManiSkill and evaluates on both. It needs both simulators
in one virtual environment, the checkpoint and LoRA adapter named in the config,
and 2 GPUs.

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh maniskill_libero_grpo_openvlaoft

What this does:

1. Installs OpenVLA-OFT with ManiSkill and LIBERO.
2. Runs LIBERO-Goal on env worker 0 and ManiSkill on env worker 1.
3. Runs two GRPO steps and logs ``env/<name>/*`` and ``eval/<name>/*`` to
   TensorBoard under ``runner.logger.log_path``.

The test uses 8-step episodes to check that everything connects. For real
training, raise the episode lengths and env counts. The ManiSkill scene also
needs its assets, as described in :doc:`../examples/embodied/maniskill`.

Limitations
-----------

- The policy has one action format and one action normalization, and it does not
  know which env a sample came from.
- NFT training needs every env configuration to send one observation format;
  use ``policy_obs`` to make them match.
- Heterogeneous training runs with the synchronous on-policy runner (PPO, GRPO,
  and similar). Async training, off-policy training (SAC, DAgger, RLT),
  env-decoupled mode, and the training pipeline need a single env
  configuration.
- PPO and SAC share one critic across env configurations; RLinf does not
  normalize rewards per env, so set ``reward_scale`` when reward scales differ.
- Smooth intervention (``smooth_intervene``) and real-time control evaluation
  (``runner.rtc``) work with a single env configuration only, and RLT simulator
  replay needs every train env configuration to be ``maniskill_rlt``. RLinf
  stops at startup when a config breaks these rules.
