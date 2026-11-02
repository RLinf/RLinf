Heterogeneous Envs
==================

This page shows you how to train or evaluate one policy on several different env
configurations in the same run, such as ManiSkill and LIBERO, two LIBERO suites,
or a simulator next to a real robot. RLinf calls this heterogeneous envs.

It is different from ``total_num_envs``, which runs many parallel copies of one
env configuration, and from :doc:`hetero`, which decides which machines the
workers run on. You'll first see how heterogeneous envs run, then configure
them, decide which env workers run which env, make the observations fit the
policy, and read the results of each env.

How It Works
------------

Normally every env worker runs the same env configuration. With heterogeneous
envs, each env worker still runs exactly one env configuration, but different
workers can run different ones.

The rest of the training loop stays the same. Rollout collects the observations
of all env workers into one batch, runs the policy once, and sends each worker
the actions for its own envs. Each worker converts those actions for its
simulator or robot and steps it. The actor then trains on all trajectories
together.

Because everything meets in one batch, two rules apply:

- Every env worker runs the same number of parallel envs.
- Every env configuration sends the policy the same kinds of observations at the
  same image size.

The sections below show how RLinf handles the first rule for you and how you
handle the second one.

Configure Heterogeneous Envs
----------------------------

List the env configurations under ``env.train.envs``, each under a name you
choose. This config trains on LIBERO-Goal and a ManiSkill scene with 8 parallel
envs each:

.. code-block:: yaml

   defaults:
     - env/libero_goal@env.train.envs.libero_goal
     - env/maniskill_put_on_plate_in_scene_25_main@env.train.envs.maniskill
     # model, FSDP, and weight-syncer presets as usual

   env:
     train:
       rollout_epoch: 1
       envs:
         libero_goal:
           total_num_envs: 8
           max_episode_steps: 8
           max_steps_per_rollout_epoch: 8
         maniskill:
           total_num_envs: 8
           max_episode_steps: 8
           max_steps_per_rollout_epoch: 8

Each ``defaults`` line loads an env preset under a name, here ``libero_goal`` and
``maniskill``. Each env configuration takes the same settings as a normal
``env.train``, and its ``total_num_envs`` is its own number of parallel envs.
``env.eval`` works the same way and is independent, so you can train on two env
configurations and evaluate on one.

A few settings belong to the shared rollout loop, so every env configuration
must give them the same value: ``max_steps_per_rollout_epoch``, ``auto_reset``,
and ``ignore_terminations``. Env presets ship different defaults, so set them on
each env configuration, as the example does. ``max_episode_steps`` may differ
between env configurations; when it does in training, also set
``env.train.max_episode_steps``, which the actor uses to normalize the loss.

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

The ``env`` placement decides where these workers run. Several env workers can
share one GPU, for example ``env: 0:0-1``. To drive a real robot, place the
workers of that env configuration on the robot's node; :doc:`hetero` shows how.

.. warning::

   Workers follow the listing order. Reordering ``envs`` or changing a count
   changes which env each worker runs, so check the placement afterwards,
   especially with real robots.

Match Observations
------------------

Rollout stacks the observations of all env workers into one batch, so they must
match. Envs from different simulators usually don't: LIBERO sends a main camera
and a wrist camera at 256x256, while the ManiSkill scene sends one 480x640
camera. Without a fix, rollout stops with an error.

Set ``policy_obs`` to choose what the policy receives:

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
not resize robot states, so a policy that reads ``states`` needs env
configurations whose states have the same size.

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
- Robot states are not resized, so they must have the same size in every env
  configuration.
- Smooth intervention (``smooth_intervene``) and real-time control evaluation
  (``runner.rtc``) work with a single env configuration only, and RLT simulator
  replay needs every train env configuration to be ``maniskill_rlt``. RLinf
  stops at startup when a config breaks these rules.
