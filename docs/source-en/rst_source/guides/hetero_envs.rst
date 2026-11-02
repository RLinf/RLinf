Heterogeneous Envs
==================

This page shows you how to run heterogeneous envs: different env workers of one
run step different env configurations, while one policy acts in all of them
during the same rollout. Use it to co-train on ManiSkill and LIBERO, to mix two
LIBERO suites, or to run a simulator next to a real robot.

By default, ``env.train`` and ``env.eval`` each hold a single env configuration.
Every env worker runs it, and ``total_num_envs`` only sets how many parallel
copies of it run; this page calls that case homogeneous. ``env.train`` or
``env.eval`` becomes heterogeneous when it lists several env configurations
under ``envs``. Heterogeneous hardware (:doc:`hetero`) is a separate feature: it
decides which nodes the workers run on, not which env each worker runs. The two
combine when an env needs particular hardware, such as a robot.

The page first explains how heterogeneous envs run, because each later step
follows from that: how to configure them, how RLinf assigns env configurations
to env workers, how observations are aligned for the policy, and how to read
per-env results. It ends with a runnable example and the current limitations.

How Heterogeneous Envs Run
--------------------------

Three terms keep the rest of the page precise:

- An **env configuration** is one complete env setup: ``env_type``, task,
  ``init_params``, reward options, and so on.
- An **env instance** is one parallel copy of an env configuration.
  ``total_num_envs`` counts env instances.
- An **env worker** is one env worker process. ``W`` denotes the number of env
  workers.

With a homogeneous ``env.train``, every env worker builds the same env
configuration and steps ``total_num_envs / W`` instances of it. Heterogeneous
envs keep this unit of work, one env configuration per env worker, and only let
different env workers build different env configurations. One rollout step then
runs as usual:

1. Each env worker steps its env instances and sends their observations to
   rollout over the env-rollout channel.
2. Each rollout worker concatenates the observations it receives into one
   batch, runs the policy once, and sends each env worker the actions for its
   rows.
3. Each env worker converts its actions for its own ``env_type`` and steps its
   env instances.
4. After the rollout, the actor trains on the trajectories of all env workers
   together.

Rollout, actor, and runner have no per-env-type code paths, and every
simulator or robot SDK runs in its own process. Steps 1 and 2 impose two
requirements on the env configurations that run together:

- **Equal batches.** The env-rollout channel expects every env worker to send a
  batch of the same size, so every env worker must host the same number of env
  instances. `Assign Env Configurations to Env Workers`_ shows how RLinf sizes
  the workers.
- **Stackable observations.** Rollout concatenates observations key by key, so
  every env configuration must send the same keys with the same shapes.
  `Align Observations for the Policy`_ shows how to conform them.

Configure Heterogeneous Envs
----------------------------

To make ``env.train`` heterogeneous, put its env configurations under ``envs``
and give each one a name. The config below trains on LIBERO-Goal and a ManiSkill
scene with 8 env instances each:

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

Each defaults line of the form ``env/<preset>@env.train.envs.<name>`` loads an
env preset as the env configuration ``<name>``. You choose the names; RLinf uses
them in metric keys and video paths. Each env configuration under ``envs`` has
the same keys a homogeneous ``env.train`` has, and its ``total_num_envs`` is its
number of env instances. ``env.eval`` is configured the same way. The two are
independent, so either one can stay homogeneous.

With ``envs`` set, ``env.train`` itself keeps only fields that describe the
whole rollout. You set some of them, and RLinf derives the others from the env
configurations:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Field of ``env.train``
     - Set by
     - Value
   * - ``envs``
     - You
     - The env configurations, keyed by name.
   * - ``rollout_epoch``
     - You
     - Rollout epochs per training step, as in a homogeneous ``env.train``.
   * - ``policy_obs``
     - You (optional)
     - The observation format the policy receives; see
       `Align Observations for the Policy`_.
   * - ``total_num_envs``
     - RLinf
     - The sum of the env configurations' ``total_num_envs``; 16 in the example.
   * - ``max_steps_per_rollout_epoch``, ``auto_reset``, ``ignore_terminations``
     - RLinf
     - Copied from the env configurations, which must all set the same value.
   * - ``max_episode_steps``
     - RLinf or you
     - Copied when all env configurations agree. In ``env.train``, set it
       yourself when they differ.

The copied fields describe the shared rollout loop, not one env. Rollout steps
every env worker in lockstep, for ``max_steps_per_rollout_epoch`` env steps per
rollout epoch. The actor applies one ``auto_reset`` and ``ignore_terminations``
rule to the whole batch when it masks finished episodes, and it normalizes the
loss by one ``max_episode_steps``. Each env configuration may still truncate its
own episodes at a different ``max_episode_steps``. Presets ship different
defaults (LIBERO-Goal runs 320 steps per rollout epoch, the ManiSkill scene 80),
so set the shared fields on every env configuration, as the example does.

RLinf derives these fields once, while it validates the config at startup, and
writes them onto ``env.train``. Rollout, actor, and runner then read them exactly
as they do for a homogeneous ``env.train``. If you set one of them yourself, it
must equal the derived value. Any other key directly under ``env.train``, such as
``env.train.video_cfg``, is rejected because no env would read it, so set it on
each env configuration instead.

Validation also checks every env configuration and stops the run with a message
that names the env configuration and the conflicting values:

- Every env configuration sets ``env_type`` and a positive integer
  ``total_num_envs``.
- The copied fields have one value across the env configurations.
- Env-type rules run on each env configuration they apply to. ManiSkill env
  configurations get ``init_params.control_mode`` from the model's
  ``policy_setup``, Behavior env configurations must use ``r1pro_behavior``, and
  with online LeRobot collection, LIBERO, RoboTwin, and Behavior env
  configurations need ``skip_intermediate_renders: False``.
- Each env configuration's ``group_size`` divides the number of env instances
  per env worker and pipeline stage, so a GRPO group never spans two env
  configurations.

Assign Env Configurations to Env Workers
----------------------------------------

Once validation has derived the totals, RLinf assigns each env configuration a
contiguous block of env worker ranks, in the order they are listed under
``envs``. With ``W`` env workers and env configurations of ``n_1, ..., n_k``
env instances:

1. Each env worker hosts ``e = (n_1 + ... + n_k) / W`` env instances. The sum
   must be divisible by ``W``.
2. Env configuration ``i`` gets ``n_i / e`` env workers. Each ``n_i`` must be a
   multiple of ``e``.
3. The first env configuration takes ranks ``0`` to ``n_1 / e - 1``, the second
   the next ``n_2 / e`` ranks, and so on.

For example, with 4 env workers, 48 LIBERO-Goal instances, and 16 ManiSkill
instances, ``e = 64 / 4 = 16``:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Env configuration
     - ``total_num_envs``
     - Env worker ranks
   * - ``libero_goal``
     - 48
     - 0, 1, 2
   * - ``maniskill``
     - 16
     - 3

With 2 env workers instead, ``e = 32``, and ``libero_goal`` would need 1.5
workers, so validation fails. Change the instance counts or the number of env
workers until every env configuration gets whole workers.

Inside its block, an env configuration runs as if it were alone on ``n_i / e``
env workers. Each env worker gets a local rank, its rank minus the first rank of
the block, and uses it wherever a homogeneous env uses the global rank:

- Seeds and reset states are partitioned across the workers of the block only.
- Item ``k`` of an env configuration's ``override_cfgs`` applies to local rank
  ``k`` of its block, so the list has one item per worker of the block.

The seeds and reset states of an env configuration therefore depend on the env
configuration itself and its block size, not on where its block starts.

Rank order follows the ``env`` placement, so the placement decides where each
block runs. Several env workers can share one GPU: ``env: 0:0-1`` places ranks 0
and 1 on GPU 0. To run an env configuration on robot hardware, place the ranks of
its block on the robot nodes; :doc:`hetero` describes node groups.

.. warning::

   Ranks are assigned by listing order. If you reorder the env configurations or
   change an instance count, the env configuration that each rank runs changes
   too, and a real-robot env configuration could end up on a GPU node. Check the
   placement whenever you edit either.

Align Observations for the Policy
---------------------------------

Every env worker now runs its own env configuration, but its observations still
end up in one policy batch. Each env worker sends rollout these keys:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Shape
   * - ``main_images``
     - ``[B, H, W, C]``
   * - ``wrist_images``
     - ``[B, H, W, C]`` or ``[B, V, H, W, C]`` for ``V`` views
   * - ``extra_view_images``
     - ``[B, V, H, W, C]``
   * - ``states``
     - ``[B, D]``
   * - ``task_descriptions``
     - ``B`` strings

When rollout concatenates the batches of several env workers, each key must be
present on all of them or on none, and its tensors must match in every dimension
after ``B``. Env configurations from different simulators rarely meet this. In
the example, LIBERO sends a 256x256 main view and a wrist view, while the
ManiSkill scene sends one 480x640 (height x width) main view and no wrist view,
so rollout stops with ``Cannot merge present and absent batch values``.

Set ``policy_obs`` on ``env.train`` or ``env.eval`` to give every env worker the
same output format:

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

``keys`` lists the keys the policy receives. Every other key is sent as
``None``, so it is absent from the batch of every env worker. ``image_size`` is
``[height, width]``. Every image key is resized to it with antialiased bilinear
interpolation over ``H`` and ``W``, and the other dimensions stay as they are.
Integer images are rounded and clamped, so ``uint8`` stays ``uint8``. Images
already at that size pass through unchanged. With this setting, both env
configurations in the example send ``main_images`` as ``[B, 256, 256, 3]``
``uint8``, plus ``task_descriptions``, which is what OpenVLA-OFT reads.

Each env worker applies ``policy_obs`` only to the copy of the observations it
sends to rollout, in both training and evaluation. Reward models, video
recording, and data collection still receive the env's original observations.
The resize runs once per image key and chunk step, on the device where the env
produced the images: the CPU for LIBERO, and the simulation GPU for ManiSkill's
GPU backend. Leave ``image_size`` out when the env configurations already agree
on resolution. ``policy_obs`` also works with a homogeneous ``env.train`` or
``env.eval``.

Actions need no matching setting. Rollout returns one action tensor, and each
env worker converts its rows for its own ``env_type`` before it steps its env.
The policy still produces one action space with one action normalization, which
for OpenVLA-OFT means one ``unnorm_key``. Every env configuration must therefore
accept the same kind of action, such as a 7-dimensional end-effector delta with a
gripper command.

Read Per-Env Results
--------------------

A training step now mixes episodes from several env configurations, so RLinf
reports each metric both pooled and per env configuration. With heterogeneous
envs, every env worker reports each metric twice, under its usual key and under
``<name>/<key>``. The runner concatenates the per-episode values of all env
workers that report a key and averages them:

- ``env/success_once`` is the mean over all episodes of all env configurations.
- ``env/libero_goal/success_once`` is the mean over the episodes of
  ``libero_goal`` only.

Evaluation reports under ``eval/`` the same way, and every env metric gets a
per-env-configuration copy, including ``episode_len`` and the interaction-delay
statistics.

Videos go to ``<video_base_dir>/seed_<n>/<name>/``. Each env configuration
numbers its seeds from 0, so the name keeps env configurations that share a
``video_base_dir`` from overwriting each other's files. Data-collection files
are named by the global env worker rank, which already tells the env
configurations apart.

Run the Example
---------------

The end-to-end test ``maniskill_libero_grpo_openvlaoft`` combines everything on
this page. It trains OpenVLA-OFT with GRPO on LIBERO-Goal and ManiSkill and
evaluates on both after every step. You need both simulators in one virtual
environment, the OpenVLA-OFT checkpoint and LoRA adapter named in the config,
and 2 GPUs.

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh maniskill_libero_grpo_openvlaoft

What this does:

1. Installs OpenVLA-OFT, ManiSkill, and LIBERO into one virtual environment.
2. Starts two env workers on GPUs 0 and 1. In both training and evaluation,
   rank 0 runs ``libero_goal`` and rank 1 runs ``maniskill``.
3. Runs two GRPO steps and logs ``env/<name>/*`` and ``eval/<name>/*`` for each
   env configuration in TensorBoard under ``runner.logger.log_path``.

The test uses 8-step episodes to check the data path. For real co-training,
raise the horizons and instance counts and keep the shared fields equal. The
ManiSkill scene also needs its assets under ``rlinf/envs/sim/maniskill/assets``,
as described in :doc:`../examples/embodied/maniskill`.

Limitations
-----------

All env configurations under one ``envs`` share one policy and one rollout loop,
which sets these limits:

- The policy has one action space and one action normalization, and it is not
  told which env configuration a sample came from.
- ``policy_obs`` resizes images only, so ``states`` must have the same size
  ``D`` in every env configuration.
- Every env configuration needs a whole number of env workers, as computed in
  `Assign Env Configurations to Env Workers`_.

The following features need a homogeneous ``env.train`` or ``env.eval``, or one
kind of env in it, and stop at startup otherwise:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Feature
     - Reason
   * - Smooth intervention (``smooth_intervene``)
     - Its actions bypass the policy and cannot share a rollout batch with the
       policy actions of other env configurations.
   * - Real-time control evaluation (``runner.rtc``)
     - The RTC rollout chooses its delay model from the ``env_type`` of a
       homogeneous ``env.eval``.
   * - RLT simulator replay
     - The actor keeps one replay layout, so either every env configuration in
       ``env.train`` is ``maniskill_rlt`` or none is.
