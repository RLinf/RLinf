Multiple Environments in One Run
================================

This page shows you how to train and evaluate one policy on several
environments at once, such as ManiSkill and LIBERO, two LIBERO suites, or a
simulator next to a real robot. A normal embodied config gives ``env.train``
and ``env.eval`` one env each. Here you'll list several envs under one mode,
decide how many parallel envs each one gets, make their observations fit a
single policy batch, and read each env's results on its own. The page follows
that order and ends with a runnable two-simulator example and the features that
still expect a single env.

List the Envs of a Mode
-----------------------

You start from env presets you already use. Instead of mounting one preset at
``env.train``, mount each preset under a name in ``env.train.envs``; the name is
yours to choose, and RLinf uses it in metric keys and video paths. ``env.eval``
works the same way, and the two modes are independent: you can train on two
envs and evaluate on one, or the reverse.

The config below trains on LIBERO-Goal and a ManiSkill scene, with 8 parallel
envs each:

.. code-block:: yaml

   defaults:
     - env/libero_goal@env.train.envs.libero_goal
     - env/maniskill_put_on_plate_in_scene_25_main@env.train.envs.maniskill
     - env/libero_goal@env.eval.envs.libero_goal
     - env/maniskill_ood_template@env.eval.envs.maniskill
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

Each entry under ``envs`` is a complete env config with the same keys a
single-env ``env.train`` has: ``env_type``, task settings, ``init_params``,
``video_cfg``, reward options. Its ``total_num_envs`` is that env's share of the
parallel envs. The mode itself keeps only what describes the whole rollout,
here ``rollout_epoch``.

When the config is validated, RLinf writes the totals that rollout, actor, and
runner read back onto the mode: ``env.train.total_num_envs`` becomes 16, the sum
of the shares, and the fields every env must agree on are copied up as well.
The next section explains which fields those are.

Keep the Rollout Loop Shared
----------------------------

Every env in a mode is stepped by one rollout loop, and the actor masks
finished episodes with one rule. A few fields therefore describe the loop rather
than an env, and every entry must give them the same value:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Field
     - Why it must agree
   * - ``max_steps_per_rollout_epoch``
     - Rollout runs the same number of chunk steps against every env worker.
   * - ``auto_reset``
     - The actor decides from one value whether episodes restart within a
       rollout epoch.
   * - ``ignore_terminations``
     - The actor builds one loss mask for the whole batch.

Env presets ship different defaults (LIBERO-Goal steps 320 times per epoch, the
ManiSkill scene 80), so set these fields on each entry, as the example does.
A mismatch fails at startup and names each env's value.

``max_episode_steps`` is different: it is each env's truncation horizon, and
envs can keep their own. The actor also normalizes the loss by a single
``env.train.max_episode_steps``. RLinf copies the value up when every train env
agrees; when they differ, set ``env.train.max_episode_steps`` yourself to the
horizon the loss should use.

The mode level accepts ``envs``, ``rollout_epoch``, ``total_num_envs``,
``max_episode_steps``, ``policy_obs``, and the shared fields above. Any other key
there, such as ``env.train.video_cfg``, would apply to no env, so RLinf rejects
it and asks you to set it on each entry.

Give Each Env Its Workers
-------------------------

With the totals settled, RLinf lays the envs out on the env workers. Each env
worker hosts exactly one env, and every worker hosts the same number of parallel
envs, because the channel between env and rollout workers splits each batch
evenly. The envs take contiguous blocks of workers in the order you list them,
sized by their share.

Say ``env`` is placed on 4 workers and the mode holds 64 envs, so each worker
hosts 16:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Env
     - ``total_num_envs``
     - Env workers
   * - ``libero_goal``
     - 48
     - 0, 1, 2
   * - ``maniskill``
     - 16
     - 3

A share that is not a whole number of workers fails at startup. With the same
envs on 2 workers (32 each), 48 LIBERO envs would fill one and a half workers,
so give the envs more workers or change the shares.

Inside its block, an env runs as it would alone on those workers. Its seeds and
reset-state shards are split among the workers hosting it, and an
``override_cfgs`` list inside the entry has one item per worker of that block,
not per worker of the whole group.

Placement decides where those workers run. Several env workers can share one
GPU, so two envs fit on a single device with ``env: 0:0-1`` (resource 0,
processes 0 and 1). When an env drives real robots, order the ``env`` placement
so the workers of that env's block land on the robot hardware;
:doc:`hetero` shows how to span node groups.

.. warning::

   Workers are assigned by listing order. If you reorder ``envs`` or change a
   share, the env that each worker rank hosts changes too, and a real-robot env
   could end up on a GPU node. Check the placement whenever you edit either.

Make Observations Fit One Policy Batch
--------------------------------------

The envs now run on their own workers, but their observations meet again at
rollout: each rollout worker concatenates the observations it receives into one
batch for the policy. Envs from different simulators rarely produce the same
views at the same resolution. LIBERO sends a 256x256 main view and a wrist view;
the ManiSkill scene above sends one 640x480 view and no wrist camera. Rollout
cannot stack those, and fails when a view is present for one env and absent for
another.

Set ``policy_obs`` on the mode to describe the observations the policy actually
reads. ``keys`` lists the observation keys to send (``main_images``,
``wrist_images``, ``extra_view_images``, ``states``, ``task_descriptions``), and
the others go as ``None``. ``image_size`` resizes every image view to one
``[height, width]``:

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

This suits OpenVLA-OFT with one input image and no proprioception, which reads
only the main view and the task. A policy that reads ``states`` needs envs whose
states have the same size; ``policy_obs`` resizes images, not states.

Only the policy's copy is conformed. Reward models, video recording, and data
collection still see each env's own observations. The resize runs on the env
worker's CPU every step, so leave ``image_size`` out when the envs already agree.

Actions take the opposite path. Rollout returns one action batch, and each env
worker formats its slice for the env it hosts, so LIBERO and ManiSkill each get
their own action conversion. The policy still produces a single action space
and one action normalization (for OpenVLA-OFT, one ``unnorm_key``), so the envs
must accept the same kind of action, such as a 7-dimensional end-effector delta
with a gripper command.

Read Results per Env
--------------------

During training, each env reports its metrics under its own name next to the
pooled ones. ``env/success_once`` averages over every episode of every env;
``env/libero_goal/success_once`` and ``env/maniskill/success_once`` average over
one env each. Evaluation follows the same pattern under ``eval/``, and every
other env metric, such as ``episode_len`` or interaction delay, gets a per-env
copy too.

Videos go to ``<video_base_dir>/seed_<n>/<name>/``. Seeds restart at zero for
each env, so the name keeps two envs that share a ``video_base_dir`` from
overwriting each other's files. Episode data from ``data_collection`` is already
named by the global env worker rank.

Run the Example
---------------

The end-to-end test ``maniskill_libero_grpo_openvlaoft`` puts the whole page
together: it trains OpenVLA-OFT with GRPO on LIBERO-Goal and ManiSkill and
evaluates on both after every step. It needs an environment with both
simulators, the OpenVLA-OFT checkpoint and LoRA adapter named in the config, and
2 GPUs.

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh maniskill_libero_grpo_openvlaoft

What this does:

1. Installs OpenVLA-OFT with ManiSkill and LIBERO into one virtual environment.
2. Starts two env workers on GPUs 0 and 1: worker 0 hosts ``libero_goal`` and
   worker 1 hosts ``maniskill``, in training and in evaluation.
3. Runs two GRPO steps and logs ``env/<name>/*`` and ``eval/<name>/*`` for each
   env in TensorBoard under ``runner.logger.log_path``.

The run checks the plumbing with 8-step episodes; for a real co-training
recipe, raise the horizons and env counts and keep the shared fields equal. The
ManiSkill scene also needs its assets under ``rlinf/envs/sim/maniskill/assets``,
as described in :doc:`../examples/embodied/maniskill`.

Features That Assume One Env
----------------------------

A few features assume one env, or one kind of env, per mode, and stop at
startup when a mode breaks that assumption:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Feature
     - Constraint
   * - Smooth intervention (``smooth_intervene``)
     - Its actions bypass the policy and cannot share a rollout batch with the
       policy actions of other envs.
   * - Real-time control evaluation (``runner.rtc``)
     - The RTC rollout picks its delay model from the type of the one eval env.
   * - RLT simulator replay
     - The actor keeps one replay layout, so either every train env is
       ``maniskill_rlt`` or none is.
