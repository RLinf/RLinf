RL with MuJoCo Warp
===================

.. figure:: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/benchmarks/humanoid/rollout_three_humanoids.webp
   :align: center
   :width: 90%

   Three humanoids simulated by MuJoCo Warp (image: `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`__).

Use this recipe to train an MLP policy with PPO on five classic MuJoCo locomotion tasks. `MuJoCo Warp (MJWarp) <https://github.com/google-deepmind/mujoco_warp>`__ runs the worlds in one RLinf env worker together on an NVIDIA GPU. Each physics substep is one batched call, and a control step repeats that call for the task's frame skip: 5 for half_cheetah, ant, and humanoid, and 4 for hopper and walker2d. This page covers how those tasks are defined, their dimensions, installation, launch commands, and validation results.

Overview
--------

Choose a task and train a low-dimensional policy through RLinf's actor, rollout, and env workers.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      half_cheetah · hopper · walker2d · ant · humanoid

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 1 NVIDIA GPU

| **You'll do:** install → pick a task → launch training → watch ``eval/env/return``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · an NVIDIA GPU for training; CPU is supported for environment development.

Tasks
~~~~~

Select a task with ``env.train.task_name`` and ``env.eval.task_name``, and match the policy dimensions to the task.

.. list-table::
   :header-rows: 1
   :widths: 24 16 16 44

   * - Task
     - Obs dim
     - Act dim
     - Termination
   * - ``half_cheetah``
     - 17
     - 6
     - Only the episode step limit.
   * - ``hopper``
     - 11
     - 3
     - Unhealthy torso height or angle, or non-finite state.
   * - ``walker2d``
     - 17
     - 6
     - Unhealthy torso height or angle, or non-finite state.
   * - ``ant``
     - 27
     - 8
     - Unhealthy torso height or non-finite state.
   * - ``humanoid``
     - 45
     - 17
     - Unhealthy torso height or non-finite state.

These five names are the tasks this recipe ships. Every world in an env worker shares one batched physics state, ``MjWarpWorld``. It loads an MJCF file or an already compiled ``MjModel``, and each physics substep advances every world together. Reward, termination, and the initial state come from a task wrapper, ``TaskWrapper``. The built-in tasks share one locomotion wrapper, ``LocomotionTask``, and ``task_name`` selects its ``TaskSpec``. A new task with this same reward form is another ``TaskSpec`` plus its MJCF, registered with the five above. A different observation or reward is a new ``TaskWrapper`` that the env constructs for that ``task_name``.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Every episode truncates at ``max_episode_steps``. The table is the observation the policy reads and the reward the env returns.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Specification
   * - Observation
     - ``qpos`` without the global x position (and y position for 3D tasks), concatenated with ``qvel``.
   * - Action
     - Continuous actuator controls, clamped to each model's control range.
   * - Reward
     - Root-x velocity plus a task-specific alive bonus, minus ``ctrl_cost_weight * ||ctrl||^2``, multiplied by ``reward_scale`` (default ``1.0``).
   * - Prompt
     - Not used; this is a low-dimensional policy-control recipe.

half_cheetah, hopper, and walker2d use the same reward terms as Gymnasium MuJoCo v4. This recipe terminates hopper and walker2d on torso height, torso angle, or a non-finite state. Gymnasium v4 also clips joint velocities to ``[-10, 10]`` and terminates when the remaining state leaves ``(-100, 100)``.

Ant's control-cost weight is ``5e-4``, and its forward velocity is the root x speed. Gymnasium Ant-v4 uses weight ``0.5`` and the torso center-of-mass speed.

Humanoid's observation is the 45-dimensional ``qpos`` and ``qvel`` vector in the task table. Before ``reward_scale``, its reward is ``1.25 + root_x_velocity - 0.1 * ||ctrl||^2``. Gymnasium Humanoid-v4 observes 376 values, and its reward is ``5.0 + 1.25 * center-of-mass velocity - 0.1 * ||ctrl||^2``.

.. note::

   MJWarp has no PGS solver. On load, a model whose solver is already PGS is switched to CG. Among the shipped assets, only humanoid sets ``solver="PGS"``, so its contact dynamics differ from Gymnasium's MuJoCo CPU backend.

Installation
------------

.. include:: _setup_common.rst

**Docker image**

Build the MuJoCo Warp target and start a container with GPU access:

.. code:: bash

   docker build -f docker/Dockerfile \
       --build-arg BUILD_TARGET=embodied-mujoco_warp \
       -t rlinf:embodied-mujoco_warp .
   docker run -it --rm --gpus all \
       --shm-size 32g --network host --name rlinf \
       -v .:/workspace/RLinf rlinf:embodied-mujoco_warp

**Custom environment**

Install the dependencies and activate the environment:

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh embodied --env mujoco_warp
   source .venv/bin/activate

The installer pins ``mujoco-warp==3.17.0`` and ``warp-lang==1.16.0``. This verified combination works with CUDA 12 drivers and avoids resolving CUDA 13-only Warp wheels during driverless Docker builds.

Run It
------

Launch the half-cheetah recipe from the repository root:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp

What this command does:

1. Loads ``examples/embodiment/config/mujoco_warp_ppo_mlp.yaml`` and ``examples/embodiment/config/env/mujoco_warp.yaml``.
2. Starts actor, rollout, and env workers on one GPU, with 512 training worlds and 16 evaluation worlds.
3. Collects PPO rollouts and updates the MLP policy and value head.

To select another task, override both env task names and the actor dimensions. For example:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp \
       env.train.task_name=humanoid env.eval.task_name=humanoid \
       actor.model.obs_dim=45 actor.model.action_dim=17

Use the task table above for the other dimension pairs. Tune ``env.train.total_num_envs``, ``actor.micro_batch_size``, and ``actor.global_batch_size`` for your GPU memory.

Visualization and Results
-------------------------

Launch TensorBoard from the repository root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

Watch ``eval/env/return`` for the mean evaluation episode return and ``env/return`` for training episode returns. See :doc:`Training metrics <../../reference/metrics>` for the shared metric definitions. For standalone checkpoint evaluation, run ``run_embodiment.sh`` with ``runner.only_eval=true`` and ``runner.ckpt_path=<path>``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Validation
     - Result
   * - One-epoch CI recipe
     - Two parallel worlds complete rollout, advantage computation, and one PPO update on one NVIDIA GPU.
   * - 20-step PPO recipe
     - On one A800, ``env/return`` increased from -268.66 at step 1 to 867.35 at step 20. ``eval/env/return`` was 726.50 at step 10 and 1246.81 at step 20.

Video recording is disabled by default. Enable it for evaluation with:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp \
       env.eval.video_cfg.save_video=true

Evaluation videos are written under ``<log_path>/video/eval``. MJWarp still runs the physics. The environment copies each world's ``qpos``, ``qvel``, and ``ctrl`` into native MuJoCo and renders RGB frames; the video wrapper writes those frames to MP4. Recording adds device-to-host synchronization and per-world rendering, so leave it off for throughput runs. Use ``render_width`` and ``render_height`` for the per-world resolution, and ``video_cfg.fps`` to override the default control frequency.

The first simulation step JIT-compiles MJWarp kernels and can take longer than later steps.
