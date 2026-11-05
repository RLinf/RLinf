RL with MuJoCo Warp
===================

.. figure:: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/benchmarks/humanoid/rollout_three_humanoids.webp
   :align: center
   :width: 90%

   Three humanoids simulated by MuJoCo Warp (image: `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`__).

Use this recipe to train an MLP policy with PPO on five classic MuJoCo locomotion tasks. `MuJoCo Warp (MJWarp) <https://github.com/google-deepmind/mujoco_warp>`__ steps all worlds in an RLinf env worker together in one batched physics call on an NVIDIA GPU. This page covers task dimensions, installation, launch commands, and validation results.

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

Select a task with ``env.train.task_name`` and ``env.eval.task_name``. Match the policy dimensions to the task.

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

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Observations and rewards follow the Gymnasium MuJoCo v4 conventions. Every episode is truncated at ``max_episode_steps``.

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
     - Forward velocity plus a task-specific alive bonus, minus the squared-control cost, multiplied by ``reward_scale`` (default ``1.0``).
   * - Prompt
     - Not used; this is a low-dimensional policy-control recipe.

.. note::

   MJWarp does not support the PGS solver. The integration switches hopper and walker2d to CG, so their contact dynamics are not identical to Gymnasium's MuJoCo CPU backend.

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

Watch ``eval/env/return`` for the mean evaluation episode return and ``env/return`` for training episode returns. See :doc:`Training metrics <../../reference/metrics>` for the shared metric definitions and the Evaluation section for standalone evaluation workflows.

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

Evaluation videos are written under ``<log_path>/video/eval``. The wrapper copies each world's state to native MuJoCo for RGB rendering while MJWarp continues to perform the physics. This adds device-to-host synchronization and per-world rendering work, so disable video for throughput benchmarks. Use ``render_width`` and ``render_height`` to change the per-world resolution, and ``video_cfg.fps`` to override the default control frequency.

The first simulation step JIT-compiles MJWarp kernels and can take longer than later steps.
