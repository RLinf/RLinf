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

| **You'll do:** install → pick a task → launch training → watch ``eval/return``.
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

To select another task, use the Python entrypoint to pass both env task names and the actor dimensions as Hydra overrides. Set the paths and rendering backend before calling Python directly from the repository root:

.. code:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD"
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config \
       --config-name mujoco_warp_ppo_mlp \
       env.train.task_name=humanoid env.eval.task_name=humanoid \
       actor.model.obs_dim=45 actor.model.action_dim=17

Use the task table above for the other dimension pairs. Tune ``env.train.total_num_envs``, ``actor.micro_batch_size``, and ``actor.global_batch_size`` for your GPU memory.

Visualization and Results
-------------------------

Launch TensorBoard from the repository root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

With the default logging path, the TensorBoard event file is under ``../results/tensorboard``. If you override ``runner.logger.log_path``, point ``--logdir`` at that directory instead. Open ``http://localhost:6006`` and select the ``env``, ``eval``, and ``time`` metric groups.

Watch ``eval/return`` for evaluation performance, ``env/return`` for training performance, and the corresponding ``episode_len`` metrics for survival time. :doc:`Training metrics <../../reference/metrics>` defines these quantities and their trajectory counts. They are means over completed episodes, not per-step rewards. Humanoid, hopper, walker2d, and ant also receive alive bonuses, so higher returns or longer survival alone do not show faster or more natural locomotion; check the evaluation video.

The training entrypoint runs evaluation and saves checkpoints every 10 iterations by default. To evaluate a saved humanoid checkpoint without further training, use the evaluation entrypoint and provide the policy dimensions under ``rollout.model``:

.. code:: bash

   python evaluations/eval_embodied_agent.py \
       --config-path examples/embodiment/config \
       --config-name mujoco_warp_ppo_mlp \
       env.eval.task_name=humanoid \
       runner.ckpt_path=<checkpoint>/actor/model_state_dict/full_weights.pt \
       +rollout.model.model_type=mlp_policy \
       +rollout.model.policy_setup=locomotion \
       +rollout.model.obs_dim=45 +rollout.model.action_dim=17 \
       +rollout.model.num_action_chunks=1 +rollout.model.hidden_dim=256 \
       +rollout.model.add_value_head=true +rollout.model.is_lora=false

Here, ``<checkpoint>`` is the ``global_step_<N>`` directory saved during training. Evaluation samples reset-state IDs in order and advances the sequence across resets, so separate evaluations need not start from identical states. Set ``env.eval.use_fixed_reset_state_ids=true`` to reuse the same initial state for each world when comparing checkpoints.

Reference Experiments
~~~~~~~~~~~~~~~~~~~~~

The following runs exercise all five tasks with the same PPO + MLP recipe. Each run uses one A800-SXM4-80GB GPU, 512 training worlds, and 16 evaluation worlds. One training iteration collects 1000 control steps per training world, then applies the default PPO updates. Evaluation and video recording run every 10 iterations, and ``max_episode_steps=1000``. The source commit is `461104f <https://github.com/RLinf/RLinf/commit/461104f2e5a106f38add2c794c19b7aa269e31a3>`__, with Python 3.11.18, PyTorch 2.11.0+cu128, MuJoCo 3.17.0, MuJoCo Warp 3.17.0, and Warp 1.16.0.

The table reports the first and last training iterations, and the first and last evaluations at iterations 10 and 50. Return is the mean undiscounted reward sum of completed episodes. The environment seed controls reset noise; actor seed 1234 and the other PPO settings stay at their defaults.

.. list-table::
   :header-rows: 1
   :widths: 20 19 19 19 23

   * - Task (env seed)
     - Train return (1 -> 50)
     - Eval return (10 -> 50)
     - Eval length (10 -> 50)
     - Media
   * - ``half_cheetah`` (42)
     - -276.82 -> 2650.85
     - 602.72 -> 3496.31
     - 1000.00 -> 1000.00
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/half_cheetah_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/half_cheetah_seed42_iterations_10_50.mp4>`__
   * - ``hopper`` (42)
     - 19.04 -> 357.37
     - 267.06 -> 374.66
     - 107.47 -> 132.96
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/hopper_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/hopper_seed42_iterations_10_50.mp4>`__
   * - ``walker2d`` (42)
     - 3.17 -> 386.73
     - 263.73 -> 454.09
     - 144.54 -> 144.41
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/walker2d_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/walker2d_seed42_iterations_10_50.mp4>`__
   * - ``ant`` (42)
     - 98.99 -> 593.47
     - 1002.27 -> 1191.58
     - 1000.00 -> 1000.00
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/ant_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/ant_seed42_iterations_10_50.mp4>`__
   * - ``humanoid`` (42)
     - 22.01 -> 155.71
     - 114.55 -> 235.88
     - 62.14 -> 110.19
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_50.mp4>`__
   * - ``humanoid`` (43)
     - 21.95 -> 145.55
     - 156.77 -> 215.50
     - 97.52 -> 111.93
     - `Curves <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed43_curves.png>`__ · `MP4 <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed43_iterations_10_50.mp4>`__

Humanoid, hopper, walker2d, and ant include an alive bonus. Higher returns or longer episodes do not necessarily indicate useful locomotion. Use the linked per-task videos to distinguish actual movement from survival behavior.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_curves.png
   :width: 100%
   :alt: Humanoid training return and episode length over 50 PPO iterations

   Humanoid with environment seed 42: training return, evaluation return, and episode length. Episode length is capped at 1000.

The following video compares iteration 10 on the left with iteration 50 on the right. `Download the humanoid comparison (MP4) <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_50.mp4>`__.

.. raw:: html

   <video controls muted playsinline preload="metadata" style="width: 100%; max-width: 960px;">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_seed42_iterations_10_50.mp4" type="video/mp4">
   </video>

The video shows evaluation world 0 on both sides. It illustrates the learned motion, not the mean over the 16 evaluation worlds. Later motion includes more stepping, but the humanoid still falls and resets; these runs do not demonstrate stable walking or convergence.

The second humanoid run changes only the environment seed from 42 to 43. It is a reset-condition comparison, not an independent multi-seed statistical benchmark.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/humanoid_two_seeds.png
   :width: 100%
   :alt: Humanoid evaluation return and length for environment seeds 42 and 43

   Humanoid evaluation with two environment reset seeds. The actor seed and training hyperparameters are unchanged.

These are 50-iteration learning curves, not converged results or comparisons with Gymnasium's MuJoCo CPU backend. The task definitions and solver differences described above also prevent direct return comparisons with Gymnasium Humanoid-v4. Some humanoid states emitted MJWarp solver iteration-limit warnings during these runs. The recipe retains the task solver settings rather than hiding these warnings.

Reproduce the Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^

Set the task, its dimensions, and the environment seed from the result table. This example reproduces humanoid with environment seed 42 on one GPU:

.. code:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD"
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   export CUDA_VISIBLE_DEVICES=0

   export TASK=humanoid ENV_SEED=42 OBS_DIM=45 ACTION_DIM=17
   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config \
       --config-name mujoco_warp_ppo_mlp \
       env.train.task_name=$TASK env.eval.task_name=$TASK \
       actor.model.obs_dim=$OBS_DIM actor.model.action_dim=$ACTION_DIM \
       env.train.seed=$ENV_SEED env.eval.seed=$ENV_SEED \
       runner.logger.log_path=../results/mujoco_warp_${TASK}_seed${ENV_SEED} \
       runner.max_steps=50 \
       runner.val_check_interval=10 runner.save_interval=10 \
       env.eval.video_cfg.save_video=true

For humanoid with environment seed 43, change only ``ENV_SEED`` to 43. Each run creates its TensorBoard events, checkpoints, and evaluation videos under the selected ``runner.logger.log_path``. Video recording adds rendering time; disable it for throughput measurements. The `combined metrics CSV <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/mujoco_warp_metrics.csv>`__ and `run metadata <https://raw.githubusercontent.com/RLinf/misc/main/pic/mujoco_warp/mujoco_warp_results.json>`__ contain the plotted values and software versions.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Validation
     - Result
   * - One-epoch CI recipe
     - Two parallel worlds complete rollout, advantage computation, and one PPO update on one NVIDIA GPU.

Video recording is disabled by default. Enable it for evaluation with:

.. code:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config \
       --config-name mujoco_warp_ppo_mlp \
       env.eval.video_cfg.save_video=true

Evaluation videos are written under ``<log_path>/video/eval/seed_<seed>``. The zero-based file counter follows the evaluation order: with ``runner.val_check_interval=10``, ``0.mp4`` records iteration 10, ``1.mp4`` records iteration 20, and so on. Each video contains a grid of the 16 evaluation worlds. A world that terminates early resets within the same video, so one video can contain several episodes.

MJWarp still runs the physics. The environment copies each world's ``qpos``, ``qvel``, and ``ctrl`` into native MuJoCo and renders RGB frames; the video wrapper writes those frames to MP4. Recording adds device-to-host synchronization and per-world rendering, so leave it off for throughput runs. Use ``render_width`` and ``render_height`` for the per-world resolution, and ``video_cfg.fps`` to override the default control frequency.

The first simulation step JIT-compiles MJWarp kernels and can take longer than later steps.
