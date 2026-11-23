RL with mjlab (Microduck)
=========================

Train a 50 Hz MLP policy with PPO on `Microduck <https://github.com/pollen-robotics/microduck_rl>`__ bipedal locomotion and trick tasks in `mjlab <https://github.com/mujocolab/mjlab>`__ (MuJoCo Warp). Simulation uses voltage-controlled Dynamixel XL330 actuator physics through `BAM <https://github.com/Rhoban/bam>`__.

Overview
--------

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

      Velocity · VelStand · StandUp · SitStand · GroundPick · BallKick · Roulade · Rollers

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 1 NVIDIA GPU

| **You'll do:** install → select a task → launch training → watch ``eval/env/return``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a CUDA-capable NVIDIA GPU for MuJoCo Warp.

Tasks
~~~~~

Select a task with ``env.train.task_id`` and ``env.eval.task_id``. Velocity, VelStand, StandUp, SitStand, GroundPick, BallKick, and Rollers also provide gear-play variants by inserting ``-Backlash`` before ``MicroDuck`` (for example, ``Mjlab-Velocity-Flat-Backlash-MicroDuck``).

.. list-table::
   :header-rows: 1
   :widths: 46 16 38

   * - Task ID
     - Terrain
     - Focus
   * - ``Mjlab-Velocity-{Flat,Rough}-MicroDuck``
     - Flat / Rough
     - Velocity and head-pose tracking
   * - ``Mjlab-VelStand-{Flat,Rough}-MicroDuck``
     - Flat / Rough
     - Walking and fall recovery in one policy
   * - ``Mjlab-StandUp-{Flat,Rough}-MicroDuck``
     - Flat / Rough
     - Stand up from the ground and hold pose
   * - ``Mjlab-SitStand-{Flat,Rough}-MicroDuck``
     - Flat / Rough
     - Commanded sit and stand transitions
   * - ``Mjlab-GroundPick-{Flat,Rough}-MicroDuck``
     - Flat / Rough
     - Crouch to touch the ground and return to stand
   * - ``Mjlab-BallKick-Flat-MicroDuck``
     - Flat
     - Kick a 70 mm ball forward
   * - ``Mjlab-Roulade-Flat-MicroDuck``
     - Flat
     - Forward roll and land back on the feet
   * - ``Mjlab-Velocity-Flat-MicroDuck-Rollers``
     - Flat
     - Roller-skating velocity tracking on passive wheels

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

All Microduck policies share the same 61-dimensional actor observation and 14-dimensional action space so exported ONNX policies can be hot-swapped on the real robot.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Specification
   * - Observation
     - Actor (61D): body angular velocity (3), projected gravity (3), relative joint positions (14), joint velocities (14), previous action (14), and commands ``[twist(3), head_pose(4), body_pose(6)]``. Critic (76D): clean actor terms without noise or bias plus base linear velocity (3), foot height (2), foot air time (2), foot contact (2), and foot contact forces (6).
   * - Action
     - 14 continuous joint position targets (5 left leg, 4 neck/head, 5 right leg) executed at 50 Hz through the BAM XL330 actuator model.
   * - Reward
     - Task-specific dense reward and regularization terms computed by ``mjlab``.
   * - Prompt
     - None.

Installation
------------

.. include:: _setup_common.rst

**Docker image**

Build the ``embodied-mjlab`` image and start a container with GPU access:

.. code:: bash

   docker build -f docker/Dockerfile \
       --build-arg BUILD_TARGET=embodied-mjlab \
       -t rlinf:embodied-mjlab .
   docker run -it --rm --gpus all \
       --shm-size 32g --network host --name rlinf \
       -v .:/workspace/RLinf rlinf:embodied-mjlab

**Custom environment**

Install the ``mjlab`` environment into a Python 3.12 virtual environment and activate it:

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh --no-flash-attn embodied --env mjlab
   source .venv/bin/activate

.. note::

   The installer clones ``pollen-robotics/microduck_rl`` into the virtual environment. Its code is licensed under Apache-2.0, and its 3D robot meshes under CC BY-NC-SA 4.0.

Run It
------

Launch flat-terrain Microduck walking from the repository root:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh mjlab_microduck_ppo_mlp

This loads ``examples/embodiment/config/mjlab_microduck_ppo_mlp.yaml`` (with environment defaults in ``examples/embodiment/config/env/mjlab_microduck.yaml``) and trains with 4096 parallel environments on one GPU.

To train another task, update ``task_id`` in ``examples/embodiment/config/env/mjlab_microduck.yaml`` (for example ``Mjlab-Velocity-Rough-Backlash-MicroDuck``) and re-run ``run_embodiment.sh``.

For a quick smoke test or a GPU with less memory, lower ``env.train.total_num_envs``, ``actor.micro_batch_size``, and ``actor.global_batch_size`` together in ``examples/embodiment/config/mjlab_microduck_ppo_mlp.yaml``.

To export a trained checkpoint to ONNX for downstream robot runtimes, see ``toolkits/mjlab/README.md``.

Visualization and Results
-------------------------

``run_embodiment.sh`` writes training logs and checkpoints under ``<train_log_path>`` (``./logs/<timestamp>-mjlab_microduck_ppo_mlp``). Launch TensorBoard from the repository root:

.. code:: bash

   tensorboard --logdir ./logs --port 6006

Track ``eval/env/return``, ``env/return``, ``env/episode_len``, and the ``Episode_Reward/*`` task terms in TensorBoard. See :doc:`Training metrics <../../reference/metrics>` for shared metric definitions.

During training, periodic evaluation runs every ``runner.val_check_interval`` steps (default ``250``). To record evaluation videos during training, set ``env.eval.video_cfg.save_video: True`` in ``examples/embodiment/config/mjlab_microduck_ppo_mlp.yaml``. Keep recording disabled during throughput runs.

The curves below show a 5000-step ``Mjlab-Velocity-Flat-MicroDuck`` run on one RTX 4090 with 4096 environments (``env/return`` reaches 124.8 and ``env/episode_len`` reaches 967.7 out of 1000 steps; small dips correspond to curriculum transitions):

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 12px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/mjlab_microduck/return_curve.png" alt="Microduck episode return curve" style="width: 100%;"/>
       <p><em>Episode Return (env/return)</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/mjlab_microduck/episode_len_curve.png" alt="Microduck episode length curve" style="width: 100%;"/>
       <p><em>Episode Length (env/episode_len)</em></p>
     </div>
   </div>

Evaluation
----------

Run standalone evaluation on a saved actor checkpoint with ``evaluations/run_eval.sh`` (see :doc:`Evaluation CLI <../../evaluations/reference/cli>` for full CLI options):

.. code:: bash

   bash evaluations/run_eval.sh mjlab mjlab_microduck_ppo_mlp_eval \
       runner.ckpt_path=<train_log_path>/mjlab_microduck_ppo_mlp/checkpoints/global_step_5000/actor/model_state_dict/full_weights.pt

Standalone evaluation writes outputs under ``<eval_log_path>`` (``./logs/<timestamp>-mjlab_microduck_ppo_mlp_eval``) and records videos under ``<eval_log_path>/video/eval`` by default (``env.eval.video_cfg.save_video: True``). Append ``env.eval.video_cfg.save_video=false`` to skip rendering, ``env.eval.task_id=<task_id>`` to evaluate another Microduck task, or ``env.eval.play_mode=true`` to switch from the training disturbance preset (3--6 s pushes on flat velocity) to the headless stress-test preset (no interactive viewer):

- **Velocity / SitStand / BallKick:** applies random pushes every 0.5--1.0 s to test push recovery.
- **Rough terrain** (``*-Rough-*``): disables the difficulty curriculum so evaluation covers the full terrain range.
- **Walk + fall recovery** (``VelStand``): keeps episodes running after a fall and applies stronger, more frequent topple pushes to test repeated stand-up recovery.

The video below shows a standalone evaluation rollout of the step-5000 checkpoint on ``Mjlab-Velocity-Flat-MicroDuck`` (``play_mode=false``, mean ``eval/return`` 149.8, mean ``eval/episode_len`` 903.8 / 1000 across 16 worlds):

.. raw:: html

   <div style="text-align: center;">
     <video controls autoplay loop muted playsinline preload="metadata" width="640">
       <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/mjlab_microduck/microduck_walk_eval.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>

Acknowledgements
----------------

This example builds on:

- `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`__ for the Microduck robot models, task definitions, reward and domain-randomization recipes, and sim2real setup (code under Apache-2.0; 3D robot meshes under CC BY-NC-SA 4.0).
- `mujocolab/mjlab <https://github.com/mujocolab/mjlab>`__ for the MuJoCo Warp RL environment framework.
- `Rhoban/bam <https://github.com/Rhoban/bam>`__ for the BAM Dynamixel XL330 actuator model.
