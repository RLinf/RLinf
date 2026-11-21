IQL (D4RL)
==========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/d4rl.png
   :align: center
   :width: 70%

   Offline RL on the D4RL benchmark.

This document explains how to run **D4RL-based offline RL training** with IQL (Implicit Q-Learning) in RLinf. It is intended for users who want to train policies directly from offline datasets without online environment interaction.

The primary objective is to train a policy that:

1. **Uses offline data only**: No environment interaction during training; data comes from D4RL datasets.
2. **Follows IQL**: Value function via expectile regression, actor via AWR-style weighting, twin Q-networks with TD targets.
3. **Fits RLinf's stack**: the IQL actor owns offline data loading; EnvWorker, RolloutWorker, and OfflineRunner handle eval; PyTorch + FSDP supported.

Overview
--------

Train a policy from D4RL offline datasets with IQL — no online environment interaction.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      IQL

   .. grid-item-card:: Models
      :text-align: center

      MLP

   .. grid-item-card:: Environments / Data
      :text-align: center

      D4RL

   .. grid-item-card:: Training
      :text-align: center

      Offline

| **You'll do:** install with D4RL → pick a config → run ``run_offline_rl.sh`` → watch ``eval/return``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · D4RL datasets (downloaded on first run).

Tasks
~~~~~

RLinf ships IQL configs for three D4RL task families; observation and action spaces are defined per task in D4RL.

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Family
     - Example task
     - Config
   * - MuJoCo locomotion
     - ``halfcheetah-medium-v2``
     - ``d4rl_iql_mujoco.yaml``
   * - AntMaze
     - ``antmaze-large-play-v0``
     - ``d4rl_iql_antmaze.yaml``
   * - Kitchen / Adroit
     - manipulation / dexterous hand
     - ``d4rl_iql_kitchen_adroit.yaml``

How IQL Works
-------------

**Core Algorithm Components**

1. **IQL (Implicit Q-Learning)**

   - **Value**: Updated with asymmetric expectile regression against the target Q-value minus the current value estimate, controlled by ``algorithm.expectile``.
   - **Actor**: Advantage-weighted regression (AWR) maximum likelihood using exponentiated advantages scaled by ``algorithm.temperature`` and clipped at 100.
   - **Critic** (twin Q): Temporal-difference (TD) regression toward the discounted next-state value estimate, controlled by ``algorithm.discount``.
   - **Target**: Polyak soft-update of the target critic with ``algorithm.tau``.

2. **Training flow**

   Each update step fetches one batch from the rank-local offline data loader, then runs IQL updates in order: Value → Actor → Critic → soft-update target critic.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: D4RL

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env d4rl && source .venv/bin/activate

      The launch script sets ``MUJOCO_GL=egl`` and ``PYOPENGL_PLATFORM=egl`` by default for headless runs.

      **2. Launch**

      RLinf ships default IQL configs for three D4RL task families (D4RL datasets download automatically on first run when ``data.dataset_path`` is ``null``):

      .. list-table::
         :header-rows: 1
         :widths: 26 74

         * - Task family
           - Config
         * - MuJoCo
           - ``examples/embodiment/config/d4rl_iql_mujoco.yaml``
         * - AntMaze
           - ``examples/embodiment/config/d4rl_iql_antmaze.yaml``
         * - Kitchen / Adroit
           - ``examples/embodiment/config/d4rl_iql_kitchen_adroit.yaml``

      Launch any task family from the repository root (defaults to ``d4rl_iql_mujoco`` when omitted; logs write to ``logs/<timestamp>-<config_name>/``):

      .. code-block:: bash

         # MuJoCo (default)
         ./examples/embodiment/run_offline_rl.sh d4rl_iql_mujoco

         # AntMaze
         ./examples/embodiment/run_offline_rl.sh d4rl_iql_antmaze

         # Kitchen / Adroit
         ./examples/embodiment/run_offline_rl.sh d4rl_iql_kitchen_adroit

.. dropdown:: Advanced configuration (optional)

   **Switching tasks and local datasets.** Set ``data.task_name`` and ``env.eval.task_name`` to switch D4RL tasks (for example ``antmaze-large-play-v0``). With ``data.dataset_path`` left as ``null``, RLinf reads ``~/.d4rl/datasets/<task_name>.hdf5`` if it exists and otherwise lets D4RL download the dataset. To train from an existing HDF5 file, set ``data.dataset_path`` to its path (matching the observation and action shapes of ``data.task_name``); standard D4RL files go through ``d4rl.qlearning_dataset`` using ``timeouts``, while files of prepared transitions with ``next_observations`` and without ``timeouts`` are used as is.

   **Resume training.** Set ``runner.resume_dir`` to a checkpoint directory (for example ``checkpoints/global_step_XXXXX``), then run the same launch command. The runner loads weights and continues from that step.

   Key default hyperparameters in the shipped YAML configs:

   .. code-block:: yaml

      runner:
        task_type: "offline"

      algorithm:
        loss_type: "offline_iql"
        batch_size: 256
        actor_lr: 3.0e-4
        value_lr: 3.0e-4
        critic_lr: 3.0e-4
        discount: 0.99
        tau: 0.005
        expectile: 0.9
        temperature: 10.0
        gamma: 0.99

      actor:
        model:
          iql_config:
            type: "actor"
            hidden_dims: [256, 256]
            dropout_rate: null
            log_std_min: -5.0
            log_std_max: 2.0

      data:
        dataset_type: "d4rl"
        task_name: "antmaze-large-play-v0"
        dataset_path: null

      env:
        task_name: "antmaze-large-play-v0"

Visualization and Results
-------------------------

Monitor ``eval/return`` with ``tensorboard --logdir ./logs --port 6006``, enable evaluation video recording with ``env.eval.video_cfg.save_video=True`` (state-only D4RL environments fall back to ``env.render()`` automatically under ``MUJOCO_GL=egl``), and see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

IQL-specific metrics:

- **Training Metrics**:

  - ``train/value_loss``: Value function expectile loss.
  - ``train/actor_loss``: AWR-style policy loss.
  - ``train/critic_loss``: Twin Q-network TD loss.
  - ``train/v``: Value function estimate (mean over batch).
  - ``train/q1``, ``train/q2``: Twin Q-network estimates.
  - ``train/adv_mean``, ``train/adv_std``: Advantage mean and standard deviation.

- **Time Metrics**:

  - ``time/step``: Wall time per training step (data fetch + actor update).
  - ``time/eval``: Wall time for evaluation (when ``runner.eval_episodes`` > 0).
  - ``time/actor/update_one_epoch``: Actor update time per step.

- **Evaluation Metrics** (when ``runner.eval_episodes`` > 0):

  - ``eval/return``: Mean episode return over evaluation rollouts.
  - ``eval/episode_len``: Mean episode length.
  - ``eval/num_trajectories``: Number of evaluation trajectories.
  - ``eval/terminated_at_end``: Fraction of episodes that terminated (not truncated); only present when the env uses ``ignore_terminations``.
