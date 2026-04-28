RL with D4RL Benchmark
======================

This document explains how to run **D4RL-based offline RL training** with IQL (Implicit Q-Learning) in RLinf. It is intended for users who want to train policies directly from offline datasets without online environment interaction.

The primary objective is to train a policy that:

1. **Uses offline data only**: No environment interaction during training; data comes from D4RL datasets.
2. **Follows IQL**: Value function via expectile regression, actor via AWR-style weighting, twin Q-networks with TD targets.
3. **Fits RLinf's stack**: the IQL actor owns offline data loading; EnvWorker, RolloutWorker, and OfflineRunner handle eval; PyTorch + FSDP supported.

Environment
-----------

**D4RL (Datasets for Deep Data-Driven Reinforcement Learning)**

RLinf uses the D4RL benchmark suite. Configs are provided for:

- **MuJoCo locomotion**: e.g. ``halfcheetah-medium-v2``, ``hopper-medium-replay-v2`` — continuous control, state-based.
- **AntMaze**: e.g. ``antmaze-large-play-v0`` — goal-conditioned navigation, sparse rewards.
- **Kitchen / Adroit**: manipulation and dexterous hand tasks — high-dimensional state and action.

Observation and action spaces are defined per task in D4RL.

Algorithm
---------

**Core Algorithm Components**

1. **IQL (Implicit Q-Learning)**

   - **Value** :math:`V(s)`: Updated with expectile regression on :math:`Q_{\mathrm{target}}(s,a) - V(s)`; weight :math:`w(d) = \tau \cdot \mathbb{I}(d>0) + (1-\tau) \cdot \mathbb{I}(d \le 0)`.
   - **Actor** :math:`\pi(a|s)`: AWR-style advantage-weighted maximum likelihood; advantage :math:`A = Q_{\mathrm{target}}(s,a) - V(s)`, weight :math:`w = \min(\exp(A \cdot \beta), 100)`.
   - **Critic** (twin Q): TD loss with target :math:`y = r + \gamma \cdot \mathrm{mask} \cdot V(s')`.
   - **Target**: Soft-update of target critic with :math:`\tau`.

2. **Training flow**

   Each update step: the actor fetches one batch from its rank-local ``DataLoader`` (built in ``EmbodiedIQLFSDPPolicy.build_offline_dataloader``), then runs IQL in the current implementation order: update Value → update Actor → update Critic → soft-update target critic.

Installation & Dependencies
----------------------------

Install the embodied stack with D4RL support:

.. code-block:: bash

   bash requirements/install.sh embodied --env d4rl
   source .venv/bin/activate

The launch script sets ``MUJOCO_GL=egl`` and ``PYOPENGL_PLATFORM=egl`` by default for headless runs.

Running the Script
----------------------------

**1. Configuration Files**

RLinf provides default IQL configs for different D4RL task families:

- **MuJoCo**: ``examples/embodiment/config/d4rl_iql_mujoco.yaml``
- **AntMaze**: ``examples/embodiment/config/d4rl_iql_antmaze.yaml``
- **Kitchen / Adroit**: ``examples/embodiment/config/d4rl_iql_kitchen_adroit.yaml``

**2. Key Parameter Configuration**

**2.1 Runner and Algorithm**

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

**2.2 Actor (Model)**

.. code-block:: yaml

   actor:
     model:
       iql_config:
         type: "actor"
         hidden_dims: [256, 256]
         dropout_rate: null
         log_std_min: -5.0
         log_std_max: 2.0

**2.3 Data**

.. code-block:: yaml

   data:
     dataset_type: "d4rl"
     task_name: "antmaze-large-play-v0"
     dataset_path: null

**2.4 Environment**

.. code-block:: yaml

   env:
     task_name: "antmaze-large-play-v0"

Set ``data.dataset_type`` to ``d4rl`` , ``data.task_name`` and ``env.eval.task_name`` to the desired D4RL task (e.g. ``antmaze-large-play-v0``).

**3. Launch Script**

- Script: ``examples/embodiment/run_offline_rl.sh``
- Default config (no argument): ``d4rl_iql_mujoco``
- Logs: ``<repo>/logs/<timestamp>-<config_name>/``
- Actual command:

.. code-block:: bash

   python examples/embodiment/train_offline_rl.py \
     --config-path examples/embodiment/config/ \
     --config-name <config_name> \
     runner.logger.log_path=<log_dir> runner.logger.experiment_name=<config_name>

**4. Launch Commands**

From the repository root:

**MuJoCo (default)**

::

   ./examples/embodiment/run_offline_rl.sh d4rl_iql_mujoco

**AntMaze**

::

   ./examples/embodiment/run_offline_rl.sh d4rl_iql_antmaze

**Kitchen / Adroit**

::

   ./examples/embodiment/run_offline_rl.sh d4rl_iql_kitchen_adroit

Resume Training
---------------

Set ``runner.resume_dir`` to a checkpoint directory (e.g. ``checkpoints/global_step_XXXXX``), then run the same launch command. The runner loads weights and continues from that step.

Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. Key Metrics Tracked**

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

**3. Video Generation**

D4RL observations are state-only (no image keys). The recorder **falls back to env.render()** when the observation has no image field, so enabling ``save_video`` is enough to generate evaluation videos. ``video_base_dir`` is optional (default: ``./video``), and you can still set it explicitly for organized outputs. Ensure the MuJoCo env is created with rendering support (e.g. ``render_mode="rgb_array"`` is set automatically when ``save_video`` is true). For headless servers, set ``MUJOCO_GL=egl`` and ``PYOPENGL_PLATFORM=egl``.

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: true
            video_base_dir: ${runner.logger.log_path}/video/eval  # optional, defaults to ./video

**4. Train Log Tool Integration**

.. code-block:: yaml

   runner:
      task_type: "offline"
      logger:
         log_path: "../results"
         project_name: rlinf
         experiment_name: "d4rl_iql_mujoco"
         logger_backends: ["tensorboard"] # wandb, swanlab