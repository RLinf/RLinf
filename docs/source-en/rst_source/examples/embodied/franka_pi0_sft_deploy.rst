Franka π₀ SFT & Deployment
==========================

Overview
--------

Run the Bin-relocation pipeline end to end with OpenPI π₀ on a Franka arm: collect LeRobot demonstrations with SpaceMouse teleoperation, compute OpenPI normalization statistics, fine-tune π₀, and deploy the checkpoint on real hardware.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      Franka Emika Panda (with gripper)

   .. grid-item-card:: Sensors & Teleop
      :text-align: center

      Intel RealSense · SpaceMouse · Keyboard

   .. grid-item-card:: Workflow
      :text-align: center

      Collect → Norm Stats → SFT → Deploy

   .. grid-item-card:: Hardware
      :text-align: center

      1× NVIDIA GPU (or Controller + GPU)

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Stage
     - Config / Format
     - Description
   * - Data collection
     - ``realworld_collect_data`` + ``pi0_realworld``
     - Collect Franka Bin-relocation episodes and export them in LeRobot format for OpenPI.
   * - SFT
     - ``realworld_bin_relocation_sft_openpi``
     - Fine-tune OpenPI π₀ on the collected real-world dataset.
   * - Deployment
     - ``realworld_pnp_eval`` / ``realworld_eval``
     - Run autonomous eval-only policy deployment on the physical Franka arm.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Wrist RealSense camera frames mapped to ``pi0_realworld`` OpenPI dataset keys and robot state.
   * - Action
     - Franka Cartesian action plus gripper control formatted by ``pi0_realworld``.
   * - Reward
     - Keyboard success label (``c``) during collection; operator-observed task completion during deployment.
   * - Prompt
     - Task instruction stored in ``override_cfg.task_description`` and dataset metadata.

Hardware and Safety
-------------------

You need a Franka Emika Panda arm with a gripper, an Intel RealSense wrist camera, an NVIDIA GPU host for SFT and rollout, and an optional SpaceMouse for teleoperation. On a single-machine setup, the GPU host also controls the robot; in a multi-node setup, a separate controller host without a GPU runs Franky and data collection. Complete the firmware check, real-time kernel setup, and FCI activation in :doc:`franka`, and obtain the base OpenPI π₀ checkpoint per :doc:`sft_openpi`.

.. warning::

   Keep the emergency stop within reach and supervise every collection and deployment run. In the Bin-relocation task, ``target_ee_pose`` marks the midpoint of the lowest point in the motion space and carves out a workspace region to prevent colliding with the container rim. Measure ``target_ee_pose`` on your own robot and clear the workspace before enabling autonomous motion.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. Setup & Collect

      **Install**

      Install the OpenPI and Franka environment on the GPU host (adding ``--env franka`` covers single-node setups; on a separate controller host, follow :doc:`franka`):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

      **Measure target pose**

      Position the arm at the Bin-relocation target midpoint in guiding mode, activate FCI in Franka Desk, and record ``[x, y, z, roll, pitch, yaw]`` via ``getpos_euler``:

      .. code-block:: bash

         export FRANKA_ROBOT_IP=192.168.1.10  # Replace with your robot IP
         python -m toolkits.realworld_check.test_franka_controller

      **Configure and collect demonstrations**

      Data collection, training, and deployment share the task template ``examples/embodiment/config/env/realworld_bin_relocation.yaml``. In ``examples/embodiment/config/realworld_collect_data.yaml``, configure your Franka IP and camera serial as in :doc:`franka`, then apply these Bin-relocation overrides (switch env, enable gripper, enable keyboard labeling, set task prompt, export LeRobot format):

      .. code-block:: yaml

         defaults:
           - env/realworld_bin_relocation@env.eval
           - override hydra/job_logging: stdout

         env:
           eval:
             no_gripper: False
             keyboard_reward_wrapper: single_stage
             override_cfg:
               target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
               task_description: "pick up the object and place it into the container"
             data_collection:
               enabled: True
               export_format: lerobot_dataset

      Run collection:

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh realworld_collect_data

      Teleoperate with the SpaceMouse, press ``c`` after each episode to mark success and reset the arm, and return the object to its start position. By default, collection stops after 20 episodes (configured by ``num_data_episodes``) and saves LeRobot shards under ``logs/<running-timestamp>/lerobot_dataset/rank_*/id_*/``.

   .. tab-item:: 2. Compute Stats & SFT

      **Prepare dataset and compute normalization statistics**

      Copy the collected LeRobot dataset to the training node (skip on a single-machine setup) under a root directory such as ``/path/to/lerobot_data/realworld_franka_bin_relocation/`` containing ``data/`` and ``meta/``. Export ``HF_LEROBOT_HOME`` to that root directory and compute ``pi0_realworld`` normalization statistics:

      .. code-block:: bash

         export HF_LEROBOT_HOME=/path/to/lerobot_data
         python -m toolkits.lerobot.calculate_norm_stats \
             --config-name pi0_realworld \
             --repo-id realworld_franka_bin_relocation

      ``config-name`` must match the OpenPI data config used in training, and ``repo-id`` must match your LeRobot dataset directory name. The script writes ``./assets/pi0_realworld/realworld_franka_bin_relocation/norm_stats.json``. Consistent with :doc:`sft_openpi`, training and evaluation do not look beside SFT ``full_weights.pt`` for this file: pass it via ``openpi_data.norm_stats_path`` (recommended) or copy it to ``<model_path>/realworld_franka_bin_relocation/norm_stats.json`` for the default fallback.

      **Run OpenPI SFT**

      Edit ``examples/sft/config/realworld_bin_relocation_sft_openpi.yaml`` or pass the dataset, base model, and normalization stats paths as Hydra CLI overrides (see :doc:`sft_openpi` for full OpenPI SFT details):

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi \
           data.train_data_paths=./lerobot_data/realworld_franka_bin_relocation \
           actor.model.model_path=./pi0-model \
           +actor.model.openpi_data.norm_stats_path=./assets/pi0_realworld/realworld_franka_bin_relocation/norm_stats.json

   .. tab-item:: 3. Deploy on Robot

      Edit ``evaluations/realworld/realworld_pnp_eval.yaml`` with your robot IP and measured target pose (RealSense cameras are auto-discovered; set ``camera_serials`` only to select a subset or fix camera order):

      .. code-block:: yaml

         cluster:
           node_groups:
             - label: franka
               hardware:
                 configs:
                   - robot_ip: ROBOT_IP

         env:
           eval:
             rollout_epoch: 20
             override_cfg:
               target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
               task_description: "pick up the object and place it into the container"

      ``evaluations/realworld/realworld_pnp_eval.yaml`` defaults to a two-node cluster (``cluster.num_nodes: 2`` with ``franka`` on ``node_rank: 1``). For multi-node deployment, start the Ray cluster as in :doc:`franka`. For single-host deployment, set ``cluster.num_nodes: 1`` and set both ``node_ranks: 0`` and ``node_rank: 0`` on the ``franka`` group (or use the single-node ``realworld_eval`` config below), otherwise Ray waits indefinitely for a second node. Launch deployment from the repository root with Hydra CLI overrides for the checkpoint, base model, and ``norm_stats.json`` (see the :doc:`real-world evaluation guide <../../evaluations/guides/realworld>` for options):

      .. code-block:: bash

         bash evaluations/run_eval.sh realworld_pnp_eval \
           runner.ckpt_path=./logs/checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt \
           rollout.model.model_path=./pi0-model \
           +rollout.model.openpi_data.norm_stats_path=./assets/pi0_realworld/realworld_franka_bin_relocation/norm_stats.json

      The policy autonomously controls the arm for ``env.eval.rollout_epoch`` episodes.

.. dropdown:: Generic Real-World SFT Environment (FrankaEnv-v1)

   To collect SFT demonstrations or deploy a trained policy on a new Franka task purely through YAML (without writing a custom environment), customize ``examples/embodiment/config/env/realworld_franka_sft_env.yaml``:

   .. code-block:: yaml

      override_cfg:
        task_description: "pick up the object and place it into the container"
        target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # goal pose [x, y, z, rx, ry, rz]
        reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]   # reset pose above goal
        max_num_steps: 300
        reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]
        action_scale: [1.0, 1.0, 1.0]                       # [xyz, rpy, gripper]
        ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
        ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

   For single-node standalone deployment with this generic environment and a π₀ policy, edit ``ROBOT_IP`` and ``MODEL_PATH`` in ``evaluations/realworld/realworld_eval.yaml`` and run (see the :doc:`real-world evaluation guide <../../evaluations/guides/realworld>`):

   .. code-block:: bash

      bash evaluations/run_eval.sh realworld_eval

Results and Troubleshooting
---------------------------

Track SFT loss in TensorBoard (``tensorboard --logdir ./logs``) and verify autonomous rollouts on hardware; see :doc:`Training metrics <../../reference/metrics>` for logging configuration.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - Fix
   * - ``calculate_norm_stats.py`` cannot find dataset
     - Export ``HF_LEROBOT_HOME`` to the parent directory containing ``<repo_id>/{data,meta}`` and match ``--repo-id`` to that folder name.
   * - Missing ``norm_stats.json`` during SFT or deployment
     - Pass the generated ``norm_stats.json`` through ``openpi_data.norm_stats_path`` (recommended, matching :doc:`sft_openpi`) or copy it into ``<model_path>/<repo_id>/norm_stats.json``.
   * - Gripper stays closed during collection
     - Set ``env.eval.no_gripper: False`` in ``realworld_collect_data.yaml``.
   * - Arm hits container rim or workspace limit
     - Re-measure ``target_ee_pose`` at the midpoint of the lowest safe point in the motion space and verify ``ee_pose_limit_min`` / ``ee_pose_limit_max``.
