Real-World Evaluation
=====================

Overview
--------

Evaluate and deploy VLA policies on Franka arms, covering Bin-relocation (pick-and-place), custom tasks via the YAML-configurable ``FrankaEnv-v1`` environment, and Dual Franka setups. Typical deployments use a Franka Emika Panda arm with Intel RealSense cameras across either a two-node layout (GPU head node + robot control node on the same LAN) or a single workstation.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      Franka PnP · Custom (``FrankaEnv-v1``) · Dual Franka

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀ / π₀.₅ · DreamZero

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/realworld/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`Franka SFT <../../examples/embodied/franka_pi0_sft_deploy>` · :doc:`Franka <../../examples/embodied/franka>` · :doc:`DreamZero <../../examples/embodied/sft_dreamzero>`

For ZED cameras or Robotiq grippers, see :doc:`../../examples/embodied/franka_zed_robotiq`. For Ray cluster setup, firmware versions, and libfranka compatibility, see :doc:`../../examples/embodied/franka` and :doc:`../../guides/realworld_robot`.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: Robot Control Node (Install)

      Install on the robot control node (direct Franka and camera access, no GPU required; uses the default Franky backend):

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

         # 2. Verify camera serials and target pose
         python -m toolkits.realworld_check.test_franka_camera
         export FRANKA_ROBOT_IP=<robot_ip>
         python -m toolkits.realworld_check.test_franka_controller

   .. tab-item:: GPU / Rollout Node (PnP / π₀)

      Install OpenPI on the GPU head node, start Ray on both nodes (see the Ray dropdown below), update ``evaluations/realworld/realworld_pnp_eval.yaml`` with your ``robot_ip``, ``camera_serials``, and ``target_ee_pose``, and launch from the head node:

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

         # 2. Evaluate (with Pi0 base model + SFT full_weights.pt)
         bash evaluations/run_eval.sh realworld_pnp_eval \
           rollout.model.model_path=/path/to/pi0-model \
           runner.ckpt_path=/path/to/full_weights.pt

   .. tab-item:: Single-Node (Custom Task)

      When GPU and robot control share one machine, replace ``ROBOT_IP`` and ``MODEL_PATH`` in ``evaluations/realworld/realworld_eval.yaml`` and run:

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

         # 2. Evaluate
         bash evaluations/run_eval.sh realworld_eval

To drive the arm through the legacy ROS backend instead of Franky, follow :doc:`../../examples/embodied/franka`. DreamZero real-robot evaluation also requires DreamZero dependencies on the GPU node (see :doc:`../../examples/embodied/sft_dreamzero`).

.. warning::

   Verify workspace limits (``ee_pose_limit_min`` / ``ee_pose_limit_max``) and the emergency stop before real-robot evaluation. Use a small ``env.eval.rollout_epoch`` on the first run.

.. dropdown:: Node Topology and Example Configs

   Two-node PnP evaluation (``realworld_pnp_eval.yaml`` and ``realworld_pnp_eval_dreamzero.yaml``) uses a **1 GPU head node + 1 Franka control node** layout; ``realworld_eval.yaml`` runs both ``env`` and ``rollout`` on a single machine.

   .. list-table::
      :header-rows: 1
      :widths: 18 30 52

      * - ``RLINF_NODE_RANK``
        - Role
        - Notes
      * - rank 0
        - GPU / head
        - Runs ``rollout``; submit ``run_eval.sh`` on this node.
      * - rank 1
        - Robot control
        - Runs ``env`` workers with direct Franka and camera access.

   Available under ``evaluations/realworld/``:

   .. list-table::
      :header-rows: 1
      :widths: 42 34 24

      * - Config file
        - Task
        - Model
      * - ``realworld_pnp_eval.yaml``
        - Bin-relocation (PnP)
        - π₀
      * - ``realworld_pnp_eval_dreamzero.yaml``
        - Bin-relocation (PnP)
        - DreamZero
      * - ``realworld_eval.yaml``
        - Custom task (``FrankaEnv-v1``)
        - π₀
      * - ``realworld_dual_franka_pi05_RTC.yaml``
        - Single-workstation Dual Franka RTC
        - π₀.₅

   If ``evaluations/realworld/<config>.yaml`` is missing, ``run_eval.sh`` falls back to ``examples/embodiment/config/`` (such as two-node ``realworld_eval_dual_franka``; see :doc:`../reference/cli`).

.. dropdown:: Pre-flight Checks and Starting the Ray Cluster

   **Pre-flight checks**

   1. **Camera serials** (control node): run ``python -m toolkits.realworld_check.test_franka_camera`` and set ``env.eval.override_cfg.camera_serials``.
   2. **Target pose** (PnP tasks, control node): export ``FRANKA_ROBOT_IP``, run ``python -m toolkits.realworld_check.test_franka_controller``, enter ``getpos_euler``, and set ``env.eval.override_cfg.target_ee_pose`` (lowest point in the PnP workspace; see :doc:`../../examples/embodied/franka_pi0_sft_deploy`).
   3. **Ray status** (all nodes): run ``ray status`` and verify both nodes appear online.
   4. **Dummy mode (optional):** set ``is_dummy: True`` in ``override_cfg`` (see ``examples/embodiment/config/realworld_dummy_franka_sac_cnn.yaml``) to verify cluster wiring without moving the arm.

   **Starting the Ray cluster**

   Align environment variables and activate ``.venv`` on every node **before** ``ray start`` (Ray freezes the Python interpreter and env vars at startup):

   .. code-block:: bash

      source ray_utils/realworld/setup_before_ray.sh
      source .venv/bin/activate
      export RLINF_NODE_RANK=<0|1>          # unique within the cluster
      # On multi-NIC hosts, pin the reachable interface:
      # export RLINF_COMM_NET_DEVICES=<network_interface>

   Then start Ray (let ``<head_ip>`` be the GPU head node IP):

   .. code-block:: bash

      # GPU node (rank 0, head)
      export RLINF_NODE_RANK=0
      ray start --head --port=6379 --node-ip-address=<head_ip>

      # Franka control node (rank 1)
      export RLINF_NODE_RANK=1
      ray start --address=<head_ip>:6379

.. _realworld-eval-config:

.. dropdown:: Evaluation Config Reference

   Minimum fields to update in ``evaluations/realworld/realworld_pnp_eval.yaml`` (``node_ranks`` and ``component_placement`` must match each node's ``RLINF_NODE_RANK``):

   .. code-block:: yaml

      cluster:
        node_groups:
          - label: "4090"
            node_ranks: 0          # GPU node rank
          - label: franka
            node_ranks: 1          # Franka control node rank
            hardware:
              type: Franka
              configs:
                - robot_ip: ROBOT_IP
                  node_rank: 1

      runner:
        ckpt_path: /path/to/full_weights.pt

      env:
        eval:
          override_cfg:
            target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
            camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
            task_description: "pick up the object and place it into the container"

      rollout:
        model:
          model_path: /path/to/pi0-model

   **Required fields**

   .. list-table::
      :header-rows: 1
      :widths: 28 32 40

      * - Field
        - Location
        - Notes
      * - ``robot_ip``
        - ``cluster.node_groups[].hardware.configs``
        - Franka robot IP.
      * - ``node_ranks`` / ``component_placement``
        - ``cluster``
        - Must match ``RLINF_NODE_RANK`` (``env`` on Franka node, ``rollout`` on GPU node).
      * - ``target_ee_pose``
        - ``env.eval.override_cfg``
        - PnP target pose ``[x,y,z,rx,ry,rz]``; used for success checks and workspace clipping.
      * - ``camera_serials``
        - ``env.eval.override_cfg``
        - RealSense serial list (lives under ``override_cfg``, not ``node_groups``).
      * - ``task_description``
        - ``env.eval.override_cfg``
        - Language instruction; must match SFT training.
      * - ``model_path``
        - ``rollout.model``
        - Pi0 base model directory containing ``<repo_id>/norm_stats.json``.
      * - ``ckpt_path``
        - ``runner``
        - SFT-exported ``full_weights.pt``.
      * - ``config_name``
        - ``rollout.model.openpi``
        - ``"pi0_realworld"`` for PnP.

   **Key evaluation fields**

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Notes
      * - ``rollout_epoch``
        - Number of evaluation rounds (default ``20``).
      * - ``max_episode_steps``
        - Max steps per trajectory (default ``200``).
      * - ``max_steps_per_rollout_epoch``
        - Steps per round; **must be divisible by** ``rollout.model.num_action_chunks`` (default ``10`` for PnP).
      * - ``total_num_envs``
        - Parallel env count (typically ``1`` on real hardware).
      * - ``teleop``
        - Takeover device; set to ``none`` for evaluation.

   ``realworld`` does not call ``setup_sim_env``. For DreamZero real-robot evaluation (``realworld_pnp_eval_dreamzero.yaml``), see :doc:`../../examples/embodied/sft_dreamzero`.

.. dropdown:: Custom Real-Robot Task Evaluation

   For custom tasks defined in YAML (without writing a Python env class), use ``FrankaEnv-v1`` via ``examples/embodiment/config/env/realworld_franka_sft_env.yaml`` and ``evaluations/realworld/realworld_eval.yaml`` (single-node layout, ``num_nodes: 1``):

   .. code-block:: yaml

      override_cfg:
        task_description: "pick up the object and place it into the container"
        target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # goal pose
        reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]   # reset pose (above goal)
        max_num_steps: 300
        reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]
        action_scale: [1.0, 1.0, 1.0]
        ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
        ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

   For data collection, SFT training, and deployment on custom tasks, see :doc:`../../examples/embodied/franka_pi0_sft_deploy`.

.. dropdown:: Dual Franka Deployment

   Two dual-arm configs are provided:

   - **Two-node fallback** (``examples/embodiment/config/realworld_eval_dual_franka.yaml``): set ``rollout.model.model_path`` to the staged checkpoint directory and ``actor.model.openpi_data.repo_id`` to the repo ID containing ``norm_stats.json``.
   - **Single-workstation RTC** (``evaluations/realworld/realworld_dual_franka_pi05_RTC.yaml``): GPU, both arms, and three cameras on one machine (20-D ``tcp_rot6d`` actions, ``openpi.config_name: pi05_dualfranka_tcp_rot6d``, main camera ``left_wrist_0_rgb``). Replace ``MODEL_PATH``, ``norm_stats``, arm IPs, camera serials, gripper ports, ``keyboard_device``, and ``TASK_DESCRIPTION``. After homing, press ``a`` on the evdev USB keyboard/pedal named by ``keyboard_device`` (or ``RLINF_KEYBOARD_DEVICE``) to start the policy, ``c`` for success, and ``b`` for failure (SSH keystrokes do not reach the evdev device). Keep ``enable_camera_player: False`` without a display and keep ``max_steps_per_rollout_epoch`` divisible by ``20``.

   .. code-block:: bash

      bash evaluations/run_eval.sh realworld_eval_dual_franka \
        rollout.model.model_path=/path/to/deploy/global_step_<N> \
        actor.model.openpi_data.repo_id=<repo_id>/tcp_rot6d_v1 \
        env.eval.override_cfg.task_description="handover the object"

      bash evaluations/run_eval.sh realworld_dual_franka_pi05_RTC

   See :doc:`../../examples/embodied/dual_franka` and :doc:`../../examples/embodied/dual_franka_pico_dagger` for collection, SFT, and checkpoint staging.

Results and Troubleshooting
---------------------------

Task metrics print to the terminal, logs go to ``logs/<timestamp>-<config_name>/eval_embodiment.log``, and videos (when ``env.eval.video_cfg.save_video: True``) go to ``video_base_dir`` or ``<log_path>/video/eval/`` (see :doc:`../reference/results`).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Unsafe motion or workspace overrun
     - Verify ``ee_pose_limit_min`` / ``ee_pose_limit_max`` and the emergency stop beforehand; use a small ``rollout_epoch`` on the first run.
   * - Node placement or rank mismatch
     - Run ``env`` workers on nodes with direct Franka access and align ``node_ranks`` with ``RLINF_NODE_RANK`` (two nodes for PnP / ``realworld_eval_dual_franka``; one node for ``realworld_eval`` / ``realworld_dual_franka_pi05_RTC``).
   * - Cameras not detected
     - Run ``python -m toolkits.realworld_check.test_franka_camera`` on the control node and update ``env.eval.override_cfg.camera_serials``.
   * - Abnormal robot actions
     - Verify ``norm_stats.json`` exists under ``model_path/<repo_id>/`` and ``openpi.config_name`` matches training.
   * - ``ray status`` shows only one node
     - Check firewall rules, ``RLINF_COMM_NET_DEVICES``, and network reachability to ``<head_ip>``.
   * - Step-count validation error
     - Set ``max_steps_per_rollout_epoch`` to a multiple of ``rollout.model.num_action_chunks``.
   * - ZED camera or Robotiq gripper setup
     - See :doc:`../../examples/embodied/franka_zed_robotiq`.
