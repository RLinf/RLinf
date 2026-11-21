Real-World Evaluation
=====================

Overview
--------

Evaluate and deploy VLA policies on Franka arms across bin-relocation (pick-and-place), YAML-defined custom tasks (``FrankaEnv-v1``), and Dual Franka RTC setups. Hardware setup, RT-kernel configuration, and camera or gripper integration live in :doc:`../../examples/embodied/franka`, :doc:`../../examples/embodied/franka_zed_robotiq`, and :doc:`../../examples/real_world_index`.

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

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: PnP Two-Node (π₀)

      Install ``--model openpi --env franka`` on the GPU head node (rank 0) and ``--env franka`` on the robot control node (rank 1, no GPU required). Start Ray on both nodes (see the Ray dropdown below), update ``robot_ip``, ``camera_serials``, and ``target_ee_pose`` in ``evaluations/realworld/realworld_pnp_eval.yaml``, and launch from the head node:

      .. code-block:: bash

         # 1. Install on GPU head node (on the control node, omit --model openpi)
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

         # 2. Launch from the GPU head node (replace paths with your local model and checkpoint)
         bash evaluations/run_eval.sh realworld_pnp_eval \
           rollout.model.model_path=./pi0-model \
           runner.ckpt_path=./full_weights.pt

   .. tab-item:: Custom Task (Single-Node)

      When GPU rollout and Franka control share one workstation, set ``robot_ip`` and task geometry in ``evaluations/realworld/realworld_eval.yaml`` (``FrankaEnv-v1``) and launch:

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

         # 2. Launch
         bash evaluations/run_eval.sh realworld_eval \
           rollout.model.model_path=./pi0-sft-model

   .. tab-item:: Dual Franka RTC (π₀.₅)

      Single-workstation dual-arm RTC evaluation runs GPU inference, both Franka arms, and three RealSense cameras on one host (``evaluations/realworld/realworld_dual_franka_pi05_RTC.yaml``). Fill in arm IPs, camera serials, gripper ports, ``keyboard_device``, ``norm_stats_path``, and ``task_description``. After homing, press ``a`` on the USB keyboard or pedal to start, ``c`` for success, and ``b`` for failure:

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

         # 2. Launch
         bash evaluations/run_eval.sh realworld_dual_franka_pi05_RTC \
           rollout.model.model_path=./pi05-dual-franka

.. warning::

   Verify workspace limits (``ee_pose_limit_min`` / ``ee_pose_limit_max``) and the emergency stop before moving the arm. Start with a small ``env.eval.rollout_epoch`` on the first run.

.. dropdown:: Pre-flight Checks and Starting Ray

   Run these checks on the robot control node before launching evaluation (example IPs: ``192.168.1.10`` for the GPU head node, ``192.168.1.20`` for the Franka controller, and ``eth0`` for multi-NIC hosts):

   .. code-block:: bash

      # 1. Inspect RealSense camera serials and record target_ee_pose (enter getpos_euler)
      python -m toolkits.realworld_check.test_franka_camera
      export FRANKA_ROBOT_IP=192.168.1.20
      python -m toolkits.realworld_check.test_franka_controller

      # 2. Prepare env on every node BEFORE starting Ray (Ray freezes env vars at startup)
      source ray_utils/realworld/setup_before_ray.sh
      source .venv/bin/activate
      # Optional on multi-NIC hosts: export RLINF_COMM_NET_DEVICES=eth0

      # 3a. GPU head node (rank 0)
      export RLINF_NODE_RANK=0
      ray start --head --port=6379 --node-ip-address=192.168.1.10

      # 3b. Franka control node (rank 1)
      export RLINF_NODE_RANK=1
      ray start --address=192.168.1.10:6379

   Run ``ray status`` to confirm both nodes are online. Set ``is_dummy: True`` under ``override_cfg`` to verify cluster wiring without moving the arm.

.. _realworld-eval-config:

.. dropdown:: Evaluation Config Reference

   Available configs under ``evaluations/realworld/`` (``realworld`` skips simulator env setup; missing names fall back to ``examples/embodiment/config/``, such as two-node ``realworld_eval_dual_franka``):

   .. list-table::
      :header-rows: 1
      :widths: 42 36 22

      * - Config file
        - Task / Topology
        - Model
      * - ``realworld_pnp_eval.yaml``
        - Bin-relocation PnP (2 nodes)
        - π₀
      * - ``realworld_pnp_eval_pi05_sft_RTC.yaml``
        - Bin-relocation PnP RTC (2 nodes)
        - π₀.₅
      * - ``realworld_pnp_eval_dreamzero.yaml``
        - Bin-relocation PnP (2 nodes)
        - DreamZero
      * - ``realworld_eval.yaml``
        - Custom ``FrankaEnv-v1`` (1 node)
        - π₀
      * - ``realworld_dual_franka_pi05_RTC.yaml``
        - Dual Franka RTC (1 workstation)
        - π₀.₅
      * - ``realworld_dual_yam_openpi_pi05_eval.yaml``
        - Dual YAM (2 nodes)
        - π₀.₅

   Core fields to update in real-robot configs:

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - Field
        - Notes
      * - ``cluster.node_groups`` / ``component_placement``
        - Set ``robot_ip`` and align ``node_ranks`` with each host's ``RLINF_NODE_RANK`` (``env`` on robot node, ``rollout`` on GPU node).
      * - ``env.eval.override_cfg``
        - Set ``camera_serials``, ``task_description`` (must match SFT), and ``target_ee_pose`` (plus ``reset_ee_pose`` and pose limits for ``FrankaEnv-v1``).
      * - ``rollout.model.model_path`` / ``runner.ckpt_path``
        - Base model directory containing ``norm_stats.json`` and optional SFT ``full_weights.pt``; match ``rollout.model.openpi.config_name`` to training.
      * - ``env.eval.rollout_epoch`` / ``max_steps_per_rollout_epoch``
        - Rounds and per-round step budget; ``max_steps_per_rollout_epoch`` must be divisible by ``rollout.model.num_action_chunks``.
      * - ``env.eval.teleop``
        - Set to ``none`` during evaluation; keep ``total_num_envs: 1`` on real hardware.

   For custom ``FrankaEnv-v1`` data collection and SFT, see :doc:`../../examples/embodied/franka_pi0_sft_deploy`. For two-node Dual Franka fallback or data collection, see :doc:`../../examples/embodied/dual_franka` and :doc:`../../examples/embodied/dual_franka_pico_dagger`. For DreamZero dependencies and checkpoints on the GPU node, see :doc:`../../examples/embodied/sft_dreamzero`.

Results and Troubleshooting
---------------------------

Task metrics print to the terminal, logs go to ``logs/<timestamp>-<config_name>/eval_embodiment.log``, and videos (when ``env.eval.video_cfg.save_video: True``) go to ``<log_path>/video/eval/`` (see :doc:`../reference/results`).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Node placement or rank mismatch
     - Run ``env`` workers on the node wired to Franka and match ``node_ranks`` to ``RLINF_NODE_RANK``.
   * - Cameras not detected
     - Run ``python -m toolkits.realworld_check.test_franka_camera`` on the control node and update ``env.eval.override_cfg.camera_serials``.
   * - Abnormal robot actions
     - Verify ``norm_stats.json`` matches training and ``rollout.model.openpi.config_name`` matches the checkpoint.
   * - ``ray status`` shows only one node
     - Check firewall rules, ``RLINF_COMM_NET_DEVICES``, and network reachability to the GPU head IP.
   * - Step-count validation error
     - Set ``max_steps_per_rollout_epoch`` to a multiple of ``rollout.model.num_action_chunks``.
   * - Dual Franka keyboard triggers ignored
     - Press keys on the USB evdev device configured in ``keyboard_device`` (or ``RLINF_KEYBOARD_DEVICE``); SSH keystrokes do not reach evdev.
