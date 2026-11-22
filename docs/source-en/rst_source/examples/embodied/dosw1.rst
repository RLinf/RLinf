Dexmal DOS-W1 Online RL
=======================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png
   :align: center
   :width: 80%

   Dexmal DOS-W1 dual-arm robot used for the Flow Matching single-arm pick workflow.

Train a Flow Matching policy on the Dexmal DOS-W1 dual-arm robot with AirBot gRPC services, RealSense cameras, keyboard-gated episodes, and SAC/RLPD real-world training.

Overview
--------

This recipe pairs the DOS-W1 controller node running AirBot services with a GPU node running synchronous or asynchronous SAC/RLPD Flow Matching training on a single-arm pick task.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      Dexmal DOS-W1 · dual-arm + leaders

   .. grid-item-card:: Sensors & Teleop
      :text-align: center

      Up to 3× RealSense · leaders · keyboard

   .. grid-item-card:: Workflow
      :text-align: center

      Joint cal · Teleop demos · SAC / RLPD Flow

   .. grid-item-card:: Hardware
      :text-align: center

      AirBot gRPC · Controller PC · GPU (≥ 4090)

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Task
     - Config / Entry Point
     - Description
   * - Dummy self-check
     - ``dosw1_dummy_sac_mlp_pick.yaml``
     - Validate config wiring and cluster plumbing without cameras or hardware calls.
   * - Data collection
     - ``dosw1_collect_data``
     - Collect optional leader-arm teleoperated demos for RLPD warm start.
   * - Training
     - ``dosw1_pick_sac_flow(_async)``
     - Train the ResNet-10 Flow Matching policy with SAC on the pick task.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - ``cam_front``, ``cam_left``, and ``cam_right`` RGB images plus dual-arm joint and gripper state.
   * - Action
     - 14-D joint-space action: six joints plus gripper width for each arm.
   * - Reward
     - Dense reach, grasp, and lift shaping with terminal success at ``target_lift_joint``.
   * - Prompt
     - ``Perform the DOSW1 dual-arm manipulation task.``

Hardware and Safety
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Requirement
   * - Robot
     - DOS-W1 dual-arm robot with leader arms for teleoperation.
   * - Cameras
     - Up to three Intel RealSense cameras (serials under ``camera_serials``; single camera supported with ``image_num: 1``).
   * - Controller node
     - DOS-W1 onboard host (or LAN PC) running AirBot gRPC services on ports ``50050`` (left leader), ``50051`` (left follower), ``50052`` (right leader), and ``50053`` (right follower).
   * - Training / Rollout node
     - Linux server with at least one CUDA GPU (RTX 4090 or better recommended) on the same local network.

.. warning::

   Keep the GPU node and DOS-W1 reachable over the same local network, have a safety operator at the emergency stop, and clear the workspace before enabling motion. Always start from the full ``dosw1_pick_sac_flow.yaml`` template (never compose partial YAML snippets), verify ``cluster.num_nodes`` and ``RLINF_NODE_RANK`` match every host, and keep ``keyboard_intervention_wrapper: True`` and ``enable_human_in_loop: True`` active during hardware runs.

.. dropdown:: Safety clamp and workspace box details

   Before sending any command to the AirBot SDK, the DOS-W1 environment applies three safety layers identically in both policy and ``TELEOP`` modes:

   1. **Per-step joint clamp**: clips target joints to ``current ± max_joint_delta`` (default ``0.1`` rad, about 5.7° per step).
   2. **Absolute joint limits**: clips targets to ``[joint_limit_min, joint_limit_max]`` (default ``±π``; tighten to match your workspace).
   3. **End-effector safety box**: binary-searches along the joint-space interpolation to find the largest fraction whose forward-kinematics pose stays inside ``left_ee_pose_limit_min/max`` and ``right_ee_pose_limit_min/max``, truncating out-of-box motions automatically.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Robot Controller Node

      **1. Install RLinf and the AirBot SDK**

      The AirBot SDK is pre-deployed on DOS-W1 at ``~/dos_w1/airbot/5.1.6/airbot_py-5.1.6-py3-none-any.whl`` and ``~/dos_w1/airbot/airbot_api`` (export ``DOSW1_SDK_WHEEL`` and ``DOSW1_API_PATH`` beforehand if stored elsewhere). Run the full installer rather than ``uv pip install -e .`` so ``evdev``, ``opencv-python``, and RealSense bindings are installed:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env dosw1 && source .venv/bin/activate

      **2. Start AirBot services and calibrate target joints**

      Start the follower and leader services on the robot node (for example ``sh ~/dos_w1/airbot/whole_start.sh``) and confirm every panel starts cleanly. Run the teleoperation check script (replace ``192.168.1.20`` with your DOS-W1 IP):

      .. code-block:: bash

         python -m toolkits.realworld_check.test_dosw1_controller \
             --robot-url 192.168.1.20 \
             --print-hz 5

      Move the left leader arm to the target grasp pose, copy the first six radian values of ``left_joint`` (the seventh value is gripper width) into ``target_grasp_joint``, then move to a safe lift pose and copy ``left_joint[:6]`` into ``target_lift_joint``.

      **3. Enable keyboard gating**

      Grant the runtime user access to ``/dev/input`` devices and re-login (see :doc:`franka` for setting ``RLINF_KEYBOARD_DEVICE`` on multi-keyboard hosts):

      .. code-block:: bash

         sudo usermod -aG input $USER

      .. list-table::
         :header-rows: 1
         :widths: 14 86

         * - Key
           - Action
         * - ``s`` / ``r`` / ``d``
           - Start a new episode from free-teleop / abort and discard / mark done and save.
         * - ``p`` / ``t`` / ``m``
           - Pause from ``MODEL`` or ``TELEOP`` / switch to ``TELEOP`` / switch back to ``MODEL`` (ignored when ``manual_episode_control_only`` is enabled).

      **4. Join the Ray cluster**

      With ``.venv`` active, export ``RLINF_NODE_RANK`` and join the Ray cluster (replace ``192.168.1.10`` with your GPU node IP):

      .. code-block:: bash

         export RLINF_NODE_RANK=1
         ray start --address=192.168.1.10:6379

      .. dropdown:: Collect teleoperated demos for RLPD (optional)

         Update ``robot_url``, the four gRPC ports, and ``camera_serials`` in ``examples/embodiment/config/dosw1_collect_data.yaml`` (single-node config, ``cluster.num_nodes: 1``; Ray starts automatically), then run on the robot node:

         .. code-block:: bash

            bash examples/embodiment/collect_data.sh dosw1_collect_data

         Position the follower arms via the leader arms, press ``s`` to start, teleoperate grasp and lift, then press ``d`` to save or ``r`` to discard. The script exits after ``runner.num_data_episodes`` (default ``20``) and writes ``<log_path>/demo_buffer/`` (and raw episodes in ``<log_path>/pickle_episode/``). Point RLPD training at the saved buffer:

         .. code-block:: yaml

            algorithm:
              demo_buffer:
                load_path: "/path/to/logs/dosw1-collect/<timestamp>/demo_buffer"

   .. tab-item:: Training & Rollout Nodes

      **1. Install RLinf and download ResNet-10**

      On the GPU node (without ``~/dos_w1/airbot``), the installer warns and skips the AirBot SDK while installing all training dependencies:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env dosw1 && source .venv/bin/activate
         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./RLinf-ResNet10-pretrained

      **2. Start the Ray head**

      Export ``RLINF_NODE_RANK=0`` before starting Ray (replace ``192.168.1.10`` with your GPU server IP) and confirm both nodes appear in ``ray status``:

      .. code-block:: bash

         export RLINF_NODE_RANK=0
         ray start --head --port=6379 --node-ip-address=192.168.1.10
         ray status

      **3. Update hardware, target joints, and backbone path**

      Edit ``examples/embodiment/config/dosw1_pick_sac_flow.yaml`` (or ``dosw1_pick_sac_flow_async.yaml`` for decoupled async weight sync):

      .. code-block:: yaml

         cluster:
           node_groups:
             - label: dosw1
               node_ranks: 1
               hardware:
                 type: DOSW1
                 configs:
                   - robot_url: "192.168.1.20"
                     camera_serials: ["SERIAL_1", "SERIAL_2", "SERIAL_3"]
                     node_rank: 1

         env:
           train:
             override_cfg:
               is_dummy: False
               target_grasp_joint: [...]   # 6 rad values from calibration
               target_lift_joint:  [...]   # 6 rad values from calibration

         actor:
           model:
             model_path: "./RLinf-ResNet10-pretrained"

      .. dropdown:: Full configuration reference (optional)

         .. code-block:: yaml

            cluster:
              num_nodes: 2
              component_placement:
                actor:   { node_group: "gpu",   placement: 0 }
                rollout: { node_group: "gpu",   placement: 0 }
                env:     { node_group: dosw1,   placement: 0 }
              node_groups:
                - label: "gpu"
                  node_ranks: 0
                - label: dosw1
                  node_ranks: 1
                  hardware:
                    type: DOSW1
                    configs:
                      - robot_url: "192.168.1.20"
                        left_arm_port: 50051
                        right_arm_port: 50053
                        left_lead_port: 50050
                        right_lead_port: 50052
                        camera_serials:
                          - "SERIAL_1"
                          - "SERIAL_2"
                          - "SERIAL_3"
                        node_rank: 1

            env:
              train:
                keyboard_intervention_wrapper: True
                override_cfg:
                  is_dummy: False
                  use_dense_reward: True
                  target_grasp_joint: [...]
                  target_lift_joint:  [...]
                  max_joint_delta: 0.1
                  action_scale: 1.0
                  left_ee_pose_limit_min: [0.1, -0.35, 0.02]
                  left_ee_pose_limit_max: [0.4,  0.08, 0.40]
                  right_ee_pose_limit_min: [0.28, -0.01, 0.16]
                  right_ee_pose_limit_max: [0.30,  0.01, 0.17]
                  enable_human_in_loop: True

            actor:
              model:
                model_path: "./RLinf-ResNet10-pretrained"
                state_dim: 14        # 6 joints + 1 gripper per arm, dual arm
                action_dim: 14
                image_num: 3         # 1 -> cam_left only; 3 -> all three cameras

      **4. Run dummy self-check (optional) and launch training**

      Validate config wiring on a single node with the state-only dummy config (auto-starts Ray), then launch two-node real-world training from the GPU node (logs write to ``logs/<timestamp>-<config>/``):

      .. code-block:: bash

         # Optional single-node dummy self-check (no hardware or cameras)
         PYTHONPATH=$PWD python examples/embodiment/train_embodied_agent.py \
             --config-path "$PWD/tests/e2e_tests/embodied" \
             --config-name dosw1_dummy_sac_mlp_pick \
             runner.max_epochs=1

         # Synchronous real-world training
         bash examples/embodiment/run_realworld.sh dosw1_pick_sac_flow

         # Asynchronous real-world training
         bash examples/embodiment/run_realworld_async.sh dosw1_pick_sac_flow_async

Results and Troubleshooting
---------------------------

Monitor training on the Ray head node with ``tensorboard --logdir ./logs --port 6006`` and watch ``env/success_once`` (primary signal), ``env/return``, ``train/sac/*``, and ``train/replay_buffer/*``; see :doc:`Training metrics <../../reference/metrics>` for metric details.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Symptom
     - Fix
   * - ``ImportError: airbot_sdk is not installed`` on robot node
     - Export ``DOSW1_SDK_WHEEL`` and ``DOSW1_API_PATH`` pointing to your SDK files and re-run ``install.sh``; for hardware-free checks set ``is_dummy: true``.
   * - ``TimeoutError`` waiting for DOS-W1 state (5 s)
     - Confirm AirBot services are running (``sh ~/dos_w1/airbot/whole_start.sh``), verify ``robot_url`` and ports 50050–50053, and test TCP connectivity from the GPU node.
   * - ``RuntimeError: DOSW1Connection is not connected``
     - Inspect the preceding connection log line for the root cause before ``connect`` failed.
   * - Camera produces no frames
     - Run ``rs-enumerate-devices`` on the robot node to verify every serial in ``camera_serials`` and check USB cables; on headless servers set ``override_cfg.enable_camera_player: false``.
   * - Reward stays 0 or stuck in ``reach``
     - Confirm ``is_dummy: false``, re-calibrate reachable ``target_grasp_joint`` with ``test_dosw1_controller.py``, and try lowering ``joint_reward_sharpness`` to ``1.0``.
   * - ``Missing key runner`` or Hydra errors
     - Start from the full ``dosw1_pick_sac_flow.yaml`` (or ``dosw1_pick_sac_flow_async.yaml``) template instead of a partial snippet.
   * - Training is unstable or diverges
     - Lower ``actor.optim.lr`` and ``actor.critic_optim.lr`` (for example ``1e-4``), increase ``algorithm.replay_buffer.min_buffer_size``, try ``algorithm.gamma`` in ``0.8–0.9``, or warm-start with RLPD via ``algorithm.demo_buffer.load_path``.

To add custom manipulation tasks on DOS-W1, see :doc:`New Real-World Tasks <../../extending/new_task>`.
