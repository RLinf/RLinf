Real-World Robot Training Launch
=================================

This page explains how to connect **multiple Franka robots** and a **GPU training
node** to one Ray cluster, configure YAML, and start real-world RL training.

For full hardware setup, dependencies, data collection, and task details, see:

- :doc:`../examples/embodied/franka` — standard RealSense + Franka gripper workflow
- :doc:`../examples/embodied/franka_zed_robotiq` — ZED camera + Robotiq gripper on split nodes
- :doc:`../examples/embodied/franka_orbbec` — Orbbec camera with aligned depth

For ``RLINF_NODE_RANK``, ``ray start``, and code sync, see :doc:`multi_node`.
For ``node_groups`` / ``component_placement`` semantics, see :doc:`hetero`.


Typical topology
----------------

Real-world training usually follows **one GPU training node + N robot control nodes**:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - ``RLINF_NODE_RANK``
     - Role
     - Notes
   * - **0** (head)
     - GPU train / rollout
     - Runs ``actor``, ``rollout`` (and optional ``reward``); **only this node** runs the training entry script
   * - **1 … N**
     - Robot control
     - Runs ``env`` workers and the Franka arm parts; one control-node rank per arm (see example docs for shared control nodes)

All nodes must be on the **same LAN** (or overlay network; see :doc:`cloud_edge`), and
``cluster.num_nodes`` must match the number of nodes joined to Ray.

.. important::

   - Control nodes need the Franka controller environment (Franky with bundled libfranka by default); see :doc:`../examples/embodied/franka`.
   - Ray freezes Python and env vars at ``ray start``; install dependencies on **each node** first.
   - Use ``ray_utils/realworld/setup_before_ray.sh`` to align per-node env before ``ray start``.


Connect robot nodes to the Ray cluster
--------------------------------------

Step 1: Prepare the environment on each node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On **every node**, before ``ray start``:

.. code-block:: bash

   # Recommended: source and edit the repo script on each machine
   source ray_utils/realworld/setup_before_ray.sh

   export RLINF_NODE_RANK=<0..N-1>          # unique; GPU node is usually 0
   # If multiple NICs, pin the reachable interface, e.g.:
   # export RLINF_COMM_NET_DEVICES=eth0

On control nodes, also activate the Franka environment if ``setup_before_ray.sh`` does not:

.. code-block:: bash

   source .venv/bin/activate

Step 2: Start Ray
~~~~~~~~~~~~~~~~~

Let ``<head_ip>`` be the GPU head's reachable IP.

**GPU node (rank 0, head):**

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_ip>

**Robot control nodes (rank 1, 2, …):**

.. code-block:: bash

   export RLINF_NODE_RANK=1   # use 2 for the second robot, etc.
   ray start --address='<head_ip>:6379'

Run ``ray status`` on any node and confirm the node count matches ``cluster.num_nodes``.


YAML configuration
------------------

Real-world training is driven by the ``cluster`` section: ``node_groups`` separate GPU
and Franka hardware; ``component_placement`` maps ``actor`` / ``rollout`` / ``env`` to
those resources.

Before running, update:

- ``robot_ip``, ``target_ee_pose`` (task goal pose)
- ``actor.model.model_path`` (pretrained ResNet, etc.)
- ``algorithm.demo_buffer`` / ``data.path`` (for RLPD and other demo-based algorithms)

Single robot (1 GPU + 1 arm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: "4090"
         placement: 0
       env:
         node_group: franka
         placement: 0          # first robot in the franka group
       rollout:
         node_group: "4090"
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0        # GPU training node
       - label: franka
         node_ranks: 1        # robot control node
         hardware:
           type: Franka
           configs:
             - robot_ip: <ROBOT_IP>
               node_rank: 1   # must match the control node's rank in node_ranks

If your GPU node is not rank 0 or the control node is not rank 1, update ``node_ranks``
and ``hardware.configs[].node_rank`` together.

Multiple robots (1 GPU + 2 arms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_2arms.yaml``:

.. code-block:: yaml

   cluster:
     num_nodes: 3
     component_placement:
       actor:
         node_group: "4090"
         placement: 0
       env:
         node_group: franka
         placement: 0-1       # one env worker per robot
       rollout:
         node_group: "4090"
         placement: 0:0-1     # two rollout processes on the same GPU
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: franka
         node_ranks: 1-2
         hardware:
           type: Franka
           configs:
             - robot_ip: <ROBOT_IP_1>
               node_rank: 1
             - robot_ip: <ROBOT_IP_2>
               node_rank: 2

Extend ``num_nodes``, ``node_ranks``, ``placement``, and ``hardware.configs`` the same
way for more arms.

Split camera and arm (ZED + Robotiq)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the **camera is on the GPU server** and the **arm and gripper are on a NUC**,
set camera/gripper types and ``controller_node_rank`` in ``hardware.configs``.
Field details and collection examples are in :doc:`../examples/embodied/franka_zed_robotiq`.

For training, place ``env`` on the Franka node group so that it receives the
Franka hardware configuration. The hardware config's ``node_rank: 0`` keeps the
env worker on the GPU node for camera capture, while ``controller_node_rank: 1``
places the arm and its end effector on the NUC. They are separate parts with
separate connections, and both follow that rank because both are wired to the
same machine:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: gpu
         placement: 0
       env:
         node_group: franka
         placement: 0
       rollout:
         node_group: gpu
         placement: 0
     node_groups:
       - label: gpu
         node_ranks: 0
       - label: franka
         node_ranks: 0-1
         hardware:
           type: Franka
           configs:
             - robot_ip: <ROBOT_IP>
               node_rank: 0
               camera_serials:
                 - "<ZED_SERIAL>"
               camera_type: zed
               gripper_type: robotiq
               gripper_connection: "/dev/ttyUSB0"
               controller_node_rank: 1   # controller on rank 1 (NUC)

.. note::

   Install the ZED SDK on the GPU node into the **same venv Ray uses**, before
   ``ray start``. Configure Robotiq serial permissions on the NUC. See
   :doc:`../examples/embodied/franka_zed_robotiq`.

Kuavo Hardware Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the Kuavo environment dependencies on the robot control node. If the
SDK is not already available through the robot's ROS workspace, point the
installer at its source checkout:

.. code-block:: bash

   KUAVO_SDK_PATH=/path/to/kuavo_humanoid_sdk \
     bash requirements/install.sh embodied --env kuavo

Select the built-in Kuavo composition with ``hardware.type: Kuavo``. The
configuration identifies the platform revision, selected arms, installed end
effector, initial head pose, control frequencies, ROS observation topics, and
safety limits. Joint-position control is currently supported.

.. code-block:: yaml

   - label: kuavo
     node_ranks: 1
     hardware:
       type: Kuavo
       configs:
         - node_rank: 1
           platform_type: 4pro       # 4pro, 5, or 5w
           which_arm: both           # left, right, or both
           end_effector_type: leju_claw  # leju_claw, qiangnao, or rq2f85
           only_arm: true
           direct_to_wbc: false
           qiangnao_dof_needed: 1
           is_binary: false
           head_position: null       # omit to leave the current head pose unchanged
           ros_rate: 10
           control_rate: 100
           controller_node_rank: 1
           feedback_timeout_s: 2.0
           feedback_stale_timeout_s: 0.5
           image_size: [848, 480]
           depth_range: [0, 1500]
           joint_limits_min: [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
           joint_limits_max: [3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
           end_effector_limits_min: [0.0, 0.0]
           end_effector_limits_max: [1.0, 1.0]
           arm_state_keys: [joint_q, gripper]
           obs_key_map:
             head_cam_h:
               topic: /cam_h/color/image_raw/compressed
               msg_type: CompressedImage
               frequency: 30
             wrist_cam_l:
               topic: /cam_l/color/image_raw/compressed
               msg_type: CompressedImage
               frequency: 30
             wrist_cam_r:
               topic: /cam_r/color/image_raw/compressed
               msg_type: CompressedImage
               frequency: 30
             joint_q:
               topic: /sensors_data_raw
               msg_type: sensorsData
               frequency: 500
             leju_claw:
               topic: /leju_claw_state
               msg_type: lejuClawState
               frequency: 500

Install the Kuavo SDK on the robot control node and ROS image message packages
on the camera node. Device dependencies load when their connections open;
constructing the hardware configuration does not connect to the robot.

This integration intentionally rejects mobile-base control, direct WBC mode,
multi-DOF Qiangnao policy actions, binary end-effector actions, and Cartesian
arm control instead of accepting settings it cannot execute.

The robot exports each selected arm under ``left.arm`` or ``right.arm``. Its
canonical observation and action field is ``arm_joint_position`` in radians;
the matching end effector is available as ``left.gripper`` or
``right.gripper`` with a normalized ``0..1`` target. ``KuavoRobot.build()``
composes these arm and end-effector parts and accepts camera descriptors through
its ``cameras`` argument.

For a dual-arm task, select
``examples/embodiment/config/env/realworld_kuavo_joint.yaml``. It uses the shared
``env_type: real`` entry point and registers ``KuavoJointEnv-v1``. This task
requires ``which_arm: both`` and ``arm_state_keys: [joint_q, gripper]``.
Task ``override_cfg.control_mode`` is ``joint``; device settings stay in
``hardware.configs``.
It reads the three camera routes returned by ``active_obs_key_map()`` and
subscribes to their ``topic`` and ``msg_type`` through ``ROSCamera`` parts.
Both ``Image`` and ``CompressedImage`` are supported. Frames are resized to
``image_size``; the camera returns BGR and the environment converts to RGB.
The head frame becomes ``main_images`` and the left/right wrist frames become
``extra_view_images`` in that order. The connection also subscribes to the
``joint_q`` and selected end-effector topics in ``obs_key_map``. Joint feedback
uses ``sensorsData`` and the configured platform's arm slices; Leju claw
feedback uses ``lejuClawState``, while Qiangnao and RQ2F85 use ``JointState``.

For Leju claws, start the ``/control_robot_leju_claw`` service with type
``controlLejuClaw`` before connecting. The connection waits up to
``feedback_timeout_s`` for this service. Commands include both ``left_claw``
and ``right_claw`` names and convert normalized targets to ``0..100``. Updating
one side preserves the other side's accepted target within the step. A rejected
service request raises an error; acceptance confirms command receipt, not that
the claw has reached its target. Check ``lejuClawState`` for measured motion.
Each response must arrive within ``feedback_timeout_s``. A timeout faults the
connection and prevents further claw commands. Close it before reconnecting;
connections in the same process refuse that service while the earlier request
is still pending. A timeout cannot cancel a command already received by the
controller; confirm the controller's state before restarting a worker process.

Connection waits up to ``feedback_timeout_s`` (default ``2.0`` seconds) for
valid arm and gripper feedback before switching control mode. Subsequent reads
reject feedback older than ``feedback_stale_timeout_s`` (default ``0.5``
seconds). Camera reads use ``camera_frame_timeout_s`` (default ``1.0`` second)
and ``camera_stale_timeout_s`` (default ``2.0`` seconds). The two freshness
limits measure time since local message receipt, not synchronization between
sensor timestamps.
Set feedback timeouts in ``hardware.configs`` and camera-read timeouts in the
task's ``override_cfg``.

The first reset opens the connections and reads the current pose without
homing the arms or opening the grippers. SDK connection switches the arms to
external control mode and applies ``head_position`` when configured. Each step
checks all 16 action values and camera freshness before sending a command,
clips targets to the hardware limits, and limits each joint target to
``max_joint_delta`` radians from the measured position (default ``0.05``).
The step interval uses ``ros_rate`` unless ``step_frequency`` is set. Camera
route ``frequency`` describes the publisher; it does not change its rate, and
``control_rate`` does not start a separate high-frequency command loop.

The base task returns zero reward. The configuration selects the existing
``keyboard_reward_wrapper: eval_control`` for operator-scored episodes; choose
the task description and episode length for your setup. Missing, stale, or
malformed camera input raises an error rather than supplying an empty image.
If observation or command dispatch fails during a step, the environment attempts
one hold command using fresh measured arm positions and then rejects further
operation until it is closed and rebuilt. Missing, stale, or out-of-limit joint
feedback also prevents that hold command; a zero target is never substituted.

.. warning::

   Keep the robot's physical emergency stop available. The one-shot hold attempt
   is not an emergency-stop mechanism and cannot guarantee that a robot stops.

   Before launching Python or starting the relevant Ray nodes, set:

   .. code-block:: bash

      export RLINF_ROS_SETUP=preserve

   Apply this setting to the driver, environment, controller, and camera nodes
   that use the existing ROS master. Start the robot's ROS services before the
   task; Kuavo and ``ROSCamera`` connect to those services without starting or
   stopping them. Starting ``roscore`` alone does not start the robot controller
   or camera publishers.

   The setting applies to the whole process, not just Kuavo. Without it, or with
   ``reset``, RLinf keeps its existing cleanup behavior: loading real-world tasks
   stops ROS core processes, and explicit calls to ``realworld_setup()`` repeat
   that cleanup. Some data-collection scripts load tasks before reading YAML,
   so setting a YAML field cannot replace the launch-time environment variable.
   The Kuavo constructor rejects a missing ``preserve`` setting, but cannot undo
   cleanup that an earlier import already performed.

   Do not change this setting during a run. After an initialization error, fix
   the launch environment and restart the process. Configure every task sharing
   the master consistently; a separate task using ``reset`` can still stop it.

For a dual-arm GR00T N1.7 checkpoint, start from
``examples/embodiment/config/model/gr00t_n1d7_kuavo.yaml``. This model config
selects the Kuavo input/output converter with a 16-dimensional state and action
layout: seven left-arm joints, the left gripper, seven right-arm joints, and
the right gripper. Actions contain absolute joint targets in radians and
gripper targets in ``0..1``. ``KuavoJointEnv-v1`` supplies head, left-wrist, and
right-wrist RGB images in that order, together with the task text. Running a
policy also requires matching checkpoint, processor, and backbone assets.

Auto-configuration from environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-machine hardware values (``robot_ip``, ``camera_serials``,
``gripper_connection``, …) do not have to be hard-coded in YAML. Any
``hardware.configs`` field left **unset** is auto-filled during enumeration from
an environment variable named after the field in **upper case**
(``robot_ip`` → ``ROBOT_IP``, ``camera_serials`` → ``CAMERA_SERIALS``), which
keeps IPs and serial numbers out of committed configs.

Rules:

- Values written in YAML always win; environment variables only fill unset fields.
- The variable is read **on the node that owns the config** (its ``node_rank``),
  so export it there *before* ``ray start`` (Ray freezes env vars at start).
- ``node_rank`` and ``controller_node_rank`` are never taken from the environment.

**One robot per node.** The whole value is used; list fields are split on commas:

.. code-block:: yaml

   - label: franka
     node_ranks: 1
     hardware:
       type: Franka
       configs:
         - node_rank: 1          # robot_ip / cameras come from the environment

.. code-block:: bash

   # on the rank-1 control node, before `ray start`
   export ROBOT_IP=<ROBOT_IP>
   export CAMERA_SERIALS=<serial_1>,<serial_2>

**Several robots on one node.** Give one comma-separated value per robot,
assigned to the configs in order (so a list field takes one item per robot). The
value count must match the number of robot configs on that node:

.. code-block:: bash

   # two arms on one control node
   export ROBOT_IP=<ip_arm0>,<ip_arm1>
   export CAMERA_SERIALS=<serial_arm0>,<serial_arm1>

**Create robots entirely from the environment.** Leave ``configs`` empty and the
robots are created from the environment — the robot count is the comma count of
the identifier variable ``ROBOT_IP``, and every field is read from its variable:

.. code-block:: yaml

   - label: franka
     node_ranks: 1
     hardware:
       type: Franka
       configs: []              # robots created from ROBOT_IP, CAMERA_SERIALS, …

.. code-block:: bash

   export ROBOT_IP=<ip_arm0>,<ip_arm1>     # two robots
   export DISABLE_VALIDATE=true,true       # one value per robot

In this mode every variable you set must provide the same number of
comma-separated values as ``ROBOT_IP``; a mismatch aborts enumeration with an
explicit error. Other robot types use their own identifier field
(``GimArm`` → ``CAN_INTERFACE``, ``DOSW1`` → ``ROBOT_URL``).


Launch real-world training
--------------------------

After the Ray cluster is up, YAML is updated, and (for RLPD) demo data is ready, on the
**head node (usually rank 0 on the GPU machine)** ``cd`` to the RLinf repo and run.

If the GPU and control nodes **do not share the same RLinf checkout**, run
``export RLINF_CODE_WORKING_DIR=auto`` in the **same terminal** before the commands
below (see :doc:`multi_node`, Step 3: Enable code sync).

**Single-arm training (peg insertion, RLPD + async SAC):**

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async

**Dual-arm parallel training:**

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async_2arms

**Other tasks (replace the config name as needed):**

.. code-block:: bash

   # Charger task
   bash examples/embodiment/run_realworld_async.sh realworld_charger_sac_cnn_async

   # Async PPO
   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_async_ppo_cnn

``<config_name>`` maps to ``examples/embodiment/config/<config_name>.yaml``; you can pass a custom name.

**Optional: dummy config to validate the cluster (single arm):**

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dummy_franka_sac_cnn

Verify cameras on control nodes and Ray/placement on the head with the dummy config before full training; see :doc:`../examples/embodied/franka` for the full checklist.
