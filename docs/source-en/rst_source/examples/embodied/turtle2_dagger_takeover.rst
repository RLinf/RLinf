Turtle2 Takeover Raw Data Collection
====================================

This example shows how to run **policy rollout + master takeover** raw episode
collection on XSquare Turtle2. It records policy-controlled execution, manual
takeover, hold/recovery behavior after takeover, and the action actually sent
to the robot for debugging, auditing, and later offline cleaning.

This example is **not an online DAgger training config**. It does not start SAC
or DAgger actor updates, and it does not modify the OpenPI loss. If the collected
data should later be used for SFT or DAgger, first clean the saved pickle
episodes into the required expert segments.

Overall Path
------------

Runtime uses one robot execution path:

.. code-block:: text

   OpenPI policy
      -> 14D absolute pose + gripper action
      -> DualAbsolutePoseActionWrapper
      -> MasterTakeoverIntervention
      -> Turtle2 direct pose backend
      -> /follow_pos_cmd_1, /follow_pos_cmd_2
      -> CollectEpisode pickle export

``MasterTakeoverIntervention`` only selects which action should be executed:

- in normal mode, it passes through the policy action;
- in takeover mode, when a fresh master pose is available, it replaces the
  policy action with the master pose;
- in takeover mode, before a fresh master pose is ready, it holds the current
  slave pose;
- after leaving takeover, it keeps holding until the current action chunk
  boundary so stale actions from the old policy chunk are not executed.

The selected action always enters the same Turtle2 direct pose backend. In other
words, policy actions, master takeover actions, and later debug/replay actions
share the same executor as long as they use the same 14D absolute pose contract.

Action and Control Semantics
----------------------------

This example uses dual-arm 14D actions:

.. code-block:: text

   [left_x, left_y, left_z, left_roll, left_pitch, left_yaw, left_gripper,
    right_x, right_y, right_z, right_roll, right_pitch, right_yaw, right_gripper]

Important details:

- ``action_mode`` must be ``absolute_pose``. The master sends absolute end-effector
  poses, which must not be interpreted as relative/delta actions.
- ``pose_control_backend`` is set to ``direct``. The direct backend publishes
  ``PosCmd``-style commands to ``/follow_pos_cmd_1`` and ``/follow_pos_cmd_2``
  and does not use the RLinf smooth interpolation path.
- The direct backend first checks shape, finite values, and safety bounds. An
  accepted action satisfies ``executed_action == raw_action``. A rejected action
  is not published; the robot holds the previous accepted action, and the info
  dict records ``action_rejected`` and ``rejection_reason``.

Pure deployment can still use the Turtle2 deploy default ``relative_pose`` if
the policy was trained with delta actions. Takeover raw collection must use
``absolute_pose`` because the master pose is already absolute. If a checkpoint
was trained with relative/delta actions, use it with normal ``relative_pose``
deployment instead of this takeover collection config; otherwise the policy
action and master action contracts will not match.

Takeover Protocol
-----------------

The RLinf env starts a TCP server, and the master program connects as a client.
The protocol keeps three X2Robot takeover frame types:

- ``MSG_MODE`` synchronizes the current mode, such as normal/policy mode or
  takeover mode.
- ``MSG_JOINT`` sends the current slave joint snapshot to the master so the
  master can align before takeover.
- ``MSG_POSE`` sends the master's 14D absolute pose to RLinf.

``MSG_JOINT`` is used only for pre-takeover synchronization. The robot action
that gets executed comes from the policy or from the master's 14D pose and is
sent through the same direct pose backend.

Mode switching is done through a ROS parameter:

.. code-block:: bash

   # policy rollout
   rosparam set /running_mode 1

   # master takeover
   rosparam set /running_mode 2

   # return to policy at the next action chunk boundary
   rosparam set /running_mode 1

When ``running_mode_source`` is ``ros_param``, the real runtime environment must
be able to import ``rospy``, and ``/running_mode`` must be convertible to an
integer. Otherwise RLinf raises an explicit error instead of silently falling
back to policy mode.

Prerequisites
-------------

Before running this example, verify the following:

- The required external repositories are prepared on the corresponding machines:

  - The master machine uses
    `gen-robot/x2robot-master@dev <https://github.com/gen-robot/x2robot-master/tree/dev>`_
    to start the master arms, gripper/joystick input node, and
    ``bi_teleop_master.py`` TCP client.
  - The Turtle2/slave control machine uses
    `gen-robot/x2robot-slave@dev/dagger <https://github.com/gen-robot/x2robot-slave/tree/dev/dagger>`_
    for Turtle2 low-level ROS, SDK, cameras, and follower-controller support.
    In this RLinf flow, do not start its ``bi_teleop_slave.py`` or
    ``socket2ros_async.py`` as an action executor.

- Turtle2 ROS, SDK, cameras, and follower controllers are running and publishing
  robot state.
- ``/follow_pos_cmd_1`` and ``/follow_pos_cmd_2`` are published only by the RLinf
  direct backend. Do not start legacy slave executors or other processes that
  write follower pose commands at the same time.
- The master program is connected to the RLinf takeover TCP server and can send
  ``MSG_POSE`` frames.
- ``/running_mode`` uses integer mode values. This example defaults to ``1`` for
  policy and ``2`` for takeover.
- The OpenPI checkpoint path is provided by ``RLINF_OPENPI_MODEL_PATH`` or by
  ``rollout.model.model_path`` in the config.

Multi-Node Startup Flow
-----------------------

The following is a generic three-end startup sequence. Replace container names,
network interfaces, IP addresses, ROS setup files, and master program paths with
the values from your deployment.

1. Turtle2 Control Node: Start Low-Level ROS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Turtle2 control node owns the ROS master, robot bring-up, cameras, and
follower controllers. If you use two containers, the control container owns
low-level ROS, while the RLinf container only runs the Ray worker and RLinf env.

Prepare the ``dev/dagger`` branch of ``x2robot-slave`` on this machine. Example:

.. code-block:: bash

   # Run on the Turtle2/slave control machine. Replace the workspace path.
   git clone -b dev/dagger https://github.com/gen-robot/x2robot-slave.git \
     <slave_ros_ws>/src/x2robot-slave
   cd <slave_ros_ws>
   catkin_make
   source devel/setup.bash

If the repository already exists, verify the branch and revision before startup:

.. code-block:: bash

   cd <slave_ros_ws>/src/x2robot-slave
   git fetch origin
   git switch dev/dagger
   git pull --ff-only
   git rev-parse --short HEAD

.. code-block:: bash

   # Run on the Turtle2 control node.
   ssh <turtle2_worker_host>

   # Enter the Turtle2 control container or an equivalent ROS environment.
   <enter_turtle2_ros_env>

   # Start a stable ROS master.
   tmux kill-session -t stable_roscore 2>/dev/null || true
   tmux new-session -d -s stable_roscore \
     'source /opt/ros/noetic/setup.bash; while true; do roscore; sleep 1; done'

   # Start low-level Turtle2 bring-up. Replace the path for your installation.
   bash <path_to_turtle2_bring_up>/run.sh S

   # Set the default policy mode.
   rosparam set /running_mode 1

   # Basic checks: state, cameras, and command topics must be visible.
   rostopic info /follow_pos_cmd_1
   rostopic info /follow_pos_cmd_2
   timeout 10 rostopic echo -n 1 /follow1_pos_back
   timeout 10 rostopic echo -n 1 /follow2_pos_back
   rostopic info /camera1/usb_cam1/image_raw
   rostopic info /camera2/usb_cam2/image_raw
   rostopic info /camera3/usb_cam3/image_raw

If a UI is used for temporary calibration, close any UI or legacy executor that
publishes ``/follow_pos_cmd_*`` before running this example. These command topics
should be owned by the RLinf direct backend during collection.

2. GPU / Ray Head Node: Start Ray Head
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPU node runs OpenPI policy inference, the rollout entry point, and the Ray
head.

.. code-block:: bash

   # Run on the GPU / Ray head node.
   ssh <gpu_head_host>
   cd <path_to_rlinf_repo>
   source .venv/bin/activate

   export RLINF_NODE_RANK=0
   export RLINF_COMM_NET_DEVICES=<head_network_interface>
   export PYTHONPATH=<path_to_rlinf_repo>:$PYTHONPATH
   export RLINF_OPENPI_MODEL_PATH=<path_to_openpi_checkpoint>

   ray stop --force || true
   ray start --head \
     --node-ip-address=<head_ip> \
     --port=6379 \
     --disable-usage-stats

3. Turtle2 / Ray Worker Node: Start RLinf Worker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RLinf worker must be started from a shell or container that can access the
Turtle2 ROS environment. Ray records the current Python interpreter and
environment variables when it starts, and later EnvWorker processes inherit
those settings.

.. code-block:: bash

   # Run on the Turtle2 worker node.
   ssh <turtle2_worker_host>

   # Enter the RLinf runtime environment and source Turtle2 ROS / SDK setup.
   <enter_rlinf_env>
   cd <path_to_rlinf_repo>
   source /opt/ros/noetic/setup.bash
   source <path_to_turtle2_ros_setup>
   source .venv/bin/activate

   export RLINF_NODE_RANK=1
   export RLINF_COMM_NET_DEVICES=<worker_network_interface>
   export PYTHONPATH=<path_to_rlinf_repo>:$PYTHONPATH

   ray stop --force || true
   ray start --address=<head_ip>:6379 \
     --node-ip-address=<worker_ip> \
     --disable-usage-stats

   rosparam set /running_mode 1

Return to the head node and run ``ray status --address=<head_ip>:6379`` to verify
that both the GPU node and the Turtle2 worker have joined the cluster.

4. Master Node: Start the Master Program
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The master node reads the operator input and connects to the RLinf env-side
takeover server as a TCP client. Start the master arm ROS nodes first, then start
the client that sends takeover frames. If gripper input depends on a separate
communication node, start that node first as well.

Prepare the ``dev`` branch of ``x2robot-master`` on this machine. Example:

.. code-block:: bash

   # Run on the master machine. Replace the workspace path.
   git clone -b dev https://github.com/gen-robot/x2robot-master.git \
     <master_ros_ws>/src/x2robot-master
   cd <master_ros_ws>
   catkin_make
   source devel/setup.bash

If the repository already exists, verify the branch and revision before startup:

.. code-block:: bash

   cd <master_ros_ws>/src/x2robot-master
   git fetch origin
   git switch dev
   git pull --ff-only
   git rev-parse --short HEAD

.. code-block:: bash

   # Run on the master node.
   ssh <master_host>
   source /opt/ros/noetic/setup.bash
   source <master_ros_ws>/devel/setup.bash

   # Terminal 1: start the master arms. Choose the command used by your install.
   roslaunch arx_x5_controller_moving open_master_moving.launch
   # Or:
   # roslaunch <master_ros_ws>/src/x2robot-master/open_master_moving.launch

.. code-block:: bash

   # Terminal 2: start the master-side gripper / joystick input node.
   ssh <master_host>
   source /opt/ros/noetic/setup.bash
   source <master_ros_ws>/devel/setup.bash
   roslaunch communication communication_MS.launch

.. code-block:: bash

   # Terminal 3: start the master TCP client and connect to the RLinf server.
   ssh <master_host>
   source /opt/ros/noetic/setup.bash
   source <master_ros_ws>/devel/setup.bash

   cd <master_ros_ws>/src/x2robot-master/scripts
   python3 bi_teleop_master.py \
     --server-host <worker_ip> \
     --server-port <master_takeover_port> \
     --send-rate 100

``master_takeover.port`` is set by ``env.eval.master_takeover.port`` in the YAML.
If a legacy slave server already occupies the default port, use the same free
port in both the YAML and the master command.

5. Head Node: Start Eval and Raw Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run on the Ray head node:

.. code-block:: bash

   cd <path_to_rlinf_repo>
   source .venv/bin/activate

   export RLINF_NODE_RANK=0
   export RLINF_COMM_NET_DEVICES=<head_network_interface>
   export PYTHONPATH=<path_to_rlinf_repo>:$PYTHONPATH
   export RLINF_OPENPI_MODEL_PATH=<path_to_openpi_checkpoint>
   export RLINF_REALWORLD_DATA_DIR=<path_to_save_pickle_episodes>

   bash examples/embodiment/run_realworld_eval.sh realworld_turtle2_dagger_takeover_collect_openpi

The example config is:

.. code-block:: text

   examples/embodiment/config/realworld_turtle2_dagger_takeover_collect_openpi.yaml

Key fields:

.. code-block:: yaml

   runner:
     only_eval: True

   rollout:
     collect_transitions: False

   env:
     eval:
       use_master_takeover: True
       action_mode: absolute_pose
       data_collection:
         enabled: True
         export_format: "pickle"
         record_executed_action: True
       master_takeover:
         running_mode_source: "ros_param"
         running_mode_param: "/running_mode"
         normal_mode_value: 1
         takeover_mode_value: 2
       override_cfg:
         pose_control_backend: direct
         use_arm_ids: [0, 1]

``algorithm.adv_type`` and ``algorithm.loss_type`` use registered placeholder
values only for generic config validation. Because ``runner.only_eval`` is true,
this example does not instantiate a training loss or run online updates.

6. Switch Takeover Mode
~~~~~~~~~~~~~~~~~~~~~~~

After the eval log shows that the master client is connected, switch
``/running_mode`` from the Turtle2 control node:

.. code-block:: bash

   # policy rollout
   rosparam set /running_mode 1
   sleep 5

   # master takeover
   rosparam set /running_mode 2
   sleep 5

   # return to policy
   rosparam set /running_mode 1

After switching back to ``1``, RLinf resumes policy inference at the action chunk
boundary.

7. Verify Saved Results
~~~~~~~~~~~~~~~~~~~~~~~

After the run finishes, check ``RLINF_REALWORLD_DATA_DIR`` or the configured
``data_collection.save_dir``. The pickle episode should contain fields such as
``observations``, ``actions``, ``infos``, ``success``, and ``intervened``. If
takeover happened, ``infos`` should include metadata such as ``intervene_flag``,
``intervene_action``, and ``takeover_active``.

Data Recording Semantics
------------------------

This example recommends ``pickle`` export for raw episodes instead of direct
LeRobot export.

The reason is that raw takeover episodes often need to preserve the full
context:

- policy-controlled segments;
- master takeover segments;
- pre-takeover synchronization and hold;
- chunk-boundary recovery after leaving takeover;
- rejected-action metadata such as ``raw_intervene_action``, ``action_rejected``,
  and ``rejection_reason``;
- the actual or held ``executed_action``.

The config sets:

.. code-block:: yaml

   data_collection:
     export_format: "pickle"
     record_executed_action: True

Therefore ``episode["actions"]`` prefers the action that was actually executed:

- if the backend accepts an action, it records the accepted ``executed_action``;
- if a takeover action is accepted, ``intervene_action`` is aligned with
  ``executed_action``;
- if a takeover action is rejected, the rejected candidate is kept only in
  ``episode["infos"]``, while ``episode["actions"]`` records the action the robot
  actually held.

``CollectEpisode`` still supports LeRobot export, but LeRobot is better suited
for cleaned demonstration datasets. This example saves raw takeover episodes. If
you later want to train OpenPI, first clean usable expert segments from the
pickle episodes, then convert those segments to LeRobot separately.

Notes
-----

- This example is eval-only raw collection. Whether a saved pickle episode is
  used for training is decided by later offline cleaning and training configs.
- Do not start another executor that publishes ``/follow_pos_cmd_1`` or
  ``/follow_pos_cmd_2`` at the same time.
- If the policy checkpoint was trained with relative/delta actions, use a normal
  ``relative_pose`` deployment config instead of this takeover collection config.
