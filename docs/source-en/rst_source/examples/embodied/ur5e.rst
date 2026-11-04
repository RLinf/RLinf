UR5e Setup and Control
======================

Set up a UR5e, a USB-RS485 Robotiq 2F-85, one environment-mounted RealSense
camera, and a SpaceMouse on one controller computer. Check the driver and
teleoperation, record LeRobot demonstrations, then fine-tune and deploy a
π₀.₅ policy with the same observations and Cartesian-delta actions.

Overview
--------

The ``UR5e`` robot composes an arm registered under ``ur_rtde``, an optional
independent Robotiq gripper, and explicitly configured cameras. Its arm action
is an absolute TCP pose in the robot base frame. A local loop sends that target
through the SDK's ``servoL`` position controller. The collection environment
converts lower-rate SpaceMouse increments into these absolute targets.

The driver uses the controller's configured TCP and payload. Prepare those
settings in PolyScope for the tool you fit. URSim with PolyScope 5.26.3 is the
controller simulation used for the integration checks described here.

Hardware Setup
--------------

Prepare a reachable controller before opening the RLinf connection. Use URSim
for initial driver checks, then apply the manufacturer's setup procedure to
your physical arm.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Requirement
   * - Controller
     - UR5e or URSim e-Series configured as UR5. Control sessions require Remote Control and NORMAL / RUNNING state; receive-only sampling requires reachable, enabled RTDE.
   * - Network services
     - Dashboard (29999), Primary / Secondary (30001 / 30002), Real-time (30003), and RTDE (30004) enabled in PolyScope and reachable from the driver. The SDK uses these controller interfaces for its connection and control script.
   * - Controller computer
     - Linux computer with an RLinf Python environment and a network path to the robot controller.
   * - Collection devices
     - SpaceMouse, a physical keyboard or a/b/c pedal, a camera, and a Robotiq 2F-85 connected through USB-RS485 on the same Linux computer.

Follow the `ur_rtde controller setup instructions
<https://sdurobotics.gitlab.io/ur_rtde/pages/getting_started/environment_setup.html>`_
to enable the services and Remote Control. Manage power, brakes, and safety
recovery through your normal controller procedure before connecting.

Installation
------------

Follow :doc:`Installation </rst_source/start/installation>` for platform
prerequisites and clone the repository. From its root, install and activate the
UR5e and OpenPI environment:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env ur5e --venv .venv-ur5e
   source .venv-ur5e/bin/activate
   export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

This installs OpenPI together with ``ur_rtde``, RealSense capture, SpaceMouse,
serial Modbus, keyboard input, and LeRobot collection dependencies. Keep this environment activated for
subsequent commands.

Check the Robot Composition
---------------------------

Before opening a controller connection, run the real robot classes against
fake SDKs:

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts UR5e --mock \
     --arg robot_ip=192.168.56.101 --arg node_rank=0

This constructs the ``UR5e`` robot, connects its ``arm``, checks the declared
observation names and shapes, and disconnects. A successful run ends with
``OK: every part connected, observed what it declares, and let go.`` The fake
endpoint is used only by the SDK fake. ``node_rank=0`` places the arm connection
on node 0 through the scheduler path.

Connect to URSim
----------------

After the composition check passes, start an e-Series URSim instance configured
as UR5. Use the pinned ``universalrobots/ursim_e-series:5.26.3`` image for the
controller version covered here; the `official URSim Docker instructions
<https://github.com/UniversalRobots/Universal_Robots_Client_Library/blob/master/doc/setup/ursim_docker.rst>`_
describe the container and its network configuration.

Run the driver on the Linux Docker host, or attach its container to the same
Docker bridge as URSim. A bridge address such as ``192.168.56.101`` belongs to
that Docker host; another computer needs an explicitly configured network path
or a controller process on the URSim host. Publishing VNC or the web interface
alone does not expose RTDE or the script interfaces.

Replace the address below with your URSim container address and first sample
feedback without opening a control connection:

.. code-block:: bash

   python -m toolkits.realworld_check.test_ur_controller \
     --robot-ip 192.168.56.101 --receive-only --frequency 10 --samples 10

This opens only ``RTDEReceiveInterface``, logs ten samples at a requested 10 Hz,
then disconnects. It reads joint positions and velocities, TCP pose and speed,
and robot and safety modes. The TCP pose in this
output is the native six-value UR representation, ``[x, y, z, rx, ry, rz]``,
with a rotation vector in radians. This mode does not upload, start, or stop a
control script and does not send robot commands.

For the full robot composition check, enable Remote Control in PolyScope and
bring the simulated robot to NORMAL / RUNNING. Run the check with the real SDK:

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts UR5e \
     --arg robot_ip=192.168.56.101 --arg node_rank=0

The check uploads the SDK control script, reads the four arm observations, and
closes both SDK connections. It sends no position target. Read the reported
shapes before proceeding to a task or motion check.

.. warning::

   Opening the control connection uploads and starts the SDK's URScript program;
   closing it stops that program. Give this driver exclusive control of the
   controller even for the observation check. It does not power on the robot,
   release brakes, or recover a safety stop.

Observation and Action
----------------------

Use the arm's named fields when building a task. Each observation is a
``float32`` array, and Cartesian values use the robot base frame.

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Field
     - Shape
     - Meaning
   * - ``tcp_pose``
     - ``(7,)``
     - Position in metres followed by quaternion ``[qx, qy, qz, qw]``. The driver converts the UR rotation vector to this representation.
   * - ``tcp_vel``
     - ``(6,)``
     - Linear velocity in m/s followed by angular velocity in rad/s.
   * - ``arm_joint_position``
     - ``(6,)``
     - Joint angles in radians.
   * - ``arm_joint_velocity``
     - ``(6,)``
     - Joint velocities in rad/s.

Send an action as ``{"arm": {"tcp_pose": pose}}`` through
``Robot.send_action``. The pose contains absolute ``[x, y, z, qx, qy, qz, qw]``
values; the driver normalizes the quaternion and converts it to the rotation
vector expected by ``servoL``. The first action starts the servo loop, which
continues to hold the latest target between task steps. Connecting or reading
observations leaves that loop idle.

``URRTDEArm`` accepts ``frequency`` (default 500 Hz), ``lookahead_time``
(default 0.1 s), and ``gain`` (default 300). These configure the local position
servo; ``gain`` is not a Cartesian stiffness. Requests for Cartesian compliance
are rejected rather than translated to these settings. The declared observations
contain the four fields above; tasks must request only those capabilities.

If the servo fails, subsequent state reads and actions report the error.
Disconnect and resolve the controller fault before reconnecting. The
collection environment's TCP reset is described under `Collect Demonstrations`_.

Run It
------

With the connection and observation checks complete, use the interactive
controller check to exercise the TCP action interface. Begin by reading the
same controller through the local driver:

.. code-block:: bash

   python -m toolkits.realworld_check.test_ur_controller \
     --robot-ip 192.168.56.101 --read-only

This prints measured joints in radians and the TCP pose in metres with an xyzw
quaternion, then disconnects. The additional ``TCP pose (m, xyz Euler rad)``
line gives the six-value representation used by ``reset_ee_pose`` and the pose
limits. Record the reset target from this line; the rotation vector printed by
``--receive-only`` is a different orientation representation. Like the
composition check, it uploads the SDK control script but sends no position
target.

For a deliberate URSim motion check, omit ``--read-only`` and enter the
following commands one at a time:

.. code-block:: bash

   python -m toolkits.realworld_check.test_ur_controller \
     --robot-ip 192.168.56.101

.. code-block:: text

   where
   offset 1 0 0
   where
   offset -1 0 0
   where
   quit

``offset 1 0 0`` translates the measured TCP position by 1 mm along the base
frame's x-axis while preserving its orientation. Each offset has a maximum
length of 5 mm and is interpolated over one second. ``where`` reads the measured
state. Wait for each motion to settle before the next offset.
``quit``, EOF, or Ctrl-C stops the servo session and disconnects.

.. warning::

   Validate the motion procedure in URSim before using it on an arm. For a
   physical test, follow the robot's bench procedure and keep its normal stop
   controls available. Exiting this script closes its control session; it is
   not the robot's emergency-stop function.

The servo loop runs locally on the computer that owns the arm connection,
independently of the environment's target-update rate. Validate control-loop
timing on that computer under its expected load using the
`ur_rtde real-time setup guide
<https://sdurobotics.gitlab.io/ur_rtde/pages/core_concepts/guides/realtime_setup.html>`_.
See :doc:`Robotics Interface </rst_source/concepts/robotics>` for composition
and placement.

Check the RealSense Camera
--------------------------

After the arm check, verify the camera stream before combining the devices for
teleoperation. The collection template uses ``camera_type: realsense`` and the
installed ``pyrealsense2`` SDK. Connect the camera to the controller computer
and query its serial number:

.. code-block:: bash

   python - <<'PY'
   import pyrealsense2 as rs

   for device in rs.context().devices:
       print(device.get_info(rs.camera_info.serial_number))
   PY

Record the serial of the camera you will use. In the following read, replace
``CAMERA_SERIAL`` with that value and set ``resolution`` (width, height) and
``fps`` to a color-stream profile supported by the device. The values shown
match the editable collection template:

.. code-block:: bash

   python - <<'PY'
   from rlinf.robotics import Camera, CameraInfo

   camera = Camera.of(CameraInfo(
       name="main",
       serial_number="CAMERA_SERIAL",
       camera_type="realsense",
       resolution=(640, 480),
       fps=15,
   ))
   try:
       camera.connect()
       frame = camera.get_observation()["frame"]
       print(frame.shape, frame.dtype)
   finally:
       camera.disconnect()
   PY

The backend requests BGR color. The output should show the selected height and
width with three channels, and dtype ``uint8``. The ``finally`` block closes
the capture stream after the read. Use the same serial, resolution, and frame
rate in the collection hardware entry's ``camera_serials``,
``camera_resolution``, and ``camera_fps``. Map that serial to ``main`` in
``camera_names`` to match this view name.

Keep the camera mounting, view, crop, image names, and color processing
consistent across demonstrations and policy execution. Configure the tool's
TCP and payload in PolyScope for the mounted gripper.

Collect Demonstrations
-----------------------

After the driver and camera checks, configure
``examples/embodiment/config/realworld_ur5e_collect_data.yaml``. The recipe keeps
the arm, gripper, environment camera, and environment on node 0. Replace ``ROBOT_IP``, the
Robotiq serial path, and ``CAMERA_SERIAL`` in the hardware entry; replace the
same serial in ``env.eval.override_cfg.camera_names`` so the chosen view is
named ``main``. ``main_image_key`` selects that name.

Fill ``reset_ee_pose``, ``ee_pose_limit_min``, and ``ee_pose_limit_max`` under
``override_cfg`` in ``env/realworld_ur5e_collect.yaml`` before launching. These
fields contain ``[x, y, z, rx, ry, rz]`` in the UR base frame, in metres and
xyz Euler radians, following Franka's pose configuration convention. Replace
their ``null`` values with the measured reset target and task limits. Both the
reset target and starting TCP must lie inside the limits; motion from outside
the range is rejected. The environment clips position and orientation targets
to this range, using a continuous window around the reset orientation. Set
``task_description`` for the demonstrations you will record.
In the hardware entry, ``camera_resolution`` requests native stream width then
height and ``camera_fps`` requests its rate. The template uses 640 × 480 at
15 fps; select a profile supported by the actual RealSense device. In
``env.eval.override_cfg``, ``image_size`` sets the recorded height then width,
defaulting to 224 × 224 after crop and resize; ``camera_names`` and
``camera_crop_regions`` configure the views there. The environment reads the
arm, gripper, and camera observations through the composed robot before
building the policy input. ``step_frequency`` and dataset ``fps`` both default to 10. Keep them
consistent when changing the collection rate. Set
``env.eval.override_cfg.enable_camera_player: true`` to preview the images.

At startup and after a saved episode, the environment returns the TCP to
``reset_ee_pose`` while preserving the gripper state. It interpolates for at
least ``reset_duration`` seconds; the per-step ``action_scale`` limit can
lengthen a distant return. After interpolation, ``reset_timeout`` sets the
arrival-check timeout against ``reset_position_tolerance`` in metres and
``reset_orientation_tolerance`` in radians. Confirm the reset path before
starting collection.

Moving or rotating the SpaceMouse puck commands base-frame Cartesian
increments with the same mapping as Franka. The left button closes the
gripper and the right button opens it. Release the puck to stop adding
increments; the first zero command after motion holds the measured pose.

.. warning::

   Connecting a Robotiq gripper runs its clear-and-activate sequence, which can
   move the fingers. Clear the space around them before connecting. TCP reset
   leaves the gripper unchanged.

Collection uses the existing Linux evdev episode listener. Give the runtime
user access to the SpaceMouse HID device, camera, serial adapter, and physical
keyboard or pedal. Select the latter with ``env.eval.keyboard_device`` or
``RLINF_KEYBOARD_DEVICE`` when discovery is ambiguous. Keys typed into an SSH
terminal are not evdev input from that device.

With the environment activated and the rig configured, start collection:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_ur5e_collect_data

The runner creates a timestamped log directory and writes its LeRobot dataset
under ``lerobot_dataset/`` there. Use the physical device's ``a`` key to start
recording, ``b`` to mark a segment, and ``c`` to finish a successful episode.
Press ``a`` again to discard the active recording; the wrapper ignores a second
``a`` during the first 1.5 s to avoid accidental double taps. The template keeps
successful episodes and targets 20 of them through ``runner.num_data_episodes``.
After a successful episode is saved, collection resets the TCP to its configured
start pose. Restore the task objects before starting the next recording.

.. warning::

   Finishing a successful episode with ``c`` starts the configured TCP reset.
   Confirm that path before saving. SpaceMouse input remains active before
   recording starts and after an episode is aborted.

Check the Recorded Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspect a short recorded episode before collecting a full dataset. The
collection action differs from the arm backend's absolute TCP pose: with the
Robotiq gripper, it has seven normalized channels,
``[dx, dy, dz, droll, dpitch, dyaw, gripper]``. Translation and rotation are
base-frame increments scaled by ``action_scale``. Its default ``[0.02, 0.1, 1]``
means at most 0.02 m and 0.1 rad per axis per full-input step. The final signal
is negative for close and positive for open.

The raw state contains ``tcp_pose`` (seven values, metres and xyzw quaternion)
and ``gripper_position`` (one opening width in metres). ``RealWorldEnv`` sorts
state keys before flattening, so the recorded eight-value state is
``[gripper_position, x, y, z, qx, qy, qz, qw]``. The environment camera supplies one RGB image under the recorded ``image``
key. Its ``main`` name selects the primary image in the environment.

The ``pi05_ur5e`` data configuration maps this image, eight-value state,
seven-value action, and task prompt to OpenPI. It preserves the recorded delta
actions. Keep the state order, camera view, task prompt, collection rate, and
``action_scale`` consistent during collection and deployment.

Fine-Tune π₀.₅
---------------

Use the recorded LeRobot directory as the training dataset. Copy it to the GPU
training host with this version of RLinf and the OpenPI environment. First
compute statistics with the same ``pi05_ur5e`` configuration that SFT and
inference will use:

.. code-block:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_ur5e \
     --repo-id /path/to/ur5e_dataset

Pass the directory containing ``meta/info.json``. The command reports where it
writes ``norm_stats.json``; use that file for both training and deployment.
Statistics use the same data transforms as training: the state keeps its eight
values, and the seven recorded delta-action channels are padded to the model
dimension.

Prepare base weights in the RLinf OpenPI layout using the download and
conversion instructions in :doc:`OpenPI SFT <sft_openpi>`. For this recipe,
convert with ``--action-dim 32``, ``--action-horizon 20``, and
``--max-token-len 200``. The model's 32-value action padding is separate from
the UR environment's seven action channels.

Edit ``examples/sft/config/realworld_ur5e_sft_openpi_pi05.yaml`` with the dataset,
base weights, and statistics paths on the training host:

.. code-block:: yaml

   data:
     train_data_paths: /path/to/ur5e_dataset
   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi_data:
         norm_stats_path: /path/to/ur5e_dataset/norm_stats.json

The template uses one RGB view, the recorded state, and a 20-step prediction
horizon. Set its batch size and GPU placement for the training host, then launch
SFT from the repository root:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh realworld_ur5e_sft_openpi_pi05

SFT saves checkpoints under the run's
``ur5e_openpi_sft/checkpoints/global_step_<N>/`` directory. Keep the dataset's
statistics file with the experiment; the deployment configuration loads it
separately from the checkpoint. See :doc:`OpenPI SFT <sft_openpi>` for training
metrics, checkpoint formats, and resuming a run.

Deploy the Trained Policy
-------------------------

Return to the robot host with the trained checkpoint and the same statistics
file. Configure
``evaluations/realworld/realworld_ur5e_openpi_pi05_eval.yaml`` with the hardware
addresses, serial paths, reset pose, pose limits, camera settings, and task description
used for collection. Its hardware and environment fields follow the collection
configuration. Keep the environment camera in the same position and preserve
``action_scale`` and ``step_frequency``.

Set the model path to the SFT checkpoint directory, and point the statistics
path to the training copy:

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/ur5e_sft/checkpoints/global_step_30000
       openpi_data:
         norm_stats_path: /path/to/ur5e_dataset/norm_stats.json

The template selects ``pi05_ur5e`` in evaluation mode, predicts a 20-step action
sequence, and executes five actions before requesting another sequence. The
environment applies its configured scale to each decoded delta. Start the
existing real-world evaluation entry:

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_ur5e_openpi_pi05_eval

The physical keyboard or pedal controls policy execution: ``a`` starts a
rollout, ``c`` marks success, and ``b`` marks failure. With ``auto_reset: true``,
either outcome ends the episode and returns the TCP to ``reset_ee_pose`` before
waiting for ``a`` again. Restore the task objects before starting another trial.
``max_steps_per_rollout_epoch`` sets the run's step budget, and Ctrl-C closes
the session, including while the operator-start prompt is waiting. The template
records an evaluation video under the run's log directory. See :doc:`Real-World Evaluation
<../../evaluations/guides/realworld>` for the shared evaluation workflow.
