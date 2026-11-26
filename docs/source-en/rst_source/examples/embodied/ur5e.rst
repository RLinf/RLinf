UR5e Setup and Control
======================

Set up a UR5e, a USB-RS485 Robotiq 2F-85, one environment-mounted Orbbec Femto
Bolt camera, and a SpaceMouse on one controller computer. Use the camera's RGB
stream, check the driver and
teleoperation, record LeRobot demonstrations, then fine-tune and deploy a
π₀.₅ policy with the same observations and Cartesian-delta actions.

Overview
--------

| **Prerequisites:** A Linux controller computer and a UR5e or URSim e-Series configured as UR5.

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

Setup and Run
-------------

Follow :doc:`Installation </rst_source/start/installation>` for platform
prerequisites and clone the repository. From its root, install and activate the
UR5e and OpenPI environment:

.. tab-set::

   .. tab-item:: UR5e and OpenPI

      .. code-block:: bash

         bash requirements/install.sh embodied --model openpi --env ur5e --venv .venv-ur5e
         source .venv-ur5e/bin/activate
         export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

This installs OpenPI together with ``ur_rtde``, Orbbec capture, SpaceMouse,
serial Modbus, keyboard input, and LeRobot collection dependencies. Keep this environment activated for
subsequent commands.

The Orbbec SDK is installed as ``pyorbbecsdk2==2.1.2`` and imports as
``pyorbbecsdk``. Device permissions and stream checks are covered below.

Check the Driver
~~~~~~~~~~~~~~~~

Check the driver before combining the arm with the collection devices. For
simulation, use URSim e-Series configured as UR5; the integration was checked
with ``universalrobots/ursim_e-series:5.26.3``. Follow the `official URSim Docker
instructions <https://github.com/UniversalRobots/Universal_Robots_Client_Library/blob/master/doc/setup/ursim_docker.rst>`_
to make the controller ports reachable from the driver computer or container.
Replace the example address below with your controller address.

You can first check robot registration and observation shapes against fake SDKs:

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts UR5e --mock \
     --arg robot_ip=192.168.56.101 --arg node_rank=0

A successful check ends with ``OK: every part connected, observed what it declares, and let go.``
To read the controller without starting its control script, use receive-only mode:

.. code-block:: bash

   python -m toolkits.realworld_check.test_ur_controller \
     --robot-ip 192.168.56.101 --receive-only --frequency 10 --samples 10

This logs ten feedback samples and disconnects. Its native UR TCP pose is
``[x, y, z, rx, ry, rz]`` with a rotation vector in radians. To obtain the xyz
Euler representation used by the reset and workspace settings, run:

.. code-block:: bash

   python -m toolkits.realworld_check.test_ur_controller \
     --robot-ip 192.168.56.101 --read-only

Record the ``TCP pose (m, xyz Euler rad)`` line for configuration. This mode
starts the SDK control script, prints the measured state, and disconnects
without sending a position target. It requires Remote Control and
NORMAL / RUNNING controller state.

.. warning::

   Give the driver exclusive controller access. Opening a control session starts
   the SDK's URScript program; disconnecting stops it. Manage power, brakes,
   and safety recovery through PolyScope before connecting.

For an interactive motion check in URSim, omit ``--read-only`` and enter:

.. code-block:: text

   where
   offset 1 0 0
   offset -1 0 0
   quit

Each offset is in millimetres in the base frame, preserves orientation, and
is interpolated over one second; the maximum offset length is 5 mm. Wait for
motion to settle between commands. ``quit``, EOF, and Ctrl-C close the session.
For a real-robot check, follow the robot's bench procedure and keep its normal
stop controls available; closing this session is not an emergency stop.

The servo loop runs on the computer that owns the arm connection. Validate its
timing under the expected load using the `ur_rtde real-time setup guide
<https://sdurobotics.gitlab.io/ur_rtde/pages/core_concepts/guides/realtime_setup.html>`_.

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

The driver accepts an absolute TCP target ``[x, y, z, qx, qy, qz, qw]`` in
the base frame and converts its quaternion to the rotation vector used by
``servoL``. The local servo holds the latest target between environment steps;
it starts on the first motion command.

Set ``frequency`` (default 500 Hz), ``lookahead_time`` (0.1 s), and ``gain``
(300) in the hardware configuration for this position servo.

If the servo fails, subsequent state reads and actions report the error.
Disconnect and resolve the controller fault before reconnecting. The
collection environment's TCP reset is described under `Collect Demonstrations`_.

Check the Orbbec RGB Camera
---------------------------

Follow :doc:`Orbbec camera setup <franka_orbbec>` to find the Femto Bolt serial
number and verify standalone RGB capture. The UR installer already provides
the SDK. The collection template requests 1280 × 720 at 15 fps with depth
disabled. The backend supplies BGR; UR collection records RGB.

Keep the camera mounting, view, crop, and image name consistent between
collection and policy execution. Enter the selected serial in the collection
configuration described below.

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
their ``null`` values with the measured reset target and task limits. The reset
target must lie inside the limits. Requested targets are clipped to the XYZ
bounds and a continuous Euler window around the reset orientation. Each command
limits the step from the measured pose toward the clipped target. Orientation
steps follow the shortest rotation. Intermediate commands can lie outside the
configured bounds. A measured pose outside the bounds does not itself stop the
run. Set ``task_description`` for the demonstrations you will record.
In the hardware entry, ``camera_resolution`` sets the camera output width then
height and ``camera_fps`` requests its rate. In
``env.eval.override_cfg``, ``image_size`` sets the recorded height then width,
defaulting to 224 × 224 after crop and resize; ``camera_names`` and
``camera_crop_regions`` configure the views there. ``step_frequency`` and dataset ``fps`` both default to 10. Keep them
consistent when changing the collection rate. Set
``env.eval.override_cfg.enable_camera_player: true`` to preview the images.

At startup and after a saved episode, the environment returns the TCP to
``reset_ee_pose``. ``open_gripper_on_reset`` defaults to ``false`` to preserve
the gripper state. Set it to ``true`` to open a closed gripper before the TCP
return, with a 0.6 s settling wait as in Franka's pick-and-place task.
The return takes at least ``reset_duration`` seconds, with steps bounded by
``action_scale`` relative to measured feedback. After interpolation, the arm
continues toward the reset pose for up to ``reset_timeout`` seconds. Arrival
uses ``reset_position_tolerance`` in metres and
``reset_orientation_tolerance`` in radians; a timeout reports failure and
attempts to hold the measured pose. Confirm the reset path before collection.

Moving or rotating the SpaceMouse puck commands base-frame Cartesian
increments with the same mapping as Franka. The left button closes the
gripper and the right button opens it. Release the puck to stop adding
increments; the first zero command after motion holds the measured pose.

.. warning::

   Connecting a Robotiq gripper runs its clear-and-activate sequence, which can
   move the fingers. Clear the space around them before connecting. Enabling
   ``open_gripper_on_reset`` also releases the grasp before each TCP reset.

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

Inspect a short episode before collecting a full dataset. With the Robotiq
gripper and the default ``use_relative_frame: True``, recorded and policy
actions are seven normalized channels:
``[dx, dy, dz, droll, dpitch, dyaw, gripper]``. Cartesian increments use the
current end-effector frame. SpaceMouse keeps Franka's base-frame mapping;
its recorded actions are converted to the same frame as policy actions.
``action_scale`` defaults to ``[0.02, 0.1, 1]``; its first two values scale
base-frame translation and Euler increments before workspace limiting. A
negative gripper signal closes the jaws and a positive signal opens them.

The recorded state is
``[gripper_position, x_rel, y_rel, z_rel, roll_rel, pitch_rel, yaw_rel]``.
The pose is relative to the measured pose returned by reset, so its six pose
values start at zero. Width and position use metres, and xyz Euler angles use
radians. The ``main`` RGB view is recorded as ``image``.

Use ``pi05_ur5e`` for training and deployment with this format. Keep the
relative-frame setting, state order, camera view, task prompt, collection
rate, and action scales consistent across both stages.

Fine-Tune π₀.₅
---------------

Use the recorded LeRobot directory as the training dataset. Copy it to the GPU
training host with this version of RLinf and the OpenPI environment. From the
repository root, compute statistics with the same ``pi05_ur5e`` configuration
that SFT and inference will use:

.. code-block:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_ur5e \
     --repo-id ./data/ur5e_dataset

Replace the example path with the dataset directory containing
``meta/info.json``. The command reports where it writes ``norm_stats.json``.
Use that file for both training and deployment.
Statistics use the same data transforms as training: the state keeps its seven
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

.. tab-set::

   .. tab-item:: Training host

      Set the batch size and GPU placement for the training host. Run SFT from
      the repository root:

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

The evaluation template sets ``open_gripper_on_reset: true``: startup and each
episode reset open a closed gripper before returning the TCP. Set it to
``false`` when the task must keep its grasp during reset.

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
