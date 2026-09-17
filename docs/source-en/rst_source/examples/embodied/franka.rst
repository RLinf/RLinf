Real-World RL with Franka
============================

This page walks you through training a CNN policy on a Franka arm with RLinf,
from demonstration collection to online RLPD training. The default setup uses
one x86-64 computer with an NVIDIA GPU running Ubuntu 20.04 or 22.04. Franky
controls the arm through Python bindings to libfranka, and the same computer
runs rollout and training. You first prepare that host
(firmware check, real-time kernel, GPU driver for that kernel), install RLinf
natively or with Docker, then run the peg-insertion example. The later sections cover a
separate controller node, the legacy ROS backend for existing deployments, and
other Franka workflows.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: Franka arm used for real-world reinforcement learning

   Franka arm used for real-world reinforcement learning.

Overview
------------

The policy learns from camera images and robot state. Successful demonstrations
fill the initial replay data.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      CNN policy

   .. grid-item-card:: Algorithms
      :text-align: center

      SAC / RLPD

   .. grid-item-card:: Tasks
      :text-align: center

      Peg insertion

   .. grid-item-card:: Hardware
      :text-align: center

      Franka · RealSense · NVIDIA GPU

Tasks
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Task
     - Config
     - Description
   * - Peg insertion
     - ``realworld_collect_data``, ``realworld_peginsertion_rlpd_cnn_async``
     - Collect SpaceMouse demonstrations, then train asynchronously on
       demonstrations and live robot experience. The SpaceMouse also provides
       human intervention during training.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - Observation
     - RGB images from the first configured camera (``wrist_1``) and robot state.
   * - Action
     - Six Cartesian position and rotation deltas; the gripper stays closed.
   * - Reward
     - Success when the end-effector pose reaches the configured target tolerance.

Hardware Setup
------------------

You need a Franka arm, a RealSense camera, a SpaceMouse, and an x86-64 computer
with an NVIDIA GPU. Connect the arm to the computer through a wired network
interface, and connect the camera and SpaceMouse by USB.

The same software can run in two layouts. This page follows the single-machine
layout; the multi-node layout reuses its steps and is described in
`Multi-Node Setup`_.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Layout
     - Machines
     - When to use it
   * - Single machine (default)
     - One Ubuntu 20.04 or 22.04 GPU host runs Franky robot control, rollout,
       and training.
     - Most setups. One computer to install and maintain.
   * - Multi-node (optional)
     - A controller computer, which needs no GPU, runs Franky robot control; a
       GPU server runs actor and rollout.
     - You want to isolate robot control from training load, or the GPU server
       is not next to the robot.

.. warning::

   Keep the emergency stop within reach and have an operator supervise every
   hardware run. Secure the peg and fixture, clear the workspace, and check
   that reset motions are safe. This task resets approximately 10 cm above
   the target with randomized horizontal position and yaw. Do not copy a
   target pose from another robot.

Prepare the Robot Host
--------------------------

Before installing RLinf, check the firmware to choose a compatible libfranka
version, then prepare the kernel for the arm's 1 kHz control loop. On a GPU
host, make sure ``nvidia-smi`` works with that kernel before running
``requirements/install.sh``; otherwise, the installer selects CPU-only PyTorch.

Check the Firmware and Choose libfranka
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open Franka Desk at ``http://<robot_ip>/desk``, go to ``SETTINGS``, and record
the version shown after ``Control`` on the dashboard:

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_firmware.png
   :align: center
   :width: 60%
   :alt: Franka Desk dashboard showing the Control firmware version

   Control firmware version in Franka Desk.

Look up that version in the
`Franka compatibility table <https://frankarobotics.github.io/docs/compatibility.html>`_
and choose a libfranka version. Franky ships as prebuilt wheels that bundle
libfranka, and wheels exist for libfranka 0.19.0 (the installer default) and
0.15.0. If your firmware needs another libfranka version, use the
`Legacy ROS Backend (Optional)`_, which builds libfranka from source.

RLinf has been tested with firmware 5.9.2 using libfranka 0.19.0 on a standard
kernel with the real-time check disabled, and with firmware 5.7.2 up to 5.9.0
using libfranka 0.15.0 on a real-time kernel. Do not change robot firmware
merely to match this example.

Install a Real-Time Kernel (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

libfranka sends a command to the arm every millisecond. A PREEMPT_RT kernel
keeps that loop on schedule while rollout and training load the CPU and GPU.
Franky also starts on a standard kernel, as described in
`Run without a Real-Time Kernel`_, but the real-time kernel remains the
recommended setup.

Install the kernel on the host, not inside Docker: containers share the host
kernel. Follow Franka's
`real-time kernel guide <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_
for your Ubuntu release. It builds a patched kernel and installs its
``linux-image`` and ``linux-headers`` packages. Keep your current kernel as a
GRUB fallback. After rebooting into the new kernel, check it:

.. code:: bash

   uname -r
   cat /sys/kernel/realtime

The second command must print ``1``. If it does not, select the real-time
kernel under GRUB's advanced options before continuing.

Install the NVIDIA Driver on the Real-Time Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After booting the real-time kernel, build the NVIDIA module against the
matching ``linux-headers`` installed above. Pass
``IGNORE_PREEMPT_RT_PRESENCE=1`` to the installation command to bypass the
driver's PREEMPT_RT build check. For an APT installation, follow NVIDIA's
`driver installation guide <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html>`_
to configure the repository and choose a package for your GPU and Ubuntu
version, then replace ``DRIVER_PACKAGE`` below with that package name:

.. code:: bash

   sudo env IGNORE_PREEMPT_RT_PRESENCE=1 apt-get install -y DRIVER_PACKAGE
   sudo reboot

If the driver is already installed through DKMS, build its module for the
running real-time kernel instead:

.. code:: bash

   sudo env IGNORE_PREEMPT_RT_PRESENCE=1 dkms autoinstall -k "$(uname -r)"
   sudo reboot

After rebooting into the real-time kernel, run ``nvidia-smi`` and confirm it
lists the GPU before installing RLinf. Apply the same environment variable
whenever a driver or kernel update rebuilds the module.

.. warning::

   ``IGNORE_PREEMPT_RT_PRESENCE=1`` only bypasses the build check; NVIDIA does
   not officially support PREEMPT_RT kernels. If the GPU is unavailable, boot
   the previous kernel to recover.

Allow Real-Time Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When RLinf starts Franky, it tries to lock the control process's memory, run
it with ``SCHED_FIFO`` priority 80, and pin it to CPU cores. Each step needs
permission, and RLinf logs a warning and continues when one is denied. Grant
the permissions by adding your login account to a dedicated group:

.. code:: bash

   getent group realtime || sudo groupadd realtime
   sudo usermod -aG realtime "$(id -un)"
   sudoedit /etc/security/limits.d/99-rlinf-realtime.conf

Add the following limits to that file, then log out and back in:

.. code:: text

   @realtime - rtprio 99
   @realtime - memlock unlimited

In the new login shell, ``ulimit -r`` should print ``99`` and ``ulimit -l``
should print ``unlimited``. These limits help on a standard kernel too.

Run without a Real-Time Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you cannot use a real-time kernel, no extra step is needed. Franky reads
libfranka's real-time mode from ``realtime_config`` in the robot's hardware
config, and the default ``ignore`` lets libfranka run on a kernel without
PREEMPT_RT. RLinf logs a warning when the running kernel is not real-time.

To make sure the arm never runs on a standard kernel, set ``enforce`` in the
hardware config described in `Run It`_; Franky then refuses to start on a
kernel without PREEMPT_RT:

.. code:: yaml

   configs:
     - robot_ip: ROBOT_IP
       node_rank: 0
       camera_serials: ["CAMERA_SERIAL"]
       realtime_config: enforce

.. warning::

   On a standard kernel, heavy training load can make libfranka miss control
   deadlines. The robot then stops with ``communication_constraints_violation``
   reflexes. Before training, verify control under your intended training load
   with an operator present.

Installation
----------------

With the driver and kernel in place, one installation on the GPU host provides
both Franky robot control and the training dependencies. You can install
natively or use the Docker image described at the end of this section. Clone
RLinf and run subsequent commands from its root directory:

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

Install the Franka Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the installer. Prefix the command with ``LIBFRANKA_VERSION=0.15.0`` only if
your firmware requires libfranka 0.15.0:

.. code:: bash

   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

What this does:

1. Creates the ``.venv`` virtual environment with RLinf and the embodied
   training dependencies, including the PyTorch build matched to your driver.
2. Installs system packages through APT, which needs sudo. Pass ``--no-root``
   only if those packages are already installed.
3. Installs the Franka dependencies, LeRobot, and a prebuilt
   ``franky-control`` wheel that bundles libfranka.

The wheel targets manylinux 2.28, so the same environment works on Ubuntu
20.04 and 22.04. The installer reads these variables:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Effect
   * - ``LIBFRANKA_VERSION``
     - ``0.19.0``
     - libfranka version of the Franky wheel. Prebuilt wheels exist only for
       ``0.15.0`` and ``0.19.0`` on x86-64; any other version or architecture
       stops the installer early unless ``FRANKY_WHEEL`` is set.
   * - ``FRANKY_WHEEL``
     - unset
     - URL or local path of a ``franky-control`` wheel to install instead of
       the prebuilt one, for example when the host cannot download from
       GitHub or you built a wheel for another libfranka version.

Pass ``--venv <name>`` to install into another directory, and ``--use-mirror``
for faster downloads from mainland China.

Your account must be able to read the camera and SpaceMouse USB devices; see
the `SpaceMouse setup <https://github.com/JakubAndrysek/PySpaceMouse#installation>`_
for its udev rule.

Check the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

In the activated environment, confirm that PyTorch sees the GPU and that Franky
loads:

.. code:: bash

   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   python -c "import franky"

The first command must print ``True``. If it prints ``False``, check
``nvidia-smi`` and re-run the installer after fixing the driver. The second
command prints nothing when Franky and its bundled libfranka load.

Use the Docker Image
~~~~~~~~~~~~~~~~~~~~~~~~

Instead of installing natively, you can run the
``rlinf/rlinf:agentic-rlinf0.4-franka`` image. It is built on CUDA 12.8 and
Ubuntu 22.04, and its environments carry CUDA PyTorch, so one container runs
the actor, rollout, and robot control on the single-machine GPU host. The host
still needs driver 570 or newer and the
`NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.
The image contains these environments, switched with
``source switch_env <name>``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Environment
     - Contents
   * - ``franky``
     - Franky backend with libfranka 0.19.0 and the dexterous-hand
       dependencies. Active by default.
   * - ``openvla``, ``openvla-oft``, ``openpi``, ``gr00t``
     - The ``franky`` contents plus that VLA policy.

Start the container with access to the GPU, robot, camera, and SpaceMouse:

.. code:: bash

   docker run -it --name rlinf-franka \
     --gpus all --network host --privileged \
     --ulimit rtprio=99 --ulimit memlock=-1 \
     -v "$PWD:/workspace/RLinf" -w /workspace/RLinf \
     rlinf/rlinf:agentic-rlinf0.4-franka bash

For mainland China downloads, replace ``rlinf/rlinf`` in the image name
with ``infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf``, keeping the tag unchanged.

The container opens in the ``franky`` environment. The ``--ulimit`` flags grant
the scheduling permissions from `Allow Real-Time Scheduling`_ inside the
container; the kernel is still the host's. The image's Franky environment
bundles libfranka 0.19.0, so firmware that needs 0.15.0 installs natively with
``LIBFRANKA_VERSION=0.15.0``. For another shell, run
``docker exec -it rlinf-franka bash``; if you switched environments, select the
same one again.

Download the Model
----------------------

Download the pretrained ResNet encoder into the repository:

.. code:: bash

   hf download RLinf/RLinf-ResNet10-pretrained \
     --local-dir ./models/RLinf-ResNet10-pretrained

The training command below supplies this directory to both actor and rollout.

Run It
----------

The run has three stages on the robot host: configure the camera and measure
the target pose, collect demonstrations, then train. Keep the environment
activated in every shell.

Check the Camera and Target Pose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, check the camera stream and record its serial number:

.. code:: bash

   python toolkits/realworld_check/test_franka_camera.py

Configure the robot in these two existing recipes:

- ``examples/embodiment/config/realworld_collect_data.yaml``
- ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``

In each file, replace only the ``label: franka`` entry under
``cluster.node_groups`` with the following block. Replace ``ROBOT_IP`` with
the arm's address and ``CAMERA_SERIAL`` with the printed serial number:

.. code:: yaml

   - label: franka
     node_ranks: 0
     hardware:
       type: Franka
       configs:
         - robot_ip: ROBOT_IP
           node_rank: 0
           camera_serials: ["CAMERA_SERIAL"]

The config needs no ``backend`` key: the arm uses the default Franky backend,
with ``realtime_config: ignore`` unless you add ``enforce`` as shown in
`Run without a Real-Time Kernel`_. Set ``cluster.num_nodes: 1`` in the training
recipe as well; collection already uses one node. Keep the training recipe's
``4090`` node group and component placement unchanged: the group names GPU
node 0, regardless of the GPU model. Use one camera for this example; it is
named ``wrist_1`` automatically.

Use the robot's guiding mode to position the peg at the desired successful
insertion pose, then unlock the arm and activate FCI in Franka Desk. Read the
pose with the controller check tool, which connects to the arm through Franky:

.. code:: bash

   export FRANKA_ROBOT_IP=192.168.1.10  # Replace with your robot's address.
   python -m toolkits.realworld_check.test_franka_controller

At the prompt, enter ``getpos_euler``, then ``q`` to release the robot.
The result is ``[x, y, z, roll, pitch, yaw]``, in metres and radians. The tool
also accepts ``getpos``, ``getjoint``, ``getstate``, ``gethand``, ``clear``,
``home``, ``open``, and ``close``, and takes ``--robot-ip`` instead of the
environment variable and ``--realtime-config enforce`` to require a real-time
kernel. Save the six measured numbers as a comma-separated list in this shell:

.. code:: bash

   export FRANKA_TARGET_POSE='[x, y, z, roll, pitch, yaw]'  # Replace all six entries.

Before proceeding, confirm that the target and the reset region described
above are within the safe workspace. Close other programs that control the arm;
only one process can hold its control connection.

.. _franka-motion-settings:

Configure Arm Motion
~~~~~~~~~~~~~~~~~~~~

Collection, training, and evaluation use the same Franky Cartesian controller.
Its defaults apply when the hardware entry has no ``compliance`` mapping, so
the recipes above need no additional controller settings. They also apply to
PICO and SpaceMouse teleoperation and to GELLO when it commands Cartesian poses.

.. list-table:: Franky Controller Defaults
   :header-rows: 1
   :widths: 30 15 55

   * - Setting
     - Default
     - Meaning
   * - ``translational_stiffness``
     - 1000 N/m
     - Initial stiffness for position errors.
   * - ``rotational_stiffness``
     - 50 Nm/rad
     - Initial stiffness for orientation errors.
   * - ``translational_clip``
     - 0.008 m
     - Initial position-error limit used to compute restoring torque.
   * - ``rotational_clip``
     - 0.04 rad
     - Initial orientation-error limit used to compute restoring torque.
   * - ``max_step``
     - 0.03 m
     - Maximum position change from the previous commanded target per call.
   * - ``max_step_rad``
     - 0.10 rad
     - Maximum orientation change from the previous commanded target per call.

The first target is limited relative to the measured pose. These target-change
limits are per call, not speeds; the error clips bound the error used by the
controller, not the workspace. The environment still applies its action scale
and workspace bounds before sending a target.

At reset, a task can request different stiffness and error clips through
``compliance_param``. Franky limits these requests using the hardware settings:
by default, stiffness is capped at 1200 N/m and 80 Nm/rad, and error clips have
floors of 5 mm and 0.02 rad. For peg insertion, reset produces those stiffness
values with position clips of 5 mm in x/y and 10 mm in z, and orientation clips
of 0.02 rad. ``max_step`` and ``max_step_rad`` remain in effect after reset.
Other tasks keep their own reset requests.

To change a hardware setting, add only its override beside ``robot_ip`` in
each recipe that should use it. For example, this limits each position-target
change to 2 cm while keeping the other Franky defaults:

.. code:: yaml

   compliance:
     max_step: 0.02

A misspelled key raises when the Franky arm is declared, before it connects.
In Python, ``FrankyArm(..., compliance={"max_step": 0.02})`` follows the same
rule. A ``CartesianCompliance`` object instead supplies a complete set of
values; its fields are used as given, including its own defaults.

The legacy ``franka_ros`` backend keeps its controller configuration and
ignores this hardware mapping. Joint commands, including dual-arm GELLO
teleoperation, use joint control rather than these Cartesian settings.

Collect Demonstrations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the node rank before starting Ray. If a hardware-check script started a
local Ray instance, stop that instance first so Ray captures the activated
environment and the correct rank:

.. code:: bash

   ray stop
   export RLINF_NODE_RANK=0
   ray start --head

Move and rotate the SpaceMouse puck to control the end effector. The task
marks success automatically when the target tolerance is reached and resets
for the next demonstration. Collect 20 successful demonstrations:

.. code:: bash

   RLINF_LOG_DIR="$PWD/logs/franka-demo" \
     bash examples/embodiment/collect_data.sh realworld_collect_data \
     "env.eval.override_cfg.target_ee_pose=$FRANKA_TARGET_POSE"

The collector saves successful trajectories under ``logs/franka-demo/demos``.
This replay-buffer directory is the input for RLPD, not the optional episode
exports under ``collected_data``. Wait for the collector to finish and release
the arm before starting training. Use a new log directory for a new collection
session to keep demonstration sets separate. To collect with a GELLO device
instead of the SpaceMouse, see :doc:`franka_gello`.

Train the Policy
~~~~~~~~~~~~~~~~~~~~

With the same environment, Ray instance, and target pose, start training:

.. code:: bash

   bash examples/embodiment/run_realworld_async.sh \
     realworld_peginsertion_rlpd_cnn_async \
     "env.train.override_cfg.target_ee_pose=$FRANKA_TARGET_POSE" \
     "algorithm.demo_buffer.load_path=$PWD/logs/franka-demo/demos" \
     "actor.model.model_path=$PWD/models/RLinf-ResNet10-pretrained" \
     "rollout.model.model_path=$PWD/models/RLinf-ResNet10-pretrained"

With these settings, actor, rollout, and reward run on GPU 0 and robot control
on node 0. Keep supervising the arm and use the SpaceMouse when intervention
is needed. To end a run, interrupt the launcher and wait for the robot to stop;
after the run exits, ``ray stop`` stops this host's Ray processes.

If a Franky impedance controller stops unexpectedly, RLinf reports the motion
error rather than silently restarting it. Resolve the cause before restarting
training. In direct Python use, call ``disconnect()`` and then ``connect()``
before resuming commands; ``clear_errors()`` does not restart failed tracking.

Label Rewards from a Keyboard (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Peg insertion computes its reward from the target pose. For a task without an
automatic success signal, an operator can label rewards from a physical
keyboard. Enable the keyboard wrapper in the training recipe:

.. code:: yaml

   env:
     train:
       keyboard_reward_wrapper: single_stage  # or multi_stage

In ``single_stage`` mode, ``a``, ``b``, and ``c`` emit failure, neutral, and
success rewards. In ``multi_stage`` mode, ``a``, ``b``, and ``c`` switch among
reward stages, and ``q`` emits a negative reward.

The listener reads a Linux input device directly, so the robot host needs the
device path before Ray starts. Find the keyboard's event device:

.. code:: bash

   ls -l /dev/input/by-id/*-event-kbd

An entry such as ``usb-Logitech_USB_Keyboard-event-kbd -> ../event20`` means
the device is ``/dev/input/event20``. Grant access to it and export the path in
the shell that runs ``ray start``:

.. code:: bash

   sudo chmod 666 /dev/input/event20
   export RLINF_KEYBOARD_DEVICE=/dev/input/event20

Visualization and Results
-----------------------------

Run TensorBoard in another activated shell:

.. code:: bash

   tensorboard --logdir ./logs --port 6006

Open ``http://localhost:6006``. Monitor ``env/success_once``, ``env/return``,
and the SAC actor and critic losses. See :doc:`/rst_source/guides/logger` for
logging configuration. The following curve and videos show representative
peg-insertion and charger runs, not a guaranteed training time.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/realworld-curve.png
   :align: center
   :width: 100%

   Real-world training curves.

.. raw:: html

   <video controls muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/peg-insertion-compressed.mp4" type="video/mp4">
   </video>
   <video controls muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/charger-compressed.mp4" type="video/mp4">
   </video>

Multi-Node Setup
--------------------

Use a separate controller computer when you want to isolate robot control from
training load, or when the GPU server is not next to the robot. The arm,
camera, and SpaceMouse connect to the controller, which runs Franky robot
control and needs no GPU. The GPU server runs actor and rollout.

Prepare Both Computers
~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare the controller as the robot host above, without the NVIDIA driver:
check the firmware and set up the real-time kernel or run without one. Then
either install natively with the command from `Installation`_, which selects
CPU-only PyTorch when no NVIDIA driver is present, or start the Docker image
from `Use the Docker Image`_ without ``--gpus all``.

On the GPU server, install its NVIDIA driver, clone the same RLinf revision,
and run the same installation:

.. code:: bash

   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

Both computers must use the same RLinf revision, Python version, and Ray
version. Collect demonstrations on the controller using the earlier collection
steps with node rank 0, before joining the multi-node cluster. Copy the
complete ``logs/franka-demo/demos`` directory to the GPU server.

Configure and Start the Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the GPU server, edit
``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``:
set ``cluster.num_nodes`` to 2, the Franka group's ``node_ranks`` to 1, and its
hardware config's ``node_rank`` to 1. Keep the arm's IP and camera serial.
Actor and rollout remain on GPU 0 of node 0.

.. warning::

   Ray records the Python interpreter and environment variables of the shell
   that runs ``ray start``, and every worker on that node inherits them. Export
   ``RLINF_NODE_RANK`` and activate the environment before ``ray start`` on each
   computer; a rank or environment missing at that moment cannot be fixed later
   without restarting Ray. ``ray_utils/realworld/setup_before_ray.sh`` is a
   template you can adapt for this.

Choose each computer's IP on the network shared by the two computers, not the
arm's IP. If a computer has several network interfaces, also export
``RLINF_COMM_NET_DEVICES`` with the interface that carries that IP. In an
activated shell on the GPU server:

.. code:: bash

   ray stop
   export RLINF_NODE_RANK=0
   export HEAD_IP=192.168.10.10  # Replace with the GPU server's address.
   ray start --head --port=6379 --node-ip-address="$HEAD_IP"

On the controller, in its activated environment:

.. code:: bash

   ray stop
   export RLINF_NODE_RANK=1
   export HEAD_IP=192.168.10.10        # Same GPU server address.
   export CONTROLLER_IP=192.168.10.11 # Replace with this computer's address.
   ray start --address="$HEAD_IP:6379" --node-ip-address="$CONTROLLER_IP"

Run ``ray status`` on the GPU server and confirm that both nodes are alive.
Then run the training command there only, using its local model and demo paths
and the measured target pose. The controller needs neither model weights nor
demonstration files for training. Stop Ray on both nodes when finished. For
several robots, see :doc:`/rst_source/guides/realworld_robot` and
:doc:`/rst_source/guides/hetero`.

Legacy ROS Backend (Optional)
---------------------------------

RLinf can also drive the arm through ROS Noetic, ``franka_ros``, and
``serl_franka_controllers`` instead of Franky. Use this backend for an existing
ROS deployment, or when your firmware needs a libfranka version other than
0.15.0 or 0.19.0. It runs on Ubuntu 20.04, the release ROS Noetic supports, and
is installed natively on the host. Everything else on this page applies once the
backend is installed and selected.

Install the ROS Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install into a separate environment so it does not replace the Franky one, and
pass the libfranka version you chose:

.. code:: bash

   LIBFRANKA_VERSION=0.15.0 bash requirements/install.sh embodied --env franka-ros --venv franka-ros
   source franka-ros/bin/activate

What this does:

1. Installs the same RLinf and training dependencies as the Franky
   environment, and ROS Noetic through APT.
2. Builds libfranka, RLinf's ``franka_ros`` fork, and
   ``serl_franka_controllers`` in the catkin workspace
   ``franka-ros/franka_catkin_ws``, and writes ``realtime_config`` into its
   ``franka_control_node.yaml``.
3. Appends the ROS and catkin ``setup.bash`` scripts to the environment's
   activate script, so activating the environment also loads ROS.

The ROS environment reads these variables:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Variable
     - Default
     - Effect
   * - ``LIBFRANKA_VERSION``
     - ``0.15.0``
     - libfranka release to build. Must match the robot firmware.
   * - ``FRANKA_ROS_VERSION``
     - ``0.10.0``
     - Branch of the ``franka_ros`` fork to build.
   * - ``FRANKA_REALTIME_CONFIG``
     - ``enforce``
     - ``ignore`` lets libfranka run on a kernel without PREEMPT_RT.
   * - ``SKIP_ROS``
     - ``0``
     - ``1`` skips ROS Noetic and the catkin build.

Check the build with ``rospack find serl_franka_controllers``, which prints the
controller package path inside the catkin workspace.

.. warning::

   With ``SKIP_ROS=1``, you provide ROS Noetic, libfranka, ``franka_ros``, and
   ``serl_franka_controllers`` yourself. Source ``/opt/ros/noetic/setup.bash``
   and your catkin workspace's ``devel/setup.bash``, and make sure libfranka is
   on ``LD_LIBRARY_PATH``, in every shell before ``ray start``. Ray workers
   inherit the environment of the shell that started Ray. For manual
   installation, see the `ROS Noetic <https://wiki.ros.org/noetic/Installation/Ubuntu>`_,
   `libfranka <https://frankarobotics.github.io/docs/libfranka/docs/installation.html>`_,
   and `serl_franka_controllers <https://github.com/rail-berkeley/serl_franka_controllers>`_
   guides.

Select the Backend and Real-Time Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``backend: franka_ros`` in each Franka hardware config:

.. code:: yaml

   configs:
     - robot_ip: ROBOT_IP
       node_rank: 0
       camera_serials: ["CAMERA_SERIAL"]
       backend: franka_ros

The ROS backend takes its real-time mode from ``FRANKA_REALTIME_CONFIG`` at
installation, not from the hardware config. ``franka_control`` reads the
installed value at every launch; the default ``enforce`` refuses a kernel
without PREEMPT_RT. To change it, re-run the installer with
``FRANKA_REALTIME_CONFIG=ignore`` or ``enforce``; no rebuild is needed. A
``realtime_config`` key in a ``franka_ros`` hardware config raises an error
that points to this variable.

The controller check tool selects the backend with a flag:

.. code:: bash

   python -m toolkits.realworld_check.test_franka_controller --backend franka_ros

Run with the ROS Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running with ROS follows `Run It`_; only the environment and the hardware config
change. Activate the ROS environment in every shell that runs RLinf, and in
particular before ``ray start``: activation loads ROS and the catkin workspace,
and Ray workers inherit that environment.

.. code:: bash

   source franka-ros/bin/activate
   ray stop
   export RLINF_NODE_RANK=0
   ray start --head

With ``backend: franka_ros`` set in both recipes, the commands in
`Collect Demonstrations`_ and `Train the Policy`_ run unchanged.

When the arm connects, RLinf starts ROS itself:

1. Reuses a running ``roscore`` or starts one, and opens a single ROS node for
   the process.
2. Runs ``roslaunch serl_franka_controllers impedance.launch`` for the arm's
   ``robot_ip``, which brings up ``franka_control``, the Franka Hand driver, and
   the Cartesian impedance controller.
3. For a joint reset, switches to ``joint.launch`` until the arm reaches the
   reset pose, then restarts the impedance controller.

Only one program can hold the arm's FCI connection, so close any other
controller first. Before the first run, confirm that the launch works with the
controller check tool:

.. code:: bash

   python -m toolkits.realworld_check.test_franka_controller --backend franka_ros

Differences from Franky
^^^^^^^^^^^^^^^^^^^^^^^^^

- The arm accepts Cartesian ``tcp_pose`` targets only. Single-arm Franka tasks
  already send these; joint-position control, such as the dual-arm joint
  tasks, needs Franky.
- ``gripper_type: franka`` reaches the Franka Hand through the
  ``/franka_gripper`` topics that the arm's launch starts, so the hand responds
  only while the arm is connected. Other end effectors, such as a Robotiq
  gripper, connect on their own as they do with Franky.
- The impedance controller owns its gains. A task's ``compliance_param`` is
  applied through ``dynamic_reconfigure``, and the hardware config's
  ``compliance`` settings have no effect.

In the multi-node layout, only the controller needs the ROS environment. The GPU
server runs actor and rollout and installs the Franky environment as in
`Prepare Both Computers`_.

Troubleshooting
^^^^^^^^^^^^^^^^^

- ``Running kernel does not have realtime capabilities``: boot the real-time
  kernel, or reinstall with ``FRANKA_REALTIME_CONFIG=ignore``.
- An incompatible libfranka version error: reinstall with the
  ``LIBFRANKA_VERSION`` that the compatibility table lists for your firmware.
- The arm never reports ready: check that FCI is active in Franka Desk, then
  run ``rostopic echo -n 1 /franka_state_controller/franka_states`` in the
  activated environment while the controller is up; it should print one state
  message.
- A crashed run can leave ``roslaunch`` processes that still hold the arm. Stop
  Ray and run ``pkill -f roslaunch`` before starting again.

Other Franka Workflows
--------------------------

The installer combines the Franka dependencies with other models. Every Franky
environment includes the dexterous-hand dependencies. For a VLA policy, install
the model and Franka together:

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env franka --venv openpi
   source openpi/bin/activate

Replace ``openpi`` with ``openvla``, ``openvla-oft``, or ``gr00t`` as needed.
These commands install dependencies; the guides below cover model weights,
task configuration, and training:

- :doc:`franka_gello` and :doc:`franka_vr` for GELLO or PICO teleoperation.
- :doc:`franka_pi0_sft_deploy` and :doc:`hg-dagger` for OpenPI policies.
- :doc:`franka_reward_model` for learned rewards.
- :doc:`franka_zed_robotiq` and :doc:`franka_dexhand` for other cameras and end effectors.
- :doc:`dual_franka` for dual-arm control and :doc:`/rst_source/guides/rtc` for overlapping action execution with inference.
