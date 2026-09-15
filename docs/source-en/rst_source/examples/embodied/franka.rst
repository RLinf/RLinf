Real-World RL with Franka
============================

This page walks you through training a CNN policy on a Franka arm with RLinf,
from demonstration collection to online RLPD training. The default setup uses a
single Ubuntu 20.04 computer with an NVIDIA GPU. It runs ROS Noetic, the robot
connection, rollout, and training. You first prepare that host (firmware check,
NVIDIA driver, real-time kernel), install RLinf, then run the peg-insertion
example. The later sections cover a separate controller node, the optional
Franky backend without ROS, and other Franka workflows.

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
     - One Ubuntu 20.04 GPU host runs ROS, the robot connection, rollout, and
       training.
     - Most setups. One computer to install and maintain.
   * - Multi-node
     - A controller computer without a GPU runs ROS and the robot connection;
       a GPU server runs actor and rollout.
     - You want to isolate robot control from training load, or the GPU server
       cannot run Ubuntu 20.04.

.. warning::

   Keep the emergency stop within reach and have an operator supervise every
   hardware run. Secure the peg and fixture, clear the workspace, and check
   that reset motions are safe. This task resets approximately 10 cm above
   the target with randomized horizontal position and yaw. Do not copy a
   target pose from another robot.

Prepare the Robot Host
--------------------------

Three decisions on the robot host come before any RLinf installation. The
firmware decides which libfranka version you build. The NVIDIA driver decides
whether the installer picks a CUDA build of PyTorch. The kernel decides whether
libfranka can run its 1 kHz control loop in real time. Complete these steps on
the Ubuntu 20.04 host in order.

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
and choose a libfranka version. The installer builds libfranka 0.15.0 by
default. You pass a different version through ``LIBFRANKA_VERSION`` during
installation. RLinf has been tested with firmware 5.7.2 up to 5.9.0 using
libfranka 0.15.0 on a real-time kernel, and with firmware 5.9.2 using libfranka
0.19.0 on a standard Ubuntu 20.04 kernel with the real-time check disabled. Do not change robot
firmware merely to match this example.

Install the NVIDIA Driver on Ubuntu 20.04
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the driver before RLinf. ``requirements/install.sh`` reads the CUDA
version reported by ``nvidia-smi`` and installs the newest PyTorch CUDA build
that driver supports: CUDA 12.6 wheels need driver 560 or newer, and CUDA 12.8
wheels need 570 or newer. Without a driver, the installer falls back to
CPU-only PyTorch, and training cannot use the GPU.

If ``nvidia-smi`` already reports driver 570 or newer, keep it and skip this
step. Otherwise, add NVIDIA's CUDA repository for Ubuntu 20.04 and install a
driver from it:

.. code:: bash

   sudo apt-get install -y build-essential dkms wget "linux-headers-$(uname -r)"
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get install -y cuda-drivers-575
   sudo reboot

What this does:

1. Installs the compiler, DKMS, and kernel headers that the driver's kernel
   module is built with.
2. Registers NVIDIA's APT repository and signing key through ``cuda-keyring``.
3. Installs driver 575, which supports CUDA 12.9, and builds its kernel module
   for the running kernel.

If Secure Boot requests Machine Owner Key (MOK) enrollment, complete it during
reboot so the module can load. After logging back in, ``nvidia-smi`` should
list your GPU. Do not mix this repository with a driver installed from a
``.run`` file or from another repository; remove the old driver first, as
described in NVIDIA's
`driver installation guide <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html>`_.

``cuda-drivers-575`` builds NVIDIA's proprietary kernel module; the Ubuntu 20.04
repository ships no open-module packages. GPUs that require the open kernel
module, such as the GeForce RTX 50 series, need a driver from NVIDIA's
`driver downloads <https://www.nvidia.com/en-us/drivers/>`_ page instead.

The CUDA toolkit is not required for this example because PyTorch wheels
bundle the CUDA runtime. Install it only if you need to compile CUDA
extensions, using the toolkit-only package so the driver stays unchanged:

.. code:: bash

   sudo apt-get install -y cuda-toolkit-12-8

Install a Real-Time Kernel (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

libfranka sends a command to the arm every millisecond. A PREEMPT_RT kernel
keeps that loop on schedule while rollout and training load the CPU and GPU.
By default, ``franka_control`` enforces this and refuses to start on a kernel
without PREEMPT_RT.

Install the kernel on the host, not inside Docker: containers share the host
kernel. For Ubuntu 20.04, follow Franka's
`real-time kernel guide <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_,
which builds a patched kernel and installs its ``linux-image`` and
``linux-headers`` packages. Keep your current kernel as a GRUB fallback. After
rebooting into the new kernel, check it:

.. code:: bash

   uname -r
   cat /sys/kernel/realtime

The second command must print ``1``. If it does not, select the real-time
kernel under GRUB's advanced options before continuing.

Use the NVIDIA Driver with the Real-Time Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The NVIDIA kernel module must also be built for the real-time kernel, and its
build refuses a PREEMPT_RT kernel unless ``IGNORE_PREEMPT_RT_PRESENCE=1`` is
set. If ``nvidia-smi`` fails after booting the real-time kernel, rebuild the
module with the override and reboot:

.. code:: bash

   sudo env IGNORE_PREEMPT_RT_PRESENCE=1 dkms autoinstall -k "$(uname -r)"
   sudo reboot

If you install the driver after the real-time kernel is running, pass the same
override to APT instead:
``sudo env IGNORE_PREEMPT_RT_PRESENCE=1 apt-get install -y cuda-drivers-575``.
Apply the override again whenever a driver or kernel update rebuilds the module.

.. warning::

   NVIDIA drivers are not officially supported on PREEMPT_RT kernels.
   ``IGNORE_PREEMPT_RT_PRESENCE=1`` bypasses the build-time check; it does not
   guarantee compatibility. Check ``nvidia-smi`` on the real-time kernel and do
   not start training if the GPU is unavailable. Boot the previous kernel to
   recover.

Allow Real-Time Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A real-time kernel only helps if your account may raise thread priority and
lock memory. Add your login account to a dedicated group:

.. code:: bash

   getent group realtime || sudo groupadd realtime
   sudo usermod -aG realtime "$(id -un)"
   sudoedit /etc/security/limits.d/99-rlinf-realtime.conf

Add the following limits to that file, then log out and back in:

.. code:: text

   @realtime - rtprio 99
   @realtime - memlock unlimited

In the new login shell, ``ulimit -r`` should print ``99`` and ``ulimit -l``
should print ``unlimited``.

Run without a Real-Time Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you cannot use a real-time kernel, install RLinf with
``FRANKA_REALTIME_CONFIG=ignore`` (see `Installation`_). The installer writes
this value to ``realtime_config`` in ``franka_control_node.yaml``, which
``franka_control`` reads at every launch, so libfranka then runs on a standard
kernel. To switch back, re-run the installer with
``FRANKA_REALTIME_CONFIG=enforce``; no rebuild is needed.

.. warning::

   On a standard kernel, heavy training load can make libfranka miss control
   deadlines. The robot then stops with ``communication_constraints_violation``
   reflexes. A real-time kernel remains the recommended setup. Before training,
   verify control under your intended training load with an operator present.

Installation
----------------

With the driver and kernel in place, one installation on the GPU host provides
both the ROS control stack and the training dependencies. Clone RLinf and run
subsequent commands from its root directory:

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

Install the Franka Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the installer with the libfranka version you chose. Add
``FRANKA_REALTIME_CONFIG=ignore`` only if you run without a real-time kernel:

.. code:: bash

   LIBFRANKA_VERSION=0.15.0 bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

What this does:

1. Creates the ``.venv`` virtual environment with RLinf, the Franka
   dependencies, and the embodied training dependencies, including the CUDA
   build of PyTorch matched to your driver.
2. Installs system packages and ROS Noetic through APT. This step needs
   Ubuntu 20.04 and sudo.
3. Builds libfranka, RLinf's ``franka_ros`` fork, and
   ``serl_franka_controllers`` in the catkin workspace
   ``.venv/franka_catkin_ws``, and sets its ``realtime_config``.
4. Appends ``source /opt/ros/noetic/setup.bash`` and the workspace's
   ``devel/setup.bash`` to ``.venv/bin/activate``, so activating the
   environment also loads ROS.

The installer reads these variables:

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

Pass ``--venv <name>`` to install into another directory, and ``--use-mirror``
for faster downloads from mainland China.

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

Your account must be able to read the camera and SpaceMouse USB devices; see
the `SpaceMouse setup <https://github.com/JakubAndrysek/PySpaceMouse#installation>`_
for its udev rule.

Check the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

In the activated environment, confirm that PyTorch sees the GPU and that ROS
finds the controllers:

.. code:: bash

   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   rospack find serl_franka_controllers

The first command must print ``True``. If it prints ``False``, check
``nvidia-smi`` and re-run the installer after fixing the driver. The second
prints the controller package path inside ``.venv/franka_catkin_ws``.

Use the Docker Image
~~~~~~~~~~~~~~~~~~~~~~~~

Instead of installing natively, you can run the
``rlinf/rlinf:agentic-rlinf0.4-franka`` image. It is built on CUDA 12.8 and
Ubuntu 20.04 with ROS Noetic, and its environments carry CUDA PyTorch, so one
container runs the actor, rollout, and robot control on the single-machine
GPU host. The host still needs driver 570 or newer from
`Install the NVIDIA Driver on Ubuntu 20.04`_ and the
`NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.
The image also serves as the controller of the multi-node layout. It contains
these environments, switched with ``source switch_env <name>``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Environment
     - Contents
   * - ``franka-0.10.0``, ``franka-0.13.3``, ``franka-0.14.1``,
       ``franka-0.15.0``, ``franka-0.18.0``, ``franka-0.19.0``
     - ROS backend with that libfranka version. ``franka-0.15.0`` is active by
       default.
   * - ``franky``
     - Optional Franky backend with libfranka 0.19.0; see
       `Franky Backend (Optional)`_.
   * - ``franka-dexhand``
     - ROS backend with dexterous-hand dependencies.

Start the container with access to the robot, camera, and SpaceMouse, then
select the environment that matches your firmware:

.. code:: bash

   docker run -it --name rlinf-franka \
     --gpus all --network host --privileged \
     --ulimit rtprio=99 --ulimit memlock=-1 \
     -v "$PWD:/workspace/RLinf" -w /workspace/RLinf \
     rlinf/rlinf:agentic-rlinf0.4-franka bash
   source switch_env franka-0.15.0

The image's ROS environments keep ``realtime_config: enforce``, so run the
container on a host with a real-time kernel. For another shell, run
``docker exec -it rlinf-franka bash`` and select the same environment again.

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

The config leaves ``backend`` unset, so the arm uses the default
``franka_ros`` backend. Set ``cluster.num_nodes: 1`` in the training recipe as
well; collection already uses one node. Keep the training recipe's ``4090``
node group and component placement unchanged: the group names GPU node 0,
regardless of the GPU model. Use one camera for this example; it is named
``wrist_1`` automatically.

Use the robot's guiding mode to position the peg at the desired successful
insertion pose, then unlock the arm and activate FCI in Franka Desk. Read the
pose with the controller check tool, which launches the ROS controller for the
arm:

.. code:: bash

   export FRANKA_ROBOT_IP=192.168.1.10  # Replace with your robot's address.
   python -m toolkits.realworld_check.test_franka_controller

At the prompt, enter ``getpos_euler``, then ``q`` to release the robot.
The result is ``[x, y, z, roll, pitch, yaw]``, in metres and radians. The tool
also accepts ``getpos``, ``getjoint``, ``getstate``, ``gethand``, ``clear``,
``home``, ``open``, and ``close``. Save the six measured numbers as a
comma-separated list in this shell:

.. code:: bash

   export FRANKA_TARGET_POSE='[x, y, z, roll, pitch, yaw]'  # Replace all six entries.

Before proceeding, confirm that the target and the reset region described
above are within the safe workspace. Close other programs that control the arm;
only one process can hold its control connection.

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
training load, or when the GPU server cannot run Ubuntu 20.04. The arm, camera,
and SpaceMouse connect to the controller, which runs ROS and robot control
without a GPU. The GPU server runs actor and rollout and needs no ROS.

Prepare Both Computers
~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare the controller as the robot host above, without the NVIDIA driver:
check the firmware and set up the real-time kernel or its fallback. Then
either start the Docker image from `Use the Docker Image`_ (omit
``--gpus all`` on a computer without a GPU), or install natively with
``bash requirements/install.sh embodied --env franka``. Without an NVIDIA
driver, the native installer selects CPU-only PyTorch.

On the GPU server, clone the same RLinf revision and install the same
environment without ROS, after its NVIDIA driver is installed:

.. code:: bash

   SKIP_ROS=1 bash requirements/install.sh embodied --env franka
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
   computer; a rank or ROS setup missing at that moment cannot be fixed later
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

Franky Backend (Optional)
-----------------------------

`Franky <https://github.com/TimSchneider42/franky>`_ controls the arm through
Python bindings to libfranka, without ROS. Consider it when the robot host
cannot run Ubuntu 20.04 and ROS Noetic, or when you do not want to build a
catkin workspace. The rest of this page applies unchanged once the backend is
selected.

The Franky environment installs a prebuilt ``franky-control`` wheel that
bundles libfranka. Wheels exist only for x86-64 and for libfranka 0.15.0 and
0.19.0 (the default), so your firmware must be compatible with one of them.
Install into a separate environment so it does not replace the ROS one:

.. code:: bash

   LIBFRANKA_VERSION=0.19.0 bash requirements/install.sh embodied --env franka-franky --venv franky
   source franky/bin/activate

Set ``FRANKY_WHEEL`` to a wheel URL or local path if the host cannot download
from GitHub. In the Docker image, run ``source switch_env franky`` instead;
it bundles libfranka 0.19.0, so firmware that needs 0.15.0 installs natively.

Select the backend in each Franka hardware config. Franky applies libfranka's
real-time mode from the same config: ``enforce``, the default, refuses a kernel
without PREEMPT_RT, and ``ignore`` runs on a standard kernel with the control
risks described in `Run without a Real-Time Kernel`_:

.. code:: yaml

   configs:
     - robot_ip: ROBOT_IP
       node_rank: 0
       camera_serials: ["CAMERA_SERIAL"]
       backend: franky
       realtime_config: ignore  # Omit on a real-time kernel.

``realtime_config`` is valid only with ``backend: franky``. The default
``franka_ros`` backend rejects it and takes the mode from
``FRANKA_REALTIME_CONFIG`` at installation instead. The controller check tool
takes the same choices as flags:

.. code:: bash

   python -m toolkits.realworld_check.test_franka_controller \
     --backend franky --realtime-config ignore

Franky also tries to lock memory and raise its thread priority, so the limits in
`Allow Real-Time Scheduling`_ apply. If a Franky impedance controller stops
unexpectedly, RLinf reports the motion error rather than silently restarting
it. Resolve the cause before restarting training. In direct Python use, call
``disconnect()`` and then ``connect()`` before resuming commands;
``clear_errors()`` does not restart failed tracking.

Other Franka Workflows
--------------------------

After completing the base example, use these guides for other setups:

- :doc:`franka_gello` and :doc:`franka_vr` for GELLO or PICO teleoperation.
- :doc:`franka_pi0_sft_deploy` and :doc:`hg-dagger` for OpenPI policies.
- :doc:`franka_reward_model` for learned rewards.
- :doc:`franka_zed_robotiq` and :doc:`franka_dexhand` for other cameras and end effectors.
- :doc:`dual_franka` for dual-arm control and :doc:`/rst_source/guides/rtc` for overlapping action execution with inference.
