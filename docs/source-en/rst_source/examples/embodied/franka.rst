Real-World RL with Franka
============================

Train a CNN policy on a Franka arm with RLinf, from demonstration collection to
online RLPD training. The default setup runs Franky robot control, rollout, and
training on one GPU computer with Ubuntu 22.04 and CUDA. ROS is not required.
Follow the peg-insertion example below, then use the later sections for a
separate controller node or other hardware and policies.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: Franka arm used for real-world reinforcement learning

   Franka arm used for real-world reinforcement learning.

Overview
------------

The policy learns from camera images and robot state, using successful
demonstrations to initialize the replay data.

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

This example inserts a peg at a measured target pose. The recipe
``realworld_peginsertion_rlpd_cnn_async`` trains asynchronously with
demonstrations and live robot experience. A SpaceMouse provides demonstrations
and human intervention during training.

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

Connect the Franka arm to the computer through a wired network interface.
Connect a RealSense camera and a SpaceMouse by USB. The commands below assume
an x86-64 Ubuntu 22.04 host. Install an NVIDIA driver using the
`Ubuntu driver guide <https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/>`_,
or follow the real-time kernel and CUDA steps below if you choose that kernel.
For Docker, also complete NVIDIA's `Container Toolkit installation and Docker
configuration <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.

.. warning::

   Keep the emergency stop within reach and have an operator supervise every
   hardware run. Secure the peg and fixture, clear the workspace, and check
   that reset motions are safe. This task resets approximately 10 cm above
   the target with randomized horizontal position and yaw. Do not copy a
   target pose from another robot.

Open Franka Desk at your robot's address, record the Control firmware version,
and choose a compatible libfranka version from the
`Franka compatibility table <https://frankarobotics.github.io/docs/compatibility.html>`_.
The image bundles libfranka 0.19.0. If your firmware requires a different
version, use the custom installation below.
Do not change robot firmware merely to match this example.

Real-Time Kernel Installation (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A PREEMPT_RT kernel is recommended for Franky's time-sensitive control loop.
It is optional in the RLinf workflow: the driver attempts real-time scheduling
and memory locking but can continue when they are unavailable. Without a
real-time kernel, control can be less responsive under heavy CPU or GPU
training load; missed control deadlines can also stop a motion.

Install the kernel on the host, not inside Docker: containers share the host
kernel. For Ubuntu 22.04, prefer the Ubuntu Pro method below. Without an Ubuntu
Pro subscription, use Franka's `manual kernel installation guide
<https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_.

.. warning::

   NVIDIA drivers are not officially supported on PREEMPT_RT kernels. The
   ``IGNORE_PREEMPT_RT_PRESENCE=1`` workaround bypasses the driver's build-time
   check; it does not guarantee compatibility. Keep your existing kernel as a
   GRUB fallback, and check both the GPU and robot before training.

Install the Ubuntu Pro Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the host, update the Pro client and attach an eligible Ubuntu Pro subscription.
Skip ``pro attach`` if the machine is already attached. These commands follow
the `Ubuntu Pro installation guide
<https://ubuntu.com/pro-client/docs/en/docs/howtoguides/enable_realtime_kernel/>`_:

.. code:: bash

   sudo apt update
   sudo apt install ubuntu-pro-client
   sudo pro attach
   sudo env IGNORE_PREEMPT_RT_PRESENCE=1 pro enable realtime-kernel

Read and confirm the prompts, including disabling Livepatch if requested. The
environment variable allows any existing NVIDIA DKMS driver to attempt a rebuild
for the new kernel. After installation succeeds, save your work and reboot:

.. code:: bash

   sudo reboot

Select the real-time kernel in GRUB if it is not selected automatically. After
logging back in, check the running kernel:

.. code:: bash

   uname -r
   cat /sys/kernel/realtime

The second command must print ``1``. If it does not, select the installed
real-time kernel under GRUB's advanced options before continuing.

Use CUDA with the Real-Time Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GPU host needs an NVIDIA driver built for the running real-time kernel.
If ``nvidia-smi`` already lists your GPU after reboot, keep that driver. Otherwise,
the following APT installation uses NVIDIA's open driver for Turing and newer
GPUs. For older GPUs or an existing runfile installation, follow NVIDIA's
`driver installation guide <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html>`_
to select a compatible driver without mixing installation methods.

.. code:: bash

   sudo apt install build-essential dkms wget "linux-headers-$(uname -r)"
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo env IGNORE_PREEMPT_RT_PRESENCE=1 apt install nvidia-open
   sudo reboot

If Secure Boot requests Machine Owner Key (MOK) enrollment, complete it during
reboot so the new driver can load. Check ``nvidia-smi`` again on the real-time
kernel. Do not start training if the GPU is unavailable; use the previous kernel
to recover. Apply the same real-time override when rebuilding the driver after
kernel updates.

For Docker, the Franka image already contains CUDA; only the driver and NVIDIA
Container Toolkit are needed on the host, as described in Hardware Setup above.
For a custom environment, also install CUDA 12.8 from NVIDIA's APT repository
configured above. If you skipped the driver installation, first run its
``wget``, ``dpkg``, and ``apt update`` commands to register the repository.
Use the `toolkit-only package
<https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/#meta-packages>`_
so this step does not replace the driver:

.. code:: bash

   sudo apt install cuda-toolkit-12-8
   export CUDA_HOME=/usr/local/cuda-12.8
   export PATH="$CUDA_HOME/bin:$PATH"
   nvcc --version

Keep these exports in each training shell, or add them to your shell startup
file. ``nvcc`` should report CUDA 12.8. The environment check below verifies
that PyTorch can also use the GPU.

Allow Real-Time Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Docker command below grants real-time priority and memory-locking
permissions. For a custom environment, add your login account to a dedicated
group on the host:

.. code:: bash

   getent group realtime || sudo groupadd realtime
   sudo usermod -aG realtime "$(id -un)"
   sudoedit /etc/security/limits.d/99-rlinf-realtime.conf

Add the following limits to that file, then log out and back in:

.. code:: text

   @realtime - rtprio 99
   @realtime - memlock unlimited

In the new login shell, ``ulimit -r`` should print ``99`` and ``ulimit -l``
should print ``unlimited``. Before training, verify control under your intended
training load with an operator present.

Installation
----------------

Clone RLinf and run subsequent commands from its root directory:

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

Choose Docker or a custom environment. Both install Franky and the dependencies
for the CNN training example.

Docker (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~

Pull the Franka image and start the container:

.. code:: bash

   docker pull rlinf/rlinf:agentic-rlinf0.4-franka

   docker run -it --name rlinf-franka --gpus all \
     --network host --privileged --shm-size 20g \
     --ulimit rtprio=99 --ulimit memlock=-1 \
     -v "$PWD:/workspace/RLinf" -w /workspace/RLinf \
     rlinf/rlinf:agentic-rlinf0.4-franka bash

The image uses CUDA and Ubuntu 22.04. Inside it, activate the bundled Franky
environment and keep using it for every step:

.. code:: bash

   source switch_env franky-0.19.0

For an additional shell, run ``docker exec -it rlinf-franka bash`` on the host
and select the same environment again.

Custom Environment
~~~~~~~~~~~~~~~~~~~~~~

Install the dependencies without Docker:

.. code:: bash

   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

The installer defaults to libfranka 0.19.0. Set
``LIBFRANKA_VERSION=0.15.0`` before the command only if your firmware requires it.
The installer uses
versioned Franky wheels with libfranka included; a separate libfranka or ROS
installation is unnecessary. Outside Docker, your account must be able to read
the camera and SpaceMouse USB devices; see the device-permission instructions in
:doc:`/rst_source/start/installation` and the
`SpaceMouse setup <https://github.com/JakubAndrysek/PySpaceMouse#installation>`_.

Check the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

In the activated environment, verify Franky and CUDA before connecting the arm:

.. code:: bash

   python -c "import franky, torch; assert torch.cuda.is_available(); print(torch.__version__)"

Download the Model
----------------------

Download the pretrained ResNet encoder into the repository:

.. code:: bash

   hf download RLinf/RLinf-ResNet10-pretrained \
     --local-dir ./models/RLinf-ResNet10-pretrained

The training command below supplies this directory to both actor and rollout.

Run It
----------

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

Set ``cluster.num_nodes: 1`` in the training recipe as well; collection already
uses one node. Keep the training recipe's ``4090`` node group and component
placement unchanged: the group names GPU node 0, regardless of the GPU model.
Use one camera for this example; it is named ``wrist_1`` automatically.

Use the robot's guiding mode to position the peg at the desired successful
insertion pose, then unlock the arm and activate FCI in Franka Desk. Read the
pose through Franky:

.. code:: bash

   export FRANKA_ROBOT_IP=192.168.1.10  # Replace with your robot's address.
   python -m toolkits.realworld_check.test_franka_controller

At the prompt, enter ``getpos_euler``, then ``q`` to release the robot.
The result is ``[x, y, z, roll, pitch, yaw]``, in metres and radians.
Save those six measured numbers as a comma-separated list in this shell:

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
session to keep demonstration sets separate.

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
training load. The controller needs no GPU, CUDA, or ROS. Keep the arm, camera,
and SpaceMouse connected to it; use the GPU computer for actor and rollout.

Prepare the Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~

On the Ubuntu 22.04 controller, clone the same RLinf revision and install
the CPU-only environment from the repository root:

.. code:: bash

   UV_TORCH_BACKEND=cpu bash requirements/install.sh embodied --env franka
   source .venv/bin/activate
   python -c "import franky, torch; assert torch.version.cuda is None"

Select the same libfranka version as before. The device-permission instructions
for custom environments and the real-time recommendations still apply.
Collect demonstrations on the controller using the earlier
collection steps with node rank 0, before joining the multi-node cluster.
Copy the complete ``logs/franka-demo/demos`` directory to the GPU computer.

Configure and Start the Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up the GPU computer with the default CUDA image. Use the same RLinf
revision, Python version, and Ray version on both nodes. On the GPU computer,
edit ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``: change
``num_nodes`` to 2, the Franka group's ``node_ranks`` to 1, and its hardware
config's ``node_rank`` to 1. Keep the arm's IP and camera serial.
Actor and rollout remain on GPU 0 of node 0.

Choose each computer's IP on the network shared by the two computers, not the
arm's IP. In an activated shell on the GPU computer:

.. code:: bash

   ray stop
   export RLINF_NODE_RANK=0
   export HEAD_IP=192.168.10.10  # Replace with the GPU computer's address.
   ray start --head --port=6379 --node-ip-address="$HEAD_IP"

On the controller, using its CPU environment:

.. code:: bash

   ray stop
   export RLINF_NODE_RANK=1
   export HEAD_IP=192.168.10.10        # Same GPU computer address.
   export CONTROLLER_IP=192.168.10.11 # Replace with this computer's address.
   ray start --address="$HEAD_IP:6379" --node-ip-address="$CONTROLLER_IP"

Run ``ray status`` on the GPU computer and confirm that both nodes are alive.
Then run the training command there only, using its local model and demo paths
and the measured target pose. The controller needs neither model weights nor
demonstration files for training. Stop Ray on both nodes when finished. See
:doc:`/rst_source/guides/hetero` for multiple network interfaces or more robots.

Legacy ROS Backend
~~~~~~~~~~~~~~~~~~~~~~

Existing ROS Noetic deployments can use the
``rlinf/rlinf:agentic-rlinf0.4-franka-ros`` Docker image or
``bash requirements/install.sh embodied --env franka-ros`` on Ubuntu 20.04.
Set ``backend: franka_ros`` in each Franka hardware config and activate the
matching ``franka-<libfranka-version>`` environment before starting Ray.
Keep the ROS controller's firmware and real-time requirements; the optional
Franky scheduling behavior does not change ROS requirements.

Other Franka Workflows
--------------------------

For VLA policies, the same Docker image includes ``openvla``, ``openvla-oft``,
``openpi``, and ``gr00t`` environments with Franka dependencies. Select the
matching environment, for example ``source switch_env openpi``. For a custom
installation, install the model and Franka together:

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env franka --venv openpi
   source openpi/bin/activate

Replace ``openpi`` in both commands with ``openvla``, ``openvla-oft``, or
``gr00t`` as needed. These commands install dependencies; follow the matching
workflow for model weights, task configuration, and training.

After completing the base example, use these guides for other setups:

- :doc:`franka_gello` and :doc:`franka_vr` for GELLO or PICO teleoperation.
- :doc:`franka_pi0_sft_deploy` and :doc:`hg-dagger` for OpenPI policies.
- :doc:`franka_reward_model` for learned rewards.
- :doc:`franka_zed_robotiq` and :doc:`franka_dexhand` for other cameras and end effectors.
- :doc:`dual_franka` for dual-arm control and :doc:`/rst_source/guides/rtc` for overlapping action execution with inference.
