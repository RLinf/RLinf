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
Connect a RealSense camera and a SpaceMouse by USB. Install an NVIDIA driver
and, for Docker, NVIDIA Container Toolkit as described in
:doc:`/rst_source/start/installation`. The commands below assume an x86-64
Ubuntu 22.04 host.

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

Real-Time Kernel (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A PREEMPT_RT kernel is recommended for Franky's time-sensitive control loop.
It is optional in the RLinf workflow: the driver attempts real-time scheduling
and memory locking but can continue when they are unavailable. Without a
real-time kernel, control can be less responsive under heavy CPU or GPU
training load; missed control deadlines can also stop a motion.

For more predictable control, follow Franka's
`real-time kernel instructions <https://frankarobotics.github.io/docs/installation_linux.html#setting-up-the-real-time-kernel>`_
on the host. Docker shares the host kernel, so installing a kernel inside the
container does not enable real-time control. The Docker command below grants
real-time priority and memory-locking permissions. Before training, verify
control under your intended training load with an operator present.

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

Build the Franka image from this checkout, then start the container:

.. code:: bash

   docker build -f docker/Dockerfile \
     --build-arg BUILD_TARGET=embodied-franka \
     --build-arg NO_MIRROR=1 -t rlinf:franka .

   docker run -it --name rlinf-franka --gpus all \
     --network host --privileged --shm-size 20g \
     --ulimit rtprio=99 --ulimit memlock=-1 \
     -v "$PWD:/workspace/RLinf" -w /workspace/RLinf \
     rlinf:franka bash

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

   LIBFRANKA_VERSION=0.19.0 bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

Set ``LIBFRANKA_VERSION=0.15.0`` instead when required. The installer uses
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

   UV_TORCH_BACKEND=cpu LIBFRANKA_VERSION=0.19.0 \
     bash requirements/install.sh embodied --env franka
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

Existing ROS Noetic deployments can use the explicit
``embodied-franka-ros`` Docker build target or
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
- :doc:`franka_zed_robotiq` and :doc:`franka_dexhand` for other cameras and end effectors.
- :doc:`franka_reward_model` for learned rewards.
- :doc:`franka_pi0_sft_deploy` and :doc:`hg-dagger` for OpenPI policies.
- :doc:`dual_franka` for dual-arm control and :doc:`/rst_source/guides/rtc` for overlapping action execution with inference.
