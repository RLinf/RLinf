Real-World RL with GimArm
============================

This document provides a comprehensive guide to launching and managing
a CNN policy training task within the RLinf framework,
focusing on training a ResNet-based CNN policy from scratch for robotic manipulation
with the GimArm 6-DOF robotic arm.

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Action Generation**: Producing absolute joint-space actions with gripper control.
3. **Reinforcement Learning**: Optimizing the policy via SAC with environment feedback.

Environment
-----------

**Real World Environment**

- **Environment**: Real world setup.

  - GimArm 6-DOF robotic arm (``gim_arm`` or ``gim_arm_xl`` variant)
  - Damiao servo motors (DM4340 / DM6248P for J1-3, DM4310 for J4-6)
  - CAN-USB adapter with SocketCAN interface
  - Intel RealSense cameras (default) or Stereolabs ZED cameras
  - Optional gripper (parallel or single-side, built-in Damiao motor)

- **Task**: Currently supports the peg-insertion task (``GimArmPegInsertionEnv-v1``).
- **Observation**:

  - RGB images (128x128) from wrist camera(s).
  - State dict containing: ``tcp_pose`` (7,), ``tcp_vel`` (6,), ``arm_joint_position`` (6,), ``gripper_position`` (1,), ``tcp_force`` (3,), ``tcp_torque`` (3,).

- **Action Space**: 7-dimensional continuous actions:

  - 6 absolute joint positions in radians, bounded by the configured joint limits.
  - 1 binary gripper command in ``[-1, 1]`` (open/close).

**Data Structure**

- **Images**: RGB tensors ``[batch_size, 128, 128, 3]``
- **Actions**: First 6 dimensions are absolute joint angles in radians; 7th dimension is gripper command in ``[-1, 1]``
- **Rewards**: Step-level rewards based on task completion


Algorithm
-----------------------------------------

**Core Algorithm Components**

1. **SAC (Soft Actor-Critic)**

   - Learning Q-values by Bellman backups and entropy regularization.

   - Learning policy to maximize entropy-regularized Q.

   - Learning temperature parameter for exploration-exploitation trade-off.

2. **Cross-Q**

   - A variant of SAC that removes the target Q network.

   - Concating curr-obs and next-obs in one batch, incorporating BatchNorm for stable training for Q.

3. **RLPD (Reinforcement Learning with Prior Data)**

   - A variant of SAC that incorporates prior data for improved learning efficiency.

   - High update-to-data ratio to leverage collected data effectively.

4. **CNN Policy Network**

   - ResNet-based architecture for processing visual inputs.

   - MLP layers for fusing images and states to output actions.

   - Q heads for critic functions.

Hardware Setup
----------------

The real-world setup requires the following hardware components:

- **Robotic Arm**: GimArm 6-DOF arm (``gim_arm`` or ``gim_arm_xl`` variant) with Damiao servo motors
- **CAN Adapter**: CAN-USB adapter (SocketCAN-compatible), connected to the controller computer
- **Cameras**: Intel RealSense cameras (default) or Stereolabs ZED cameras
- **Gripper (Optional)**: Parallel or single-side gripper with built-in Damiao motor
- **Computing Unit**: A computer with GPU support for training the CNN policy
- **Robot Controller**: A computer connected to the GimArm via CAN bus (no GPU required)

.. warning::

  Ensure the controller node and the training node are in the same local network.
  The GimArm robot is connected to the controller node via CAN bus, not Ethernet.

.. note::

   Unlike the Franka setup, GimArm does **not** require a real-time kernel or ROS.
   Communication uses the Linux SocketCAN interface directly.

Dependency Installation
-------------------------

The controller node and the training/rollout node(s) should be set up with different software dependencies.

Robot Controller Node
~~~~~~~~~~~~~~~~~~~~~~

1. CAN Interface Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CAN bus must be initialized before using the robot.
The ``gim_arm_control`` SDK provides a convenience script, or you can run the commands manually.

Using the script from the ``gim_arm_control`` repository:

.. code:: bash

   bash sh/init_can.sh can0

Or manually:

.. code:: bash

   sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
   sudo ip link set can0 txqueuelen 1000
   sudo ip link set can0 up

This sets a 1 Mbps standard bitrate and 5 Mbps CAN FD data bitrate.

.. warning::

  The CAN interface must be re-initialized after every system reboot.
  You can verify the interface is up with:

  .. code:: bash

     ip link show can0

2. Motor Zero Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before first use (or after replacing a motor), you must calibrate the motor zero positions.
This sets the current physical position as the zero reference for Damiao motors.

From the ``gim_arm_control`` repository:

.. code:: bash

   # Zero a single motor (CAN ID in hex)
   bash sh/set_zero.sh can0 001

   # Zero all motors (001-008)
   bash sh/set_zero.sh can0 --all

.. warning::

  Calibration should only be done with the arm in its mechanical home position.
  Incorrect calibration can cause the arm to move unexpectedly.
  Requires ``can-utils`` (install with ``sudo apt install can-utils``).

3. Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

a. Clone RLinf Repository
__________________________

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

b. Install RLinf Dependencies
________________________________

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag for better download speed.

   bash requirements/install.sh embodied --env maniskill_libero
   source .venv/bin/activate

c. Install gim_arm_control SDK
________________________________

The ``gim_arm_control`` package provides the low-level CAN communication driver and Python bindings
for controlling the GimArm robot.

.. code:: bash

   cd /path/to/gim_arm_control/python
   pip install -e .

This builds the C++ core via CMake and installs Python bindings using nanobind.

**Build requirements**: ``scikit-build-core>=0.5``, ``nanobind>=2.0``, a C++17 compiler (GCC >= 7 or Clang >= 5).

**Runtime dependencies**: ``numpy``, ``pinocchio`` (imported as ``pin``).

.. note::

   ``pinocchio`` is required for forward kinematics and Jacobian computation used by the controller.
   It is automatically installed as a dependency of the SDK.
   For older systems requiring NumPy 1.x compatibility, install with:

   .. code:: bash

      pip install -e ".[pin270]"

Training/Rollout Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~

a. Clone RLinf Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

b. Install Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag for better download speed.

   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

.. note::

   Training/rollout nodes do **not** need the ``gim_arm_control`` SDK or ``pinocchio``.

Running the Experiment
-----------------------

Prerequisites
~~~~~~~~~~~~~~~

**Get the Target Pose for the Task**

To acquire the target end-effector pose for the peg-insertion task, you can use the hardware test script.

First, initialize the CAN interface (see above), then run:

.. code-block:: bash

   python toolkits/realworld_check/test_gim_arm_env.py --can can0 --variant gim_arm_xl

The script launches the controller, performs forward kinematics on the current joint positions,
and prints the TCP position and quaternion.
Manually move the arm to the desired target pose, then record the printed values.
Convert the quaternion to Euler XYZ angles for use in the ``target_ee_pose`` configuration field.

Data Collection
~~~~~~~~~~~~~~~~~

For RLPD experiments, you need to first collect some initial demonstration data.

Follow the VR teleoperation instructions in `XRoboToolkit <https://github.com/NVlabs/XRoboToolkit>`_ for data collection with the GimArm robot.

Configuration File
~~~~~~~~~~~~~~~~~~~~~~

Before starting the experiment, you need to create or modify a configuration YAML file.
The key section is the cluster hardware configuration, which specifies the GimArm robot:

.. code-block:: yaml

  cluster:
    num_nodes: 2
    component_placement:
      actor:
        node_group: "4090"
        placement: 0
      env:
        node_group: gim_arm
        placement: 0
      rollout:
        node_group: "4090"
        placement: 0
    node_groups:
      - label: "4090"
        node_ranks: 0
      - label: gim_arm
        node_ranks: 1
        hardware:
          type: GimArm
          configs:
            - can_interface: can0
              arm_variant: gim_arm_xl
              camera_serials: ["YOUR_CAMERA_SERIAL"]
              camera_type: realsense
              enable_gripper: true
              gripper_type: parallel
              node_rank: 1

Set the ``target_ee_pose`` in the environment override configuration:

.. code-block:: yaml

  env:
    train:
      override_cfg:
        target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
        reset_joint_qpos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        safe_retract_qpos: [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
        is_dummy: false

    eval:
      override_cfg:
        target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]

Key configuration fields:

- ``target_ee_pose``: Target end-effector pose ``[x, y, z, rx, ry, rz]`` (meters / Euler XYZ radians).
- ``reset_joint_qpos``: Joint configuration for the start of each episode.
- ``safe_retract_qpos``: Joint configuration for safe retraction during peg-insertion reset.
- ``is_dummy``: Set to ``true`` for testing without hardware.

Testing the Setup (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide several test scripts to verify that the setup is correct before starting the experiment. This step is optional but recommended.

1. Verify the CAN interface is up:

.. code-block:: bash

   ip link show can0

2. Test the robot controller:

.. code-block:: bash

   python toolkits/realworld_check/test_gim_arm_env.py --can can0 --variant gim_arm_xl

This script tests: controller launch, ``is_robot_up()``, ``get_state()`` output shapes, ``move_joints()``, ``reset_joint()``, and gripper open/close.

3. Test the camera connection:

.. code-block:: bash

   python -m toolkits.realworld_check.test_franka_camera

4. Test with dummy mode by setting ``is_dummy: true`` in the configuration and running:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh <your_gim_arm_config_name>

Running the Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~

After verifying the setup, you can start the real-world training experiment by running the following command on the head node:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh <your_gim_arm_config_name>
