Real-World RL with GimArm
============================

This document covers the hardware setup, dependency installation, and experiment
configuration for the GimArm 6-DOF robotic arm within the RLinf framework.

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

- **Reward**: Computed in Cartesian space by comparing FK-based TCP pose to ``target_ee_pose``. Sparse (0/1) by default with optional dense exponential falloff.

Peg Insertion Task
~~~~~~~~~~~~~~~~~~~~

The peg-insertion task (``GimArmPegInsertionEnv``, registered as ``GimArmPegInsertionEnv-v1``) is implemented
in ``rlinf/envs/realworld/gim_arm/tasks/peg_insertion.py``. It extends ``GimArmEnv`` with task-specific
reset and reward logic:

- **Reset**: The gripper closes on the peg, the arm retracts to ``safe_retract_qpos`` to clear the hole,
  then moves to ``reset_joint_qpos``. When ``enable_random_reset`` is enabled (default), small
  joint-space perturbations (controlled by ``random_joint_noise``, default 0.02 rad) are applied
  to the reset configuration for diversity.

- **Reward**: Computed in Cartesian space by comparing the FK-derived TCP pose against
  ``target_ee_pose``. Success is determined per-axis using ``reward_threshold``
  (default: 1 cm on XYZ position). The ``reward_threshold`` config accepts a
  6-element ``[x, y, z, rx, ry, rz]`` array for Franka-API parity, but only
  the XYZ entries are currently consulted; orientation entries are reserved
  for future use.

Hardware Setup
----------------

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

1. Installation
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

   bash requirements/install.sh embodied --env gim_arm
   source .venv/bin/activate

c. Install gim_arm_control SDK
________________________________

The ``gim_arm_control`` package provides the low-level CAN communication driver and Python bindings
for controlling the GimArm robot. It also ships the helper shell scripts used in the next steps
(``sh/init_can.sh``, ``sh/set_zero.sh``), so install it before proceeding.

.. code:: bash

   # Clone the SDK alongside RLinf (the example below assumes ~/gim_arm_control).
   cd ~
   git clone https://github.com/RLinf/gim_arm_control.git
   cd ~/gim_arm_control/python
   pip install -e .

This builds the C++ core via CMake and installs Python bindings using nanobind.

**Build requirements**: ``scikit-build-core>=0.5``, ``nanobind>=2.0``, a C++17 compiler (GCC >= 7 or Clang >= 5).

**Runtime dependencies**: ``numpy``, ``pinocchio`` (imported as ``pin``).

.. note::

   ``pinocchio`` is required for forward kinematics and Jacobian computation used by the controller.
   It is automatically installed as a dependency of the SDK.
   For systems requiring NumPy 1.x compatibility, install with:

   .. code:: bash

      pip install -e ".[pin270]"

2. CAN Interface Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CAN bus must be initialized before using the robot.
The ``gim_arm_control`` SDK (installed in the previous step) provides a convenience script,
or you can run the commands manually.

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

3. Motor Zero Calibration
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

Training/Rollout Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the RLinf repository (same as above), then install dependencies.

Install Dependencies
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

   bash requirements/install.sh embodied --env gim_arm
   source .venv/bin/activate

   # To install model-specific dependencies (e.g. OpenVLA), add the --model flag:
   # bash requirements/install.sh embodied --model openvla --env maniskill_libero


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

Follow the provided VR teleoperation code, which are adapted from `XRoboToolkit <https://github.com/XR-Robotics>`_ for data collection with the GimArm robot.

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
              camera_serials: ["YOUR_CAMERA_SERIAL"]  # Use [] if no cameras are available
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

.. note::

   Camera support is optional. If ``camera_serials`` is set to an empty list
   ``[]`` or omitted, the environment will run without camera observations.
   The ``frames`` key in the observation space will be an empty dictionary.
   This is useful for state-only policies or when cameras are not yet set up.

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

.. note::

   Camera setup has not been fully tested yet. To run the peg-insertion
   experiment, cameras should be available and configured via
   ``camera_serials`` in the hardware config.
