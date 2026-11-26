GimArm Online RL
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gim-arm.png
   :align: center
   :width: 80%

   GimArm 6-DoF arm used for SocketCAN control, motor calibration, and peg-insertion training.

Run RLinf on a GimArm 6-DoF arm over Linux SocketCAN to calibrate motors, record target poses, collect VR demonstrations, and train real-world peg-insertion policies.

Overview
--------

This guide configures a two-node setup: a controller computer wired to GimArm over CAN FD and a GPU server that runs SAC/RLPD training and rollout.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      GimArm 6-DoF · parallel gripper

   .. grid-item-card:: Sensors & Teleop
      :text-align: center

      RealSense (optional) · PICO 4 Ultra VR

   .. grid-item-card:: Workflow
      :text-align: center

      CAN init · Zero cal · VR teleop · SAC / RLPD

   .. grid-item-card:: Hardware
      :text-align: center

      SocketCAN FD · Controller PC · GPU node

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Task
     - Config / Entry Point
     - Description
   * - Hardware check
     - ``test_gim_arm_env.py``
     - Verify controller launch, state reads, joint motion, reset, and gripper commands.
   * - Data collection
     - ``gim_arm_teleop``
     - Collect VR teleoperation demonstrations with a PICO 4 Ultra headset.
   * - Training
     - GimArm real-world config
     - Train a CNN peg-insertion policy with target-pose reward.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Optional RealSense camera frames plus GimArm joint and TCP state.
   * - Action
     - 6-DoF arm control with gripper command, depending on config.
   * - Reward
     - Sparse or dense forward-kinematics distance to ``target_ee_pose``.
   * - Prompt
     - Peg-insertion task text in the env config.

Hardware and Safety
-------------------

Connect the GimArm robot directly to the controller node through a Linux SocketCAN FD adapter (1 Mbps standard bitrate, 5 Mbps data bitrate), and place the controller node and GPU training node on the same local network.

.. warning::

   Keep an emergency stop within reach and clear the workspace before enabling motion. Run motor zero calibration only when the arm rests at its mechanical home position; incorrect calibration causes unexpected arm motion. Re-initialize the CAN interface after every system reboot.

.. dropdown:: Manual CAN FD setup and single-motor calibration

   If you prefer not to use ``sh/init_can.sh`` from ``gim_arm_control``, bring up ``can0`` manually:

   .. code-block:: bash

      sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
      sudo ip link set can0 txqueuelen 1000
      sudo ip link set can0 up

   Motor zero calibration requires ``can-utils`` (``sudo apt install can-utils``). To zero a single Damiao motor by hexadecimal CAN ID instead of all motors (``001``–``008``):

   .. code-block:: bash

      bash ~/gim_arm_control/sh/set_zero.sh can0 001

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Robot Controller Node

      **1. Install RLinf and the gim_arm_control SDK**

      Install RLinf's GimArm environment, then build the C++ CAN driver and Python bindings (requires ``scikit-build-core>=0.5``, ``nanobind>=2.0``, and a C++17 compiler; installs ``numpy`` and ``pinocchio`` automatically):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env gim_arm && source .venv/bin/activate

         git clone https://github.com/RLinf/gim_arm_control.git ~/gim_arm_control
         uv pip install -e ~/gim_arm_control/python

      For systems requiring NumPy 1.x compatibility, install the SDK with ``uv pip install -e "~/gim_arm_control/python[pin270]"``.

      **2. Initialize CAN FD and calibrate motor zero**

      Bring up ``can0`` using the helper script shipped in ``gim_arm_control`` and verify the link is up:

      .. code-block:: bash

         bash ~/gim_arm_control/sh/init_can.sh can0
         ip link show can0

      Before first use or after replacing a motor, place the arm in its mechanical home position and zero all Damiao motors (``001``–``008``):

      .. code-block:: bash

         bash ~/gim_arm_control/sh/set_zero.sh can0 --all

      **3. Run hardware check and record the target pose**

      Launch the hardware check script to verify controller startup, connection status, state output shapes, joint motion, joint reset, and gripper open/close:

      .. code-block:: bash

         python -m toolkits.realworld_check.test_gim_arm_env --can can0 --variant gim_arm_xl

      The script runs forward kinematics on the live joint angles and prints the TCP position and quaternion. Move the arm manually to the target peg-insertion pose, record the printed values, and convert the quaternion to Euler XYZ radians for ``target_ee_pose``.

      .. dropdown:: VR teleoperation data collection (optional)

         Collect demonstrations with `gim_arm_teleop <https://github.com/RLinf/gim_arm_teleop>`_ on Ubuntu 22.04 (x86) using a PICO 4 Ultra headset on the same LAN (enable headset camera permissions if you need passthrough VST):

         1. Install and start `XRoboToolkit-PC-Service <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.1.1/XRoboToolkit-PC-Service-1.1.1.deb>`__:

            .. code-block:: bash

               sudo dpkg -i XRoboToolkit-PC-Service-1.1.1.deb
               bash /opt/apps/roboticsservice/runService.sh

         2. Install `XRoboToolkit-PICO-1.1.1.apk <https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk>`__ on the headset, connect to the controller PC IP, and enable ``head``, ``hand``, and ``controller`` as needed.

         3. Follow the `gim_arm_teleop <https://github.com/RLinf/gim_arm_teleop>`__ setup and run hardware teleoperation to log VR control signals and arm state:

            .. code-block:: bash

               cd gim_arm_teleop
               python scripts/hardware/teleop_gim_arm_hardware.py

   .. tab-item:: Training & Rollout Nodes

      **1. Install RLinf**

      Install the GimArm environment on the GPU node (add ``--model <model>`` if you train a specific VLA model):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env gim_arm && source .venv/bin/activate

      **2. Configure hardware and target pose**

      Update the hardware connection and task poses in your GimArm training YAML:

      .. code-block:: yaml

         cluster:
           node_groups:
             - label: gim_arm
               node_ranks: 1
               hardware:
                 type: GimArm
                 configs:
                   - can_interface: can0
                     arm_variant: gim_arm_xl
                     camera_serials: ["YOUR_CAMERA_SERIAL"]  # Use [] for state-only runs
                     node_rank: 1

         env:
           train:
             override_cfg:
               target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]  # [x, y, z, rx, ry, rz] (m / rad)
               reset_joint_qpos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
               safe_retract_qpos: [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
               is_dummy: false

      - ``target_ee_pose`` sets the peg-insertion goal pose in meters and Euler XYZ radians.
      - ``reset_joint_qpos`` and ``safe_retract_qpos`` define the episode start pose and safe retraction pose during reset.
      - ``is_dummy: true`` runs without hardware; ``camera_serials: []`` runs state-only with an empty ``frames`` dict.

      .. dropdown:: Full configuration reference (optional)

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

      **3. Start Ray and launch training**

      No standalone GimArm training YAML ships in ``examples/embodiment/config/`` yet—start from an existing real-world SAC/RLPD config (such as ``realworld_peginsertion_rlpd_cnn_async.yaml``) and apply the hardware and env blocks above. Bring up the two-node Ray cluster following :doc:`../../guides/multi_node` (head on GPU rank ``0``, worker on controller rank ``1``), then launch from the head node:

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh <your_config>

Results and Troubleshooting
---------------------------

Monitor training on the Ray head node with ``tensorboard --logdir ./logs --port 6006`` and check ``env/success_once``, ``env/return``, and SAC losses; see :doc:`Training metrics <../../reference/metrics>` for metric definitions.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - Fix
   * - CAN interface missing or down after reboot
     - Re-run ``bash ~/gim_arm_control/sh/init_can.sh can0`` and confirm ``UP`` with ``ip link show can0``.
   * - Unexpected arm motion after motor replacement
     - Return the arm to its mechanical home position and re-run ``bash ~/gim_arm_control/sh/set_zero.sh can0 --all``.
   * - ``pinocchio`` or NumPy 1.x version conflict
     - Reinstall the SDK bindings with ``uv pip install -e "~/gim_arm_control/python[pin270]"``.
   * - Running without RealSense cameras attached
     - Set ``camera_serials: []`` in the hardware config to receive an empty ``frames`` dict for state-only checks.

To define custom tasks on GimArm, see :doc:`New Real-World Tasks <../../extending/new_task>`.
