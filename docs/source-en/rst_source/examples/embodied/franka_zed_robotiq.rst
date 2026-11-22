Franka ZED & Robotiq
====================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robotiq_zed.jpeg
   :align: center
   :width: 80%

   Franka setup with Stereolabs ZED cameras and a Robotiq 2F gripper for sensor-rich data collection.

Overview
--------

Replace the default RealSense camera and Franka gripper with Stereolabs ZED cameras and a Robotiq 2F gripper in a two-node Franka deployment. Node 0 (GPU server) captures ZED RGB streams, while Node 1 (NUC controller) drives the Franka arm and Robotiq gripper over USB-RS485.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Hardware
      :text-align: center

      Franka · ZED cameras · Robotiq 2F

   .. grid-item-card:: Connection
      :text-align: center

      USB 3.0 (ZED) · USB-RS485 Modbus RTU

   .. grid-item-card:: Workflow
      :text-align: center

      Two-node data collection & RL/SFT

   .. grid-item-card:: Requires
      :text-align: center

      :doc:`franka` · ZED SDK 4.2.x · 2 nodes

Hardware and Safety
-------------------

This configuration uses **two nodes** (``num_nodes: 2``):

- **Node 0 (GPU server)**: Connect ZED cameras over USB 3.0 and check ``lsusb`` for ``Stereolabs`` devices with the camera LED lit.
- **Node 1 (NUC controller)**: Connect the Franka arm over Ethernet and the Robotiq 2F gripper through a USB-to-RS485 adapter (``ls /dev/ttyUSB*``, typically ``/dev/ttyUSB0``). Grant serial access with ``sudo chmod 666 /dev/ttyUSB0`` or ``sudo usermod -aG dialout $USER`` (log out and back in).

.. warning::

   Keep the Franka e-stop button within reach, clear the workspace, and start from a safe pose before testing the gripper or collecting data. Use **ZED SDK 4.2.x** (such as v4.2.5): ZED SDK 5.0+ bundles a NumPy version that conflicts with the ``numpy<2`` constraint required by Franka cross-node serialization.

.. _franka-zed-robotiq-data-collection:

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. Setup & Hardware Check

      **Node 0 (GPU server) — Install ZED SDK 4.2.x and find camera serials**

      Download ZED SDK 4.2.x matching your OS and CUDA version from the `Stereolabs download page <https://www.stereolabs.com/developers/release>`_ (see the `official ZED Python API guide <https://www.stereolabs.com/docs/development/python/install>`_ for details), run the installer, and press **Y** when prompted to install the Python API:

      .. code-block:: bash

         # Example for CUDA 12.x on Ubuntu 22.04
         chmod +x ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run
         ./ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run

      To install ``pyzed`` into your RLinf virtual environment, activate it from the repository root and run the helper script or install the matching wheel:

      .. code-block:: bash

         source .venv/bin/activate
         python3 /usr/local/zed/get_python_api.py
         # Or direct wheel install:
         # uv pip install --reinstall /usr/local/zed/pyzed-*.whl

      List connected ZED cameras and record their serial numbers:

      .. code-block:: bash

         python -c "
         import pyzed.sl as sl
         for dev in sl.Camera.get_device_list():
             print(f'Serial: {dev.serial_number}  Model: {dev.camera_model}')
         "

      **Node 1 (NUC) — Install RLinf Franka env and verify Robotiq gripper**

      The ``franka`` environment automatically installs ``pymodbus`` for Modbus RTU control:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      Grant serial access and run a quick open/close check on the NUC:

      .. code-block:: bash

         sudo chmod 666 /dev/ttyUSB0
         python -c "
         from rlinf.robotics.parts.end_effectors.grippers.robotiq import RobotiqGripper
         g = RobotiqGripper(port='/dev/ttyUSB0')
         g.connect()
         print(f'Position: {g.position:.4f} m, Ready: {g.is_ready()}')
         g.open()
         import time; time.sleep(1)
         g.close()
         g.disconnect()
         "

   .. tab-item:: 2. Collect Data & Run

      Update the required hardware and target fields in ``examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml``:

      .. code-block:: yaml

         cluster:
           node_groups:
             - label: franka
               node_ranks: 0-1
               hardware:
                 type: Franka
                 configs:
                   - robot_ip: ROBOT_IP           # your robot's IP
                     camera_serials:
                       - "ZED_SERIAL_1"           # your ZED camera serial(s)
                     camera_type: zed
                     gripper_type: robotiq
                     gripper_connection: "/dev/ttyUSB0"
                     controller_node_rank: 1

         env:
           eval:
             override_cfg:
               target_ee_pose: TARGET_EE_POSE     # your target end-effector pose

      On **both nodes**, from the RLinf repository root with ``.venv`` active, export the node variables and start Ray:

      .. code-block:: bash

         # Node 0 (GPU server, Ray head; replace IP and NIC for your network)
         export RLINF_NODE_RANK=0
         export RLINF_COMM_NET_DEVICES=eth0  # optional, omit if single NIC
         ray start --head --port=6379 --node-ip-address=192.168.1.10

         # Node 1 (NUC controller)
         export RLINF_NODE_RANK=1
         export RLINF_COMM_NET_DEVICES=eth0  # optional, omit if single NIC
         ray start --address=192.168.1.10:6379

      Launch data collection on **Node 0 (head node)** and teleoperate with the SpaceMouse:

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh realworld_collect_data_zed_robotiq

      Pass another config name if you copy the YAML. Episodes save under ``logs/<timestamp>-<config_name>/`` as ``demo_buffer/`` and ``lerobot_dataset/``; point ``algorithm.demo_buffer.load_path`` to that path for downstream training.

.. dropdown:: Full multi-node YAML and hardware/collection fields

   Complete two-node cluster placement example with ZED cameras on Node 0 and arm/gripper control on Node 1:

   .. code-block:: yaml

      cluster:
        num_nodes: 2
        component_placement:
          actor:
            node_group: gpu
            placement: 0
          env:
            node_group: franka
            placement: 0
          rollout:
            node_group: gpu
            placement: 0
        node_groups:
          - label: gpu
            node_ranks: 0
          - label: franka
            node_ranks: 0-1
            hardware:
              type: Franka
              configs:
                - robot_ip: <ROBOT_IP>
                  node_rank: 0
                  camera_serials:
                    - "<ZED_SERIAL_1>"
                    - "<ZED_SERIAL_2>"
                    - "<ZED_SERIAL_3>"
                  camera_type: zed            # "realsense" or "zed"
                  gripper_type: robotiq       # "franka" or "robotiq"
                  gripper_connection: "/dev/ttyUSB0"
                  controller_node_rank: 1     # arm control runs on the NUC
                  disable_validate: false

   **Hardware config fields**

   .. list-table::
      :header-rows: 1
      :widths: 25 15 60

      * - Field
        - Default
        - Description
      * - ``camera_type``
        - ``"realsense"``
        - Camera backend. Set to ``"zed"`` for ZED cameras.
      * - ``gripper_type``
        - ``"franka"``
        - Gripper backend. Set to ``"robotiq"`` for Robotiq grippers.
      * - ``gripper_connection``
        - ``null``
        - Serial port for Robotiq (such as ``"/dev/ttyUSB0"``). Ignored when ``gripper_type`` is ``"franka"``.
      * - ``controller_node_rank``
        - ``null``
        - Node rank hosting arm control. Co-locates with the env worker when ``null``; set to ``1`` when the arm and cameras live on separate machines.

   **Data-collection fields**

   Key fields in ``realworld_collect_data_zed_robotiq.yaml``:

   .. list-table::
      :header-rows: 1
      :widths: 30 18 52

      * - Field
        - Default
        - Description
      * - ``runner.num_data_episodes``
        - ``20``
        - Number of episodes to collect before exiting.
      * - ``env.eval.teleop``
        - ``spacemouse``
        - Teleoperation device driving the arm during collection.
      * - ``env.eval.data_collection.save_dir``
        - ``${runner.logger.log_path}``
        - Output root (``{save_dir}/lerobot_dataset`` and ``{save_dir}/demo_buffer``).
      * - ``env.eval.data_collection.export_format``
        - ``[replay_buffer, lerobot_dataset]``
        - Real-robot collector accepts ``replay_buffer``, ``pickle_episode``, and ``lerobot_dataset``.
      * - ``env.eval.data_collection.only_success``
        - ``True``
        - Save only successful episodes.

.. dropdown:: Cluster setup notes

   - **Node 0 (GPU server)**: Install ZED SDK 4.2.x and ``pyzed`` into the virtual environment **before** running ``ray start``.
   - **Node 1 (NUC)**: Verify ``pymodbus`` is installed and the Robotiq serial port is readable **before** running ``ray start``.

   .. warning::

      Ray captures the Python interpreter and environment variables at ``ray start`` time. SDKs or libraries installed after ``ray start`` are invisible to Ray workers. Always install dependencies first, then start Ray.

   See :doc:`franka` and :doc:`../../guides/hetero` for full multi-node Ray setup details.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Symptom
     - Fix
   * - ZED camera not detected
     - Check the USB 3.0 cable and camera LED, run ``lsusb`` for ``Stereolabs`` devices, and confirm the ZED SDK version matches your CUDA version.
   * - NumPy conflict after ZED SDK install
     - Install **ZED SDK 4.2.x** (such as v4.2.5) instead of 5.0+, which bundles an incompatible NumPy version against Franka's ``numpy<2`` requirement.
   * - Robotiq gripper not responding
     - Confirm ``ls /dev/ttyUSB*`` shows the adapter, grant access with ``sudo chmod 666 /dev/ttyUSB0``, verify ``pymodbus`` is ``>=3.0,<4.0``, and try ``baudrate=9600`` if serial traffic is noisy.
   * - Controller node rank mismatch
     - Set ``controller_node_rank: 1`` whenever the arm is on the NUC (node 1) and ZED cameras are on the GPU server (node 0); otherwise arm control attempts to start on node 0 with no robot link.
