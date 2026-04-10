Real-World Franka with ZED Cameras and Robotiq Gripper
======================================================

This guide explains how to set up and use **Stereolabs ZED cameras** and a
**Robotiq 2F-85/2F-140 gripper** with the Franka real-world environment in
RLinf.  It extends the base :doc:`franka` documentation with
hardware-specific installation, configuration, and data collection.

.. note::

   If you have not read the base Franka guide yet, please start with
   :doc:`franka` first.  This page only covers the **additional** steps
   required for ZED and Robotiq hardware.


Hardware Overview
-----------------

A typical ZED + Robotiq deployment uses **two nodes**:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Node
     - Role
     - Hardware
   * - **GPU server** (node 0)
     - Actor, rollout, env worker; camera capture
     - NVIDIA GPU (e.g. RTX 4090), 1-3 ZED cameras
   * - **NUC** (node 1)
     - FrankaController, Robotiq gripper
     - Franka arm, Robotiq 2F via USB-RS485

The GPU server runs the ZED cameras because the ZED SDK leverages GPU
acceleration for depth and image processing.  The Robotiq gripper is
connected to the NUC (or whichever machine is physically wired to the arm)
via a USB-to-RS485 adapter.


ZED Camera Installation
-----------------------

The ZED SDK and its Python API must be installed on every node that captures
images (typically the GPU server node).  Full details are available in the
`official ZED Python API installation guide <https://www.stereolabs.com/docs/development/python/install>`_.

1. Install the ZED SDK
^^^^^^^^^^^^^^^^^^^^^^^^

Download the SDK installer from the
`Stereolabs download page <https://www.stereolabs.com/developers/release>`_
and choose the version that matches your OS and CUDA version.

.. warning::

   Use **ZED SDK 4.2.x** (e.g. v4.2.5).  ZED SDK 5.0+ ships its own
   bundled NumPy which conflicts with the ``numpy<2`` pin required by
   Franka cross-node serialization.  Stick with the 4.2 series to avoid
   dependency issues.

.. code-block:: bash

   # Example for CUDA 12.x on Ubuntu 22.04
   chmod +x ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run
   ./ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run

Follow the on-screen prompts.  When asked
``Do you want to install the Python API (recommended) [Y/n] ?``,
press **Y** to install the Python bindings automatically.

2. Install the Python API (if not done during SDK setup)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you skipped the Python API during SDK installation, or need to install
it into a different virtual environment, run the bundled helper script:

.. code-block:: bash

   # Activate your virtual environment first
   source /path/to/your/venv/bin/activate

   # Run the official install script
   cd /usr/local/zed/
   python3 get_python_api.py

The script auto-detects your platform, Python version, and ZED SDK version,
then downloads and installs the matching ``pyzed`` wheel.

Alternatively, you can install the wheel directly:

.. code-block:: bash

   python -m pip install --ignore-installed /usr/local/zed/pyzed-*.whl

.. note::

   The ``pyzed`` wheel is tied to a specific Python version and CUDA
   version.  Make sure you install it into the **same virtual environment**
   that Ray will use on this node.  If you are using a virtual environment,
   activate it **before** running the install script.

3. Verify camera detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

List connected ZED cameras and note their **serial numbers**:

.. code-block:: bash

   python -c "
   import pyzed.sl as sl
   for dev in sl.Camera.get_device_list():
       print(f'Serial: {dev.serial_number}  Model: {dev.camera_model}')
   "

Record the serial numbers; they will be used in the YAML configuration.


Robotiq Gripper Installation
-----------------------------

The Robotiq gripper communicates over **Modbus RTU** through a USB-to-RS485
adapter.  The required Python dependency ``pymodbus`` is **automatically
installed** when you run the Franka installation script:

.. code-block:: bash

   bash requirements/install.sh embodied --env franka

The following steps set up the serial device on the node that controls the
gripper (typically the NUC).

1. Set up the serial device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plug the USB-to-RS485 adapter into the NUC.  Identify the serial device:

.. code-block:: bash

   ls /dev/ttyUSB*
   # Typically /dev/ttyUSB0

Grant permission:

.. code-block:: bash

   sudo chmod 666 /dev/ttyUSB0
   # Or add your user to the dialout group for persistent access:
   sudo usermod -aG dialout $USER

3. Verify the gripper
^^^^^^^^^^^^^^^^^^^^^^^

A quick sanity check (run on the NUC):

.. code-block:: bash

   python -c "
   from rlinf.envs.realworld.common.gripper.robotiq_gripper import RobotiqGripper
   g = RobotiqGripper(port='/dev/ttyUSB0')
   print(f'Position: {g.position:.4f} m, Ready: {g.is_ready}')
   g.open()
   import time; time.sleep(1)
   g.close()
   g.cleanup()
   "


YAML Configuration
-------------------

The key differences from a standard (RealSense + Franka gripper) config are
the new ``camera_type``, ``gripper_type``, ``gripper_connection``, and
``controller_node_rank`` fields inside the ``hardware.configs`` block.

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: gpu
         placement: 0
       env:
         node_group: gpu
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
               controller_node_rank: 1     # FrankaController runs on the NUC
               disable_validate: false

.. list-table:: New hardware config fields
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``camera_type``
     - ``"realsense"``
     - Camera backend.  Set to ``"zed"`` for ZED cameras.
   * - ``gripper_type``
     - ``"franka"``
     - Gripper backend.  Set to ``"robotiq"`` for Robotiq grippers.
   * - ``gripper_connection``
     - ``null``
     - Serial port for Robotiq (e.g. ``"/dev/ttyUSB0"``).  Ignored when
       ``gripper_type`` is ``"franka"``.
   * - ``controller_node_rank``
     - ``null``
     - Node rank where ``FrankaController`` runs.  When ``null``, it
       co-locates with the env worker.  Set this when the arm and cameras
       are on different machines.


.. _franka-zed-robotiq-data-collection:

Data Collection
-----------------

A dedicated data collection script and configuration are provided for
collecting teleoperation demonstrations using the ZED + Robotiq hardware
setup.  This mirrors the base Franka data collection workflow described in
:doc:`franka`, but with ZED camera and Robotiq gripper defaults
pre-configured.

**Configuration**

The config file is
``examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml``.
Before running, update the following fields to match your setup:

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

Key data-collection settings in the same file:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Description
   * - ``runner.num_data_episodes``
     - ``20``
     - Number of episodes to collect before the script exits.
   * - ``env.eval.use_spacemouse``
     - ``True``
     - Enable SpaceMouse teleoperation during collection.
   * - ``env.data_collection.save_dir``
     - ``${runner.logger.log_path}/collected_data``
     - Directory where collected trajectories are saved.
   * - ``env.data_collection.export_format``
     - ``"lerobot"``
     - Export format for the collected data.
   * - ``env.data_collection.only_success``
     - ``True``
     - Only save successful episodes.

**Running data collection**

.. important::

   Unlike the base Franka setup described in :doc:`franka` (which runs on
   a single controller node with ``num_nodes: 1``), the ZED + Robotiq
   configuration requires **two nodes** (``num_nodes: 2``):

   - **Node 0** – GPU server with ZED camera connected.
   - **Node 1** – NUC / controller node connected to the robot arm and
     Robotiq gripper (``controller_node_rank: 1``).

   Both nodes must have Ray running before you launch the script.

1. On **each node**, source the virtual environment and set the required
   environment variables:

.. code-block:: bash

   source <path_to_your_venv>/bin/activate

   export PYTHONPATH=<path_to_your_RLinf_repo>:$PYTHONPATH
   export RLINF_NODE_RANK=<0_or_1>
   # Optional – only needed if you have multiple network interfaces
   export RLINF_COMM_NET_DEVICES=<network_device>

2. Start Ray on each node:

.. code-block:: bash

   # On the head node (node 0, GPU server)
   ray start --head --port=6379 --node-ip-address=<head_node_ip>

   # On the controller node (node 1, NUC)
   ray start --address=<head_node_ip>:6379

3. Run the data collection script **on the head node (node 0)**:

.. code-block:: bash

   bash examples/embodiment/collect_data_zed_robotiq.sh

The script defaults to the ``realworld_collect_data_zed_robotiq`` config.
You can pass a different config name as an argument:

.. code-block:: bash

   bash examples/embodiment/collect_data_zed_robotiq.sh <config_name>

During collection, use the SpaceMouse to teleoperate the robot.  The script
will terminate after the configured number of episodes and save the data
under ``logs/<timestamp>-<config_name>/collected_data/``.

After collection, upload the data to the training node and set the
``algorithm.demo_buffer.load_path`` field in your training config to the
data path.


Cluster Setup Notes
---------------------

The cluster setup procedure is the same as described in
:doc:`franka`, with one key difference:

- On the **GPU server** (node 0): make sure the ZED SDK and ``pyzed`` are
  installed in the virtual environment **before** running ``ray start``.
- On the **NUC** (node 1): make sure ``pymodbus`` is installed, and the
  Robotiq serial device is accessible.

.. warning::

   Remember that Ray captures the Python interpreter and environment
   variables at ``ray start`` time.  Any SDK or library installed
   **after** ``ray start`` will not be visible to Ray workers.  Always
   install dependencies first, then start Ray.

For multi-node Ray setup details, refer to :doc:`franka` and
:doc:`../../tutorials/advance/hetero`.


Troubleshooting
----------------

**ZED camera not detected**

- Verify the USB 3.0 cable is connected and the camera LED is on.
- Run ``lsusb`` and look for ``Stereolabs`` devices.
- Ensure the ZED SDK version matches your CUDA version.

**Robotiq gripper not responding**

- Check ``ls /dev/ttyUSB*`` to confirm the serial device is present.
- Verify permissions: ``sudo chmod 666 /dev/ttyUSB0``.
- Ensure ``pymodbus`` version is ``>=3.0,<4.0``.
- Try reducing the baud rate if communication is unreliable:
  ``RobotiqGripper(port='/dev/ttyUSB0', baudrate=9600)``.

**Controller node rank mismatch**

- If the arm is on a NUC (node 1) but cameras are on the GPU server
  (node 0), you **must** set ``controller_node_rank: 1`` in the hardware
  config.  Otherwise the controller will try to start on node 0, which
  does not have ROS or the arm connection.
