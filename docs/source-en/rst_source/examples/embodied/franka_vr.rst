Real-World Franka with VR Teleoperation
========================================

This guide explains how to set up and use **VR / PICO** teleoperation devices
with the Franka real-world environment in RLinf. It extends the base
:doc:`franka` documentation and only covers the **additional** steps required
for the VR teleoperation pipeline.

.. note::

   If you have not read the base Franka guide yet, please start with
   :doc:`franka` first. This page assumes that Franka control, ROS, Ray, and
   camera setup have already been configured according to the base guide.


Hardware Overview
-----------------

The current VR teleoperation pipeline uses a PICO headset and controller as
the example device. PICO data is collected by an external service and
published through ZeroMQ. The PICO intervention wrapper in RLinf subscribes
to that data stream and converts relative controller motion into normalized
end-effector delta actions for the Franka environment.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Node
     - Role
     - Hardware / Software
   * - **GPU server** (node 0)
     - Actor, rollout, optional camera capture
     - NVIDIA GPU, RLinf
   * - **Franka controller node** (node 1 or single-node setup)
     - FrankaController, env worker, VR data subscriber
     - Franka, ROS Noetic, serl_franka_controllers, pyzmq
   * - **VR / PICO PC**
     - Runs XRoboToolkit and the VR data publisher
     - PICO headset, controller, VR publisher

If the VR publisher and RLinf env worker run on the same machine, you can use
an IPC address. If they run on different machines, use a TCP address.


VR Software Setup
------------------------------

The VR data publisher is not launched by RLinf directly. It first reads
headset and controller data from PICO / XRoboToolkit, then publishes the data
to ZeroMQ.

1. Prepare PICO / XRoboToolkit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Complete the following setup on the PICO headset and the machine running the
publisher:

- On the PICO headset, install a compatible APK from
  `XRoboToolkit-Unity-Client releases <https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases>`_.
- On the machine that sends and receives PICO data, install a compatible PC
  Service from
  `XRoboToolkit-PC-Service releases <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases>`_.
- Confirm that the PICO headset and controllers are connected and that pose
  and button data update continuously.
- Ensure that the publisher machine is on the same network as the Franka
  controller node, or that it runs on the same machine as the RLinf env worker.

After the PICO headset is connected to the data publisher machine, start the
XRoboToolkit PC Service on that machine:

.. code-block:: bash

   cd /opt/apps/roboticsservice
   bash runService.sh

Choose an installation directory and clone the repository:

.. code-block:: bash

   cd /path/to/install/pico
   git clone git@github.com:tiny-xie/pico_software.git
   cd pico_software

Set up the environment:

.. code-block:: bash

   bash setup_uv.sh

Then start the VR data publisher on the same machine:

.. code-block:: bash

   cd /path/to/pico_software
   source .venv/bin/activate
   python -m vr_teleop.vr_data_publisher --config configs/vr_bridge.yaml

2. Configure the ZeroMQ address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For same-machine deployment, IPC can be used:

.. code-block:: yaml

   zmq:
     ipc_addr: "ipc:///tmp/vr_data.ipc"
   publish_rate: 80

For cross-machine deployment, the publisher should bind to TCP:

.. code-block:: yaml

   zmq:
     ipc_addr: "tcp://0.0.0.0:<port>"
   publish_rate: 80

The RLinf consumer connects to the publisher machine:

.. code-block:: yaml

   pico:
     zmq_addr: "tcp://<vr_publisher_ip>:<port>"

.. warning::

   The publisher bind address and the RLinf connect address must match. Do
   not use ``ipc:///tmp/vr_data.ipc`` for cross-machine setups, because IPC
   files are only valid on the same machine.


3. Install RLinf-side dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the RLinf environment that runs the PICO intervention with the
``franka-vr`` env. It includes the base ``franka`` real-world dependencies
and additionally installs ``pyzmq`` for the VR / PICO pipeline:

.. code-block:: bash

   bash requirements/install.sh embodied --env franka-vr
   source .venv/bin/activate

If you use Ray, install it and source the corresponding environment **before**
running ``ray start``.

.. warning::

   Ray captures the Python interpreter and environment variables at
   ``ray start`` time. If ``pyzmq``, ROS environment variables, or
   ``PYTHONPATH`` are configured after ``ray start``, worker processes may
   fail to import ``PicoIntervention`` or connect to ZeroMQ.


4. Verify the PICO data stream
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After starting the VR data publisher, run the built-in RLinf check script on
the node that will run ``PicoIntervention`` to confirm that PICO / ZeroMQ data
is being received:

.. code-block:: bash

   python toolkits/realworld_check/test_pico_data.py \
       --zmq-addr ipc:///tmp/vr_data.ipc

For cross-machine publisher and RLinf env-worker setups, replace
``--zmq-addr`` with the TCP address used by ``pico.zmq_addr`` in YAML:

.. code-block:: bash

   python toolkits/realworld_check/test_pico_data.py \
       --zmq-addr tcp://<vr_publisher_ip>:<port>

You should see continuously updating output like:

.. code-block:: text

   [000012] recv_rate=79.8Hz | headset: pos=[0.120 1.430 0.210] quat=[0.000 0.707 0.000 0.707] | right: pos=[0.320 1.120 0.850] quat=[0.010 0.690 0.020 0.724] grip=0.000 trigger=0.000 | buttons_active=none

If the output keeps refreshing, RLinf can receive the PICO data stream. When
you move the controller or press ``grip`` / ``trigger`` / ``A`` / ``B``, the
corresponding values or button states should update with the incoming data.


YAML Configuration
-------------------

To use PICO for data collection, use the config file
``examples/embodiment/config/realworld_collect_data_pico.yaml``.

The key configuration is:

.. code-block:: yaml

   env:
     eval:
       use_spacemouse: False
       use_pico: True

       pico:
         zmq_addr: "tcp://<vr_publisher_ip>:<port>"
         position_scale: 1.0
         rotation_scale: 1.0
         calibration:
           button: "trigger"


Gripper Configuration
---------------------

If ``no_gripper: true``, PICO only controls the 6D end-effector motion and
does not control the gripper.

To control the gripper through VR, make sure the environment action space
contains the gripper dimension and use a supported end-effector configuration:

.. code-block:: yaml

   env:
     eval:
       no_gripper: False
       pico:
         gripper_close_button: "A"
         gripper_open_button: "B"
         gripper_close_threshold: 0.5

Current button semantics:

.. code-block:: text

   A -> close gripper
   B -> open gripper

If neither A nor B is pressed, the gripper action is ``0.0``. Releasing the
button does not automatically open the gripper.


Cluster Setup Notes
---------------------

The cluster setup procedure is the same as described in :doc:`franka`. The
main additional requirements are:

- The node running ``PicoIntervention`` must be able to reach the VR publisher
  ZeroMQ address.
- If the VR publisher uses TCP, make sure the firewall and network route allow
  access to the configured port.
- If the VR publisher uses IPC, the publisher and RLinf env worker must run on
  the same machine.
- ``RLINF_NODE_RANK``, ``cluster.node_groups[*].node_ranks`` in YAML, and
  ``node_rank`` in the Franka hardware config must be consistent.


Safety Notes
------------

- For the first real-robot run, lower ``position_scale`` to a conservative
  value such as ``0.3`` to ``0.5``.
- Check the workspace safety box, per-step action scale, and Franka Desk state
  before placing task objects in the workspace.
- After changing Ray environments, Python dependencies, ROS environment
  variables, or ZeroMQ addresses, stop Ray first and then restart it.
- If the control direction is clearly wrong, release ``grip``, reposition
  yourself, and press ``trigger`` again for calibration. Do not try to correct
  the frame while actively controlling the robot.


Startup Order
-------------

1. On the Franka controller node, configure ROS, the catkin workspace, the
   RLinf virtual environment, and ``PYTHONPATH``.
2. Confirm that the ``franka-vr`` environment is installed and sourced before
   starting Ray.
3. Start the Ray cluster. Single-node and multi-node startup follow the same
   steps as :doc:`franka`.
4. Start the PICO / XRoboToolkit PC Service and confirm that the headset and
   controller are connected.
5. Start the VR data publisher.
6. On the first run, or after changing the ZeroMQ address, run
   ``test_pico_data.py`` to confirm that PICO data is reachable.
7. On the Ray head node, start data collection, and confirm that the data
   collection script has integrated ``PicoIntervention``.

.. code-block:: bash

   cd /path/to/RLinf
   bash examples/embodiment/collect_data.sh realworld_collect_data_pico

.. warning::

   Only one program should hold Franka control at a time. When running RLinf
   data collection, do not simultaneously run ``robo_avatar.slave.main``,
   another ``franka_control_node``, or another RLinf real-world task.


VR Operation
------------

For the first real-robot test, remove task objects and only validate small
motions.

1. Stand in the desired operator pose with the headset facing the front of the
   robot workspace.
2. Press ``trigger`` to calibrate the PICO base frame.
3. Hold controller ``grip`` to start intervention. When ``grip`` is pressed,
   the system records the controller reference pose and the current TCP
   reference pose.
4. Move the controller slightly and confirm the direction mapping:

.. code-block:: text

   Move controller forward  -> end-effector +X
   Move controller backward -> end-effector -X
   Move controller left/right -> end-effector +/-Y
   Move controller up/down -> end-effector +/-Z
   Rotate controller -> end-effector roll / pitch / yaw

5. If gripper control is enabled, confirm:

.. code-block:: text

   A -> close gripper
   B -> open gripper

6. Releasing ``grip`` stops intervention. Holding ``grip`` again records a new
   reference pose.


Troubleshooting
---------------

**No VR data after startup**

- Confirm that XRoboToolkit PC Service / runService is running.
- Confirm that the VR publisher process has not exited.
- Run ``test_pico_data.py`` first to confirm that the RLinf side can receive
  PICO data directly.
- Confirm that ``pico.zmq_addr`` matches the publisher address.
- For TCP setups, confirm that the publisher IP and port are reachable from
  the RLinf env worker node.

**The robot does not move when holding grip**

- Confirm that ``use_spacemouse: False`` and ``use_pico: True``.
- Confirm that the current code creates ``PicoIntervention`` in ``apply.py``.
- Confirm that the ``grip`` value exceeds ``control_threshold``.
- Confirm that calibration has completed. If ``calibration.required=True`` and
  calibration has not completed, PICO will not take over.
- Check whether the motion is being limited by the Franka safety box or
  ``action_scale``.

**Wrong direction or jump at activation**

- Release ``grip``, return to a comfortable pose, and press ``trigger`` again
  to calibrate.
- Hold ``grip`` again so the system records a fresh reference pose.
- If the entire control frame is rotated, adjust ``operator_to_robot_yaw``,
  for example ``1.5708``, ``3.1416``, or ``-1.5708``.

**The gripper does not respond**

- Confirm that ``no_gripper: False``.
- Confirm that the environment action space includes the seventh gripper
  action dimension.
- Confirm that no other wrapper is overriding the gripper action.

**FrankaHW initialization failed / UDP timeout**

If the log contains:

.. code-block:: text

   FrankaHW: Failed to initialize libfranka robot. libfranka: UDP receive: Timeout

This is usually not a VR data issue. It means the Franka controller node did
not establish FCI UDP communication with the robot. Check ``robot_ip``,
programmable mode in Franka Desk, controller-node network routing, and whether
another process is already holding Franka control.
