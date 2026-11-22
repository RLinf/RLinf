Franka PICO VR Teleop
=====================

Overview
--------

Teleoperate a real Franka arm with a PICO VR headset and controller for demonstration collection and human-in-the-loop intervention. An external VR publisher reads headset and controller poses from XRoboToolkit and streams them over ZeroMQ to the RLinf environment worker, which maps relative controller motion to normalized end-effector deltas. Complete base Franka control, camera, and Ray setup in :doc:`franka` first.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Hardware
      :text-align: center

      Franka · PICO headset & controller

   .. grid-item-card:: Connection
      :text-align: center

      ZeroMQ (IPC same host / TCP cross-host)

   .. grid-item-card:: Workflow
      :text-align: center

      VR teleop collection & intervention

   .. grid-item-card:: Requires
      :text-align: center

      :doc:`franka` · XRoboToolkit · Ubuntu 22.04/24.04

Hardware and Safety
-------------------

The VR pipeline runs across up to three roles. Use an IPC address when the VR publisher and RLinf env worker share a machine, or a TCP address when they run on separate hosts on the same network.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Node
     - Role
     - Hardware / Software
   * - **GPU server** (node 0)
     - Actor, rollout, optional camera capture
     - NVIDIA GPU, RLinf
   * - **Franka controller node** (node 1 or single-node)
     - Franka arm, env worker, VR subscriber
     - Franky, ``pyzmq``
   * - **VR / PICO PC**
     - XRoboToolkit PC Service and VR data publisher
     - PICO headset, controller, ``pico_software``

.. warning::

   Keep the Franka e-stop button within reach and clear task objects from the workspace before the first run. Lower ``position_scale`` to ``0.3``–``0.5`` on your first real-robot test, and check the workspace safety box, per-step ``action_scale``, and Franka Desk state before placing objects. If the control direction looks wrong, release ``grip``, reposition, and press ``trigger`` to recalibrate—never try to correct the frame while holding ``grip``. After changing Ray environments, Python packages, robot settings, or ZeroMQ addresses, run ``ray stop`` and restart Ray.

.. _pico-software-setup:

Prepare the PICO headset and XRoboToolkit PC Service on the publisher PC, then verify the XRT data stream before launching the ZeroMQ bridge:

.. dropdown:: Install PICO APK, XRoboToolkit PC Service, and verify XRT

   1. Install a compatible APK on the PICO headset from `XRoboToolkit-Unity-Client releases <https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases>`_.
   2. On the publisher machine (Ubuntu 22.04 or Ubuntu 24.04; other OS versions require local adaptation), install the matching PC Service from `XRoboToolkit-PC-Service releases <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases>`_. Ensure the publisher machine shares a network with the controller node or runs on the same host as the RLinf env worker.
   3. Start the XRoboToolkit PC Service before connecting the headset:

      .. code-block:: bash

         (cd /opt/apps/roboticsservice && bash runService.sh)

   4. Connect the PICO headset to the PC Service and confirm the headset and controllers report live poses and button states.
   5. Clone ``pico_software`` (which wraps `XR-Robotics/XRoboToolkit-Teleop-Sample-Python <https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python>`_), build its environment, and verify the XRT pipeline:

      .. code-block:: bash

         git clone https://github.com/tiny-xie/pico_software.git && cd pico_software
         (cd XRoboToolkit-Teleop-Sample-Python && bash setup_uv.sh)
         source XRoboToolkit-Teleop-Sample-Python/.venv/bin/activate
         python test_pico_xrt_pipeline.py

      Moving the headset and controllers should print a continuously updating stream in the terminal.

.. _pico-publisher-setup:

Configure the publisher-side ZeroMQ bind address in ``pico_software/configs/vr_bridge.yaml`` and start the VR data publisher:

.. dropdown:: Configure and start the ZeroMQ VR publisher

   For same-machine deployment, bind to IPC:

   .. code-block:: yaml

      zmq:
        ipc_addr: "ipc:///tmp/vr_data.ipc"

   For cross-machine deployment, bind to TCP:

   .. code-block:: yaml

      zmq:
        ipc_addr: "tcp://0.0.0.0:<port>"

   From the ``pico_software`` directory on the VR / PICO PC, activate the virtual environment created above and launch the publisher:

   .. code-block:: bash

      source XRoboToolkit-Teleop-Sample-Python/.venv/bin/activate
      python -m vr_data_publisher --config configs/vr_bridge.yaml

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. Setup & VR Bridge

      Install the ``franka`` environment on the Franka controller node (includes base Franka dependencies and ``pyzmq``):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      .. warning::

         Install and activate this environment **before** running ``ray start``. Ray captures the Python interpreter and environment variables at startup; configuring ``pyzmq`` or the Franka environment afterward causes workers to miss the controller or ZeroMQ stream.

      With the VR data publisher running, verify ZeroMQ reception on the node that reads the controller:

      .. code-block:: bash

         # Same-machine IPC
         python -m toolkits.realworld_check.test_pico_data \
             --zmq-addr ipc:///tmp/vr_data.ipc

         # Cross-machine TCP (replace with your publisher IP:port; matches env.eval.pico.zmq_addr)
         python -m toolkits.realworld_check.test_pico_data \
             --zmq-addr tcp://192.168.1.100:5555

      Confirm continuously refreshing output as you move the controller and press ``grip``, ``trigger``, ``A``, or ``B``:

      .. code-block:: text

         [000012] recv_rate=79.8Hz | headset: pos=[0.120 1.430 0.210] quat=[0.000 0.707 0.000 0.707] | right: pos=[0.320 1.120 0.850] quat=[0.010 0.690 0.020 0.724] grip=0.000 trigger=0.000 | buttons_active=none

   .. tab-item:: 2. Configure & Operate

      Edit ``examples/embodiment/config/realworld_collect_data_pico.yaml`` and match ``env.eval.pico.zmq_addr`` to your publisher (use ``ipc:///tmp/vr_data.ipc`` on the same host, or ``tcp://<vr_publisher_ip>:<port>`` across hosts—never ``tcp://0.0.0.0:<port>``):

      .. code-block:: yaml

         env:
           eval:
             teleop: pico
             pico:
               zmq_addr: "tcp://<vr_publisher_ip>:<port>"
               position_scale: 1.0
               rotation_scale: 1.0
               calibration:
                 button: "trigger"

      **Startup order**

      1. Activate the ``franka`` environment (``source .venv/bin/activate``) on the controller node.
      2. For multi-node runs, start the Ray cluster as in :doc:`franka` (single-node runs auto-start Ray).
      3. Start the XRoboToolkit PC Service and verify the headset and controller are connected.
      4. Start ``vr_data_publisher``.
      5. On the first run or after changing the ZeroMQ address, verify with ``test_pico_data.py``.
      6. Launch collection on the Ray head node from the RLinf repository root:

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh realworld_collect_data_pico

      .. warning::

         Only one program can hold Franka control at a time. Do not run ``robo_avatar.slave.main``, another ``franka_control_node``, or another RLinf real-world job alongside collection.

      **VR controller mapping**

      Stand facing the front of the robot workspace, validate small motions with no task objects first, and use these controls:

      .. list-table::
         :header-rows: 1
         :widths: 34 66

         * - Input
           - Robot action
         * - Press ``trigger``
           - Calibrate the PICO base frame to your standing pose.
         * - Hold / release ``grip``
           - Start intervention (records controller and TCP reference poses) / stop intervention. Holding ``grip`` again sets a fresh reference pose.
         * - Move forward / backward
           - End-effector ``+X`` / ``-X``.
         * - Move left / right
           - End-effector ``+/-Y``.
         * - Move up / down
           - End-effector ``+/-Z``.
         * - Rotate controller
           - End-effector roll / pitch / yaw.
         * - Press ``A`` / ``B``
           - Close gripper / open gripper (when gripper control is enabled).

.. dropdown:: Franka motion settings

   The collection config uses Franky's default motion settings, including target-change limits of 3 cm and 0.10 rad per call, with no extra ``compliance`` block needed for PICO. Collection and policy execution share these defaults; see :ref:`Configure Arm Motion <franka-motion-settings>` for values, reset behavior, and hardware overrides.

   ``pico.position_scale`` and ``pico.rotation_scale`` scale controller input (including human interventions during training). The environment applies its action scale and workspace bounds before passing targets to Franky.

.. dropdown:: Gripper configuration

   When ``no_gripper: true``, PICO drives only 6D end-effector motion. To control the gripper from the controller, ensure the environment action space includes the gripper dimension and set:

   .. code-block:: yaml

      env:
        eval:
          no_gripper: False
          pico:
            gripper_close_button: "A"
            gripper_open_button: "B"
            gripper_close_threshold: 0.5

   ``A`` closes the gripper and ``B`` opens it. When neither button is pressed, the gripper command is ``0.0``; releasing a button does not open the gripper automatically.

.. dropdown:: Cluster setup notes and YAML reference

   - The node reading the controller must reach ``env.eval.pico.zmq_addr``.
   - For TCP, open the configured port in firewall and routing rules; for IPC, run the publisher and RLinf env worker on the same host.
   - Keep ``RLINF_NODE_RANK``, ``cluster.node_groups[*].node_ranks`` in YAML, and ``node_rank`` in the Franka hardware config consistent.
   - Start from ``examples/embodiment/config/realworld_collect_data_pico.yaml`` and update ``robot_ip``, ``camera_serials``, ``target_ee_pose``, and ``env.eval.pico.zmq_addr`` for your setup.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Symptom
     - Fix
   * - No VR data after startup
     - Verify XRoboToolkit PC Service (``runService.sh``) and ``vr_data_publisher`` are running, run ``test_pico_data.py``, check ``pico.zmq_addr`` matches the publisher, and verify TCP reachability from the env worker node.
   * - Robot does not move when holding ``grip``
     - Confirm ``teleop: pico`` in the env config, check ``grip`` exceeds ``control_threshold``, complete ``trigger`` calibration (required when ``calibration.required=True``), and check Franka safety box limits and ``action_scale``.
   * - Wrong direction or jump at activation
     - Release ``grip``, return to a comfortable stance, press ``trigger`` to recalibrate, and hold ``grip`` again for a fresh reference pose. If the whole frame is rotated, adjust ``operator_to_robot_yaw`` (for example ``1.5708``, ``3.1416``, or ``-1.5708``).
   * - Gripper does not respond
     - Confirm ``no_gripper: False``, verify the environment action space includes the seventh gripper dimension, and check that no other wrapper overrides gripper actions.
   * - ``FrankaHW: Failed to initialize libfranka robot. libfranka: UDP receive: Timeout``
     - The controller node failed to establish FCI UDP communication with the arm. Check ``robot_ip``, programmable mode in Franka Desk, controller network routing, and whether another process holds Franka control.
