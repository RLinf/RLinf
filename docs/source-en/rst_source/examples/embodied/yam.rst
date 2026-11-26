YAM PICO Data Collection
========================

Collect dual-arm YAM joint-space demonstrations with PICO controllers and three RealSense RGB cameras on one YAM host, then merge successful shards into a LeRobot dataset ready for :doc:`yam_openpi`.

Overview
--------

This recipe teleoperates two YAM follower arms (with ``flexible_4310`` grippers) through per-arm inverse kinematics driven by PICO controllers and exports uncompressed LeRobot shards for downstream OpenPI SFT or HG-DAgger.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      2× YAM · ``flexible_4310`` grippers

   .. grid-item-card:: Teleop & Cameras
      :text-align: center

      PICO dual controllers · 3× RealSense RGB

   .. grid-item-card:: Workflow
      :text-align: center

      PICO IK Collect → Filter & Merge LeRobot

   .. grid-item-card:: Hardware
      :text-align: center

      1× YAM host · 2× CAN (1 Mbit/s) · PICO publisher

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Description
   * - Observation
     - Top, left-wrist, and right-wrist RealSense RGB views plus 14-D absolute joint positions in ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]`` order (radians; grippers ``0=closed, 1=open``).
   * - Action
     - 14-D absolute joint and gripper targets in the same order, computed per arm via YAM inverse kinematics (``override_cfg.ik``).
   * - Episode controls
     - Hold left/right ``grip`` to take over the matching arm; press right menu button to start/end an episode and left menu button to discard the active take (``yam_episode``).

Hardware and Safety
-------------------

Connect two YAM follower arms (``flexible_4310`` grippers) over two 1 Mbit/s CAN interfaces (`i2rt CAN setup <https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/README.md#can-bus-setup>`_) and three RealSense RGB cameras (top, left wrist, right wrist) matched by physical chassis labels. Prepare XRoboToolkit, PC Service, and the ZeroMQ publisher by following :ref:`the PICO software setup <pico-software-setup>` and :ref:`the PICO publisher setup <pico-publisher-setup>`. Release any other process holding CAN interfaces or cameras before collection.

.. warning::

   The collector opens both follower CAN devices and all three cameras on its first reset. Keep both arms supported in a verified safe pose, clear the motion area, and keep an operator at the emergency stop. Automatic reset and parking are disabled in this recipe—before ending a session, confirm both arms remain safely supported when motor output is released.

.. dropdown:: Hardware calibration details (optional)

   Verify the installed robot model, tool center point, joint limits, motion scaling, and operator-to-robot yaw before enabling motion:

   * **Joint limits and TCP site:** open the arm and gripper XML paths printed by ``i2rt`` below and compare ``joint1`` through ``joint6`` ``range`` values (radians) against both arms' ``joint_limit_min`` / ``joint_limit_max`` in ``examples/embodiment/config/env/realworld_dual_yam_joint.yaml`` (task limits must stay inside model and hardware limits). Confirm the gripper XML ``grasp_site`` matches the installed tool and ``env.eval.override_cfg.ik.site_name``.
   * **Scaling and per-step delta:** ``env.eval.pico.position_scale`` and ``rotation_scale`` default to ``0.3`` (1 cm controller motion requests 3 mm TCP displacement). ``env.eval.override_cfg.max_joint_delta`` caps each step's joint target change relative to current feedback in radians.
   * **Publisher address:** run ``ip -brief address`` on the publisher PC for a YAM-reachable IP and take the port from ``zmq.ipc_addr`` in ``configs/vr_bridge.yaml``. Set ``YAM_PICO_ZMQ_ADDR`` to ``tcp://<publisher-PC-address>:<port>`` (never the headset IP or ``0.0.0.0``; a matching IPC path works on the same host).
   * **Per-arm operator-to-robot yaw:** measure left and right arms separately. At calibration, the headset's horizontal facing direction is operator ``+X`` (``+Y`` left, ``+Z`` up). Express operator forward as ``(dx, dy)`` in each arm base's horizontal plane and set yaw to ``atan2(dy, dx)`` in radians (positive follows the right-hand rule around base ``+Z``; multiply degrees by ``π/180``; use ``0`` only when horizontal axes align; yaw cannot compensate for a tilted base). Face the measured direction with both grips released at startup, and remeasure if your operating direction changes.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Dual-Arm YAM + PICO

      **1. Install the YAM runtime** (includes the pinned ``i2rt`` backend and PICO transport dependencies):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env yam && source .venv/bin/activate

      **2. Check CAN interfaces, RealSense serials, and model XML paths** (read-only queries; motors stay disabled):

      .. code-block:: bash

         ip -details link show type can
         python -c 'import pyrealsense2 as rs; print([device.get_info(rs.camera_info.serial_number) for device in rs.context().devices])'
         python - <<'PY'
         from i2rt.robots.utils import ArmType, GripperType

         print(ArmType.YAM.get_xml_path(), GripperType.FLEXIBLE_4310.get_xml_path(), sep="\n")
         PY

      Confirm each CAN interface is ``UP`` at 1 Mbit/s with no ``BUS-OFF`` errors, identify left/right CAN from physical wiring, and match camera serials to physical top/left/right labels (see the calibration dropdown under Hardware and Safety for XML and yaw checks).

      **3. Export station variables, verify PICO, and launch collection** (export all variables **before** launching collection so workers inherit them; you can alternatively set ``left_can_channel``, ``right_can_channel``, and ``camera_serials`` directly in ``examples/embodiment/config/realworld_dual_yam_collect_data_pico.yaml``):

      .. code-block:: bash

         export LEFT_CAN_CHANNEL='can0'                      # replace with left arm CAN interface
         export RIGHT_CAN_CHANNEL='can1'                     # replace with right arm CAN interface
         export CAMERA_SERIALS='12345678,23456789,34567890'  # replace with top,left,right serials
         export YAM_PICO_ZMQ_ADDR='tcp://192.168.1.100:5555' # replace with publisher IP:port
         export YAM_LEFT_OPERATOR_TO_ROBOT_YAW='0.0'         # replace with measured yaw (rad)
         export YAM_RIGHT_OPERATOR_TO_ROBOT_YAW='0.0'        # replace with measured yaw (rad)

         # Verify live headset, controller, and button updates (Ctrl-C to exit)
         python -m toolkits.realworld_check.test_pico_data --zmq-addr "$YAM_PICO_ZMQ_ADDR"

         # Launch collection
         bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data_pico

      On your first takeover, test one arm at a time with small motions to verify direction and scaling. If direction is wrong, release both grips and stop collection before editing yaw. If either controller drops out, both arms hold until both grips are released. After any fault stop, confirm the robot state on site before resuming.

      **4. Merge successful shards** into one dataset (unsuccessful or truncated takes are also written with ``is_success: false``; replace ``20261121-120000`` with your run timestamp):

      .. code-block:: bash

         python -m toolkits.lerobot.merge_lerobot_datasets \
             --source-dir "logs/20261121-120000-realworld_dual_yam_collect_data_pico/lerobot_dataset" \
             --output-dir logs/yam_merged \
             --only-success  # replace timestamp with your run directory

Results and Troubleshooting
---------------------------

Each run writes LeRobot shards under ``logs/<timestamp>-realworld_dual_yam_collect_data_pico/lerobot_dataset/rank_0/`` as uncompressed PNGs (~0.9 MB per 640×480 frame, ~5 GB per minute across three cameras). Inspect all three RGB views before training—``is_success`` does not replace visual verification. See :doc:`../../guides/data_collection` for export format options and :doc:`yam_openpi` for Pi0.5 SFT and evaluation.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Arm moves in the wrong direction on takeover
     - Release both grips, stop collection, remeasure ``YAM_LEFT_OPERATOR_TO_ROBOT_YAW`` / ``YAM_RIGHT_OPERATOR_TO_ROBOT_YAW`` using ``atan2(dy, dx)``, re-export, and relaunch.
   * - CAN interface shows ``BUS-OFF`` or fails to open
     - Release any other process using CAN, re-apply 1 Mbit/s configuration per the ``i2rt`` CAN guide, and verify ``UP`` state with ``ip -details link show type can``.
   * - ``test_pico_data.py`` receives no updates
     - Verify XRoboToolkit PC Service and ``vr_data_publisher`` are running and set ``YAM_PICO_ZMQ_ADDR`` to a reachable ``tcp://<publisher-ip>:<port>`` (not the headset IP or ``0.0.0.0``).
   * - Left/right camera views appear swapped
     - Match serials to physical camera chassis labels rather than enumeration order and update ``CAMERA_SERIALS`` in ``top,left,right`` order before launching.
   * - Unfinished or faulted takes appear in raw shards
     - Pass ``--only-success`` to ``merge_lerobot_datasets.py`` and inspect the merged RGB videos before SFT.
