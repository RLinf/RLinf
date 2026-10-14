Collect Dual-Arm YAM Demonstrations with PICO
=============================================

Collect dual-arm YAM demonstrations with PICO controllers and three RealSense RGB cameras. The recipe records absolute joint position and action vectors in the order ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``; each vector has 14 values, and grippers use ``0=closed, 1=open``.

Prepare the station
-------------------

Use two YAM follower arms with ``flexible_4310`` grippers, three RealSense cameras, a PICO headset with two controllers, and a ZeroMQ PICO publisher. Follow :ref:`the PICO software setup <pico-software-setup>` to prepare XRoboToolkit, its PC Service, and the publisher. Install the YAM runtime in a Python environment:

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

The installer includes the pinned ``i2rt`` backend and PICO transport dependencies. Check that the installed robot model, tool center point, joint limits, operator-to-robot-base yaw, and motion scales match your hardware before enabling motion.

.. warning::

   The collector opens both follower CAN devices and all three cameras on its first reset. Keep the arms supported in a verified safe pose, clear the motion area, and have an operator at the emergency stop. Automatic reset and parking are disabled in this recipe. Before ending a session, the operator must confirm that both arms can safely remain supported when motor output is released.

Check the hardware
------------------

Before starting Ray, run these queries at the repository root on the YAM node with the YAM environment activated. They list CAN interfaces, camera serial numbers, and installed model files without enabling motors or starting camera streams:

.. code-block:: bash

   ip -details link show type can
   python -c 'import pyrealsense2 as rs; print([device.get_info(rs.camera_info.serial_number) for device in rs.context().devices])'
   python - <<'PY'
   from i2rt.robots.utils import ArmType, GripperType

   print(ArmType.YAM.get_xml_path(), GripperType.FLEXIBLE_4310.get_xml_path(), sep="\n")
   PY

Identify the left and right CAN interfaces from the physical wiring. Check the ``UP`` flag, the SDK's required 1 Mbit/s bitrate, and the absence of faults such as ``BUS-OFF``; see the `i2rt CAN setup <https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/README.md#can-bus-setup>`_. Match camera serial numbers to the labels on the top, left-wrist, and right-wrist cameras rather than enumeration order, and release CAN devices and cameras held by other tasks before collection.

Confirm that the hardware is YAM with ``flexible_4310`` grippers. Open the two XML files printed above and compare the ``range`` values for the arm's ``joint1`` through ``joint6`` with both arms' ``joint_limit_min/max`` in ``examples/embodiment/config/env/realworld_dual_yam_joint.yaml``. Limits are in radians; task limits must stay within the model and hardware limits. The gripper XML defines the TCP as ``grasp_site``. Check its position and orientation against the installed tool and the collection config's ``env.eval.override_cfg.ik.site_name``; validate the model before using a different gripper or tool.

The collection config's ``env.eval.pico.position_scale`` and ``rotation_scale`` scale controller translation and rotation into requested targets. Both are ``0.3`` in the example, so a 1 cm controller movement requests a 3 mm TCP displacement. ``env.eval.override_cfg.max_joint_delta`` limits each step's joint target relative to current feedback, in radians, rather than the total travel over a session.

Obtain the publisher address using :ref:`the PICO publisher setup <pico-publisher-setup>`: run ``ip -brief address`` on the publisher PC to find an address reachable from the YAM node, and take the port from ``zmq.ipc_addr`` in ``configs/vr_bridge.yaml``. Set the consumer address to ``tcp://<publisher-PC-address>:<port>``, not the headset IP or ``0.0.0.0``; on the same host, a matching IPC address also works.

Measure yaw separately for each arm. During calibration, the headset's horizontal facing direction becomes operator +X, with +Y to the left and +Z up. Use the model and manufacturer's coordinate diagrams to express operator forward as ``(dx, dy)`` in each robot base's horizontal plane, then set yaw to ``atan2(dy, dx)`` in radians. Positive yaw follows the right-hand rule around base +Z; multiply degrees by π/180 to convert them. Use 0 only when the horizontal axes align; yaw cannot compensate for a tilted base. Face the measured direction and release both grips at startup, and remeasure and recalibrate if the operating direction changes.

Configure and collect
---------------------

Set the two follower CAN interfaces, the three camera serial numbers in top, left, right order, the PICO publisher address, and the measured operator-to-robot yaw for each arm. Export these values on the YAM node **before** starting Ray, so hardware enumeration and the environment worker receive them:

.. code-block:: bash

   export LEFT_CAN_CHANNEL='<left-can-interface>'
   export RIGHT_CAN_CHANNEL='<right-can-interface>'
   export CAMERA_SERIALS='<top-serial>,<left-serial>,<right-serial>'
   export YAM_PICO_ZMQ_ADDR='tcp://<publisher-host>:<publisher-port>'
   export YAM_LEFT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'
   export YAM_RIGHT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'

The first three fill the unset fields of the ``DualYam`` hardware config, which can also be written in the YAML as ``left_can_channel``, ``right_can_channel``, and ``camera_serials``. Replace every placeholder with a value checked at your station. The example configuration is ``examples/embodiment/config/realworld_dual_yam_collect_data_pico.yaml``. It assigns the environment worker and hardware to one Ray node.

Before launching collection, check the PICO data stream on the YAM node:

.. code-block:: bash

   python toolkits/realworld_check/test_pico_data.py \
       --zmq-addr "$YAM_PICO_ZMQ_ADDR"

Confirm that the headset pose, both controller poses, and button values change as you operate them, then press Ctrl-C to exit the check. On a fresh single-node host, start Ray and launch collection:

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head
   bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data_pico

On the first takeover, move one arm at a time through small motions to check direction and scaling. If the direction is wrong, release both grips and stop collection before correcting the config. After a fault stop, have the on-site operator confirm the robot state before resuming.

Hold a controller's grip to control its corresponding arm; release it to stop taking over. The recipe gives ``pico`` one entry per arm, and each turns its controller's pose into joint targets through the YAM model's inverse kinematics, set in ``override_cfg.ik``. If either controller drops out, both arms hold until both grips are released. The ``yam_episode`` block maps the right menu button to start/end an episode and the left menu button to discard the current recording. Confirm these button events on your PICO setup before collecting task data. Discard removes only the current unfinished episode; faulted or truncated recordings can still be saved as unsuccessful episodes.

The launcher writes logs under ``logs/<timestamp>-realworld_dual_yam_collect_data_pico/`` by default. This recipe sets ``export_format: lerobot_dataset``. Each recorded take is streamed into its own shard under that run's ``lerobot_dataset/rank_0/``. Unsuccessful takes are saved too, with ``is_success`` false. Merge the shards into one dataset for OpenPI SFT or HG-DAgger, leaving those out. Replace ``<timestamp>`` with the timestamp in your run's log directory:

.. code-block:: bash

   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir "logs/<timestamp>-realworld_dual_yam_collect_data_pico/lerobot_dataset" \
       --output-dir logs/yam_merged \
       --only-success

Frames are stored as uncompressed PNGs, about 0.9 MB per 640x480 view, so a one-minute take uses about 5 GB. Inspect the three RGB views before training; an ``is_success`` flag does not replace the operator's task-result check. See :doc:`../../guides/data_collection` for how the three formats are used.
