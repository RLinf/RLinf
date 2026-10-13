Collect Dual-Arm YAM Demonstrations with PICO
=============================================

Collect dual-arm YAM demonstrations with PICO controllers and three RealSense RGB cameras. The recipe records absolute joint position and action vectors in the order ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``; each vector has 14 values, and grippers use ``0=closed, 1=open``.

Prepare the station
-------------------

Use two YAM follower arms with ``flexible_4310`` grippers, three RealSense cameras, a PICO headset with two controllers, and a ZeroMQ PICO publisher. Follow :doc:`the Franka VR setup <franka_vr>` to prepare XRoboToolkit, its PC Service, and the publisher. Install the YAM runtime in a Python environment:

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

The installer includes the pinned ``i2rt`` backend and PICO transport dependencies. Check that the installed robot model, tool center point, joint limits, controller-to-base yaw, and motion scales match your hardware before enabling motion.

.. warning::

   The collector opens both follower CAN devices and all three cameras on its first reset. Keep the arms supported in a verified safe pose, clear the motion area, and have an operator at the emergency stop. Automatic reset and parking are disabled in this recipe. Before ending a session, the operator must confirm that both arms can safely remain supported when motor output is released.

Configure and collect
---------------------

Set the two follower CAN interfaces, the three camera serial numbers in top, left, right order, the PICO publisher address, and the measured operator-to-robot yaw for each arm. Export these values on the YAM node **before** starting Ray, so hardware enumeration and the environment worker receive them:

.. code-block:: bash

   export LEFT_CAN_CHANNEL='<left-can-interface>'
   export RIGHT_CAN_CHANNEL='<right-can-interface>'
   export CAMERA_SERIALS='<top-serial>,<left-serial>,<right-serial>'
   export YAM_PICO_ZMQ_ADDR='tcp://<publisher-host>:5555'
   export YAM_LEFT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'
   export YAM_RIGHT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'

The first three fill the unset fields of the ``DualYam`` hardware config, which can also be written in the YAML as ``left_can_channel``, ``right_can_channel``, and ``camera_serials``. Replace every placeholder with a value checked at your station. The example configuration is ``examples/embodiment/config/realworld_dual_yam_collect_data_pico.yaml``. It assigns the environment worker and hardware to one Ray node. On a fresh single-node host, start Ray and launch collection:

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head
   bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data_pico

Hold a controller's grip to control its corresponding arm; release it to stop taking over. The recipe gives ``pico`` one entry per arm, and each turns its controller's pose into joint targets through the YAM model's inverse kinematics, set in ``override_cfg.ik``. If either controller drops out, both arms hold until both grips are released. The ``yam_episode`` block maps the right menu button to start/end an episode and the left menu button to discard the current recording. Confirm these button events on your PICO setup before collecting task data. Discard removes only the current unfinished episode; faulted or truncated recordings can still be saved as unsuccessful episodes.

The launcher writes logs under ``logs/<timestamp>-realworld_dual_yam_collect_data_pico/`` by default. This recipe sets ``export_format: lerobot_dataset``, so it does not write ``demo_buffer/``. Each recorded take is streamed into its own shard under that run's ``lerobot_dataset/rank_0/``. Unsuccessful takes are saved too, with ``is_success`` false. Merge the shards into one dataset for OpenPI SFT or HG-DAgger, leaving those out:

.. code-block:: bash

   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir logs/<timestamp>-realworld_dual_yam_collect_data_pico/lerobot_dataset \
       --output-dir logs/yam_merged \
       --only-success

Frames are stored as uncompressed PNGs, about 0.9 MB per 640x480 view, so a one-minute take uses about 5 GB. Inspect the three RGB views before training; an ``is_success`` flag does not replace the operator's task-result check. See :doc:`../../guides/data_collection` for how the three formats are used.
