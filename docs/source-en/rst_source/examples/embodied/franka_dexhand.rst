Real-World RL with Franka + Dexterous Hand
==========================================

This page summarizes the configuration differences when the Franka arm uses a Ruiyan dexterous hand.
For the end-to-end real-world workflow, see :doc:`franka` and :doc:`franka_reward_model`.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The dexterous-hand setup keeps the same real-world RL and reward-model workflow as Franka.
The main differences are in the end-effector, teleoperation, and action space:

- The action space is 12-D.
- The first 6 dimensions control arm pose deltas.
- The last 6 dimensions control the dexterous hand.
- ``RuiyanHand`` handles the hand hardware.
- ``DexHandIntervention`` combines SpaceMouse input and glove input into expert actions.

Teleoperation
-------------

Dexterous-hand teleoperation uses:

- SpaceMouse for 6-D arm motion
- a data glove for 6-D finger control
- the SpaceMouse left button to enable relative glove control

Reward Model
------------

The reward-model path is the same as the Franka real-world reward-model workflow described in :doc:`franka_reward_model`.

For the dexterous-hand pick-and-place environment:

- the default reward image follows ``env.main_image_key``
- ``main_image_key`` defaults to ``wrist_1`` in ``examples/embodiment/config/env/realworld_dex_pnp.yaml``
- ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml`` uses the reward model through the ``reward`` section

Configurations
--------------

Use ``examples/embodiment/config/realworld_collect_dexhand_data.yaml`` for data collection.
This config includes:

- ``end_effector_type: "ruiyan_hand"``
- glove settings for teleoperation
- ``data_collection`` for raw episode export in ``pickle`` format

Use ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml`` for RL training.
Before running, fill in:

- ``robot_ip``
- ``target_ee_pose``
- policy ``model_path``
- reward ``model.model_path``
- dexterous-hand serial ports in ``end_effector_config`` and ``glove_config``

Camera naming and crop are configured directly in ``override_cfg`` when needed.
This PR does not ship any serial-specific defaults so that other projects are
not affected. Without ``camera_names``, default names follow the
``camera_serials`` list order: the first serial is ``wrist_1``, the second is
``wrist_2``. Serial numbers are not sorted. For example:

.. code-block:: yaml

   camera_names:
     "SERIAL1": global
     "SERIAL2": wrist_1
   camera_crop_regions:
     "SERIAL1": [0.4, 0.3, 0.85, 0.7]

If you rename a camera to ``global``, update ``main_image_key`` to ``global``
in the task YAML as well.

Workflow
--------

1. On the Franka control node, install the Franka DexHand environment:

   .. code-block:: bash

      bash requirements/install.sh embodied --env franka-dexhand

   This command installs the base Franka dependencies plus ``RLinf-dexterous-hands``, which includes the Ruiyan dexterous-hand and data-glove drivers.
2. Put the Franka robot into programming mode, manually move it to the task target pose, then run the script on the Franka control node to acquire the target end-effector pose:

   .. code-block:: bash

      python -m toolkits.realworld_check.test_franka_controller \
        --robot-ip <FRANKA_IP> \
        --end-effector-type ruiyan_hand \
        --hand-port /dev/ttyUSB0

   After the script starts, enter ``getpos_euler``, record the Euler-angle pose it prints, and fill that value into ``target_ee_pose``.
3. On the Franka control node, fill in the collection-time task configuration: ``robot_ip``, ``target_ee_pose``, ``end_effector_config``, ``glove_config``, and related settings.
4. On the Franka control node, collect expert demos with:

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh realworld_collect_dexhand_data

5. On the Franka control node, collect reward raw episodes with the same entrypoint. For this pass, increase ``env.eval.override_cfg.success_hold_steps`` and use a separate log directory.
6. Copy the collected reward raw data from the Franka control node to the training node, or place it on shared storage in advance.
7. On the training node, preprocess the raw reward episodes with ``examples/reward/preprocess_reward_dataset.py`` as described in :doc:`franka_reward_model`.
8. On the training node, train the reward model with ``examples/reward/run_reward_training.sh``.
9. Before the final RL run, follow the cluster setup section in :doc:`franka` to start a two-node Ray cluster with the training node as head and the Franka control node as worker.
10. On the training node, launch RL with:

   .. code-block:: bash

      bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async
