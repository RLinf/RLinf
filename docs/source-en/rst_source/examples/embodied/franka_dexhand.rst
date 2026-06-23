Using Dexterous Hand with Franka
================================
.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dexhand.jpg
   :align: center
   :width: 80%

   Franka arm paired with a Ruiyan five-finger dexterous hand for real-world manipulation.

Adapt the Franka real-world workflow to a Ruiyan five-finger dexterous hand. You'll keep the same cluster and reward-model flow, then change the end effector, teleoperation input, action layout, and dex-hand configs.

Overview
--------

Run the Franka real-world recipe with a dexterous hand end effector.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      CNN policy · reward model

   .. grid-item-card:: Algorithms
      :text-align: center

      SAC/RLPD · reward-model assist

   .. grid-item-card:: Tasks
      :text-align: center

      Dexterous pick-and-place

   .. grid-item-card:: Hardware
      :text-align: center

      Franka · Ruiyan dex hand · glove

| **You'll do:** install Franka deps → install dex-hand deps → configure glove/hand → collect data → train.
| **Prerequisites:** :doc:`franka` · :doc:`franka_reward_model` · Ruiyan hand driver · glove device.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - Task
     - Config / entry point
     - Description
   * - Collection
     - ``realworld_collect_dexhand_data``
     - Collect dexterous-hand demonstrations.
   * - Training
     - ``realworld_dexpnp_rlpd_cnn_async``
     - Train a CNN policy with dex-hand actions.
   * - Reward model
     - Franka reward-model workflow
     - Reuse the reward-model path for dexterous manipulation.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - Field
     - Description
   * - Observation
     - Franka camera frames plus dex-hand/glove state when configured.
   * - Action
     - Arm action plus dexterous-hand joint or command vector.
   * - Reward
     - Task completion or reward-model prediction.
   * - Prompt
     - Task text from the real-world env config.

Installation
------------

Install the base Franka dependencies from :doc:`franka`, then install the dexterous-hand runtime in the workflow below.

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

Run It
------

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
