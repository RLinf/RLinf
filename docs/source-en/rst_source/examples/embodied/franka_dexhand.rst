Franka Dexterous Hand
=====================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dexhand.jpg
   :align: center
   :width: 80%

   Franka arm paired with a Ruiyan five-finger dexterous hand for real-world manipulation.

Overview
--------

Adapt the Franka real-world RLPD and reward-model workflow to a Ruiyan five-finger dexterous hand. You keep the same two-node cluster and reward-model pipeline while switching the end effector, SpaceMouse + data-glove teleoperation input, action layout, and dexterous-hand task configs.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      Franka Emika Panda + Ruiyan 5-Finger Hand

   .. grid-item-card:: Sensors & Teleop
      :text-align: center

      RealSense · SpaceMouse · Data Glove

   .. grid-item-card:: Workflow
      :text-align: center

      Collect → Reward Model → Online RLPD

   .. grid-item-card:: Hardware
      :text-align: center

      Controller Node + NVIDIA GPU Node

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - Stage
     - Config
     - Description
   * - Collection
     - ``realworld_collect_dexhand_data``
     - Collect dexterous pick-and-place demonstrations with SpaceMouse and data glove.
   * - Reward model
     - :doc:`franka_reward_model`
     - Preprocess raw episodes and train a ResNet reward model for dexterous manipulation.
   * - Training
     - ``realworld_dexpnp_rlpd_cnn_async``
     - Train a CNN policy asynchronously with arm + dexterous-hand actions and learned rewards.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Franka camera frames (``wrist_1`` by default) plus dexterous-hand and glove state when configured.
   * - Action
     - 6-D Franka arm Cartesian motion plus 6-D Ruiyan dexterous-hand finger commands.
   * - Reward
     - Learned reward-model prediction (using ``env.main_image_key``) or target-pose completion.
   * - Prompt
     - Task instruction defined in ``examples/embodiment/config/env/realworld_dex_pnp.yaml``.

Hardware and Safety
-------------------

You need a Franka arm fitted with a Ruiyan five-finger dexterous hand (connected over USB serial, such as ``/dev/ttyUSB0``), a compatible data glove, a SpaceMouse, Intel RealSense camera(s), a Franka controller node, and an NVIDIA GPU training node. Complete the base Franka firmware check, real-time kernel setup, and FCI activation in :doc:`franka`, and review the reward-model pipeline in :doc:`franka_reward_model`.

.. warning::

   Keep the emergency stop within reach and supervise every hardware run. Clear the workspace, confirm your account has read/write permission to the hand and glove serial ports (``/dev/ttyUSB*``), and verify finger-closure limits before starting teleoperation or autonomous RL.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Franka + Dexterous Hand

      **1. Install**

      Install the ``franka`` environment on the Franka control node (and GPU training node). It automatically includes ``RLinf-dexterous-hands`` for the Ruiyan hand and data-glove drivers:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. Teleoperation and target pose setup**

      During teleoperation, the SpaceMouse controls 6-D arm motion, the data glove controls 6-D finger articulation, and holding the SpaceMouse left button enables relative glove control.

      Put the Franka arm into programming mode, move it manually to the target pose, unlock FCI in Franka Desk, and read the target pose (enter ``getpos_euler`` at the prompt):

      .. code-block:: bash

         python -m toolkits.realworld_check.test_franka_controller \
           --robot-ip 192.168.1.10 \
           --end-effector-type ruiyan_hand \
           --hand-port /dev/ttyUSB0

      Fill ``robot_ip``, ``target_ee_pose``, ``end_effector_config``, and ``glove_config`` (serial ports) in ``examples/embodiment/config/realworld_collect_dexhand_data.yaml`` and ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml``, plus policy ``model_path`` and reward ``model.model_path`` in the RL config.

      **3. Collect data, train reward model, and launch RL**

      On the Franka control node, collect expert demonstrations first, then collect reward raw episodes with a larger ``env.eval.override_cfg.success_hold_steps`` in a separate log directory:

      .. code-block:: bash

         # Expert demos
         bash examples/embodiment/collect_data.sh realworld_collect_dexhand_data

         # Reward raw episodes (increase success_hold_steps and use a separate RLINF_LOG_DIR)
         RLINF_LOG_DIR="$PWD/logs/dexhand-reward-raw" \
           bash examples/embodiment/collect_data.sh realworld_collect_dexhand_data \
           "env.eval.override_cfg.success_hold_steps=20"

      Copy the reward raw episodes to the training node (or use shared storage), preprocess with ``examples/reward/preprocess_reward_dataset.py``, and train the reward model with ``examples/reward/run_reward_training.sh`` per :doc:`franka_reward_model`.

      ``realworld_dexpnp_rlpd_cnn_async.yaml`` defaults to **2 nodes** (``cluster.num_nodes: 2``: GPU training node as head, Franka control node as worker). Start the 2-node Ray cluster per :doc:`franka`, then launch async RLPD on the training node:

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async

      .. dropdown:: Advanced configuration (optional)

         **Shipped config files**

         .. list-table::
            :header-rows: 1
            :widths: 52 48

            * - Config
              - Purpose
            * - ``examples/embodiment/config/realworld_collect_dexhand_data.yaml``
              - Collection with ``end_effector_type: "ruiyan_hand"``, glove teleoperation, and export to ``replay_buffer`` (``demo_buffer/``) and ``pickle_episode`` (``pickle_episode/``).
            * - ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml``
              - Async RLPD CNN training with the ``reward`` section enabled.
            * - ``examples/embodiment/config/env/realworld_dex_pnp.yaml``
              - Dexterous pick-and-place env defaults (``env.main_image_key`` defaults to ``wrist_1``).

         **Camera naming and crop regions**

         Without ``camera_names``, cameras follow the ``camera_serials`` list order (``wrist_1``, then ``wrist_2``; serials are not sorted). To rename or crop views in ``override_cfg``:

         .. code-block:: yaml

            camera_names:
              "SERIAL1": global
              "SERIAL2": wrist_1
            camera_crop_regions:
              "SERIAL1": [0.4, 0.3, 0.85, 0.7]

         If you rename the primary view to ``global``, update ``main_image_key`` to ``global`` in the task YAML so the reward model reads the same view.

Results and Troubleshooting
---------------------------

Monitor ``env/success_once``, reward-model predictions, and SAC losses in TensorBoard (``tensorboard --logdir ./logs``); see :doc:`Training metrics <../../reference/metrics>` for logging options.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - Fix
   * - Permission denied opening ``/dev/ttyUSB0``
     - Add your user to the ``dialout`` group (``sudo usermod -aG dialout "$USER"``) and re-login, or verify the port path in ``end_effector_config`` / ``glove_config``.
   * - Dexterous hand fingers do not follow the data glove
     - Hold the SpaceMouse left button to engage relative glove control and confirm the glove serial port matches ``glove_config``.
   * - Reward model evaluates the wrong camera view
     - Ensure ``env.main_image_key`` matches your camera name (update it to ``global`` if you renamed a camera in ``camera_names``).
