XSquare Turtle2 Online RL
=========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg
   :align: center
   :width: 80%

   XSquare Turtle2 dual-arm robot used for SAC/CNN real-world button-pressing training.

Train a visual SAC/CNN button-pressing policy on the XSquare Turtle2 dual-arm robot across a vendor controller node and a GPU training node.

Overview
--------

This recipe pairs the XSquare ROS controller container on the robot host with an asynchronous SAC, Cross-Q, or RLPD training loop using a pretrained ResNet-10 backbone on the GPU node.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      XSquare Turtle2 · 1–2 arms

   .. grid-item-card:: Sensors & Teleop
      :text-align: center

      Up to 3 RGB cameras (IDs 0–2)

   .. grid-item-card:: Workflow
      :text-align: center

      Dummy check · SAC / Cross-Q / RLPD · CNN

   .. grid-item-card:: Hardware
      :text-align: center

      Controller PC (XSquare Docker) · GPU node

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Task
     - Config / Entry Point
     - Description
   * - Dummy check
     - ``realworld_dummy_turtle2_sac_cnn``
     - Validate config and cluster plumbing without hardware motion.
   * - Training
     - ``realworld_button_turtle2_sac_cnn``
     - Train button pressing with one or two active arms.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Up to three RGB camera streams plus TCP pose per active arm.
   * - Action
     - 6-D continuous delta pose per arm for button pressing (12-D dual arm, gripper fixed closed; the arm hardware supports 7-D with gripper).
   * - Reward
     - ``1.0`` on button-press success; optional dense exponential shaping.
   * - Prompt
     - ``Press the button with the end-effector.``

Hardware and Safety
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Requirement
   * - Robot & cameras
     - XSquare Turtle2 dual-arm robot with up to three mounted RGB cameras (IDs 0–2).
   * - Controller node
     - Small Linux computer (no GPU required) running the official XSquare Docker container on the same LAN.
   * - Training / Rollout node
     - Linux GPU server on the same local network running the CNN policy.

.. warning::

   Keep the controller node and training node on the same local network, keep the physical emergency stop within reach, and clear the workspace around the arms before disabling ``is_dummy``. Button pressing resets each active arm slightly above ``target_ee_pose`` along the Z axis; verify the recorded pose before starting motion.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Robot Controller Node

      **1. Enter the XSquare Docker container and install RLinf**

      Enter the official XSquare Docker container (contact `XSquare <https://x2robot.com>`__ for the image and startup commands) before installing RLinf:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env xsquare_turtle2 && source .venv/bin/activate

      **2. Record the target end-effector pose**

      Move the active arm(s) to the button-press pose through the XSquare control interface and read back the pose in XSquare's Euler convention ``[x, y, z, rz, ry, rx]``. Record both arms when running in dual-arm mode.

      **3. Start the Ray worker**

      Ray freezes the active Python interpreter and environment variables at ``ray start`` time. Activate ``.venv`` and export the variables below from the repo root (replace ``192.168.1.10`` with your head node IP and ``eth0`` if you have multiple NICs):

      .. code-block:: bash

         source .venv/bin/activate
         export RLINF_NODE_RANK=1
         export RLINF_COMM_NET_DEVICES=eth0  # Optional if only one NIC
         ray start --address='192.168.1.10:6379'

   .. tab-item:: Training & Rollout Nodes

      **1. Install RLinf and download ResNet-10**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env xsquare_turtle2 && source .venv/bin/activate
         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./RLinf-ResNet10-pretrained

      **2. Start the Ray head**

      Export the cluster variables from the repo root (replace ``192.168.1.10`` with your head node IP) and verify both nodes join with ``ray status``:

      .. code-block:: bash

         export RLINF_NODE_RANK=0
         export RLINF_COMM_NET_DEVICES=eth0  # Optional if only one NIC
         ray start --head --port=6379 --node-ip-address=192.168.1.10
         ray status

      **3. Configure arms, target pose, and backbone path**

      Update the required fields in ``examples/embodiment/config/realworld_button_turtle2_sac_cnn.yaml``:

      .. code-block:: yaml

         env:
           train:
             override_cfg:
               is_dummy: False
               use_arm_ids: [1]          # 0=left, 1=right; [0, 1] for dual arm
               use_camera_ids: [2]       # Camera IDs to use (0, 1, or 2)
               target_ee_pose:           # [[left], [right]], Euler [x, y, z, rz, ry, rx]
                 - [0, 0, 0, 0, 0, 0]
                 - [0.3, 0.0, 0.15, 0.0, 1.0, 0.0]

         actor:
           model:
             model_path: "./RLinf-ResNet10-pretrained"
             state_dim: 6    # 6 for single arm, 12 for dual arm
             action_dim: 6   # 6 for single arm, 12 for dual arm

      .. dropdown:: Full configuration reference (optional)

         .. code-block:: yaml

            cluster:
              num_nodes: 2  # 1 training/rollout node + 1 controller node
              component_placement:
                actor:
                  node_group: "gpu"
                  placement: 0
                rollout:
                  node_group: "gpu"
                  placement: 0
                env:
                  node_group: turtle2
                  placement: 0
              node_groups:
                - label: "gpu"
                  node_ranks: 0
                - label: turtle2
                  node_ranks: 1
                  hardware:
                    type: Turtle2
                    configs:
                      - node_rank: 1

            env:
              train:
                override_cfg:
                  is_dummy: False
                  use_arm_ids: [1]          # 0=left arm, 1=right arm; use [0,1] for dual arm
                  use_camera_ids: [2]       # camera IDs to use (0, 1, or 2)
                  target_ee_pose:           # [[left_arm_pose], [right_arm_pose]], Euler [x,y,z,rz,ry,rx]
                    - [0, 0, 0, 0, 0, 0]
                    - [0.3, 0.0, 0.15, 0.0, 1.0, 0.0]

            actor:
              model:
                model_path: "./RLinf-ResNet10-pretrained"
                state_dim: 6    # 6 for single arm (xyz+euler); 12 for dual arm
                action_dim: 6   # 6 for single arm (xyz_delta+rpy_delta); 12 for dual arm

      **4. Run dummy check and launch training**

      Validate the cluster and model pipeline without robot motion first (uses ``is_dummy: True``), then start real-world training on the head node:

      .. code-block:: bash

         # Pipeline check without hardware motion
         bash examples/embodiment/run_realworld_async.sh realworld_dummy_turtle2_sac_cnn

         # Real-world training
         bash examples/embodiment/run_realworld_async.sh realworld_button_turtle2_sac_cnn

Results and Troubleshooting
---------------------------

Monitor training on the Ray head node with ``tensorboard --logdir ./logs --port 6006`` and track ``env/success_once``, ``env/return``, ``train/sac/*``, and ``train/replay_buffer/*``; see :doc:`Training metrics <../../reference/metrics>` for all metrics.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - Fix
   * - Missing XSquare SDK or ROS packages on controller node
     - Run installation and ``ray start`` inside the official XSquare Docker container so SDK packages stay on the Python path.
   * - Ray workers miss ``RLINF_NODE_RANK`` or ``.venv``
     - Source ``.venv/bin/activate`` and export ``RLINF_NODE_RANK`` before running ``ray start``.
   * - State or action dimension mismatch in dual-arm mode
     - Set both ``actor.model.state_dim`` and ``actor.model.action_dim`` to ``12`` when ``use_arm_ids: [0, 1]``.

To build additional tasks on Turtle2, see :doc:`New Real-World Tasks <../../extending/new_task>`.
