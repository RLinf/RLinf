Real-World RL with Franka + Dexterous Hand
=============================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document describes how to set up **dexterous hand end-effectors**
(Aoyi Hand, Ruiyan Hand) on a Franka arm within the RLinf framework,
use **data glove + SpaceMouse** teleoperation for data collection and
human intervention during training, and train a **visual reward classifier**
for automated success/failure judgment in dexterous manipulation tasks.

Please read :doc:`franka` first if you have not yet set up the basic
Franka real-world environment.

.. contents:: Contents
   :local:
   :depth: 2

Overview
-----------

In the default Franka setup, the end-effector is a parallel gripper with
a 7-dimensional action space (6 arm + 1 gripper). With dexterous hand
integration, the action space expands to **12 dimensions**
(6 arm + 6 finger), enabling more complex manipulation tasks.

**Key Features:**

1. **End-effector abstraction layer** — A unified ``EndEffector`` interface
   that allows switching between Franka gripper, Aoyi Hand, and Ruiyan Hand
   via a single configuration field.
2. **Glove teleoperation** — ``GloveExpert`` reads 6-DOF finger angles
   from a PSI data glove, combined with ``SpaceMouseExpert`` to form
   12-dimensional expert actions.
3. **Dexterous hand intervention wrapper** — ``DexHandIntervention``
   automatically replaces ``SpacemouseIntervention`` and provides full
   12-dimensional expert actions during human intervention.
4. **Visual reward classifier** — For dexterous hand tasks where
   end-effector position alone cannot determine success or failure,
   a ResNet-10 based binary classifier judges task completion from
   camera images.

Environment
-----------

- **Task**: Dexterous manipulation tasks (e.g., grasping, fine assembly)
- **Observation**: Wrist or third-person camera RGB images (128×128) +
  6-dimensional hand state
- **Action Space**: 12-dimensional continuous actions:

  - 3D position control (x, y, z)
  - 3D rotation control (roll, pitch, yaw)
  - 6D finger control (thumb rotation, thumb bend, index, middle, ring,
    pinky), normalized ``[0, 1]``

Algorithm
-----------

The dexterous hand setup uses the same algorithm stack as the Franka
gripper (SAC / Cross-Q / RLPD). The difference is that the policy
outputs 12-dimensional actions, and a visual classifier can optionally
provide the reward signal.
See :doc:`franka` for algorithm details.


Hardware Setup
----------------

In addition to the standard hardware listed in :doc:`franka`, the
dexterous hand setup requires:

- **Dexterous hand** — Aoyi Hand (Modbus RTU serial) or Ruiyan Hand
  (custom serial protocol)
- **Data glove** — PSI data glove, USB serial connection (typically
  mounted as ``/dev/ttyACM0``)

Controller Node Connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following hardware should be connected to the controller node:

1. **Franka arm** — Ethernet
2. **Dexterous hand** — USB serial (Aoyi: Modbus, Ruiyan: custom protocol)
3. **SpaceMouse** — USB
4. **Data glove** — USB serial
5. **RealSense camera** — USB

**Serial port permissions:**

.. code-block:: bash

   # Add user to the dialout group for serial access
   sudo usermod -a -G dialout $USER
   # Re-login for the change to take effect

   # Or temporarily change permissions
   sudo chmod 666 /dev/ttyUSB0  # dexterous hand
   sudo chmod 666 /dev/ttyACM0  # data glove

**Check device connections:**

.. code-block:: bash

   # List serial devices
   ls -la /dev/ttyUSB* /dev/ttyACM*

   # Check SpaceMouse (HID device)
   lsusb | grep -i 3dconnexion


Dependency Installation
-------------------------

The dexterous hand setup builds on the standard installation from
:doc:`franka`, with an additional driver package for the dexterous hands
and data glove.

Controller Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After completing the base installation from :doc:`franka`, install the
following in the virtual environment on the controller node:

.. code-block:: bash

   # Dexterous hand + data glove drivers (includes all serial deps)
   pip install "RLinf-dexterous-hands[all]"

``RLinf-dexterous-hands`` bundles drivers for the Ruiyan hand, Aoyi hand, and
PSI data glove, along with the required serial libraries (pyserial,
pymodbus, pyyaml, etc.). For finer-grained control over optional
dependencies:

- ``pip install RLinf-dexterous-hands`` — base only (pyserial + numpy)
- ``pip install "RLinf-dexterous-hands[glove]"`` — adds data glove deps (pyyaml)
- ``pip install "RLinf-dexterous-hands[aoyi]"`` — adds Aoyi hand deps (pymodbus)
- ``pip install "RLinf-dexterous-hands[all]"`` — all dependencies

Training / Rollout Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same as :doc:`franka` — no additional dependencies required.


Model Download
---------------

The dexterous hand setup uses the same pretrained ResNet-10 backbone as
:doc:`franka` for the policy's visual encoder:

.. code-block:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # Method 2: Using huggingface-hub
   # For mainland China users:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

After downloading, make sure to correctly specify the model path in the
configuration YAML file.

.. note::

   Pretrained models for dexterous hand tasks are still being trained and
   will be published on |huggingface| `HuggingFace <https://huggingface.co/RLinf>`_ later.
   For now you can train from scratch using the ResNet-10 backbone above.


Running the Experiment
-----------------------

Prerequisites
~~~~~~~~~~~~~~~

**1. Get the Target Pose**

Use the diagnostic tool to get the target end-effector pose and verify
dexterous hand status.

Set environment variables and run the diagnostic script:

.. code-block:: bash

   export FRANKA_ROBOT_IP=<your_robot_ip>
   export FRANKA_END_EFFECTOR_TYPE=ruiyan_hand  # or aoyi_hand
   export FRANKA_HAND_PORT=/dev/ttyUSB0
   python -m toolkits.realworld_check.test_franka_controller

In the interactive prompt:

- Enter ``getpos_euler`` to get the current end-effector pose (Euler angles)
- Enter ``gethand`` to view the current finger positions
- Enter ``handinfo`` to verify the hand connection
- Enter ``help`` for all available commands

**2. Test Hardware Connections**

.. code-block:: bash

   # Test the camera
   python -m toolkits.realworld_check.test_franka_camera

Data Collection
~~~~~~~~~~~~~~~~~

For data collection with a dexterous hand, the SpaceMouse controls the
arm and the data glove controls the fingers.
``DexHandIntervention`` automatically merges both inputs into
12-dimensional actions.

**Teleoperation controls:**

.. list-table::
   :widths: 18 30 52
   :header-rows: 1

   * - Input Device
     - Button / Action
     - Effect
   * - SpaceMouse
     - 6D joystick
     - Control arm pose delta (x, y, z, roll, pitch, yaw)
   * - SpaceMouse
     - **Left button** (hold)
     - Activate relative glove control: captures the glove baseline at
       press instant, then applies only deltas to fingers
   * - SpaceMouse
     - **Left button** (release)
     - Freeze fingers at their current position
   * - Data Glove
     - Finger bending
     - (Only active while left button is held) Finger angle deltas are
       added to the current hand pose

.. note::

   The data glove works in **relative control mode**: it only applies
   glove reading **deltas** while the SpaceMouse left button is held.
   When released, fingers stay in place. This avoids sudden finger jumps
   that occur in absolute mode and is better suited for fine manipulation.

1. Activate the virtual environment:

.. code-block:: bash

   source /opt/venv/franka-0.15.0/bin/activate
   # If using ROS: source <your_catkin_ws>/devel/setup.bash

2. Edit the configuration file ``examples/embodiment/config/realworld_collect_data.yaml``:

.. code-block:: yaml

   # examples/embodiment/config/realworld_collect_data.yaml
   defaults:
     - env/realworld_dex_pnp@env.eval      # reference env sub-config
     - override hydra/job_logging: stdout

   cluster:
     num_nodes: 1
     component_placement:
       env:
         node_group: franka
         placement: 0
     node_groups:
       - label: franka
         node_ranks: 0
         hardware:
           type: Franka
           configs:
             - robot_ip: 172.16.0.2          # ← replace with your robot IP
               node_rank: 0

   runner:
     task_type: embodied
     num_data_episodes: 20

   env:
     group_name: "EnvGroup"
     eval:
       ignore_terminations: False            # False for regular collection
       auto_reset: False                     # manual reset control
       use_spacemouse: True
       glove_config:
         left_port: "/dev/ttyACM0"           # data glove serial port
         frequency: 30                       # glove polling frequency (Hz)
       override_cfg:
         target_ee_pose: [0.8188, 0.1384, 0.1188, -3.1331, -1.1213, -0.0676]
         end_effector_type: "ruiyan_hand"    # or aoyi_hand
         end_effector_config:
           port: "/dev/ttyUSB0"              # dexterous hand serial port
           baudrate: 460800
           motor_ids: [1, 2, 3, 4, 5, 6]
           default_velocity: 2000
           default_current: 800
           default_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
         hand_target_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
         hand_reset_state: [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]  # finger pose at reset
         hand_action_scale: 1.0
         joint_reset_qpos: [0, 0, 0, -1.9, 0, 2, 0]

Key fields:

- ``target_ee_pose`` — Target arm TCP pose (obtain via ``test_controller``'s ``getpos_euler``)
- ``end_effector_type`` — Dexterous hand type: ``ruiyan_hand`` or ``aoyi_hand``
- ``hand_reset_state`` — Finger pose to reset to at the start of each episode (6-D, ``[0, 1]``)
- ``hand_target_state`` — Target finger pose for reward computation
- ``default_velocity`` / ``default_current`` — Motor speed and current limits

3. Run the data collection script:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

During collection, use the SpaceMouse to move the arm and hold the left
button while wearing the data glove to control fingers. Collected data is
saved to ``logs/[running-timestamp]/demos/``.

.. tip::

   If your task requires a visual classifier for success/failure detection
   (recommended for dexterous hand tasks), skip this section and follow
   the full workflow in `Visual Reward Classifier`_:

   1. Collect classifier training data → 2. Train classifier →
   3. Collect demos with classifier → 4. Train RL

Cluster Setup
~~~~~~~~~~~~~~~~~

Cluster setup is identical to :doc:`franka`.
Make sure all environment variables are properly set before running
``ray start`` on each node
(see ``ray_utils/realworld/setup_before_ray.sh``).

Configuration Files
~~~~~~~~~~~~~~~~~~~~~~

Configuration files typically consist of a main YAML config and an
``env/`` sub-config YAML.

**Env sub-config** ``examples/embodiment/config/env/realworld_dex_pnp.yaml``:

.. code-block:: yaml

   env_type: realworld
   auto_reset: True
   ignore_terminations: True
   max_episode_steps: 100          # max steps per episode
   use_spacemouse: True
   main_image_key: wrist_1         # main camera observation key

   init_params:
     id: "DexpnpEnv-v1"

**Full Ruiyan Hand configuration** (main config ``override_cfg`` section):

.. code-block:: yaml

   override_cfg:
     end_effector_type: "ruiyan_hand"
     end_effector_config:
       port: "/dev/ttyUSB0"
       baudrate: 460800
       motor_ids: [1, 2, 3, 4, 5, 6]
       default_velocity: 2000          # motor speed (higher = faster)
       default_current: 800            # motor current limit
       default_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     hand_reset_state: [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]
     hand_target_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     hand_action_scale: 1.0
     target_ee_pose: [0.8188, 0.1384, 0.1188, -3.1331, -1.1213, -0.0676]
     joint_reset_qpos: [0, 0, 0, -1.9, 0, 2, 0]

**Full Aoyi Hand configuration:**

.. code-block:: yaml

   override_cfg:
     end_effector_type: "aoyi_hand"
     end_effector_config:
       port: "/dev/ttyUSB0"
       node_id: 2
       baudrate: 115200
     hand_reset_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     hand_target_state: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     hand_action_scale: 1.0

**Glove configuration** (under ``env.eval`` or ``env.train``):

.. code-block:: yaml

   use_spacemouse: True
   glove_config:
     left_port: "/dev/ttyACM0"        # left-hand glove serial port
     frequency: 30                     # polling frequency (Hz)

Also set the ``model_path`` field in the ``rollout`` and ``actor``
sections to the path of the downloaded pretrained model.

Launch Training
~~~~~~~~~~~~~~~~~~~~~~~~~~

Start the experiment on the head node (using the dex_pnp task as an
example):

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async

This config already includes the classifier reward and demo path. See
``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml`` for
the full configuration.


Visual Reward Classifier
-----------------------

In dexterous hand tasks, the end-effector pose alone is insufficient to
determine success (e.g., whether an object is stably grasped). A visual
classifier automatically provides the reward signal.

Overview
~~~~~~~~~~~~~~~

The visual reward classifier uses a frozen ResNet-10 backbone to extract
image features, applies ``SpatialLearnedEmbeddings`` for spatial pooling,
and passes the result through a binary classification head to output a
success probability.

**Hardware Topology**

A typical dexterous hand setup uses two nodes (either two physical
machines or two Docker containers on the same host):

- **Training / GPU node** (node_rank=0) — Has GPU, runs actor, rollout,
  and **reward classifier inference**
- **Controller node** (node_rank=1) — Connected to Franka arm and
  dexterous hand, runs the env worker, **no GPU**

Classifier inference runs on the GPU node via a ``ClassifierRewardServer``
(a Ray actor). The env worker on the controller node obtains classifier
predictions through Ray remote calls. This leverages GPU inference speed
while avoiding CUDA model loading on the GPU-less controller node.

.. note::

   If your controller node has a GPU (or you are running a single-node
   setup), you can set ``classifier_reward_wrapper.remote: false`` and
   ``device: cuda`` in the config. The classifier will then be loaded
   directly inside the env worker process.

**Complete workflow (execute in order):**

1. **Collect classifier training data** — Single-node run on the
   controller node; teleoperate the robot and mark success/failure frames
   in real time using the SpaceMouse right button
2. **Human review** — Review frame labels in an OpenCV window on a
   machine with a display
3. **Train classifier** — **Run on the GPU node**; train the ResNet-10
   binary classifier
4. **Start Ray cluster** — Start Ray on both nodes to form a cluster
5. **Collect demo data** — Run on the 2-node cluster; classifier infers
   on GPU, env worker teleoperates on the controller node
6. **Train RL** — Run on the 2-node cluster; specify the demo path and
   classifier checkpoint path

Step 1: Collect Classifier Training Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   This step **only requires the controller node** (the node connected to
   the robot). No GPU node is needed. Start Ray in single-node mode on the
   controller node.

Classifier data collection uses the same teleoperation environment
(SpaceMouse + data glove + Franka) as regular data collection, but
uses the SpaceMouse **right button** to label frame classes in real time:

- **Hold right button** → Current frame is labeled as **success**
- **Release right button** → Current frame is labeled as **failure** (default)

.. important::

   During classifier data collection, you **cannot** rely on end-effector
   proximity to ``target_pos`` for success/failure detection — the visual
   classifier is meant to replace pose-based detection. Therefore the
   script automatically sets ``ignore_terminations=True`` so that every
   episode **always runs to** ``max_episode_steps`` before resetting.
   The operator manually marks success frames with the right button.

**On the controller node:**

.. code-block:: bash

   # 1. Activate the virtual environment
   source /opt/venv/franka-0.15.0/bin/activate
   # If using ROS: source <your_catkin_ws>/devel/setup.bash

   # 2. Start Ray in single-node mode on the controller node
   ray start --head --port=6379

   # 3. Run the classifier data collection script
   bash examples/embodiment/collect_classifier_data.sh

   # 4. Stop Ray after collection is complete
   ray stop

Data is saved to ``logs/<timestamp>-reward-classifier-<env_name>/``.

**Collection workflow:**

1. After the script starts, the robot resets to its initial pose and the
   terminal shows an episode progress bar
2. At the start of each episode the terminal prints a prominent prompt
   telling the operator to begin teleoperation
3. The operator teleoperates the task; when they believe the task is
   in a success state, they **hold** the SpaceMouse right button
4. Each episode runs for ``max_episode_steps`` and then automatically
   resets
5. Once enough success frames are collected (default 200), the review
   phase starts automatically

**Example terminal output:**

.. code-block:: text

   ##################################################
     Episode 3  Success: 42/200  Failure: 158
     >>> Begin teleoperation — right-click to mark success <<<
   ##################################################
   Ep3 [S:42/200 F:158]:  35%|████████          | 35/100

Step 2: Human Review
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After data collection, the script automatically opens an OpenCV review
window. If the window does not appear (e.g., on a headless controller
node), copy the data directory to a machine with a display and run the
review script manually:

.. code-block:: bash

   bash examples/embodiment/review_classifier_data.sh \
       logs/<timestamp>-reward-classifier-dex_pnp

**Review controls:**

.. list-table::
   :widths: 10 50
   :header-rows: 1

   * - Key
     - Action
   * - ``n``
     - Next frame
   * - ``p``
     - Previous frame
   * - ``g``
     - Mark as **keep** (good)
   * - ``b``
     - Mark as **discard** (bad)
   * - ``1``
     - Show success frames only
   * - ``2``
     - Show failure frames only
   * - ``0``
     - Show all
   * - ``s``
     - Save
   * - ``q`` / ``ESC``
     - Finish review and save

After review, the final data is saved as both images and pickle files in
the same directory:

.. code-block:: text

   logs/<timestamp>-reward-classifier-dex_pnp/
   ├── raw_frames.pkl               # raw collected data
   ├── success/                     # success frame images
   │   ├── 2026-03-03_04-41-36_00000.png
   │   └── ...
   ├── failure/                     # failure frame images
   │   ├── 2026-03-03_04-41-36_00000.png
   │   └── ...
   ├── success_202_2026-03-03_04-41-36.pkl   # pickle for training
   └── failure_741_2026-03-03_04-41-36.pkl

Step 3: Train the Classifier (on the GPU Node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   Classifier training should be run on the **GPU node** to leverage GPU
   acceleration. If the controller node and GPU node are different
   machines, copy the data directory to the GPU node first (if they share
   a mounted volume, no copy is needed).

**On the GPU node:**

.. code-block:: bash

   # Activate the virtual environment (may not be needed if using Docker)
   source /opt/venv/openvla/bin/activate

   # Train the classifier
   python examples/embodiment/train_reward_classifier.py \
       --log_dir logs/<timestamp>-reward-classifier-dex_pnp \
       --pretrained_ckpt RLinf-ResNet10-pretrained/resnet10_pretrained.pt \
       --image_keys global wrist_1 \
       --num_epochs 200 \
       --device cuda

The script auto-detects GPU availability. If ``--device`` is not
specified, the default is ``cuda`` when a GPU is available, falling back
to ``cpu`` otherwise.

**Training parameters:**

.. list-table::
   :widths: 22 12 50
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``--log_dir``
     - (required)
     - Data directory (created by ``collect_classifier_data.sh``); also
       used as the model save directory
   * - ``--pretrained_ckpt``
     - ``RLinf-ResNet10-pretrained/resnet10_pretrained.pt``
     - ResNet-10 pretrained weights path
   * - ``--image_keys``
     - ``wrist_1``
     - Camera observation keys (must match the camera ``name`` in the
       env config)
   * - ``--image_size``
     - ``128``
     - Input image size
   * - ``--num_epochs``
     - ``200``
     - Number of training epochs
   * - ``--batch_size``
     - ``64``
     - Batch size
   * - ``--lr``
     - ``1e-4``
     - Learning rate
   * - ``--device``
     - auto-detect
     - Training device; explicitly setting ``cuda`` is recommended

During training, each epoch's loss and accuracy are printed. The best
model is saved as ``<log_dir>/reward_classifier.pt``.

.. code-block:: text

   Loading data from logs/...-reward-classifier-dex_pnp ...
     success: 202   failure: 741   total: 943
   Building classifier ...
   Training for 200 epochs ...
   Epoch  10/200  loss=0.4123  acc=0.8125
   ...
   Epoch 200/200  loss=0.0512  acc=0.9688

   Best accuracy: 0.9844
   Checkpoint saved to logs/...-reward-classifier-dex_pnp/reward_classifier.pt

Step 4: Start the Ray Cluster (2 Nodes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From this step onward, two nodes must cooperate. Make sure both nodes
(or Docker containers) are on the same network.

.. warning::

   This step is critical! Environment variables must be set correctly on
   each node **before** starting Ray, because Ray snapshots the
   environment at startup and all subsequently created processes inherit
   that snapshot.

**On the GPU node (node_rank=0, head node):**

.. code-block:: bash

   # Activate the virtual environment
   source /opt/venv/openvla/bin/activate

   # Set RLinf environment variables
   export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
   export RLINF_NODE_RANK=0

   # Start the Ray head
   ray start --head --port=6380 --node-ip-address=<head_ip>

**On the controller node (node_rank=1, worker node):**

.. code-block:: bash

   # Activate the virtual environment (franka environment)
   source /opt/venv/franka-0.15.0/bin/activate

   # Set RLinf environment variables
   export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
   export RLINF_NODE_RANK=1

   # Join the Ray cluster
   ray start --address='<head_ip>:6380'

Use ``ray status`` to confirm both nodes are ready.

.. tip::

   If you are using two Docker containers on the same host with
   ``--network host``, ``<head_ip>`` can be the host machine's LAN IP
   (e.g., ``172.16.0.1``).

Step 5: Collect Demo Data (2 Nodes, Classifier on GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After starting the Ray cluster, run the demo collection script on the
**head node (GPU node)**. The script will automatically:

1. Start a ``ClassifierRewardServer`` on the GPU node and load the
   classifier model onto the GPU
2. Start a ``DataCollector`` (env worker) on the controller node that
   calls the classifier remotely via Ray

During demo collection:

- **Classifier judges success** → episode terminates, saved as a
  success demo
- **Episode reaches** ``max_episode_steps`` → truncated, saved as a
  failed demo and automatically resets

**Configuration file**
``examples/embodiment/config/realworld_collect_dexpnp_demo.yaml``
key fields:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       env:
         node_group: franka     # env worker on the controller node
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0          # GPU node
       - label: franka
         node_ranks: 1          # controller node
         env_configs:
           - node_ranks: 1
             env_vars:
               - PYTHONPATH: "..."
               - LD_LIBRARY_PATH: "..."
         hardware:
           type: Franka
           configs:
             - robot_ip: 172.16.0.2
               node_rank: 1

   env:
     eval:
       ignore_terminations: False   # let classifier-triggered termination take effect
       classifier_reward_wrapper:
         checkpoint_path: "/path/to/reward_classifier.pt"
         device: cuda               # classifier infers on GPU
         remote: true               # key: remote call to GPU node via Ray
         threshold: 0.75

Update ``checkpoint_path`` to the classifier path from Step 3, and
``robot_ip``, ``env_vars``, etc. to your actual values, then run on the
**head node**:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_dexpnp_demo

.. important::

   ``ignore_terminations`` must be ``False``, otherwise the classifier's
   termination signal is ignored. This is the **opposite** of Step 1
   (classifier data collection) where ``ignore_terminations: True``.

Collected demos are saved to ``logs/<timestamp>/demos/``.

**Example terminal output:**

.. code-block:: text

   [collect_real_data] ClassifierRewardServer 'ClassifierRewardServer' ready on GPU node.
   ...
   ✅ SUCCESS  classifier_reward=0.912
       Success: 5/20  Total episodes: 8
   ⏱️  TRUNCATED (max steps)
       Success: 5/20  Total episodes: 9

Step 6: Train RL (with Demos and Classifier Reward)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After collecting demos, **make sure the Ray cluster is still running**
(or restart it), then specify in the training config YAML:

1. ``classifier_reward_wrapper.remote: true`` — the classifier also
   infers on the GPU node during online training
2. ``demo_buffer.load_path`` — path to the demos collected in Step 5

**Training configuration**
``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml``
key fields:

.. code-block:: yaml

   env:
     train:
       classifier_reward_wrapper:
         checkpoint_path: /path/to/reward_classifier.pt
         device: cuda
         remote: true              # classifier infers on GPU node
         threshold: 0.75

   algorithm:
     demo_buffer:
       load_path: "/path/to/logs/<timestamp>/demos"

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: "4090"
         placement: 0
       env:
         node_group: franka
         placement: 0
       rollout:
         node_group: "4090"
         placement: 0

After updating ``checkpoint_path`` and ``demo_buffer.load_path``, run on
the **head node**:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async

.. note::

   When using ``classifier_reward_wrapper``, there is no need to also set
   ``keyboard_reward_wrapper`` — the two are mutually exclusive reward
   sources.


Quick Reference
-----------------------

The following table summarizes the node and command for each step of the
workflow:

.. list-table::
   :widths: 6 26 18 50
   :header-rows: 1

   * - Step
     - Operation
     - Where to Run
     - Command
   * - 1
     - Collect classifier training data
     - Controller node (single-node Ray)
     - ``ray start --head`` → ``bash examples/embodiment/collect_classifier_data.sh`` → ``ray stop``
   * - 2
     - Human review
     - Node with display
     - ``bash examples/embodiment/review_classifier_data.sh logs/<dir>``
   * - 3
     - Train classifier
     - GPU node
     - ``python examples/embodiment/train_reward_classifier.py --log_dir ... --device cuda``
   * - 4
     - Start Ray cluster
     - Both nodes
     - head: ``ray start --head`` / worker: ``ray start --address=...``
   * - 5
     - Collect demos (classifier on GPU)
     - Head node (GPU)
     - ``bash examples/embodiment/collect_data.sh realworld_collect_dexpnp_demo``
   * - 6
     - Train RL
     - Head node (GPU)
     - ``bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async``


Diagnostic Tool
---------------------------

``test_controller`` is an interactive CLI diagnostic tool for querying
the arm and dexterous hand states in real time.

.. code-block:: bash

   export FRANKA_ROBOT_IP=<your_robot_ip>
   export FRANKA_END_EFFECTOR_TYPE=ruiyan_hand
   python -m toolkits.realworld_check.test_franka_controller

.. list-table:: Available Commands
   :widths: 20 60
   :header-rows: 1

   * - Command
     - Description
   * - ``getpos``
     - Get arm TCP pose (quaternion, 7-D)
   * - ``getpos_euler``
     - Get arm TCP pose (Euler angles, 6-D)
   * - ``getjoints``
     - Get arm joint positions and velocities (7-D each)
   * - ``getvel``
     - Get arm TCP velocity (6-D)
   * - ``getforce``
     - Get arm TCP force and torque (3-D each)
   * - ``gethand``
     - Get finger positions [0, 1]
   * - ``gethand_detail``
     - Get detailed motor status (position, velocity, current, status code)
   * - ``handinfo``
     - Show hand configuration info (type, port, baudrate, etc.)
   * - ``state``
     - Show full robot state
   * - ``help``
     - Show help
   * - ``q``
     - Quit

Finger DOF Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 8 20 50
   :header-rows: 1

   * - #
     - DOF Name
     - Description
   * - 1
     - ``thumb_rotation``
     - Thumb lateral rotation (adduction/abduction)
   * - 2
     - ``thumb_bend``
     - Thumb flexion/extension
   * - 3
     - ``index``
     - Index finger flexion/extension
   * - 4
     - ``middle``
     - Middle finger flexion/extension
   * - 5
     - ``ring``
     - Ring finger flexion/extension
   * - 6
     - ``pinky``
     - Pinky finger flexion/extension

All position values are normalized to ``[0, 1]``: ``0`` = fully open,
``1`` = fully closed.


End-Effector Architecture
---------------------------

All end-effectors implement a unified ``EndEffector`` abstract base class:

.. code-block:: python

   class EndEffectorType(str, Enum):
       FRANKA_GRIPPER = "franka_gripper"   # 7-D actions
       AOYI_HAND      = "aoyi_hand"        # 12-D actions
       RUIYAN_HAND    = "ruiyan_hand"      # 12-D actions

The factory function ``create_end_effector(end_effector_type, **kwargs)``
creates the appropriate instance. After switching the end-effector,
``FrankaEnv`` automatically adjusts its action and observation spaces.

**Supported dexterous hands:**

- **Aoyi Hand** — Modbus RTU serial, 6 DOF, ``[0, 1]`` continuous control
- **Ruiyan Hand** — Custom serial protocol, 6 DOF, ``[0, 1]`` continuous control

Teleoperation Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dexterous hand teleoperation uses **SpaceMouse + Data Glove**:

- **SpaceMouse** — 6-D end-effector pose delta (x, y, z, roll, pitch, yaw)
- **Data Glove** — 6-D finger angles (relative mode)

``DexHandIntervention`` merges both into a 12-dimensional expert action.
The system automatically selects the correct intervention wrapper based
on the ``end_effector_type`` in the configuration — no manual code
changes are needed.

**Relative Glove Control**

The glove operates in relative control mode, toggled by the SpaceMouse
left button:

1. **Press left button** — Capture the current glove reading as the
   "baseline" and record the current finger positions
2. **Hold left button** — The delta between the glove reading and the
   baseline is added to the finger positions
3. **Release left button** — Fingers freeze at their last position

This ensures the glove does not cause sudden finger jumps. The operator
can press and release the left button multiple times to fine-tune finger
poses.

**Button Mapping**

In the ``pyspacemouse`` library, ``buttons[0]`` corresponds to the
**physical right button** and ``buttons[1]`` corresponds to the
**physical left button** (counter-intuitive). ``DexHandIntervention``
corrects this mapping internally:

.. code-block:: python

   # pyspacemouse: buttons[0] = physical right, buttons[1] = physical left
   self.left, self.right = bool(buttons[1]), bool(buttons[0])

Both buttons are exposed to downstream components via
``info["left"]`` / ``info["right"]``.

**Finger State Synchronization Across Episodes**

``DexHandIntervention.reset()`` synchronizes the internal finger tracking
state to the configured ``hand_reset_state`` at the start of each episode,
ensuring the operator starts from the reset pose rather than the ending
pose of the previous episode.


Visualization and Results
-------------------------

**TensorBoard Logging**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**Key Metrics**

- ``env/success_once``: Recommended metric; reflects the episodic success rate
- ``env/return``: Episode return
- ``env/reward``: Step-level reward

See :doc:`franka` for the full list of metrics.

.. note::

   Training results and demo videos for dexterous hand tasks will be
   provided in a future update.
