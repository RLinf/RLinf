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
:doc:`franka`, with additional serial communication dependencies.

Controller Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After completing the base installation from :doc:`franka`, install the
following in the virtual environment on the controller node:

.. code-block:: bash

   # Serial communication (dexterous hand + data glove)
   pip install pyserial pymodbus pyyaml

   # Data glove driver
   pip install psi_glove_driver

.. note::

   If ``psi_glove_driver`` is not available via pip, please contact
   the glove manufacturer for the driver package and install manually.

Training / Rollout Nodes
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
the dexterous hand connection.

Set environment variables and run the diagnostic script:

.. code-block:: bash

   export FRANKA_ROBOT_IP=<your_robot_ip>
   export FRANKA_END_EFFECTOR_TYPE=ruiyan_hand  # or aoyi_hand
   export FRANKA_HAND_PORT=/dev/ttyUSB0
   python -m toolkits.realworld_check.test_controller

In the interactive prompt:

- Enter ``getpos_euler`` to get the current end-effector pose (Euler angles)
- Enter ``gethand`` to view the current finger positions
- Enter ``handinfo`` to verify the hand connection
- Enter ``help`` for all available commands

**2. Test Hardware Connections**

.. code-block:: bash

   # Test the camera
   python -m toolkits.realworld_check.test_camera

Data Collection
~~~~~~~~~~~~~~~~~

For data collection with a dexterous hand, the SpaceMouse controls the
arm and the data glove controls the fingers.
``DexHandIntervention`` automatically merges both inputs into
12-dimensional actions.

1. Source the virtual environment and the ROS setup scripts:

.. code-block:: bash

   source <path_to_your_venv>/bin/activate
   source <your_catkin_ws>/devel/setup.bash

2. Modify the configuration file ``examples/embodiment/config/realworld_collect_data.yaml``:

.. code-block:: yaml

   env:
     train:
       override_cfg:
         end_effector_type: ruiyan_hand   # or aoyi_hand
         robot_ip: YOUR_ROBOT_IP
         target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
       use_spacemouse: true
       glove_config:
         left_port: "/dev/ttyACM0"
         frequency: 60

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
             - robot_ip: YOUR_ROBOT_IP
               node_rank: 0

3. Run the data collection script:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh

During collection, use the SpaceMouse to move the arm while wearing
the data glove to control the fingers. Collected data is saved to
``logs/[running-timestamp]/data.pkl``.

Cluster Setup
~~~~~~~~~~~~~~~~~

Cluster setup is identical to :doc:`franka`.
Make sure all environment variables are properly set before running
``ray start`` on each node
(see ``ray_utils/realworld/setup_before_ray.sh``).

Configuration File
~~~~~~~~~~~~~~~~~~~~~~

Set the end-effector type in the configuration YAML and fill in the
parameters for your specific hand model:

**Ruiyan Hand example:**

.. code-block:: yaml

   env:
     train:
       override_cfg:
         end_effector_type: ruiyan_hand
         ruiyan_port: "/dev/ttyUSB0"
         ruiyan_baudrate: 460800
       use_spacemouse: true
       glove_config:
         left_port: "/dev/ttyACM0"

**Aoyi Hand example:**

.. code-block:: yaml

   env:
     train:
       override_cfg:
         end_effector_type: aoyi_hand
         aoyi_port: "/dev/ttyUSB0"
         aoyi_node_id: 2
         aoyi_baudrate: 115200
       use_spacemouse: true
       glove_config:
         left_port: "/dev/ttyACM0"

Also set the ``model_path`` field in the ``rollout`` and ``actor``
sections to the path of the downloaded pretrained model.

Running the Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Start the experiment on the head node:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh <your_dexhand_config_name>


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
success probability. The workflow is:

1. Collect success and failure images
2. Train the binary classifier
3. Use the classifier output as the reward in the environment

Step 1: Collect Classifier Training Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the data collection tool to gather success/failure images. Press the
space bar to mark a frame as success:

.. code-block:: bash

   python -m toolkits.realworld_check.record_classifier_data \
       --save_dir /path/to/classifier_data \
       --successes_needed 200

Data is saved as pickle files in the specified directory.

Step 2: Train the Classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m toolkits.realworld_check.train_reward_classifier \
       --data_dir /path/to/classifier_data \
       --save_dir /path/to/classifier_ckpt \
       --pretrained_ckpt RLinf-ResNet10-pretrained/resnet10_pretrained.pt \
       --image_keys wrist_1 \
       --num_epochs 200

The ``--pretrained_ckpt`` argument points to the pretrained ResNet-10
policy backbone weights, which are used to initialize the classifier's
frozen visual encoder. The trained classifier is saved under
``--save_dir``.

Step 3: Use the Classifier Reward in the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a ``classifier_reward_wrapper`` section to the configuration YAML.
The system will automatically replace the keyboard reward with the
classifier output:

**Single-stage task:**

.. code-block:: yaml

   env:
     train:
       classifier_reward_wrapper:
         checkpoint_path: /path/to/classifier_ckpt/best_classifier.pt
         image_keys: [wrist_1]
         device: cuda
         threshold: 0.75   # sigmoid output above this threshold = success

**Multi-stage task:**

.. code-block:: yaml

   env:
     train:
       classifier_reward_wrapper:
         multi_stage: true
         checkpoint_paths:
           - /path/to/stage1_classifier.pt
           - /path/to/stage2_classifier.pt
         image_keys: [wrist_1]
         device: cuda

.. note::

   When using ``classifier_reward_wrapper``, there is no need to set
   ``keyboard_reward_wrapper`` — the two are mutually exclusive reward
   sources.


Diagnostic Tool
---------------------------

``test_controller`` is an interactive CLI diagnostic tool for querying
the arm and dexterous hand states in real time.

.. code-block:: bash

   export FRANKA_ROBOT_IP=<your_robot_ip>
   export FRANKA_END_EFFECTOR_TYPE=ruiyan_hand
   python -m toolkits.realworld_check.test_controller

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
- **Data Glove** — 6-D finger angles

``DexHandIntervention`` merges both into a 12-dimensional expert action.
The system automatically selects the correct intervention wrapper based
on the ``end_effector_type`` in the configuration — no manual code
changes are needed.


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
