Running Pi0 SFT with Franka
===========================
.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 80%

   OpenPI π₀ SFT and deployment workflow used for Franka bin-relocation experiments.

Run the Bin-relocation pipeline end to end with OpenPI π₀: collect Franka data, convert it to a LeRobot-style dataset, compute normalization stats, fine-tune, and deploy the checkpoint on real hardware.

Overview
--------

Create a real-world Franka dataset, fine-tune π₀, and deploy the result.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀

   .. grid-item-card:: Algorithms
      :text-align: center

      SFT · eval-only deployment

   .. grid-item-card:: Tasks
      :text-align: center

      Bin relocation · generic SFT env

   .. grid-item-card:: Hardware
      :text-align: center

      Franka · cameras · gripper

| **You'll do:** get target pose → collect data → compute norm stats → run SFT → run real-world eval.
| **Prerequisites:** :doc:`franka` · :doc:`sft_openpi` · OpenPI base checkpoint · real-world dataset path.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - Task
     - Config / entry point
     - Description
   * - Data conversion
     - ``pi0_realworld``
     - Represent Franka data in the OpenPI data format.
   * - SFT
     - ``realworld_sft_openpi``
     - Fine-tune π₀ on real-world Franka data.
   * - Deployment
     - ``realworld_pnp_eval`` / ``realworld_eval``
     - Run eval-only deployment on the real robot.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - Field
     - Description
   * - Observation
     - Real-world camera frames mapped to OpenPI dataset keys.
   * - Action
     - Franka action format selected by ``pi0_realworld`` metadata.
   * - Reward
     - Eval success or operator-observed deployment outcome.
   * - Prompt
     - Task text stored in the SFT dataset/config.

Installation
------------

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Robot arm**: Franka Emika Panda.
- **Camera**: Intel RealSense camera (wrist camera for observation).
- **Compute node**: A GPU-equipped machine for SFT training and rollout.
- **Robot control node**: A small computer on the same LAN as the robot
  (no GPU required) for controlling the Franka arm.
- **SpaceMouse (optional)**: For remote teleoperation during data collection.

.. note::

   For detailed hardware setup instructions (ROS Noetic, libfranka,
   serl_franka_controllers, etc.), refer to the **Hardware Setup** and
   **Dependency Installation** sections in :doc:`franka`.

Software Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **control node** (data collection) requires Franka control dependencies;
see the dependency installation section in :doc:`franka`.

The **training / rollout node** (SFT training + deployment) requires OpenPI
model dependencies:

.. code:: bash

   # For mainland China users, you can add `--use-mirror` to the install.sh command.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

.. note::

   **Note on training / rollout node installation**: Make sure to specify
   ``openpi`` (not ``openvla``) in the ``--model`` argument.

Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Data collection, training, and deployment all rely on the Franka real-world
environment config template at
``examples/embodiment/config/env/realworld_bin_relocation.yaml``.
This config defines key Bin-relocation task parameters such as end-effector pose
limits and success thresholds. Adjust fields under ``override_cfg`` as needed
for your task.

Run It
------

The following steps cover the full Bin-relocation pipeline.

Step 1: Obtain the Target Pose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before collecting data, you must determine the target end-effector pose
(``target_ee_pose``).

Note that in the Bin-relocation task, the target end-effector pose represents
the midpoint of the lowest point in the motion space. Specifically, to prevent
the Franka end-effector from colliding with the rim of the container, a
workspace region is carved out around the target pose to limit the robot's
range of motion. See ``rlinf/envs/realworld/franka/tasks/franka_bin_relocation.py``
for details.

Follow the **Obtain the target pose** section in :doc:`franka` and use the
``toolkits.realworld_check.test_franka_controller`` script to obtain the
target pose. Record this pose for use in subsequent configuration steps.

Step 2: Collect Expert Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the **Data Collection** section in :doc:`franka` to collect expert
data on the control node.

In addition to the base configuration, make the following modifications for
the Bin-relocation pick-and-place task:

1. Switch the environment from peg insertion to bin relocation:

.. code-block:: yaml

  defaults:
    - env/realworld_bin_relocation@env.eval
    - override hydra/job_logging: stdout

2. Enable the gripper degree of freedom for the pick-and-place task:

.. code:: yaml

  env:
    eval:
      no_gripper: False

3. Use the keyboard to label trajectories during data collection:
   press ``c`` to mark the current trajectory as successful and reset the robot:

.. code:: yaml

  env:
    eval:
      keyboard_reward_wrapper: single_stage

4. Set the task description:

.. code:: yaml

   env:
     eval:
       override_cfg:
         task_description: "pick up the object and place it into the container"

5. Export data in LeRobot format:

.. code:: yaml

  env:
    data_collection:
      enabled: True
      export_format: "lerobot"

During collection, use the SpaceMouse to teleoperate the robot and perform
the task. After each episode, press ``c`` to mark it as successful and reset
the robot pose. Remember to return the target object to the starting position.

The script stops after collecting 20 episodes by default (configurable via
``num_data_episodes``). Collected LeRobot-format data is saved under
``logs/<running-timestamp>/collected_data``.

Step 3: SFT Training Pi0
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a Real-World Dataset Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step follows the **Supported datasets** section in :doc:`sft_openpi`.
For real-world Franka environments, you can create the ``pi0_realworld``
dataset format, defined in:

1. ``rlinf/models/embodiment/openpi/__init__.py``
2. ``rlinf/models/embodiment/openpi/dataconfig/realworld_dataconfig.py``

To unify the policy call interface between real-world and simulated
environments, RLinf provides
3. ``rlinf/models/embodiment/openpi/policies/realworld_policy.py``.

Compute Normalization Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following the **Normalization statistics for new LeRobot datasets** section in
:doc:`sft_openpi`, you must compute normalization statistics for your newly
collected LeRobot dataset before launching SFT.

First, upload the data from the control node to the training node's data
directory, e.g. ``/path/to/lerobot_data``. The file structure should be:

.. code::

   /path/to/lerobot_data
   |-- realworld_franka_bin_relocation
      |-- data
      |-- meta
    |-- franka_dagger
        |-- data
        |-- meta
    |-- ...

Here ``realworld_franka_bin_relocation`` corresponds to the ``repo_id`` field in the
``TrainConfig`` defined in ``rlinf/models/embodiment/openpi/__init__.py``.

Then run on the training node:

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

Notes:

- ``HF_LEROBOT_HOME`` must be set before running the script.
- ``config_name`` must match the OpenPI dataconfig used by training.
- ``repo_id`` must match your LeRobot-format dataset name.

The script writes stats to
``<assets_dir>/<exp_name>/<repo_id>/norm_stats.json``.

The OpenPI loader reads normalization stats from ``<model_path>/<repo_id>`` at
runtime.

Run OpenPI SFT
~~~~~~~~~~~~~~~

With the ``pi0_realworld`` dataset format, modify the SFT training config
``examples/sft/config/realworld_sft_openpi.yaml``:

.. code:: yaml

   data:
     train_data_paths: "/path/to/lerobot_data"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi:
         config_name: "pi0_realworld"

Place the normalization stats under the model path; the OpenPI loader reads
them from ``<model_path>/<repo_id>``. The file structure should be:

.. code::

  /path/to/pi0-model
  |-- config.json
  |-- model.safetensors
  |-- realworld_franka_bin_relocation
    |-- norm_stats.json
  |-- franka_dagger
    |-- norm_stats.json
  |-- ...

Run the SFT training script:

.. code:: bash

   bash examples/sft/run_vla_sft.sh realworld_sft_openpi

The checkpoint exported by SFT will be used in the deployment step.
See :doc:`sft_openpi` for more details on OpenPI datasets and SFT training.

Step 5: Real-World Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify ``evaluations/realworld/realworld_pnp_eval.yaml``
to match your cluster, camera, and target pose:

.. code-block:: yaml

   cluster:
     node_groups:
       - label: franka
         hardware:
           configs:
             - robot_ip: ROBOT_IP

   env:
     eval:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
         task_description: "pick up the object and place it into the container"

After SFT training completes, update the model checkpoint path in the deploy
config:

.. code:: yaml

   runner:
     ckpt_path: "/path/to/your/sft/checkpoint/full_weights.pt"

   rollout:
     model:
       model_path: "/path/to/pi0-model"

After starting the Ray cluster (see the **Cluster configuration** section in
:doc:`franka`), run deployment through the :doc:`real-world evaluation guide
<../../evaluations/guides/realworld>` with ``realworld_pnp_eval``. The policy
will autonomously control the robot to complete the Bin-relocation task.

You can control the number of evaluation episodes via the
``env.eval.rollout_epoch`` parameter:

.. code:: yaml

   env:
     eval:
       rollout_epoch: 20

Generic Real-World SFT Environment and Deployment
-------------------------------------------------

Beyond the Bin-relocation task, RLinf provides a **generic SFT environment**
(``FrankaEnv-v1``) that lets you define new real-world tasks entirely through
YAML configuration, without writing a custom environment class. It is useful for:

- Collecting SFT demonstration data on new tasks
- Deploying (evaluating) a trained policy on the real robot

Generic SFT Environment
~~~~~~~~~~~~~~~~~~~~~~~~

The env config template lives at
``examples/embodiment/config/env/realworld_franka_sft_env.yaml``.
Key fields you should customise for your task:

.. code:: yaml

   override_cfg:
     task_description: "pick up the object and place it into the container"
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # goal pose [x,y,z,rx,ry,rz]
     reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]    # reset pose (above goal)
     max_num_steps: 300
     reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]  # success tolerance
     action_scale: [1.0, 1.0, 1.0]                         # [xyz, rpy, gripper]
     ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
     ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

Under the hood, ``FrankaEnv`` accepts ``override_cfg`` as a plain dict and uses
a class-variable ``CONFIG_CLS`` to instantiate the dataclass config (defaults to
``FrankaRobotConfig``). Subclasses such as ``PegInsertionEnv`` and ``BottleEnv``
override ``CONFIG_CLS`` to their own dataclass while sharing the same
constructor.

Standalone Evaluation
~~~~~~~~~~~~~~~~~~~~~

A full evaluation config is provided at
``evaluations/realworld/realworld_eval.yaml``. It pairs the generic SFT env
with a Pi0 actor for deployment.

Before running, replace the placeholders:

- ``ROBOT_IP`` — your Franka robot's IP address.
- ``MODEL_PATH`` — path to your trained checkpoint.

Launch and override examples are maintained in the :doc:`real-world evaluation
guide <../../evaluations/guides/realworld>`.
