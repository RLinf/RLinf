Reward Model Guide (Real-World)
===============================

This document describes how to collect and preprocess a reward model training dataset
directly on a real-world Franka robot. Two data collection approaches are supported:
a **general-purpose keyboard-labeling** approach and a **fixed-pose** approach that
uses a predetermined target pose to drive episode success/failure.

Before getting started, it is strongly recommended to read the following documents:

1. :doc:`../../examples/embodied/franka` — to familiarize yourself with the end-to-end real-world Franka training pipeline.
2. :doc:`reward_model` — to understand the canonical reward model workflow in RLinf (data collection via ``pickle``, offline preprocessing, training, RL inference).
3. :doc:`../../examples/embodied/franka_reward_model` — to understand the full real-world RL pipeline that follows after you have a trained reward model.

Workflow Overview
-----------------

The collection script combines data collection, labeling, and dataset generation into one end-to-end run (Approach 1) or a streamlined two-step pipeline (Approach 2).

.. code-block:: text

   RealWorld dataset collection (this guide)
   ├── Approach 1: Keyboard labeling (general-purpose)
   │   1. Launch a single RealWorld episode with SpaceMouse/keyboard teleop.
   │   2. Press 'c' (success) or 'a' (fail) to label each frame.
   │   3. Stop when thresholds are reached, or max_steps is exhausted.
   │   4. Apply fail:success ratio sampling and train/val split.
   │   5. Save train.pt / val.pt directly (no .pkl intermediate).
   │
   └── Approach 2: Fixed-pose (target-driven)
       1. Configure a target end-effector pose (no keyboard labeling needed).
       2. Episode auto-terminates on reaching the pose.
       3. Save collected episodes as .pkl files.
       4. Automatically extract success/fail frames from episode trajectories.
       5. Run preprocess_reward_dataset.py to generate train.pt / val.pt.

Prerequisites
-------------

Follow the **Prerequisites** and **Hardware Setup** sections in :doc:`../../examples/embodied/franka`
up to and including the robot connection and environment validation steps.

Data Collection
---------------

Approach 1: Keyboard Labeling (General-Purpose)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This approach uses keyboard keys to manually label each frame during a live episode.
It is task-agnostic and works for any manipulation task.

**Configuration file** — ``examples/reward/config/realworld_collect_dataset.yaml``,
inheriting environment parameters from ``env/realworld_bin_relocation.yaml``:

.. code-block:: yaml

   defaults:
     - env/realworld_bin_relocation@env.eval
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
             - robot_ip: ROBOT_IP
               node_rank: 0

   runner:
     task_type: embodied
     logger:
       log_path: null
       project_name: rlinf
       experiment_name: "collect-dataset"
       logger_backends: ["tensorboard"]
     num_success_frames: 50    # target number of success frames to collect
     num_fail_frames: 150      # target number of fail frames to collect
     val_split: 0.2            # fraction of frames reserved for validation
     fail_success_ratio: 2.0   # downsample fail frames to 2x success frames
     random_seed: 42

   env:
     group_name: "EnvGroup"
     eval:
       no_gripper: False
       use_spacemouse: True
       max_episode_steps: 10000
       keyboard_reward_wrapper: single_stage
       override_cfg:
         target_ee_pose: TARGET_EE_POSE

**Key configuration fields:**

- ``runner.num_success_frames`` / ``runner.num_fail_frames`` — target numbers of labeled
  frames to collect. Collection stops when both thresholds are reached.
- ``runner.val_split`` — fraction of all labeled frames held out as validation data.
- ``runner.fail_success_ratio`` — during training-set post-processing, fail frames are
  downsampled so that ``num_fail = num_success * fail_success_ratio``. Set to ``0`` to
  disable downsampling.
- ``env.eval.keyboard_reward_wrapper`` — set to ``single_stage`` (or the appropriate
  stage key for your task) to enable the keyboard labeling interface.
- ``env.eval.use_spacemouse`` — whether SpaceMouse is used for teleoperation (the
  ``intervene_action`` in step info overrides the zero dummy action).
- ``env.eval.override_cfg.target_ee_pose`` — the target end-effector pose for the task.

**Launching:**

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh

Or with an explicit config name:

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh realworld_collect_dataset

A progress bar prints live to the terminal:

.. code-block:: text

   success: 12/50 [############----------------]  fail: 28/150 [#####################-----------]

Use the following keys during the episode:

- ``c`` — label the current frame as **success**.
- ``a`` — label the current frame as **fail**.
- Keyboard actions from the ``keyboard_reward_wrapper`` also control whether the episode
  continues or resets.

When both ``num_success_frames`` and ``num_fail_frames`` are reached, the script
automatically stops, splits the data, and saves the ``.pt`` files.


Approach 2: Fixed-Pose (Target-Driven)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This approach is specifically designed for tasks with a **fixed target pose** (e.g., reaching a
predetermined bin location). Instead of manual keyboard labeling, the episode automatically
drives success/failure based on whether the robot reaches the configured ``target_ee_pose``.
``success_hold_steps`` can be set to require the robot to maintain the pose for a certain
number of steps before declaring success, which helps collect more diverse successful samples.

This approach follows the same data collection pipeline as described in
:doc:`../../examples/embodied/franka_reward_model`, but with a simplified preprocessing step
that uses the same script as Approach 1 (``realworld_collect_process_dataset.py``).


Step 1: Fixed-Pose Reward Data Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To obtain a high-quality reward model, additional data needs to be collected for training
and evaluation. On top of the expert trajectory collection above, make the following
modifications to the collection script:

Increase the ``success_hold_steps`` field so that, within a limited number of collection
episodes, more diverse successful data can be obtained. The robot arm end-effector will not
be immediately marked as successful upon reaching the target pose — it must maintain the
target pose for a certain number of steps (``success_hold_steps``) before being marked as
successful. If the arm exits the target zone mid-hold, the counter resets.

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         success_hold_steps: 20

Collection tips:

- Move the robot arm slowly to obtain more diverse failure samples.
- When reaching the target pose, make small-range movements while maintaining the pose
  to obtain more diverse successful samples.

Step 2: Preprocessing into a Reward Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The collected ``.pkl`` episodes are converted into ``train.pt`` / ``val.pt`` using
``preprocess_reward_dataset.py``. It is recommended to increase ``fail-success-ratio`` to ``3``:

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data \
       --fail-success-ratio 3

This produces:

.. code-block:: text

   logs/xxx/processed_reward_data/
   ├── train.pt
   └── val.pt

The generated ``.pt`` files follow the ``RewardDatasetPayload`` schema:

.. code-block:: python

   {
       "images": list[torch.Tensor],
       "labels": list[int],
       "metadata": dict[str, Any],
   }

Where:

- ``images`` — training images.
- ``labels`` — binary labels (1 = success, 0 = fail).
- ``metadata`` — source path, sampling arguments, split ratio, etc.


Output
~~~~~~

After collection (both approaches), the output consists of two ``.pt`` files saved to
``runner.logger.log_path`` (defaults to the Hydra run dir):

.. code-block:: text

   logs/<timestamp>-collect-dataset/
   ├── train.pt
   └── val.pt
   └── run_collect_process.log   # (Approach 1 only)

Each ``.pt`` file follows the ``RewardDatasetPayload`` schema:

.. code-block:: python

   {
       "images": list[torch.Tensor],
       "labels": list[int],             # 1 = success, 0 = fail
       "metadata": dict,                # collection stats and config
   }

The ``metadata`` dict includes:

- ``num_success_frames`` / ``num_fail_frames`` — raw counts before ratio sampling.
- ``fail_success_ratio`` / ``val_split`` / ``random_seed`` — sampling parameters.
- ``num_train_samples`` / ``num_val_samples`` — final dataset sizes.

These ``.pt`` files can be fed directly into ``RewardBinaryDataset`` for training,
exactly as described in :doc:`reward_model` Section 2.

Comparison of Data Collection Approaches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * -
     - Keyboard labeling
     - Fixed-pose (target-driven)
   * - **Labeling**
     - Manual per-frame (``c`` / ``a``)
     - Automatic (episode success/fail signal)
   * - **Episode termination**
     - Driven by keyboard wrapper
     - Driven by reaching ``target_ee_pose``
   * - **Success hold**
     - N/A
     - ``success_hold_steps`` to capture diverse successes
   * - **Output pipeline**
     - Direct .pt (one script)
     - ``.pkl`` episodes → ``preprocess_reward_dataset.py`` → .pt
   * - **Use case**
     - Any manipulation task
     - Tasks with a fixed target pose

Reward Model Training
---------------------

After completing the above steps, continue with Section 2
(**Reward Model Training**) in :doc:`reward_model` using the generated
``train.pt`` / ``val.pt`` files.

After training, you can use the trained reward model in two real-world ways:

- **Real-world teleoperation with live inference** (see below) — teleoperate the robot with
  SpaceMouse while the reward model runs on a GPU node, streaming real-time success
  probabilities to the terminal. No RL training loop is needed.
- **Real-world RL training** (see :doc:`../../examples/embodied/franka_reward_model`) —
  integrate the reward model into the full RL training loop on the physical Franka.

Real-World Teleoperation with Live Reward Inference
---------------------------------------------------

Once a reward model checkpoint is available, ``examples/reward/eval_realworld_teleop.py``
provides a teleoperation mode where SpaceMouse drives the robot while the reward model
runs on a GPU node, printing per-step success probabilities in real time.

This is useful for:

- Sanity-checking the reward model's accuracy on live robot observations.
- Collecting human-aligned success/fail data for further dataset expansion.
- Qualitatively evaluating whether the reward model generalizes to the current scene.

Cluster Configuration
---------------------

The teleop script requires **two nodes**: one for the Franka robot and one for the GPU
that runs the reward model inference:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       env:
         node_group: franka
         placement: 0
       reward:
         node_group: "4090"
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: franka
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

The reward worker is launched on the GPU node (``"4090"``) alongside the teleop worker
on the robot node (``franka``). This is a disaggregated placement — the reward model does
not share a node with the robot.

Configuration File
------------------

The default config is ``examples/reward/config/realworld_teleop.yaml``,
which inherits environment parameters from ``env/realworld_bin_relocation.yaml``:

.. code-block:: yaml

   defaults:
     - env/realworld_bin_relocation@env.eval
     - override hydra/job_logging: stdout

   cluster:
     num_nodes: 2
     component_placement:
       env:
         node_group: franka
         placement: 0
       reward:
         node_group: "4090"
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: franka
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

   env:
     group_name: "EnvGroup"
     eval:
       no_gripper: True
       use_spacemouse: True
       max_episode_steps: 10000
       override_cfg:
         target_ee_pose: TARGET_EE_POSE
         camera_serials: ["0123456789"]

   reward:
     use_reward_model: True
     use_reward_prob: True    # log raw sigmoid probs to terminal
     standalone_realworld: True
     reward_mode: "per_step"
     reward_threshold: 0.2
     model:
       model_path: path/to/reward_model_checkpoint
       model_type: "resnet"
       arch: "resnet18"
       image_size: [3, 128, 128]

Key fields for the reward model in teleop mode:

- ``reward.use_reward_model: True`` — enable reward model inference.
- ``reward.use_reward_prob: True`` — print raw sigmoid probabilities to the terminal each step.
- ``reward.standalone_realworld: True`` — use the reward model to directly drive success/failure and resets.
- ``reward.reward_threshold`` — probability below which success is suppressed. Adjust based on model calibration.
- ``reward.model.model_path`` — path to the trained reward model checkpoint.

Launching
---------

Set environment variables and run:

.. code-block:: bash

   bash examples/reward/run_realworld_teleop.sh

Or with an explicit config:

.. code-block:: bash

   bash examples/reward/run_realworld_teleop.sh realworld_teleop

The terminal prints per-step output:

.. code-block:: text

   [TeleopWorker] Starting teleoperation loop.
   [TeleopWorker] EmbodiedRewardWorker ready: type=EmbodiedRewardWorker | reward_threshold=0.200
   Step 0      | rm_reward: 0 | success: False
   Step 1      | rm_reward: 0 | success: False
   Step 10     | rm_reward: 0 | success: False
   Step 123    | rm_reward: 1 | success: True
   Step 124    | rm_reward: 1 | success: True

SpaceMouse controls:

- **Move** — teleoperate the robot arm.
- **Left button** — close gripper.
- **Right button** — open gripper.
- **Ctrl+C** — stop.

How It Works
------------

Inside ``TeleopWorker``:

1. ``RealWorldEnv`` is initialized with ``use_spacemouse=True``, wrapping the gym env with
   ``SpacemouseIntervention``. Non-zero SpaceMouse input (or a button press) overrides the
   zero dummy action for 0.5 seconds.
2. ``EmbodiedRewardWorker`` is launched on the GPU node via
   ``EmbodiedRewardWorker.launch_for_realworld(...)`` and initialized once at startup.
3. Each teleop step, the wrist camera image (``obs["main_images"]``) is extracted and sent
   to the reward worker for inference.
4. The raw sigmoid probability is printed to the terminal. When ``standalone_realworld=True``,
   the reward model also directly drives success/failure and triggers environment resets.

Compared with the full RL pipeline in :doc:`../../examples/embodied/franka_reward_model`,
the teleop script runs no policy, no actor, and no rollout worker — it is purely
human-in-the-loop evaluation of the reward model.
