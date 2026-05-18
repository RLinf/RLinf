Real-World RL with Franka (Reward Model)
=========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document describes how to use a reward model when training on a Franka robotic arm in the real world.
The focus is on training and deploying a ResNet-based reward model from scratch to assist robotic manipulation tasks.

Before getting started, it is strongly recommended to read the following documents:

1. :doc:`franka` — to familiarize yourself with the end-to-end real-world Franka training pipeline.
2. :doc:`../../tutorials/extend/reward_model` — to understand the complete reward model workflow in RLinf's simulated environments.

Prerequisites
-----------------------

Follow all steps in the :doc:`franka` document up to and including **Data Collection** (i.e., everything before the "Running the Experiment" section).

Data Collection
-----------------------

Two types of data need to be collected: (1) expert trajectories for the demo buffer, and
(2) reward model training/evaluation data.

Expert Trajectory Data Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expert trajectory data is collected first and stored in the demo buffer during training.
Follow the steps in the **Data Collection** section under **Running the Experiment** in
:doc:`franka`. Make sure that in ``examples/embodiment/config/realworld_collect_data.yaml``,
``data_collection`` under the ``env`` section is enabled:

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: True

Reward Model Dataset Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collecting reward model training and evaluation data supports two approaches.
For full details, see the **Data Collection** section in
:doc:`../../tutorials/extend/reward_model_realworld`.
The core difference lies in the labeling method: Approach 1 uses manual keyboard labeling
and is task-agnostic; Approach 2 uses pose-based automatic labeling and is designed for
tasks with a fixed target pose.

Approach 1: Keyboard Labeling (General-Purpose)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This approach manually labels each frame during a live episode via keyboard keys.
It is task-agnostic and works for any manipulation task. It combines data collection,
labeling, and dataset generation into one end-to-end run with no separate offline preprocessing.

**Key configuration:**

- ``runner.num_success_frames`` / ``runner.num_fail_frames`` — target numbers of frames;
  collection stops when both thresholds are reached.
- ``runner.val_split`` — fraction of labeled frames held out for validation.
- ``runner.fail_success_ratio`` — fail-frame downsampling ratio during training-set post-processing.
- ``env.eval.keyboard_reward_wrapper`` — set to ``single_stage`` to enable the keyboard interface.
- ``env.eval.use_spacemouse`` — whether SpaceMouse is used for teleoperation.
- ``env.eval.override_cfg.target_ee_pose`` — the target end-effector pose for the task.

**Launching:**

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh realworld_collect_dataset

**Key bindings:**

- ``c`` — label the current frame as **success**.
- ``a`` — label the current frame as **fail**.

Once the target frame counts are reached, the script automatically stops, splits the data,
and saves ``train.pt`` / ``val.pt``. See **Approach 1** in
:doc:`../../tutorials/extend/reward_model_realworld` for full configuration details.

Approach 2: Fixed-Pose (Target-Driven)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This approach is designed for tasks with a **fixed target pose**. No manual keyboard
labeling is required — the episode automatically drives success/failure based on whether
the robot reaches the configured ``target_ee_pose``. ``success_hold_steps`` can be set to
require the robot to maintain the pose for a number of steps before declaring success,
which helps collect more diverse successful samples. It uses a streamlined two-step pipeline.

**Step 1: Fixed-Pose Reward Data Collection**

On top of the expert trajectory collection, increase the ``success_hold_steps`` field:

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         success_hold_steps: 20

Collection tips:

- Move the robot arm slowly to obtain more diverse failure samples.
- When reaching the target pose, make small-range movements while maintaining the pose
  to obtain more diverse successful samples.

**Step 2: Preprocessing into a Reward Dataset**

Run ``preprocess_reward_dataset.py`` to convert ``.pkl`` episodes into ``.pt`` files.
It is recommended to set ``fail-success-ratio`` to ``3``:

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data \
       --fail-success-ratio 3

The resulting ``.pt`` files follow the ``RewardDatasetPayload`` schema, containing
``images``, ``labels`` (1 = success, 0 = fail), and ``metadata``.
See **Approach 2** in :doc:`../../tutorials/extend/reward_model_realworld` for the full example.

Reward Model Training
-----------------------

This step is identical to **Section 2 — Reward Model Training** in :doc:`../../tutorials/extend/reward_model`.

In particular, for real-world scenarios, it is recommended to lower the ``min_delta`` of ``early_stop``, for example:

.. code-block:: yaml

  runner:
    early_stop:
      min_delta: 1e-6

For real-world teleoperation with live reward model inference (SpaceMouse + GPU node, no RL loop),
see **Real-World Teleoperation with Live Reward Inference** in :doc:`../../tutorials/extend/reward_model_realworld`.

Cluster Configuration
-----------------------

This step is identical to the **Cluster Configuration** section under **Running the Experiment** in :doc:`franka`.

Configuration File
-----------------------

This step is identical to the **Configuration File** section under **Running the Experiment** in :doc:`franka`, applied to ``examples/embodiment/config/realworld_charger_sac_cnn_async_standalone_reward.yaml``.
In addition, enable the reward model parameters under the ``reward`` section:

.. code-block:: yaml

   reward:
     use_reward_model: True
     group_name: "RewardGroup"
     standalone_realworld: True
     reward_mode: "per_step"
     reward_threshold: 0.8

     model:
       model_path: /path/to/reward_model_checkpoint
       model_type: "resnet"

Where:

- ``reward_mode`` controls whether the reward model runs inference at every step or only on terminal frames.
- ``standalone_realworld`` uses the reward model to directly determine task success and trigger environment resets.
- ``reward_threshold`` applies threshold filtering on the success probability output by the reward model; values below the threshold are set to ``0``.
- ``model_path`` points to the reward model checkpoint used for online inference.

Starting the Experiment
-----------------------

Once training begins, the reward model directly judges task success/failure based on image observations and drives environment resets.
The remaining steps follow the **Running the Experiment** section of :doc:`franka`.

Worker Interaction During Rollout
----------------------------------------------

Unlike **Section 3.2 — Worker Interaction During Rollout** and **Section 3.3 — Final Reward Computation** in :doc:`../../tutorials/extend/reward_model`:
in real-world systems with ``standalone_realworld`` enabled, the reward model does **not** combine env rewards with reward model outputs.

In other words, the reward model does **not** act as an additional reward source inside the env worker when constructing the final reward,
because the system bypasses the weighted sum of ``env_reward`` and ``reward_model_output`` entirely.
Therefore, ``reward_mode``, ``reward_weight``, and ``env_reward_weight`` all have no effect.
The final reward is generated directly by FrankaEnv based on the reward model's success/failure determination.

From a system perspective, the actual behavior in the real-world system can be understood as:
directly replacing the ``env_reward`` inside the env worker, re-using the original ``env_reward`` logic to assign rewards and trigger environment resets, thereby fundamentally integrating the reward model.
