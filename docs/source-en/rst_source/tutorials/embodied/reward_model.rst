Reward Model Guide
==================

This document describes how to use reward models in RLinf, covering both simulation-based
and real-world robot workflows. It covers image classification rewards such as
``ResNetRewardModel`` and VLM rewards such as QwenTrend / ``HistoryVLMRewardModel``.

.. contents::
   :depth: 2
   :local:

Simulation Reward Model
-----------------------

The full simulation workflow has four stages:

1. Data collection: collect raw episode data during RL runs.
2. Dataset conversion: convert raw episodes into either image classification data or VLM SFT data.
3. Reward model training: train a ResNet reward model or fine-tune a VLM reward model.
4. Reward model inference in RL: plug the trained model into online rollout and use it in final reward computation.

1. Data Collection
^^^^^^^^^^^^^^^^^^

Reward model training data is typically built from episode-level data collection. RLinf provides
a unified collection wrapper, and the related usage is documented in :doc:`the data collection tutorial <data_collection>`.

For reward model use cases, we recommend saving raw episodes in ``pickle`` format first, then converting
them into processed training splits with the preprocessing script.

Enable Data Collection
""""""""""""""""""""""

Enable ``data_collection`` under ``env`` in your YAML config:

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: False

After training or evaluation starts, the environment will automatically save episodes into ``save_dir``.
When ``export_format="pickle"``, each episode is written as an individual ``.pkl`` file for later offline preprocessing.

For QwenTrend VLM rewards, RLinf also provides a ready-to-run collection config:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_collect

This config keeps ``reward.use_reward_model: false`` and enables data collection on the
evaluation environment.

Preprocess into a ResNet Reward Dataset
"""""""""""""""""""""""""""""""""""""""

Raw ``pickle`` files cannot be consumed by reward model training directly. Use
``examples/reward/preprocess_reward_dataset.py`` to convert collected ``.pkl`` episodes into
``.pt`` files that can be loaded by ``RewardBinaryDataset``.

Example:

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data

By default, this produces ``train.pt`` and ``val.pt`` following the ``RewardDatasetPayload`` schema:

.. code-block:: python

   {
       "images": list[torch.Tensor],
       "labels": list[int],
       "metadata": dict[str, Any],
   }

Convert into a QwenTrend VLM Dataset
""""""""""""""""""""""""""""""""""""

QwenTrend uses short dual-view history windows rather than single images. Use
``examples/reward/preprocess_qwentrend_reward_dataset.py`` to slice collected
episodes into 5-frame windows.

Example:

.. code-block:: bash

   python examples/reward/preprocess_qwentrend_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_qwentrend_reward_data \
       --window-size 5 \
       --stride 1 \
       --delta-threshold 0.05

2. Reward Model Training
^^^^^^^^^^^^^^^^^^^^^^^^

RLinf supports two reward training paths.

Fine-Tune the ResNet Reward Model
"""""""""""""""""""""""""""""""""

Before training, edit ``examples/reward/config/reward_training.yaml`` so it points to your processed splits:

.. code-block:: yaml

   data:
     train_data_paths: "logs/processed_reward_data/train.pt"
     val_data_paths: "logs/processed_reward_data/val.pt"

For the ResNet path, set ``actor.model.model_type`` to ``"resnet"``:

.. code-block:: yaml

   actor:
     model:
       model_type: "resnet"
       arch: "resnet18"
       pretrained: False
       image_size: [3, 128, 128]

The online reward-worker registry contains the following model types:

.. code-block:: python

   reward_model_registry = {
       "resnet": ResNetRewardModel,
       "vlm": VLMRewardModel,
       "history_vlm": HistoryVLMRewardModel,
   }

Launch training:

.. code-block:: bash

   bash examples/reward/run_reward_training.sh

Fine-Tune the QwenTrend VLM Reward Model
""""""""""""""""""""""""""""""""""""""""

After converting collected episodes with ``preprocess_qwentrend_reward_dataset.py``,
launch VLM SFT:

.. code-block:: bash

   export DUALVIEW_SFT_DATA_ROOT=/path/to/processed_qwentrend_reward_data
   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_qwentrend

3. Reward Model Inference in RL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RLinf provides several example configs for integrating a reward model into RL training.
These configs show how to enable a reward worker while keeping the policy on state observations
and the reward model on image or VLM observations.

Key Config Fields
"""""""""""""""""

.. code-block:: yaml

   reward:
     use_reward_model: True
     group_name: "RewardGroup"
     reward_mode: "terminal"   # or "per_step" / "history_buffer"
     reward_threshold: 0.5
     reward_weight: 1.0
     env_reward_weight: 0.0

     model:
       model_path: /path/to/reward_model_checkpoint
       model_type: "resnet"    # or "vlm" / "history_vlm"

Where:

- ``reward_mode`` accepts ``"per_step"``, ``"terminal"``, or ``"history_buffer"``.
- ``reward_weight`` and ``env_reward_weight`` control how learned reward and environment reward are combined.
- ``reward_threshold`` filters reward model probabilities.

Worker Interaction During Rollout
"""""""""""""""""""""""""""""""""

During online RL, the ``env``, ``rollout``, and ``reward`` workers collaborate as follows:

.. code-block:: text

   Env worker
      | 1. Interacts with the environment and gets obs / env reward / done
      | 2. Sends obs to the Rollout worker to produce actions
      | 3. When reward model is enabled, sends a reward input dict to the Reward worker
      v
   Reward worker
      | 4. Runs ``compute_reward(...)`` and returns reward model output
      v
   Env worker
      | 5. Receives bootstrap values from the Rollout worker
      | 6. Combines env reward with reward model output
      v
   Final reward -> stored in rollout results and used by later RL updates

Final Reward Computation
""""""""""""""""""""""""

When the reward channel is enabled, the final reward is computed as:

.. code-block:: python

   reward = env_reward_weight * env_reward + reward_weight * reward_model_output

Deploy QwenTrend for MLP RL
"""""""""""""""""""""""""""

For VLM reward inference, install embodied dependencies with VLM reward support:

.. code-block:: bash

   bash requirements/install.sh embodied --env maniskill_libero --vlm-reward

Then configure the reward section to use ``history_vlm``. Launch with:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_reward


Real-World Reward Model
-----------------------

This section describes how to collect and preprocess a reward model training dataset
directly on a real-world Franka robot. Two data collection approaches are supported:
a **general-purpose keyboard-labeling** approach and a **fixed-pose** approach that
uses a predetermined target pose to drive episode success/failure.

Before getting started, it is strongly recommended to read:

1. :doc:`../../examples/embodied/franka` — to familiarize yourself with the end-to-end real-world Franka training pipeline.
2. :doc:`../../examples/embodied/franka_reward_model` — to understand the full real-world RL pipeline that follows after you have a trained reward model.

Workflow Overview
^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^

Follow the **Prerequisites** and **Hardware Setup** sections in :doc:`../../examples/embodied/franka`
up to and including the robot connection and environment validation steps.

Approach 1: Keyboard Labeling (General-Purpose)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This approach uses keyboard keys to manually label each frame during a live episode.
It is task-agnostic and works for any manipulation task.

**Configuration file** — ``examples/reward/config/realworld_collect_dataset.yaml``:

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
     num_success_frames: 50
     num_fail_frames: 150
     val_split: 0.2
     fail_success_ratio: 2.0
     random_seed: 42

   env:
     eval:
       no_gripper: False
       use_spacemouse: True
       max_episode_steps: 10000
       keyboard_reward_wrapper: single_stage

**Key configuration fields:**

- ``runner.num_success_frames`` / ``runner.num_fail_frames`` — target numbers of labeled frames.
- ``runner.val_split`` — fraction of frames held out as validation.
- ``runner.fail_success_ratio`` — downsampling factor for fail frames.
- ``env.eval.keyboard_reward_wrapper`` — enables the keyboard labeling interface.
- ``env.eval.use_spacemouse`` — whether SpaceMouse is used for teleoperation.

**Launching:**

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh

Use the following keys during the episode:

- ``c`` — label the current frame as **success**.
- ``a`` — label the current frame as **fail**.

When both ``num_success_frames`` and ``num_fail_frames`` are reached, the script
automatically stops, splits the data, and saves the ``.pt`` files.

Approach 2: Fixed-Pose (Target-Driven)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This approach is for tasks with a **fixed target pose** (e.g., reaching a predetermined bin location).
Instead of manual keyboard labeling, the episode automatically drives success/failure based on
whether the robot reaches the configured ``target_ee_pose``.

Step 1: Fixed-Pose Reward Data Collection
"""""""""""""""""""""""""""""""""""""""""

Increase ``success_hold_steps`` to obtain more diverse successful data:

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         success_hold_steps: 20

Collection tips:

- Move the robot arm slowly to obtain more diverse failure samples.
- When reaching the target pose, make small-range movements while maintaining the pose.

Step 2: Preprocessing into a Reward Dataset
"""""""""""""""""""""""""""""""""""""""""""

Convert collected ``.pkl`` episodes into ``train.pt`` / ``val.pt``:

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data \
       --fail-success-ratio 3

Comparison of Data Collection Approaches
""""""""""""""""""""""""""""""""""""""""

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
^^^^^^^^^^^^^^^^^^^^^

After completing the above steps, continue with **2. Reward Model Training** in the
Simulation Reward Model section above, using the generated ``train.pt`` / ``val.pt`` files.

Real-World Teleoperation with Live Reward Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a reward model checkpoint is available, ``examples/reward/eval_realworld_teleop.py``
provides a teleoperation mode where SpaceMouse drives the robot while the reward model
runs on a GPU node, printing per-step success probabilities in real time.

This is useful for:

- Sanity-checking the reward model's accuracy on live robot observations.
- Collecting human-aligned success/fail data for further dataset expansion.
- Qualitatively evaluating whether the reward model generalizes to the current scene.

The teleop script requires **two nodes**: one for the Franka robot and one for the GPU.

Key fields for the reward model in teleop mode:

.. code-block:: yaml

   reward:
     use_reward_model: True
     use_reward_prob: True    # log raw sigmoid probs to terminal
     standalone_realworld: True
     reward_mode: "per_step"
     reward_threshold: 0.2
     model:
       model_path: path/to/reward_model_checkpoint
       model_type: "resnet"

Launching:

.. code-block:: bash

   bash examples/reward/run_realworld_teleop.sh

SpaceMouse controls:

- **Move** — teleoperate the robot arm.
- **Left button** — close gripper.
- **Right button** — open gripper.
- **Ctrl+C** — stop.

Compared with the full RL pipeline in :doc:`../../examples/embodied/franka_reward_model`,
the teleop script runs no policy, no actor, and no rollout worker — it is purely
human-in-the-loop evaluation of the reward model.

Summary
-------

The full workflow is:

1. Enable ``data_collection`` in the environment config and save raw data in ``pickle`` format.
2. For ResNet rewards, use ``preprocess_reward_dataset.py`` to build ``train.pt`` / ``val.pt`` and train with ``run_reward_training.sh``.
3. For QwenTrend VLM rewards, use ``preprocess_qwentrend_reward_dataset.py`` to build dual-view history-window data and fine-tune with ``run_vlm_sft.sh``.
4. Enable ``reward.use_reward_model=True`` in your RL YAML and plug the trained reward worker into online RL inference.
5. For real-world scenarios, use keyboard labeling or fixed-pose approaches to collect data, then train and deploy the reward model on the physical robot.
