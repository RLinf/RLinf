Using Reward Models with Franka
===============================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_reward_model.jpg
   :align: center
   :width: 80%

   Franka reward-model loop: collect labeled real-robot data, train a reward model, then let it produce the reward during real-world RL.

Replace the hand-coded success signal on a Franka arm with a learned reward model.
This page documents **two** reward-model types. They differ in what the model looks at,
what signal it produces, and how that signal reaches the RL loop — pick one before you
start collecting data, because the label format differs too.

.. list-table::
   :header-rows: 1
   :widths: 16 30 18 36

   * - Type
     - Reward signal
     - Reward model
     - Pick this when
   * - :ref:`ResNet <franka-resnet-reward-model>`
     - Per-frame success/failure probability that also drives environment resets.
     - ResNet image classifier
     - The task has a visually unambiguous success state and you can label frames for it.
   * - :ref:`Qwen VLM <franka-vlm-reward-model>`
     - Motion-trend judgment over a 5-frame sliding window, plus a success bonus.
     - Qwen3-VL-4B with LoRA
     - Success alone is too sparse and you want dense shaping toward the goal.

Overview
--------

Use a trained reward model as the real-world success signal for Franka tasks.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      CNN policy · ResNet reward model · Qwen3-VL-4B (LoRA)

   .. grid-item-card:: Algorithms
      :text-align: center

      SAC/RLPD · reward-model inference · VLM trend judgment

   .. grid-item-card:: Tasks
      :text-align: center

      Charger · Peg Insertion

   .. grid-item-card:: Hardware
      :text-align: center

      Franka · cameras · keyboard labels · dual RealSense

| **You'll do:** collect expert demos → collect reward labels → preprocess data → train reward model → launch real-world RL.
| **Prerequisites:** :doc:`franka` through data collection · :doc:`Reward model tutorial <../../extending/reward_model>`.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - Task
     - Config / entry point
     - Description
   * - Keyboard labels
     - ``realworld_collect_dataset``
     - Label success/failure frames during live teleoperation.
   * - Fixed pose labels
     - ``realworld_charger_sac_cnn_async_standalone_reward``
     - Use target-pose reachability to generate reward-model data.
   * - RL with reward model
     - ``realworld_charger_sac_cnn_async_standalone_reward``
     - Use reward-model success predictions in the Franka env.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - Field
     - Description
   * - Observation
     - Camera frames used by both policy and reward model.
   * - Action
     - Same Franka Cartesian action as the base real-world env.
   * - Reward
     - Reward-model success/failure prediction replaces the hand-coded success signal.
   * - Prompt
     - Task-specific env text or fixed target pose, depending on config.

.. _franka-resnet-reward-model:

ResNet Reward Model (Frame-Level Success)
-----------------------------------------

Train a ResNet classifier on labeled frames, then use its prediction as the real-world
success signal for Franka tasks.

Installation
~~~~~~~~~~~~

Follow all steps in the :doc:`franka` document up to and including **Data Collection** (i.e., everything before the "Running the Experiment" section).

Data Collection
~~~~~~~~~~~~~~~

Two types of data need to be collected: (1) expert trajectories for the demo buffer, and
(2) reward model training/evaluation data.

For the reward model dataset, two approaches are supported. For full details, see the
**Data Collection** section in :doc:`../../extending/reward_model`.
The core difference lies in the labeling method: Approach 1 uses manual keyboard labeling
and is task-agnostic; Approach 2 uses pose-based automatic labeling and is designed for
tasks with a fixed target pose.

Expert Trajectory Data Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
:doc:`../../extending/reward_model` for full configuration details.

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
See **Approach 2** in :doc:`../../extending/reward_model` for the full example.

Reward Model Training
~~~~~~~~~~~~~~~~~~~~~

This step is identical to **Section 2 — Reward Model Training** in :doc:`../../extending/reward_model`.

In particular, for real-world scenarios, it is recommended to lower the ``min_delta`` of ``early_stop``, for example:

.. code-block:: yaml

  runner:
    early_stop:
      min_delta: 1e-6

For real-world teleoperation with live reward model inference (SpaceMouse + GPU node, no RL loop),
see **Real-World Teleoperation with Live Reward Inference** in :doc:`../../extending/reward_model`.

Cluster Setup
~~~~~~~~~~~~~

This step is identical to the **Cluster Configuration** section under **Running the Experiment** in :doc:`franka`.

Configuration File
~~~~~~~~~~~~~~~~~~

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

Run It
~~~~~~

Once training begins, the reward model directly judges task success/failure based on image observations and drives environment resets.
The remaining steps follow the **Running the Experiment** section of :doc:`franka`.

Worker Interaction During Rollout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike **Section 3.2 — Worker Interaction During Rollout** and **Section 3.3 — Final Reward Computation** in :doc:`../../extending/reward_model`:
in real-world systems with ``standalone_realworld`` enabled, the reward model does **not** combine env rewards with reward model outputs.

In other words, the reward model does **not** act as an additional reward source inside the env worker when constructing the final reward,
because the system bypasses the weighted sum of ``env_reward`` and ``reward_model_output`` entirely.
Therefore, ``reward_mode``, ``reward_weight``, and ``env_reward_weight`` all have no effect.
The final reward is generated directly by FrankaEnv based on the reward model's success/failure determination.

From a system perspective, the actual behavior in the real-world system can be understood as:
directly replacing the ``env_reward`` inside the env worker, re-using the original ``env_reward`` logic to assign rewards and trigger environment resets, thereby fundamentally integrating the reward model.

.. _franka-vlm-reward-model:

Qwen VLM Reward Model (Action Trend Judgment)
---------------------------------------------

Unlike the ResNet reward model above, which classifies single frames as
"success" or "failure," the Qwen VLM reward model guides learning through
**action trend judgment**. Every 5 frames form a sliding history window, and the
Qwen3-VL model judges whether the robot's motion trend within the window is
``positive`` (moving toward the target), ``negative`` (moving away), or
``unclear`` (indeterminate), then converts the trend label into a scalar reward
for RL training.

.. code-block:: text

   VLM Output         Reward Value    Meaning
   ─────────────────────────────────────────────
   positive           1.0             Correct trend, positive signal
   negative           -0.2            Wrong trend, mild penalty
   unclear            0.0             Ambiguous trend, no signal
   invalid            0.0             Unparseable output, no signal

When the robot arm reaches the target pose and holds it (``terminated=True``),
``RealWorldEnv`` records ``success_once`` in the episode metrics passed to the
reward model. ``gt_success_bonus`` (default +20.0) adds a large bonus on top,
helping the RL agent strongly associate the success state with high reward.

Workflow
~~~~~~~~

The full pipeline has three stages:

1. **Data Collection** — Collect episodes with dual-view (wrist + global) image sequences on the real robot.
2. **Supervised Fine-Tuning (SFT)** — Preprocess collected data into VLM trend format and fine-tune Qwen3-VL-4B with LoRA.
3. **Real-World RL Training** — Integrate the fine-tuned VLM reward model into RLPD training with ``history_buffer`` mode for online inference.

Stage 1: Data Collection
~~~~~~~~~~~~~~~~~~~~~~~~

Use the real-world pipeline described in :doc:`franka` to collect episode data.
Enable ``data_collection`` to save each episode as a ``.pkl`` file:

.. code-block:: yaml

   env:
     eval:
       data_collection:
         enabled: True
         save_dir: /path/to/collected_data
         export_format: "pickle"
         only_success: False

Collection Tips
^^^^^^^^^^^^^^^

- **Move the robot arm slowly** so the collected data contains rich intermediate states for VLM trend learning.
- **Ensure both camera views are clear**: ``main_images`` (wrist camera) and ``extra_view_images`` (global camera) should clearly show the end-effector and the target hole.
- **Collect enough episodes** (50+ recommended), covering both successful and failed outcomes.
- **Correctly configure** ``camera_names``, matching ``SERIAL1`` / ``SERIAL2`` to the actual camera serial numbers:

.. code-block:: yaml

   env:
     train:
       override_cfg:
         camera_names:
           "SERIAL1": "global"    # Replace with your global camera serial
           "SERIAL2": "wrist_1"   # Replace with your wrist camera serial

Why two cameras:

- **Wrist camera** (``main_images``): Close-up view of the gripper and insertion hole for fine manipulation detail.
- **Global camera** (``extra_view_images``): Full workspace context for spatial awareness.

Dual-view input allows the VLM to simultaneously focus on local manipulation details
and global spatial relationships, improving trend judgment accuracy.

Launch collection with the standard real-world entry script after updating
``examples/embodiment/config/realworld_collect_data.yaml`` (camera serials,
``target_ee_pose``, and ``data_collection.save_dir``):

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

Stage 2: Supervised Fine-Tuning (SFT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preprocess into VLM Trend Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The collected ``.pkl`` episodes must be converted to VLM trend SFT format using
``preprocess_vlm_trend_reward_dataset.py``. Activate the virtual environment and
set ``PYTHONPATH`` before running:

.. code-block:: bash

   source .venv/bin/activate
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

This script slices episodes into 5-frame windows, extracts dual-view images,
and auto-labels based on GAE, reward, or TCP-distance changes:

.. code-block:: bash

   python examples/reward/preprocess_vlm_trend_reward_dataset.py \
       --raw-data-path /path/to/collected_data \
       --output-dir /path/to/processed_vlm_trend_reward_data \
       --window-size 5 \
       --task-description "Pick up the peg and insert it into the hole."

When collected data has no GAE signal, use ``--target-ee-pose`` to rely on
TCP-to-target distance instead of rewards as the trend signal:

.. code-block:: bash

   python examples/reward/preprocess_vlm_trend_reward_dataset.py \
       --raw-data-path /path/to/collected_data \
       --output-dir /path/to/processed_vlm_trend_reward_data \
       --window-size 5 \
       --target-ee-pose "X,Y,Z,RX,RY,RZ"

Replace ``X,Y,Z,RX,RY,RZ`` with your task target pose. To obtain it:

.. code-block:: bash

   python -m toolkits.realworld_check.test_franka_controller

Move the robot arm to the target position using SpaceMouse; the current TCP pose is printed to the terminal. If you only need position values, pass ``X,Y,Z`` (3 values) — orientation is ignored.

Output directory structure:

.. code-block:: text

   processed_vlm_trend_reward_data/
   ├── dataset_info.json
   ├── train/
   │   ├── segments.jsonl
   │   └── pkl/              # one pkl per window, contains main_frames + extra_view_frames
   └── eval/
       ├── segments.jsonl
       └── pkl/

Fine-Tune Qwen3-VL-4B
^^^^^^^^^^^^^^^^^^^^^

Update paths in the SFT config ``examples/sft/config/qwen3vl_sft_vlm_trend_reward.yaml``:

.. code-block:: yaml

   data:
     type: vlm
     dataset_name: "vlm_trend_reward_sft"
     train_data_paths: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}/train/segments.jsonl"
     val_data_paths: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}/eval/segments.jsonl"
     video_root: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}"

   actor:
     model:
       model_type: qwen3_vl
       model_path: /path/to/Qwen3-VL-4B-Instruct
       is_lora: true
       lora_rank: 16
       attn_implementation: flash_attention_2

   fsdp_config:
     gradient_checkpointing: true     # save GPU memory
     sharding_strategy: no_shard      # for RTX 4090 single-GPU

Set the environment variable and start training:

.. code-block:: bash

   export VLM_TREND_REWARD_DATA_ROOT=/path/to/processed_vlm_trend_reward_data
   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_vlm_trend_reward

After training, note the LoRA checkpoint path (e.g., ``checkpoints/global_step_3000``)
for use as ``reward.model.lora_path`` in the RL config.

Stage 3: Real-World Reinforcement Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration File
^^^^^^^^^^^^^^^^^^

Use ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_vlm_reward.yaml``
as the RL training config. The core reward section:

.. code-block:: yaml

   reward:
     use_reward_model: true            # Enable reward model
     worker_type: model                # Local HuggingFace inference
     group_name: "RewardGroup"
     standalone_realworld: False
     reward_mode: history_buffer       # Trend judgment via history window
     history_reward_assign: true       # Back-assign VLM reward to history steps
     reward_weight: 1.0                # VLM reward weight
     env_reward_weight: 0.0            # Native env reward weight (VLM-only)
     reward_threshold: 0.5

     model:
       model_path: "/path/to/Qwen3-VL-4B-Instruct"
       model_type: "buffered_vlm"
       lora_path: "/path/to/sft_output/checkpoints"
       gt_success_bonus: 20.0
       precision: "bf16"

       input_builder_name: vlm_trend_reward_input_builder
       input_builder_params:
         default_task_description: "Pick up the peg and insert it into the hole."

       reward_parser_name: vlm_trend_reward_parser
       reward_parser_params:
         positive_reward: 1.0
         negative_reward: -0.2
         unclear_reward: 0.0
         invalid_reward: 0.0

       history_buffers:
         history_window:
           history_size: 5
           min_history_size: 5
           input_interval: 1
           history_keys:
             - main_images
             - extra_view_images
           input_on_done: false

       interval_reward: 0.0
       max_new_tokens: 16
       do_sample: false
       temperature: 0.0

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: "4090"
       env:
         node_group: franka
       reward:
         node_group: "4090"

Key Configuration Fields
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - ``reward_mode: history_buffer``
     - The env worker maintains a sliding window per environment and sends data to the VLM only when ``min_history_size`` frames are accumulated.
   * - ``history_reward_assign: true``
     - Back-assigns the VLM trend reward to every step within the history window.
   * - ``env_reward_weight: 0.0``
     - The Franka native reward is sparse (1.0 only at target). Training relies primarily on VLM trend judgment. Success is rewarded via ``gt_success_bonus``.
   * - ``gt_success_bonus: 20.0``
     - Adds +20.0 when ``env_infos["episode"]["success_once"]`` reports success.
   * - ``history_buffers.history_window.history_keys``
     - ``[main_images, extra_view_images]`` — the history window caches frames from both camera views for VLM inference.
   * - ``history_buffers.history_window``
     - Caches the last 5 frames of ``main_images`` and ``extra_view_images``, triggers inference only after at least 5 frames.
   * - ``worker_type: model``
     - Loads the model directly in the reward worker for local HuggingFace inference.

Reward Computation Flow
^^^^^^^^^^^^^^^^^^^^^^^

At each RL training step, the final reward is composed through the following pipeline:

.. code-block:: text

   FrankaEnv.step()
     -> native reward and termination
          |
          v
   RealWorldEnv.step()
     -> env_infos["episode"]["success_once"]
          |
          v
   EnvWorker.get_reward_model_output()
     -> HistoryManager builds history_input
     -> reward worker receives reward_input
          |
          v
   EmbodiedRewardWorker.compute_image_rewards()
     -> BufferedVLMRewardModel.compute_reward()
        -> incomplete history: 0.0
        -> complete history: Qwen3-VL parses to 1.0 / -0.2 / 0.0
     -> apply_gt_success_bonus(): success_once adds 20.0
          |
          v
   final_reward = env_reward_weight * env_reward
                + reward_weight * vlm_reward_with_bonus

Success Signal
^^^^^^^^^^^^^^

``FrankaEnv`` returns a sparse reward of 1.0 when the target is reached.
``RealWorldEnv`` converts that reward into the sticky episode metric
``success_once``. The env worker forwards the episode metrics to the reward
model, where ``apply_gt_success_bonus`` reads ``success_once``.

Starting Training
^^^^^^^^^^^^^^^^^

Once hardware deployment and configuration are verified, run on the Ray head node:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh \
       realworld_peginsertion_rlpd_cnn_async_vlm_reward

After training starts, the logs will show VLM inference outputs, reward
distributions, and success signals.
