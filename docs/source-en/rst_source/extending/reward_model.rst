Reward Model Guide
==================

Use reward models in RLinf — both image-classification rewards such as
``ResNetRewardModel`` and VLM rewards such as QwenTrend /
``HistoryVLMRewardModel``.
Here, QwenTrend means using a Qwen3-VL model to judge the action trend in a short
history video and convert that judgment into a scalar reward.

Simulation Reward Model
-----------------------
Recommended QwenTrend Success Pipeline
--------------------------------------

Use terminal-success classification when you need a reward that is stable enough
for PPO. This route keeps the VLM task-independent: the task text comes from each
episode, while labels come only from environment success and trajectory timing.
The older ``positive`` / ``negative`` / ``unclear`` trend route remains available
below, but it is not the recommended PPO configuration.

1. Collect one 50-step rollout from checkpoints that cover the policy distribution:

.. code-block:: bash

   export CHECKPOINT_TEMPLATE_EARLY='/path/to/clean_gt_0_120/checkpoints/global_step_%d/actor/model_state_dict/full_weights.pt'
   export CHECKPOINT_TEMPLATE_LATE='/path/to/clean_gt_0_200/checkpoints/global_step_%d/actor/model_state_dict/full_weights.pt'
   export OUTPUT_ROOT=/path/to/qwentrend_uniform_collection
   export CUDA_DEVICES=0,1,2,3
   export PLACEMENT=0-3
   NUM_ENVS=1024 SEED=0 \
       bash examples/embodiment/qwentrend_success/run_collect_uniform_checkpoints.sh

The default checkpoint steps are ``0, 20, ..., 200``. Step 0 uses a randomly
initialized policy; steps 20--120 use the early template and steps 140--200 use
the late template. Each checkpoint uses seed 0 and one four-GPU Ray evaluation
job. Rollouts use ``simple`` observations, ignore early termination, and run for
50 environment steps, so an early success cannot silently produce a short
negative example. To replace a failed checkpoint, rerun the same collection
command with, for example,
``STEPS="80 120 160"``; matching episode filenames are overwritten.

2. Build 5-frame, dual-view binary SFT samples:

.. code-block:: bash

   export UNIFORM_DATA_ROOT=/path/to/qwentrend_uniform_collection
   export DUALVIEW_SFT_DATA_ROOT=/path/to/qwentrend_success_sft
   bash examples/embodiment/qwentrend_success/run_preprocess_success_dataset.sh

The preprocessor splits by source episode before creating windows and exits
nonzero if any episode appears in both train and eval. The default online-aligned
mode emits one 5-frame window at each post-action observation index ``5, 10, ..., 50``. Index 0 is the reset observation and is never used as the first online window. Each window is labeled from ``infos[end_step]["success"]``; the natural class ratio is preserved and no near-terminal positive or sampled hard negative is added. Set ``UNIFORM_STEPS`` to preprocess a subset while debugging; the default remains checkpoints 0 through 200. A validated step-100 pilot produced 2,700/6,560 train positives/negatives and 221/759 eval positives/negatives.
Manifests reference the original episode pickle and frame range instead of
copying images. Only load trusted pickle data because Python pickle can execute
code during deserialization.

3. Fine-tune Qwen3-VL-4B with the validated LoRA recipe:

.. code-block:: bash

   export DUALVIEW_SFT_DATA_ROOT=/path/to/qwentrend_success_sft
   export QWEN_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
   export PLACEMENT=0-3
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
       bash examples/embodiment/qwentrend_success/run_train_success_vlm.sh

The launcher uses LoRA rank 16, learning rate ``1e-5``, micro batch size 4,
global batch size 256, ``format_ce_coef=2``, class-ratio-weighted success loss,
and 400 optimization steps. It evaluates and saves every 100 steps. Select the
checkpoint with the highest balanced accuracy while checking positive recall
and negative accuracy separately; do not use aggregate accuracy alone.

4. Run PPO from the policy checkpoint:

.. code-block:: bash

   export QWEN_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
   export QWENTREND_SUCCESS_CHECKPOINT=/path/to/global_step_300
   export QWENTREND_POTENTIAL_CHECKPOINT=/path/to/potential/global_step_400
   export QWENTREND_SCALAR_HEAD=/path/to/scalar_head/best.pt
   export POLICY_CHECKPOINT=/path/to/policy/full_weights.pt
   export PPO_OUTPUT_ROOT=/path/to/ppo-output
   CUDA_VISIBLE_DEVICES=0,1,2,3 PLACEMENT=0-3 NUM_ENVS=128 MAX_STEPS=160 \
       INFER_BATCH_SIZE=32 \
       bash examples/embodiment/run_qwentrend_success_reward.sh

Online inference uses the same prompt and ``0`` / ``1`` generation contract as
SFT. The potential checkpoint and scalar head produce the bounded dense shaping
term, while the independently trained success LoRA generates the sparse success
term. A generated ``1`` contributes a one-shot ``+1`` bonus; ``0`` and invalid
output contribute zero. Environment reward is disabled, inference runs every
five steps, and fixed-50-step training keeps the online distribution aligned
with preprocessing.

The PPO reward combines bounded Qwen potential differences with the generated
success bonus. To continue a run, set ``RESUME_DIR`` to a saved
``checkpoints/global_step_N`` directory and set ``MAX_STEPS`` to the desired
total step, for example 160 when resuming from step 60.

The launcher evaluates every 5 steps and saves every 20 steps. It fixes actor,
rollout, environment, and reward placement to GPUs 0-3 by default. Override
``PLACEMENT`` only when the hardware layout differs.
The online Qwen micro-batch defaults to 32. The launcher uses the main-branch
``history_buffer`` dispatch mode; the binary parser still makes the effective
reward sparse by mapping only generated ``1`` to ``+1``.


The full workflow has four stages:

1. Data collection: collect raw episode data during RL runs.
2. Dataset conversion: convert raw episodes into either image classification data or VLM SFT data.
3. Reward model training: train a ResNet reward model or fine-tune a VLM reward model.
4. Reward model inference in RL: plug the trained model into online rollout and use it in final reward computation.

1. Data Collection
^^^^^^^^^^^^^^^^^^

Reward model training data is typically built from episode-level data collection. RLinf provides
a unified collection wrapper, and the related usage is documented in :doc:`the data collection tutorial <../guides/data_collection>`.

For reward model use cases, we recommend saving raw episodes in ``pickle`` format first, then converting
them into processed training splits with the preprocessing script.

1.1 Enable Data Collection
""""""""""""""""""""""""""

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
evaluation environment. The saved episodes include the dual-view image observations
used later by the VLM pipeline, such as ``main_images`` and ``extra_view_images``.

1.2 Preprocess into a ResNet Reward Dataset
"""""""""""""""""""""""""""""""""""""""""""

Raw ``pickle`` files cannot be consumed by reward model training directly. Use
``examples/reward/preprocess_reward_dataset.py`` to convert collected ``.pkl`` episodes into
``.pt`` files that can be loaded by ``RewardBinaryDataset``. In the current implementation,
the script extracts ``main_images`` from observations and builds binary labels from per-step
``info["success"]``.

Example:

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data

By default, this produces:

.. code-block:: text

   logs/xxx/processed_reward_data/
   ├── train.pt
   └── val.pt

The generated ``.pt`` files follow the canonical ``RewardDatasetPayload`` schema:

.. code-block:: python

   {
       "images": list[torch.Tensor],
       "labels": list[int],
       "metadata": dict[str, Any],
   }

Where:

- ``images`` stores the training images.
- ``labels`` stores the binary labels.
- ``metadata`` stores source path, sampling arguments, split ratio, and related preprocessing info.

``RewardBinaryDataset`` then loads these ``train.pt`` / ``val.pt`` files directly.

1.3 Convert into a QwenTrend VLM Dataset
""""""""""""""""""""""""""""""""""""""""

QwenTrend uses short dual-view history windows rather than single images. Use
``examples/reward/preprocess_qwentrend_reward_dataset.py`` to slice collected
episodes into 5-frame windows, extract ``main_images`` and ``extra_view_images``,
and assign each window one of ``positive``, ``negative``, or ``unclear``.

Example:

.. code-block:: bash

   python examples/reward/preprocess_qwentrend_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_qwentrend_reward_data \
       --window-size 5 \
       --stride 1 \
       --delta-threshold 0.05

By default, this produces JSONL manifests and per-sample pickle files:

.. code-block:: text

   logs/xxx/processed_qwentrend_reward_data/
   ├── dataset_info.json
   ├── train/
   │   ├── segments.jsonl
   │   └── pkl/
   └── eval/
       ├── segments.jsonl
       └── pkl/

The train/eval split is done by episode, so windows from the same episode are not
mixed across splits.

2. Reward Model Training
^^^^^^^^^^^^^^^^^^^^^^^^

RLinf supports two reward training paths. ``examples/reward/run_reward_training.sh``
trains the ResNet image reward model, while ``examples/sft/run_vlm_sft.sh``
fine-tunes a VLM reward model such as QwenTrend.

2.1 Fine-Tune the ResNet Reward Model
"""""""""""""""""""""""""""""""""""""

2.1.1 Configure ResNet Dataset Paths
........................................

Before training, edit ``examples/reward/config/reward_training.yaml`` so it points to your processed splits:

.. code-block:: yaml

   data:
     train_data_paths: "logs/processed_reward_data/train.pt"
     val_data_paths: "logs/processed_reward_data/val.pt"

.. note::

   At present, ``run_reward_training.sh`` mainly prepares the launch command and log directory.
   The dataset paths are taken from ``reward_training.yaml``, specifically
   ``data.train_data_paths`` and ``data.val_data_paths``.

2.1.2 Configure the ResNet Model
....................................

For the ResNet path, set ``actor.model.model_type`` to ``"resnet"``:

.. code-block:: yaml

   actor:
     model:
       model_type: "resnet"
       arch: "resnet18"
       pretrained: False
       image_size: [3, 128, 128]

If you want to continue training from existing weights, set ``model_path`` to a checkpoint.
If you want to train from scratch, keep ``model_path: null``.

The online reward-worker registry currently contains the following model types:

.. code-block:: python

   reward_model_registry = {
       "resnet": ResNetRewardModel,
       "vlm": VLMRewardModel,
       "history_vlm": HistoryVLMRewardModel,
   }

``resnet`` is the image classifier path. ``vlm`` runs a VLM on the current
observation. ``history_vlm`` runs a VLM on history windows built by the env worker.

2.1.3 Launch ResNet Training
................................

Once the dataset and model are configured, run:

.. code-block:: bash

   bash examples/reward/run_reward_training.sh

Training logs are written to a newly created ``logs/<timestamp>-reward_training`` directory.

2.2 Fine-Tune the QwenTrend VLM Reward Model
""""""""""""""""""""""""""""""""""""""""""""

After converting collected episodes with ``preprocess_qwentrend_reward_dataset.py``,
point ``DUALVIEW_SFT_DATA_ROOT`` to the processed output root and launch VLM SFT:

.. code-block:: bash

   export DUALVIEW_SFT_DATA_ROOT=/path/to/processed_qwentrend_reward_data
   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_qwentrend

The corresponding config reads the JSONL manifests and per-sample pickle files:

.. code-block:: yaml

   data:
     type: vlm
     dataset_name: "qwentrend_progress_sft"
     train_data_paths: "${oc.env:DUALVIEW_SFT_DATA_ROOT}/train/segments.jsonl"
     val_data_paths: "${oc.env:DUALVIEW_SFT_DATA_ROOT}/eval/segments.jsonl"
     video_root: "${oc.env:DUALVIEW_SFT_DATA_ROOT}"
     video_nframes: 5

   actor:
     model:
       model_type: qwen3_vl
       model_path: /path/to/Qwen3-VL-4B-Instruct
       attn_implementation: flash_attention_2
       is_lora: true
       lora_rank: 16

The trained LoRA checkpoint can then be passed to the online reward config through
``reward.model.lora_path``.

3. Reward Model Inference in RL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RLinf provides several example configs for integrating a reward model into RL:

- ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``
- ``examples/embodiment/config/maniskill_sac_mlp_resnet_reward_async.yaml``
- ``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml``

These configs show how to enable a reward worker in RL training while keeping the policy on state observations
and the reward model on image or VLM observations.

3.1 Key Config Fields
"""""""""""""""""""""

Reward-model-related settings live under the ``reward`` section:

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

- ``reward_mode`` accepts ``"per_step"``, ``"terminal"``, or ``"history_buffer"``: run inference every step, only on terminal frames, or on history windows.
- ``reward_weight`` and ``env_reward_weight`` control how learned reward and environment reward are combined.
- ``reward_threshold`` filters reward model probabilities; values below the threshold are set to ``0``.
- ``model_path`` points to the reward model checkpoint used for online inference.

3.2 Worker Interaction During Rollout
"""""""""""""""""""""""""""""""""""""

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

In the implementation, ``EnvWorker`` requests reward model outputs during rollout and then computes the final reward centrally.

3.3 Final Reward Computation
""""""""""""""""""""""""""""

When the reward channel is enabled, ``EnvWorker`` first fetches ``reward_model_output``,
then merges it with the original environment reward inside ``compute_bootstrap_rewards``:

.. code-block:: python

   reward = env_reward_weight * env_reward + reward_weight * reward_model_output

If bootstrap is enabled by the algorithm config, RLinf may also add bootstrap values to the last step reward.

From a system perspective, the reward model does not replace the original bootstrap reward. Instead, it serves as
an additional reward source inside the env worker and participates in final reward construction.

3.4 Deploy QwenTrend for MLP RL
"""""""""""""""""""""""""""""""

For VLM reward inference, install embodied dependencies with VLM reward support:

.. code-block:: bash

   bash requirements/install.sh embodied --env maniskill_libero --vlm-reward

Then configure the reward section to use ``history_vlm``. The QwenTrend example
uses ``reward_mode: history_buffer`` so the env worker maintains per-env history
windows and sends them to the reward worker only when a valid window is available:

.. code-block:: yaml

   reward:
     use_reward_model: true
     group_name: "RewardGroup"
     reward_mode: history_buffer
     history_reward_assign: true
     reward_weight: 1.0
     env_reward_weight: 0.0
     model:
       model_path: "/path/to/Qwen3-VL-4B-Instruct"
       model_type: "history_vlm"
       lora_path: "/path/to/qwen3-vl-lora-checkpoint"
       gt_success_bonus: 20.0
       precision: "bf16"
       input_builder_name: qwentrend_input_builder
       input_builder_params:
         default_task_description: "Pick up the red cube and place it on the green spot on the table."
       reward_parser_name: qwentrend_reward_parser
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
           input_on_done_full_window: false
       interval_reward: 0.0
       infer_micro_batch_size: 64
       max_new_tokens: 16
       do_sample: false
       temperature: 0.0
       use_chat_template: true

Important fields:

- ``history_buffers`` defines which observation keys are cached, the window length, and the minimum valid history length.
- ``input_on_done_full_window`` makes a done-triggered request use the last full history window; it is disabled by default for compatibility.
- ``input_builder_name`` converts the history window into dual-view VLM inputs.
- ``reward_parser_name`` maps generated labels to scalar rewards using ``positive_reward``, ``negative_reward``, ``unclear_reward``, and ``invalid_reward``.
- ``gt_success_bonus`` optionally adds a success bonus from environment info.

Launch the MLP RL run with:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_reward

4. Summary
^^^^^^^^^^

The full workflow is:

1. Enable ``data_collection`` in the environment config and save raw data in ``pickle`` format.
2. For ResNet rewards, use ``preprocess_reward_dataset.py`` to build ``train.pt`` / ``val.pt`` and train with ``run_reward_training.sh``.
3. For QwenTrend VLM rewards, use ``preprocess_qwentrend_reward_dataset.py`` to build dual-view history-window data and fine-tune with ``run_vlm_sft.sh``.
4. Enable ``reward.use_reward_model=True`` in your RL YAML and plug the trained reward worker into online RL inference.


Real-World Reward Model
-----------------------

Collect and preprocess a reward model training dataset directly on a real-world
Franka robot. Two data collection approaches are supported:
a **general-purpose keyboard-labeling** approach and a **fixed-pose** approach that
uses a predetermined target pose to drive episode success/failure.

Before getting started, it is strongly recommended to read the following documents:

1. :doc:`../examples/embodied/franka` — to familiarize yourself with the end-to-end real-world Franka training pipeline.
2. :doc:`reward_model` — to understand the canonical reward model workflow in RLinf (data collection via ``pickle``, offline preprocessing, training, RL inference).
3. :doc:`../examples/embodied/franka_reward_model` — to understand the full real-world RL pipeline that follows after you have a trained reward model.

Workflow Overview
^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^

Follow the **Prerequisites** and **Hardware Setup** sections in :doc:`../examples/embodied/franka`
up to and including the robot connection and environment validation steps.

Data Collection
^^^^^^^^^^^^^^^

Approach 1: Keyboard Labeling (General-Purpose)
"""""""""""""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""""""""

This approach is specifically designed for tasks with a **fixed target pose** (e.g., reaching a
predetermined bin location). Instead of manual keyboard labeling, the episode automatically
drives success/failure based on whether the robot reaches the configured ``target_ee_pose``.
``success_hold_steps`` can be set to require the robot to maintain the pose for a certain
number of steps before declaring success, which helps collect more diverse successful samples.

This approach follows the same data collection pipeline as described in
:doc:`../examples/embodied/franka_reward_model`, but with a simplified preprocessing step
that uses the same script as Approach 1 (``realworld_collect_process_dataset.py``).


Step 1: Fixed-Pose Reward Data Collection
.........................................

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
...........................................

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
""""""

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
exactly as described in the Simulation Reward Model Section 2.

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

After completing the above steps, continue with Section 2
(**Reward Model Training**) in the Simulation Reward Model section above using the generated
``train.pt`` / ``val.pt`` files.

After training, you can use the trained reward model in two real-world ways:

- **Real-world teleoperation with live inference** (see below) — teleoperate the robot with
  SpaceMouse while the reward model runs on a GPU node, streaming real-time success
  probabilities to the terminal. No RL training loop is needed.
- **Real-world RL training** (see :doc:`../examples/embodied/franka_reward_model`) —
  integrate the reward model into the full RL training loop on the physical Franka.

Real-World Teleoperation with Live Reward Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a reward model checkpoint is available, ``examples/reward/eval_realworld_teleop.py``
provides a teleoperation mode where SpaceMouse drives the robot while the reward model
runs on a GPU node, printing per-step success probabilities in real time.

This is useful for:

- Sanity-checking the reward model's accuracy on live robot observations.
- Collecting human-aligned success/fail data for further dataset expansion.
- Qualitatively evaluating whether the reward model generalizes to the current scene.

Cluster Configuration
^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^

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
^^^^^^^^^^^^

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

Compared with the full RL pipeline in :doc:`../examples/embodied/franka_reward_model`,
the teleop script runs no policy, no actor, and no rollout worker — it is purely
human-in-the-loop evaluation of the reward model.
