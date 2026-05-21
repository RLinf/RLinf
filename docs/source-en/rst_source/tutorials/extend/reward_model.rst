Reward Model Guide
========================

This document describes how to use reward models in RLinf. It covers both image
classification rewards such as ``ResNetRewardModel`` and VLM rewards such as
QwenTrend / ``HistoryVLMRewardModel``.
Here, QwenTrend means using a Qwen3-VL model to judge the action trend in a short
history video and convert that judgment into a scalar reward.

The full workflow has four stages:

1. Data collection: collect raw episode data during RL runs.
2. Dataset conversion: convert raw episodes into either image classification data or VLM SFT data.
3. Reward model training: train a ResNet reward model or fine-tune a VLM reward model.
4. Reward model inference in RL: plug the trained model into online rollout and use it in final reward computation.

1. Data Collection
----------------------------

Reward model training data is typically built from episode-level data collection. RLinf provides
a unified collection wrapper, and the related usage is documented in :doc:`the data collection tutorial <../components/data_collection>`.

For reward model use cases, we recommend saving raw episodes in ``pickle`` format first, then converting
them into processed training splits with the preprocessing script.

1.1 Enable Data Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
----------------------------

RLinf supports two reward training paths. ``examples/reward/run_reward_training.sh``
trains the ResNet image reward model, while ``examples/sft/run_vlm_sft.sh``
fine-tunes a VLM reward model such as QwenTrend.

2.1 Fine-Tune the ResNet Reward Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.1.1 Configure ResNet Dataset Paths
""""""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""""""

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
""""""""""""""""""""""""""""

Once the dataset and model are configured, run:

.. code-block:: bash

   bash examples/reward/run_reward_training.sh

Training logs are written to a newly created ``logs/<timestamp>-reward_training`` directory.

2.2 Fine-Tune the QwenTrend VLM Reward Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
---------------------------------

RLinf provides several example configs for integrating a reward model into RL:

- ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``
- ``examples/embodiment/config/maniskill_sac_mlp_resnet_reward_async.yaml``
- ``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml``

These configs show how to enable a reward worker in RL training while keeping the policy on state observations
and the reward model on image or VLM observations.

3.1 Key Config Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the reward channel is enabled, ``EnvWorker`` first fetches ``reward_model_output``,
then merges it with the original environment reward inside ``compute_bootstrap_rewards``:

.. code-block:: python

   reward = env_reward_weight * env_reward + reward_weight * reward_model_output

If bootstrap is enabled by the algorithm config, RLinf may also add bootstrap values to the last step reward.

From a system perspective, the reward model does not replace the original bootstrap reward. Instead, it serves as
an additional reward source inside the env worker and participates in final reward construction.

3.4 Deploy QwenTrend for MLP RL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For VLM reward inference, install embodied dependencies with VLM reward support:

.. code-block:: bash

   bash requirements/install.sh embodied --model qwen3_vl --env maniskill_libero --vlm-reward

This installs the Qwen3-VL reward runtime used by the MLP policy example. If the
training policy is OpenPI, use ``--model openpi --vlm-reward`` instead so RLinf
also applies the OpenPI patch for the Qwen3-VL-compatible ``transformers``
runtime.

Then configure the reward section to use ``history_vlm`` for the Hugging Face
backend or ``history_vlm_sglang`` for the SGLang backend. The QwenTrend example
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
       interval_reward: 0.0
       infer_micro_batch_size: 64
       max_new_tokens: 16
       do_sample: false
       temperature: 0.0
       use_chat_template: true

Important fields:

- ``model_type`` selects the reward inference backend: ``history_vlm`` uses
  Hugging Face/Transformers, while ``history_vlm_sglang`` uses SGLang.
- ``history_buffers`` defines which observation keys are cached, the window length, and the minimum valid history length.
- ``input_builder_name`` converts the history window into dual-view VLM inputs.
- ``reward_parser_name`` maps generated labels to scalar rewards using ``positive_reward``, ``negative_reward``, ``unclear_reward``, and ``invalid_reward``.
- ``gt_success_bonus`` optionally adds a success bonus from environment info.

To switch the same ManiSkill YAML to SGLang reward inference, change only the
reward model type:

.. code-block:: yaml

   reward:
     model:
       model_type: "history_vlm_sglang"

The SGLang backend reuses the same QwenTrend input builder, reward parser, and
history buffer settings. It uses ``model_path`` and optional
``sglang_engine_kwargs`` for engine construction; the Hugging Face ``lora_path``
loading path does not apply to ``history_vlm_sglang``.

Launch the MLP RL run with:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_reward

Summary
----------------------------

The full workflow is:

1. Enable ``data_collection`` in the environment config and save raw data in ``pickle`` format.
2. For ResNet rewards, use ``preprocess_reward_dataset.py`` to build ``train.pt`` / ``val.pt`` and train with ``run_reward_training.sh``.
3. For QwenTrend VLM rewards, use ``preprocess_qwentrend_reward_dataset.py`` to build dual-view history-window data and fine-tune with ``run_vlm_sft.sh``.
4. Enable ``reward.use_reward_model=True`` in your RL YAML and plug the trained reward worker into online RL inference.
