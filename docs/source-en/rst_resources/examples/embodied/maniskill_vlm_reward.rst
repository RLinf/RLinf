ManiSkill PPO with VLM Reward Model
===================================

This document provides a complete guide for running ManiSkill PPO in RLinf with an **MLP policy + Qwen3-VL reward model**.
The main reference config is ``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml``.

The main goals of this setup are:

1. **State-based policy learning**: the actor still uses a lightweight ``mlp_policy`` over ``states``.
2. **Visual reward judgement**: the reward worker uses Qwen3-VL to judge image history together with task text.
3. **History-based scoring**: learned rewards are assigned over a short trajectory segment through ``history_buffer``.
4. **Backend selection**: the reward model can run as an in-process Hugging Face model
   (``history_vlm``) or as an in-process ``sglang.Engine`` backend
   (``history_vlm_sglang``) when that backend is configured.
5. **RL optimization**: PPO updates the policy with reward-worker outputs.

Environment
-----------

**ManiSkill3 Environment**

- **Environment**: ManiSkill3 simulation platform
- **Task**: robotic manipulation tasks such as ``PickCube3View-v1``
- **Policy Observation**: ``states``
- **Reward Observation**: ``main_images`` and ``extra_view_images`` together with task text
- **Action Space**: 8-dimensional continuous actions

**Reward Input Structure**

- **States**: state vectors for ``mlp_policy``
- **Main Images**: main-view image history for the Qwen3-VL reward worker
- **Extra View Images**: third-person image history paired with the main view
- **Task Descriptions**: task text descriptions
- **History Buffer**: short video segments organized by ``history_size`` and ``input_interval``

Algorithm
-----------------------------------------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - GAE-based advantage estimation
   - Clipped policy update
   - Critic optimization

2. **MLP Policy**

   - the actor only consumes ``states``
   - policy training stays lightweight

3. **Qwen3-VL Reward Model**

   - the reward worker uses ``HistoryVLMRewardModel``
   - inputs are task text plus short video history
   - outputs are parsed into scalar rewards by a reward parser
   - async runs keep the reward worker alive and serve queued history requests through RLinf channels

Dependency Installation
-----------------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

For OpenPI + Qwen3-VL Hugging Face reward experiments, use the OpenPI embodied
environment and pin the Qwen3-VL-compatible ``transformers`` version. RLinf's
repo-local OpenPI patch is compatible with this unified environment, so the Qwen3-VL
reward worker can run in the same venv:

.. code:: bash

   source switch_env openpi
   uv pip install --upgrade "transformers==4.57.1" "tokenizers>=0.22,<0.23"

SGLang is optional and only required when ``reward.model.model_type=history_vlm_sglang``.
RLinf uses an in-process ``sglang.Engine`` reward backend, not an external SGLang server
path. The target stack is ``sglang==0.5.4`` with ``transformers==4.57.1`` plus the
matching SGLang torch, xgrammar, and flashinfer runtime, so keep it separate from the
default Hugging Face reward install unless you select that backend.

**Option 2: Custom Environment**

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero --vlm-reward
   source .venv/bin/activate

The ``--vlm-reward`` flag pins ``transformers==4.57.1`` and installs the tokenizers
range required by Qwen3-VL. It is intended to work with OpenPI through RLinf's
repo-local OpenPI patch, without a dedicated Qwen reward venv. It does not download
reward checkpoints or install SGLang.

For the unified OpenPI + Qwen3-VL + in-process SGLang reward environment, use
``--vlm-reward-sglang`` instead:

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env maniskill_libero --vlm-reward-sglang
   source .venv/bin/activate

This SGLang path targets ``sglang==0.5.4`` with ``transformers==4.57.1`` and installs
the SGLang-specific torch, xgrammar, and flashinfer runtime. Use it only when the reward
config sets ``reward.model.model_type: history_vlm_sglang``.

Assets Download
----------------

Download the ManiSkill assets by running the following command:

.. code:: bash

   cd <path_to_RLinf>/rlinf/envs/maniskill
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

Model Download
--------------

Before starting training, prepare the base model and LoRA weights used by the reward worker:

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com

   # Download the base model
   hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /path/to/Qwen3-VL-4B-Instruct

   # Replace this with your own reward-model LoRA directory
   ls /path/to/Qwen3-VL-4B-Instruct_lora

After downloading, make sure the config yaml correctly sets:

- ``reward.model.model_path``
- ``reward.model.lora_path``

If you still need to prepare or fine-tune the Qwen3-VL checkpoint / LoRA used by the reward worker,
the QwenTrend-specific data and SFT flow is described below. For the general VLM SFT runner,
please also refer to :doc:`/rst_source/examples/embodied/sft_vlm`.

QwenTrend Reward Model Data and SFT Workflow
------------------------------------------------------------

You can skip this section if you already have a trained QwenTrend reward checkpoint. Otherwise,
the reward model is prepared in three stages: collect raw ManiSkill episode pickle files,
convert them into 5-frame dual-view progress labels, and run VLM SFT.

**1. Collect raw episode pickle files**

.. code-block:: bash

   cd <path_to_RLinf>
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_collect

This command launches ``examples/embodiment/train_embodied_agent.py`` with
``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_collect.yaml``.
It is a data-collection run, not the online VLM-reward PPO run:

- ``reward.use_reward_model`` is ``false``, so no VLM reward worker is used.
- ``env.eval.data_collection.enabled`` is ``true`` while training collection is disabled.
- At each configured evaluation interval, the eval environment is wrapped by ``CollectEpisode``.
- Raw pickle files are written under ``logs/<timestamp>-maniskill_ppo_mlp_qwentrend_collect/collected_data`` by default.
- Each pickle episode stores observations, actions, rewards, termination flags, info dicts, and success metadata. With ``obs_mode: rgb`` and ``use_3rd_view_as_extra: true``, the observations include ``main_images`` and ``extra_view_images``.

**2. Convert episodes into QwenTrend SFT data**

.. code-block:: bash

   python examples/reward/preprocess_qwentrend_reward_dataset.py \
      --raw-data-path logs/<timestamp>-maniskill_ppo_mlp_qwentrend_collect/collected_data \
      --output-dir logs/processed_qwentrend_reward_data \
      --max-samples-per-label 5000 \
      --load-workers 32 \
      --write-workers 32

The preprocessing script searches for ``*.pkl`` episodes, slices each episode into
dual-view windows, and writes:

- ``<output-dir>/train/segments.jsonl``
- ``<output-dir>/train/pkl/*.pkl``
- ``<output-dir>/eval/segments.jsonl``
- ``<output-dir>/eval/pkl/*.pkl``

Important defaults are:

- ``--window-size 5`` exports five frames per view.
- ``--stride 1`` creates overlapping windows.
- ``--delta-threshold 0.05`` labels small progress deltas as ``unclear``.
- ``--tail-unclear-ratio 0.15`` forces the tail windows of each episode to ``unclear``.
- ``--val-split 0.1`` splits episodes into train and eval sets.
- ``--balance-labels`` and ``--reverse-positive-as-negative`` are enabled by default.
- ``--fps`` is kept only for backward-compatible CLI calls; the pkl export path does not use FPS resampling.

If the raw episodes do not contain task text, pass ``--task-description`` to avoid the generic
fallback prompt.

**3. Train the Qwen3-VL QwenTrend reward model**

Set the processed dataset root, check ``actor.model.model_path`` and ``runner.output_dir`` in
``examples/sft/config/qwen3vl_sft_qwentrend.yaml``, then launch VLM SFT:

.. code-block:: bash

   export DUALVIEW_SFT_DATA_ROOT=/path/to/processed_qwentrend_reward_data
   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_qwentrend

The launch script runs ``examples/sft/train_vlm_sft.py`` with the selected config. The
``qwen3vl_sft_qwentrend`` config uses ``dataset_name: qwentrend_progress_sft`` and reads:

- ``${DUALVIEW_SFT_DATA_ROOT}/train/segments.jsonl``
- ``${DUALVIEW_SFT_DATA_ROOT}/eval/segments.jsonl``

The dataset loads each sample's ``pkl_path`` and passes the two in-memory 5-frame video arrays
directly to the Qwen3-VL processor. The config uses ``video_nframes: 5`` and ``video_fps: null``.

The SFT checkpoints are saved under:

.. code-block:: text

   logs/<sft-timestamp>/<experiment_name>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt

For online PPO with the trained reward model, set ``reward.model.model_path`` to the base
Qwen3-VL model directory and set ``reward.model.lora_path`` to the checkpoint step directory,
for example ``logs/<sft-timestamp>/<experiment_name>/checkpoints/global_step_<N>``.

Running the Script
-------------------

**1. Key Parameters Configuration**

.. code-block:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         actor: 0-0
         env: 0-0
         rollout: 0-0
         reward: 0-0

   env:
      train:
         wrap_obs_mode: simple_prompt
         use_full_state: True
         init_params:
            id: PickCube3View-v1
            obs_mode: rgb
         use_3rd_view_as_extra: True

   reward:
      use_reward_model: True
      reward_mode: history_buffer
      history_reward_assign: True
      reward_weight: 1.0
      env_reward_weight: 0.0
      model:
         model_type: history_vlm
         model_path: /path/to/Qwen3-VL-4B-Instruct
         lora_path: /path/to/Qwen3-VL-4B-Instruct_lora
         gt_success_bonus: 20.0
         input_builder_name: qwentrend_input_builder
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
         infer_micro_batch_size: 64

These parameters matter because:

- ``component_placement.reward`` places the online reward worker.
- ``wrap_obs_mode: simple_prompt`` exposes ``states``, ``main_images``, ``extra_view_images``, and ``task_descriptions`` together.
- ``use_full_state: True`` keeps the actor on ``states`` with ``mlp_policy``.
- ``use_3rd_view_as_extra: True`` makes the ManiSkill third-person camera available as ``extra_view_images``.
- ``reward_mode: history_buffer`` means the reward worker scores a short trajectory segment instead of a single frame.
- ``history_reward_assign: True`` back-fills the reward to earlier steps covered by the current history window.
- ``infer_micro_batch_size`` limits the Hugging Face reward-model micro-batch size. Tune it for reward-worker GPU memory.

**Reward backend variants**

The history-buffer fields, input builder, and reward parser are shared by both VLM reward
backends:

- ``reward.model.model_type: history_vlm`` loads Qwen3-VL in the reward worker process
  through Hugging Face ``AutoModelForVision2Seq``.
- ``reward.model.model_type: history_vlm_sglang`` loads Qwen3-VL through an in-process
  ``sglang.Engine`` inside the reward worker. It is not an external server endpoint path.
  Keep the same ``history_buffers``, ``input_builder_name``, and ``reward_parser_name``
  fields, and use the ``--vlm-reward-sglang`` environment because this backend targets
  ``sglang==0.5.4`` with its own torch, xgrammar, and flashinfer runtime.

**2. Configuration Files**

You can directly refer to the following config:

- Main QwenTrend example: ``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml``

**3. Launch Commands**

After choosing a config, start training with:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_reward

Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. Key Metrics Tracked**

- ``env/success_once``: the most direct success-rate metric to watch.
- ``env/reward``: raw environment step reward.
- ``rollout/rewards``: mixed rollout reward after reward-model integration.
- ``train/actor/policy_loss``: policy optimization status.

Online Call Chain
-----------------

The VLM reward path is:

.. code-block:: text

   train_embodied_agent.py
      -> EmbodiedRunner
      -> EnvWorker(history_buffer)
      -> HistoryManager
      -> EmbodiedRewardWorker
      -> HistoryVLMRewardModel
      -> InputBuilder + Qwen3-VL generate() + RewardParser

The concrete flow is:

1. ``train_embodied_agent.py`` creates ``EmbodiedRewardWorker`` when ``reward.use_reward_model=True``.
2. ``EmbodiedRunner.run`` activates the reward channel once ``global_step >= reward.use_output_step``.
3. ``EnvWorker.get_reward_model_output`` appends observations into ``HistoryManager`` when ``reward_mode="history_buffer"``.
4. ``HistoryManager.build_history_input`` extracts the configured history windows.
5. ``EmbodiedRewardWorker`` instantiates the configured history VLM reward model from
   ``reward.model.model_type``. Use ``history_vlm`` for the in-process Hugging Face path
   or ``history_vlm_sglang`` for the in-process ``sglang.Engine`` path.
6. ``HistoryVLMRewardModel.compute_reward`` builds multimodal inputs with the configured ``input_builder_name``, runs ``AutoModelForVision2Seq.generate()``, and parses the generated text with ``reward_parser_name``.
7. ``EnvWorker.compute_bootstrap_rewards`` writes the reward-model output to the current step. If ``history_reward_assign=True``, ``EnvWorker.assign_history_reward`` also back-fills the same reward to earlier steps covered by the current history window.

In async embodied runs, the runner starts ``EmbodiedRewardWorker.compute_rewards_async`` once
and keeps it serving reward requests. Env workers enqueue split reward inputs with stable
``train_reward_input`` channel keys, while reward workers return split outputs with
``reward_output`` keys. This is the queued reward-worker path; it avoids restarting the VLM
for every rollout step, but the final reward is still assembled in ``EnvWorker``.

Current Implementation Notes
----------------------------

- ``reward_threshold`` is configured at the top-level ``reward`` section in these YAML files, but the ``history_vlm`` implementation does not currently apply that threshold during reward inference.
- ``qwentrend_input_builder`` consumes both ``main_images`` and ``extra_view_images`` from ``history_input``, so the history buffer must record both keys to form the synchronized two-view prompt.
- ``qwentrend_reward_parser`` maps the generated label directly to signed scalar rewards using ``positive_reward``, ``negative_reward``, ``unclear_reward``, and ``invalid_reward``. It does not clamp outputs into ``[0, 1]``.
- ``gt_success_bonus`` is configured under ``reward.model`` and is applied inside the reward model path rather than in the reward worker front-end.
- ``history_vlm_sglang`` should use the same prompt/input and parser contract as ``history_vlm``. Differences are limited to the in-process SGLang runtime and batching behavior; validate generated labels before comparing reward curves across backends.
