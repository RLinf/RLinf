Qwen-VL SFT
===========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/release_0.2/qwen2_5_sft_vlm.png
   :align: center
   :width: 85%

   Qwen2.5-VL supervised fine-tuning on the Robo2VLM visual-QA dataset.

Run supervised fine-tuning for vision-language models with RLinf: **full-parameter** SFT and evaluation for Qwen2.5-VL, Qwen3-VL, and Qwen3-VL-MoE on Robo2VLM visual QA, or **full-parameter and LoRA** SFT for Qwen3.8-27B on multi-turn RPent planner data, then export the trained checkpoint to HuggingFace format.

Overview
--------

Both workflows run FSDP supervised fine-tuning through ``examples/sft/run_vlm_sft.sh``; select a tab below for setup and configuration.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      Qwen2.5-VL · Qwen3-VL · Qwen3-VL-MoE · Qwen3.8-27B

   .. grid-item-card:: Methods
      :text-align: center

      Full-parameter SFT · LoRA

   .. grid-item-card:: Data
      :text-align: center

      Robo2VLM (visual QA) · RPent

   .. grid-item-card:: Hardware
      :text-align: center

      1–2 nodes · GPUs (4 × 80 GB+ for 27B LoRA)

| **You'll do:** install the environment → prepare weights and data (Robo2VLM or RPent JSONL) → launch ``run_vlm_sft.sh`` → monitor loss/eval metrics and export to HuggingFace.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Qwen-VL or Qwen3.8-27B weights · Robo2VLM or RPent dataset.

.. _rpent-planner-sft:

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Qwen2.5-VL (Robo2VLM)

      **1. Install and Download**

      Install the agentic/VLM environment and download `Qwen2.5-VL-3B-Instruct <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>`_ and `Robo2VLM-1 <https://huggingface.co/datasets/keplerccc/Robo2VLM-1>`_:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct
         hf download keplerccc/Robo2VLM-1 --repo-type dataset --local-dir ./Robo2VLM-1

      .. warning::

         After downloading Robo2VLM, train and eval parquet files are mixed in one directory (for example ``train-00000-of-00262.parquet`` and ``test-0000X-of-00003.parquet``). Split them into separate ``train_data`` and ``test_data`` folders before launching, or RLinf may load the entire dataset.

      .. note::

         To train **qwen3_vl** or **qwen3_vl_moe**, ensure ``transformers >= 4.57.1``.

      **2. Configure**

      Set the dataset and model paths in ``examples/sft/config/qwen2_5_vl_sft_vlm.yaml``:

      .. code-block:: yaml

         data:
           train_data_paths: "./Robo2VLM-1/train_data"
           val_data_paths: "./Robo2VLM-1/test_data"

         actor:
           model:
             model_path: "./Qwen2.5-VL-3B-Instruct"

      **3. Launch**

      Run from the repository root (logs write to ``logs/<timestamp>/``; pass another config name to run a custom YAML, or set ``data.train_data_paths: null`` for eval-only mode):

      .. code-block:: bash

         bash examples/sft/run_vlm_sft.sh qwen2_5_vl_sft_vlm

   .. tab-item:: Qwen3.8-27B (RPent)

      Fine-tune Qwen3.8-27B (``qwen3_5`` architecture) on multi-turn `RPent <https://github.com/RLinf/RPent>`_ planner data with ``examples/sft/config/qwen3_8_27b_rpent_sft.yaml``. Loss applies only to the target assistant reply and tool calls. LoRA SFT is verified on 4 × A800 80 GB GPUs; 27B full-parameter SFT requires more GPUs.

      **1. Install**

      Install the default ``agentic --engine sglang`` environment (which pins ``transformers==5.6.1`` with ``qwen3_5`` support):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate

      **2. Download and Prepare Data**

      Download the Qwen3.8-27B weights from HuggingFace:

      .. code-block:: bash

         hf download Qwen/Qwen3.8-27B --local-dir ./Qwen3.8-27B

      Prepare an RPent JSONL file or directory (image paths resolve relative to each JSONL file):

      .. list-table::
         :header-rows: 1
         :widths: 22 78

         * - Field
           - Description
         * - ``episode_id``, ``turn_id``
           - Episode and turn identifiers.
         * - ``tools``
           - Tool definitions with ``name``, ``description``, and ``input_schema``.
         * - ``messages``
           - Conversation history of messages with ``role``, ``content`` blocks (``text`` or ``image``), optional ``tool_calls``, and optional ``tool_call_id``.
         * - ``target``
           - Target assistant turn with ``role: "assistant"``, ``content``, and optional ``tool_calls``.

      Minimal single-record JSONL example (formatted across lines for readability):

      .. code-block:: json

         {
           "episode_id": "ep_001",
           "turn_id": "P1",
           "tools": [{"name": "look", "description": "Inspect view", "input_schema": {"type": "object"}}],
           "messages": [
             {"role": "user", "content": [{"type": "text", "text": "Find the cup."}]},
             {"role": "assistant", "content": [], "tool_calls": [{"id": "c1", "name": "look", "arguments": {"view": 1}}]},
             {"role": "tool", "tool_call_id": "c1", "content": [
               {"type": "image", "artifact": {"path": "images/view1.png"}},
               {"type": "text", "text": "Cup on left"}
             ]}
           ],
           "target": {"role": "assistant", "content": [{"type": "text", "text": "Picking up the cup."}]}
         }

      **3. Launch**

      Point ``RPENT_SAMPLES_PATH`` at your JSONL file or directory and ``QWEN3_8_MODEL_PATH`` at the downloaded weights, then launch full-parameter or rank-16 LoRA SFT from the repository root:

      .. code-block:: bash

         export RPENT_SAMPLES_PATH=./rpent_samples.jsonl
         export QWEN3_8_MODEL_PATH=./Qwen3.8-27B

         # Full-parameter SFT (default)
         bash examples/sft/run_vlm_sft.sh qwen3_8_27b_rpent_sft

         # LoRA SFT (rank 16 on language-model attention and MLP layers)
         bash examples/sft/run_vlm_sft.sh qwen3_8_27b_rpent_sft actor.model.is_lora=true

      After training, export or merge the checkpoint under *Checkpoint conversion (FSDP to HuggingFace)* below, then evaluate the planner in RPent / RoboTwin.

.. dropdown:: Advanced configuration & field reference

   The shipped ``examples/sft/config/qwen2_5_vl_sft_vlm.yaml`` already sets column keys, batch sizes, optimizer, and FSDP defaults:

   .. code-block:: yaml

      cluster:
        num_nodes: 1
        component_placement:
          actor: all

      runner:
        task_type: sft
        max_epochs: -1
        max_steps: 6000
        val_check_interval: 1000
        save_interval: 1000

      data:
        type: vlm
        dataset_name: "robo2vlmsft"
        prompt_key: "question"
        choice_key: "choices"
        answer_key: "correct_answer"
        image_keys: ["image"]
        apply_chat_template: True
        use_chat_template: True
        max_prompt_length: 1024
        lazy_loading: true
        num_workers: 16

      actor:
        group_name: "ActorGroup"
        training_backend: "fsdp"
        micro_batch_size: 4
        eval_batch_size: 4
        global_batch_size: 256
        seed: 42
        model:
          model_type: "qwen2.5_vl"
          precision: fp32
          is_lora: False
        optim:
          lr: 1e-5
          adam_beta1: 0.9
          adam_beta2: 0.999
          adam_eps: 1.0e-08
          weight_decay: 0.01
          clip_grad: 1.0
          lr_scheduler: "cosine"
          total_training_steps: ${runner.max_steps}
          lr_warmup_steps: 200
        fsdp_config:
          strategy: "fsdp"
          sharding_strategy: "full_shard"
          use_orig_params: False
          gradient_checkpointing: False
          mixed_precision:
            param_dtype: bf16
            reduce_dtype: fp32
            buffer_dtype: bf16

   **Key fields:**

   - ``micro_batch_size``: per-GPU batch size per forward/backward.
   - ``global_batch_size``: total batch size across all GPUs (must be divisible).
   - ``max_epochs``: number of full passes over the dataset.
   - ``save_interval``: checkpoint save frequency (in steps).
   - ``model_path``: local model directory (must exist).
   - ``train_data_paths`` / ``val_data_paths``: dataset directory or file path (set ``train_data_paths: null`` for eval-only mode).

.. dropdown:: Checkpoint conversion (FSDP to HuggingFace)

   FSDP SFT saves full weights to ``full_weights.pt``. Run the converter from the repository root with the run's saved ``config.yaml``:

   .. code-block:: bash

      export REPO_PATH=$(pwd)
      python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
          +convertor.train_config_path=./logs/<timestamp>/tensorboard/config.yaml \
          convertor.ckpt_path=./logs/<timestamp>/<experiment_name>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt \
          convertor.save_path=./hf_converted_model

   Replace ``<timestamp>``, ``<experiment_name>``, and ``<N>`` with the values from your run directory. For LoRA checkpoints, the converter merges adapters into the base weights by default; pass ``convertor.merge_lora_weighs=False`` to save only the adapter under ``lora_adapter/``. See :doc:`Checkpoint conversion <../../guides/convertor>` for details.

.. dropdown:: Troubleshooting

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Issue
        - Fix
      * - Model path not found
        - Verify ``actor.model.model_path`` is correct and readable.
      * - Dataset key mismatch
        - Verify ``prompt_key``, ``choice_key``, ``answer_key``, and ``image_keys`` match your dataset columns.
      * - OOM (out of memory)
        - Reduce ``micro_batch_size`` first, then ``num_workers``; if it persists, use a smaller model or shorter input length.
      * - Quick verification
        - Use a small data subset, set ``max_epochs: 1``, and lower ``save_interval`` for faster feedback.

Visualization and Results
-------------------------

A healthy run shows the **loss** decreasing and the **eval accuracy** climbing. Visualize ``logs/<timestamp>`` with TensorBoard (``tensorboard --logdir ./logs --port 6006``). For every logged metric, see :doc:`Training metrics <../../reference/metrics>`.

Reference runs across model scales (Robo2VLM):

.. list-table::
   :header-rows: 1
   :widths: 34 20 14 32

   * - Model
     - Hardware
     - Iters
     - Eval accuracy (before → after)
   * - Qwen2.5-VL-3B
     - 8 × H100
     - 6000
     - — → 89.96%
   * - Qwen3-VL-4B
     - 4 × H100
     - 6000
     - — → 96.9%
   * - Qwen3-VL-30B-A3B (MoE)
     - 2 × 8 × A100
     - 1000
     - 58.4% → 91.3%

For **Qwen3.8-27B (RPent)**, monitor ``train/loss`` in TensorBoard. On 4 × A800 80 GB GPUs, a 48-step LoRA run (``actor.model.lora_rank=592``) reached a final ``train/loss`` of 0.199 and, after merging weights, raised solved RoboTwin planner episodes from 8/50 (base model) to 14/50.

**Qwen2.5-VL-3B** — eval accuracy, grad_norm, and loss every 1000 iterations:

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/sft_vlm_eval_accuracy.png
   :alt: Qwen2.5-VL-3B VLM SFT eval accuracy
   :width: 85%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/sft_vlm_eval_grad_norm.png
   :alt: Qwen2.5-VL-3B VLM SFT grad norm
   :width: 85%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/sft_vlm_eval_loss.png
   :alt: Qwen2.5-VL-3B VLM SFT loss
   :width: 85%
   :align: center

**Qwen3-VL-4B** — eval accuracy, grad_norm, and loss every 1000 iterations:

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3_sft_vlm_eval_accuracy.png
   :alt: Qwen3-VL-4B VLM SFT eval accuracy
   :width: 85%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3_sft_vlm_eval_grad_norm.png
   :alt: Qwen3-VL-4B VLM SFT grad norm
   :width: 85%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3_sft_vlm_eval_loss.png
   :alt: Qwen3-VL-4B VLM SFT loss
   :width: 85%
   :align: center

**Qwen3-VL-30B-A3B (MoE)** — grad_norm and loss over 1000 iterations:

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3_moe_sft_vlm_eval_grad_norm.png
   :alt: Qwen3-VL-30B-A3B MoE VLM SFT grad norm
   :width: 85%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3_moe_sft_vlm_eval_loss.png
   :alt: Qwen3-VL-30B-A3B MoE VLM SFT loss
   :width: 85%
   :align: center
