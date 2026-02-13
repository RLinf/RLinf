VLM Supervised Fine-Tuning (SFT)
================================

This document explains how to run **full-parameter supervised fine-tuning (Full-parameter SFT)** for VLM models in RLinf.

This tutorial mainly focuses on two files:

- Launch script: ``examples/sft/run_vlm_sft.sh``
- Training config: ``examples/sft/config/custom_sft_vlm.yaml``

Launch Script: ``examples/sft/run_vlm_sft.sh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The script uses ``examples/sft/config/custom_sft_vlm.yaml`` by default.
- Logs are redirected to: ``<repo>/logs/<timestamp>/``
- Actual command:

.. code:: bash

   python examples/sft/train_vlm_sft.py \
     --config-path examples/sft/config/ \
     --config-name <your_config_name> \
     runner.logger.log_path=<auto_generated_log_dir>

Config Template: ``examples/sft/config/custom_sft_vlm.yaml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The VLM config structure is similar to other RLinf training configs.  
You mainly need to adapt ``data`` and ``actor.model`` for your VLM use case.

Preparation Before Running
--------------------------

1. Prepare the environment. Pull the RLinf Docker image:
   ``rlinf/rlinf:math-rlinf0.1-torch2.6.0-sglang0.4.6.post5-vllm0.8.5-megatron0.13.0-te2.1``.
2. Prepare model weights:
   ``https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct``.
3. Prepare Robo2VLM dataset:
   ``https://huggingface.co/datasets/keplerccc/Robo2VLM-1``.
4. Edit ``examples/sft/config/custom_sft_vlm.yaml`` and run
   ``examples/sft/run_vlm_sft.sh``.

Example YAML
------------

Important note: after downloading Robo2VLM, train and eval parquet files are mixed in one directory
(e.g., ``train-00000-of-00262.parquet`` and ``test-0000X-of-00003.parquet``).
Please split them into different folders. Otherwise, RLinf may load the entire dataset.

In the example below, fields you must modify are already commented.
Keep other parameters unchanged for a baseline run.

.. code:: yaml

   defaults:
     - override hydra/job_logging: stdout

   hydra:
     run:
       dir: .
     output_subdir: null

   cluster:
     num_nodes: 1
     component_placement:
       actor: all

   runner:
     task_type: sft
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "qwen2_5_vl_sft_demo"
       logger_backends: ["tensorboard"]

     max_epochs: 6000
     max_steps: -1
     val_check_interval: 1000
     save_interval: 1000

   data:
     type: vlm
     dataset_name: "robo2vlmsft"

     # Data paths: split train and eval files into different directories
     train_data_paths: "/path/to/Robo2VLM-1/train_data"
     # For eval-only runs, set train_data_paths to null
     eval_data_paths: "/path/to/Robo2VLM-1/test_data"

     # Keys must match dataset columns
     prompt_key: "question"
     choice_key: "choices"
     answer_key: "correct_answer"
     image_keys: ["image"]

     apply_chat_template: True
     use_chat_template: True
     max_prompt_length: 1024
     lazy_loading: false
     num_workers: 4

   algorithm:
     adv_type: gae

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
       # Download model weights locally and set the path here
       model_path: "/path/to/Qwen2.5-VL-3B-Instruct"
       is_lora: False

     optim:
       lr: 1e-5
       adam_beta1: 0.9
       adam_beta2: 0.999
       adam_eps: 1.0e-08
       weight_decay: 0.01
       clip_grad: 1.0
       lr_scheduler: "cosine"
       total_training_steps: ${runner.max_epochs}
       lr_warmup_steps: 200

     fsdp_config:
       strategy: "fsdp"
       sharding_strategy: "no_shard"
       use_orig_params: False
       gradient_checkpointing: False
       mixed_precision:
         param_dtype: bf16
         reduce_dtype: fp32
         buffer_dtype: bf16

   reward:
     use_reward_model: False

   critic:
     use_critic_model: False

Step 2: Start Training
----------------------

Run from repository root:

.. code:: bash

   bash examples/sft/run_vlm_sft.sh

Notes:

- If no argument is provided, the script uses ``custom_sft_vlm`` by default.
- If your config name is different (e.g., ``my_vlm_config.yaml``), pass it as an argument:

.. code:: bash

   bash examples/sft/run_vlm_sft.sh my_vlm_config

Step 3: Check Whether Training Is Healthy
-----------------------------------------

1. Check if loss decreases in terminal logs.
2. Check the generated log directory (script creates ``logs/<timestamp>`` automatically).
3. Visualize with TensorBoard:

.. code:: bash

   tensorboard --logdir /path/to/RLinf/logs --port 6006

Open in browser: ``http://localhost:6006``

Eval-Only Mode (No Training)
----------------------------

If you only want evaluation, update config as:

- ``data.train_data_paths: null``
- ``data.eval_data_paths: "/eval_data_path"``

Use the same launch command:

.. code:: bash

   bash examples/sft/run_vlm_sft.sh <config_name>

Experiment Results
------------------

We provide a reference experiment run on a single machine with 8 × H100 GPUs for 6000 iterations.

Evaluation accuracy on test_data every 1000 iterations:

.. image:: /_static/images/sft/sft_vlm_eval_accuracy.png
   :alt: VLM SFT eval accuracy
   :width: 85%
   :align: center

grad_norm curve:

.. image:: /_static/images/sft/sft_vlm_eval_grad_norm.png
   :alt: VLM SFT grad norm
   :width: 85%
   :align: center

loss curve:

.. image:: /_static/images/sft/sft_vlm_eval_loss.png
   :alt: VLM SFT loss
   :width: 85%
   :align: center

Final eval accuracy: ``0.8995802998542786`` (about ``89.96%``).

Checkpoint Notes
----------------

SFT with FSDP saves checkpoints in FSDP format (for example, ``full_weights.pt``).

If you need HuggingFace format, use the built-in converter:

- Script: ``toolkits/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh``
- Config: ``toolkits/ckpt_convertor/fsdp_convertor/config/fsdp_model_convertor.yaml``

Update these fields first:

- ``convertor.ckpt_path``: path to ``full_weights.pt``
- ``convertor.save_path``: output HF model directory
- ``model.model_path``: base model path
- ``model.model_type``: model type (e.g., ``qwen2.5_vl``)

Run:

.. code:: bash

   bash toolkits/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh

Field Reference
---------------

- ``micro_batch_size``: per-GPU batch size per forward/backward
- ``global_batch_size``: total batch size across all GPUs (must be divisible)
- ``max_epochs``: number of full passes over dataset
- ``save_interval``: checkpoint save frequency (in steps)
- ``model_path``: local model directory (must exist)
- ``train_data_paths/eval_data_paths``: dataset directory or file path

Common Issues and Fixes
-----------------------

1. **Model path not found**
   - Verify ``actor.model.model_path`` is correct and readable.

2. **Dataset key mismatch**
   - Verify ``prompt_key/choice_key/answer_key/image_keys`` match your dataset columns.

3. **OOM (out of memory)**
   - Reduce ``micro_batch_size`` first.
   - Reduce ``num_workers`` if needed.
   - If still OOM, use a smaller model or shorter input length.

4. **You only want a quick smoke test**
   - Use a very small data subset.
   - Set ``max_epochs`` to 1.
   - Set smaller ``save_interval`` for faster feedback.