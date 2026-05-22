GRPO Training for Qwen3-VL Visual Language Reasoning
=====================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document introduces how to train vision-language models (VLMs) for geometric reasoning using reinforcement learning (RL) within the RLinf framework.

Environment & Installation
----------------------------

**Dependency Versions**

The Qwen3-VL model family requires recent versions of several key dependencies:

- ``torch >= 2.8.0``
- ``sglang == 0.5.4``
- ``transformers == 4.57.1``

Older versions of sglang or transformers either do not support or cannot correctly handle Qwen3-VL models.

**One-Click Install**

RLinf provides the ``requirements/install.sh`` script for one-click environment setup:

.. code-block:: bash

   export MEGATRON_PATH=/path/to/Megatron-LM
   bash requirements/install.sh agentic \
       --torch 2.8.0 \
       --sglang 0.5.4 \
       --transformers 4.57.1 \
       --no-apex

The script performs the following steps automatically.

.. note::

   ``MEGATRON_PATH`` must point to an existing Megatron-LM repository clone (the ``core_r0.13.0`` branch is recommended).
   If the directory does not exist, the install script will exit early without installing flash-attn or apex.

.. tip::

   Since this example uses FSDP2 as the training backend, apex is not required. The ``--no-apex`` flag
   skips apex installation. You may also need this flag when the system CUDA toolkit version
   (``nvcc --version``) does not match the CUDA version that PyTorch was compiled with,
   which can cause apex to fail building from source.

**Post-Install Verification**

After installation, verify that the key dependencies are working:

.. code-block:: bash

   source .venv/bin/activate
   python -c "
   import torch; print('torch:', torch.__version__)
   import transformers; print('transformers:', transformers.__version__)
   import sglang; print('sglang:', sglang.__version__)
   print('All good')
   "

Dataset
-------------

We use the geo3K dataset (download from https://huggingface.co/datasets/CAIR-HKISI/geo3k), which contains geometric problems along with their corresponding images and answers.

An example training sample looks like:

.. code-block:: json

   {
      "problem": "<image>\nProblem description",
      "images": An numpy.ndarray of image bytes,
      "answer": "\\boxed{x}"
   }

.. note::

  The geo3k dataset is available in multiple storage formats. If you download it from a different source,
  images may be stored as base64 strings or PIL.Image objects, and you may need to adapt accordingly.

We support several dataset format configurations:

- **Prompt key and answer key configuration**

  By default, the configuration expects the dataset to use ``problem`` and ``answer`` keys for prompts and answers.
  Modify the ``prompt_key`` and ``answer_key`` values in the YAML config to point to the corresponding fields in your dataset.

  .. code-block:: yaml

      prompt_key: "problem"
      answer_key: "answer"

- **Image data configuration**

  For vision-language tasks, configure the ``image_keys`` parameter to specify the image field name:

  .. code-block:: yaml

      image_keys: ["images"]

- **Image placeholder parsing**

  The dataset supports ``<image>`` placeholders in prompts to mark image positions. The framework automatically
  parses the placeholders and interleaves text with images. If the prompt does not contain placeholders,
  images are placed before the text.

- **System prompt configuration**

  To prepend a unified instruction before the dataset prompt, use the ``system_prompt`` configuration:

  .. code-block:: yaml

      system_prompt: 'Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer}.'

Algorithm
--------

We use standard GRPO (Group Relative Policy Optimization) with **TIS** (Truncated Importance Sampling) enabled to stabilize training:

  .. code-block:: yaml

      importance_sampling_fix: True
      importance_sampling_clip: 2

Experiments show that enabling TIS prevents unbounded entropy growth and excessive sequence length increase, leading to more stable training.

Reward function:

- Correct: +1 (configurable via ``reward_max_val``)
- Incorrect: 0 (configurable via ``reward_min_val``)

Running the Script
-------------------

**1. Configuration File**

Recommended example config:

- ``examples/reasoning/config/vqa/qwen3-vl-2b-grpo-fsdp-geo3k.yaml``

**2. Key Parameters**

Before launching, review the configuration file. The main fields include:

- Paths: ``rollout.model.model_path`` (local path to the base model), ``data.train_data_paths`` (training data paths), etc.
- Model type: ``rollout.model.model_type`` must be set to ``qwen3_vl``.

**3. Launch Command**

Run the following commands to start a Ray cluster and begin training:

.. code-block:: bash

   cd /path_to_RLinf/ray_utils;
   rm /path_to_RLinf/ray_utils/ray_head_ip.txt;
   export TOKENIZERS_PARALLELISM=false
   bash start_ray.sh;
   if [ "$RANK" -eq 0 ]; then
       bash check_ray.sh 4
       bash examples/reasoning/run_main_grpo_vqa.sh qwen3-vl-2b-grpo-fsdp-geo3k
   else
     sleep 10d
     rm ray_utils/ray_head_ip.txt;
   fi

   sleep 10d

Technical Details
------------------

**Sequence Packing**

To improve training efficiency, the framework supports dynamic sequence packing. When ``enable_dynamic_batch_size: True`` is enabled,
multiple short sequences are packed into a single long sequence for training. The following parameters control packing behavior:

- ``max_tokens_per_mbs``: Maximum number of tokens per micro-batch.
- ``variable_seq_lengths``: When set to ``True``, padding is removed for compact packing. We set this to ``True`` in this example.

**FSDP2 Training**

Qwen3-VL training uses FSDP2 as the training backend. Key configurations include:

.. code-block:: yaml

   actor:
     fsdp_config:
       strategy: "fsdp2"
       gradient_checkpointing: True
       gradient_checkpointing_use_reentrant: False

**KL Monitoring between Rollout and Training**

The framework provides the ``actor/rollout_train_kl`` metric to monitor the difference between logprobs during the rollout phase and the training phase,
helping diagnose training stability issues. This curve is automatically displayed when both ``recompute_logprobs`` and ``return_logprobs`` are enabled.

Results
--------

We conducted experiments using Qwen3-VL-2B-Instruct. The training curves are shown below:

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/qwen3vl_grpo_reward.jpeg" style="width: 50%;"/>
       <p><em>reward with TIS</em></p>
     </div>
   </div>
   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/qwen3vl_grpo_entropy.jpeg" style="width: 50%;"/>
       <p><em>entropy with TIS</em></p>
     </div>
   </div>

After 750 training steps, the reward still shows an upward trend, and the entropy converges.

Without TIS, there is a high probability that the reward curve will collapse after a certain number of steps, and the entropy will be more unstable:

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/qwen3vl_grpo_reward_wo_tis.jpeg" style="width: 50%;"/>
       <p><em>reward without TIS</em></p>
     </div>
   </div>
   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/qwen3vl_grpo_entropy_wo_tis.jpeg" style="width: 50%;"/>
       <p><em>entropy without TIS</em></p>
     </div>
   </div>
