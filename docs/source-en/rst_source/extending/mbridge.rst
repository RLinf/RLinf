Megatron-Bridge
===============

RLinf supports `Megatron-Bridge <https://github.com/NVIDIA/Megatron-Bridge>`__
through the Megatron-LM training backend. This integration lets users start
Megatron-LM training directly from HuggingFace-format checkpoints, use model
architectures supported by Megatron-Bridge, and keep RLinf's training loop, data
pipeline, logging, and checkpoint workflow unchanged.

Use Megatron-Bridge when:

- the actor-side model is large and FSDP or FSDP2 becomes a performance bottleneck;

- the model architecture is not yet supported by RLinf's native Megatron-LM integration.

Megatron-Bridge resources:

- `Megatron-Bridge upstream repository <https://github.com/NVIDIA/Megatron-Bridge>`__

- `Megatron-Bridge version 0.3.0 used by RLinf <https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/v0.3.0>`__

- `Corresponding Megatron-LM version b0cc2706ddc60d2aefd5fff346445b5c013036a8 <https://github.com/NVIDIA/Megatron-LM/tree/b0cc2706ddc60d2aefd5fff346445b5c013036a8>`__

Environment Setup
-----------------

MBridge currently uses RLinf's agentic environment. Install the base
environment first:

.. code:: bash

   bash requirements/install.sh agentic
   source .venv/bin/activate

Install the extra Python packages required by the MBridge path:

.. code:: bash

   uv pip install transformers==4.57.1 bitsandbytes

The reasoning image does not include the ``megatron.bridge`` package.
Clone Megatron-Bridge and the matching Megatron-LM revision, then add both
source trees to ``PYTHONPATH``:

.. code:: bash

   export MBRIDGE_ROOT=/workspace/dataset/Megatron-Bridge-0.3.0
   export MEGATRON_LM_ROOT=/workspace/dataset/Megatron-LM-b0cc2706ddc60d2aefd5fff346445b5c013036a8

   mkdir -p "$(dirname "${MBRIDGE_ROOT}")" "$(dirname "${MEGATRON_LM_ROOT}")"
   git clone --branch v0.3.0 https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "${MBRIDGE_ROOT}"
   git clone https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_LM_ROOT}"
   git -C "${MEGATRON_LM_ROOT}" checkout b0cc2706ddc60d2aefd5fff346445b5c013036a8

   export PYTHONPATH="${MBRIDGE_ROOT}/src:${MEGATRON_LM_ROOT}:${PYTHONPATH}"
   export CUDA_DEVICE_MAX_CONNECTIONS=1
   python -c "from megatron.bridge import AutoBridge; print('Megatron-Bridge OK')"

If your cluster image already mounts these repositories, keep the same
``PYTHONPATH`` exports and skip the two ``git clone`` commands.

Download the model and dataset used by the reasoning example:

.. code:: bash

   # For faster downloads in mainland China, you can set:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
       --local-dir /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B

   mkdir -p /dataset/boba
   hf download inclusionAI/AReaL-boba-Data AReaL-boba-106k.jsonl \
       --repo-type dataset \
       --local-dir /dataset/boba

Download the model and dataset used by the VLM SFT example:

.. code:: bash

   # For faster downloads in mainland China, you can set:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download Qwen/Qwen2.5-VL-3B-Instruct \
       --local-dir /path/to/Qwen2.5-VL-3B-Instruct

   hf download keplerccc/Robo2VLM-1 \
       --repo-type dataset \
       --local-dir /path/to/Robo2VLM-1

.. warning::

   The Robo2VLM download places training and evaluation files in the same
   directory, for example ``train-00000-of-00262.parquet`` and
   ``test-0000X-of-00003.parquet``. Move them into separate directories before
   training. Otherwise, RLinf will read the whole dataset as training data.

The example SFT config expects ``/path/to/Robo2VLM-1/data`` and
``/path/to/Robo2VLM-1/eval_data``. If you store the dataset elsewhere, update
``data.train_data_paths`` and ``data.eval_data_paths`` accordingly.

Overview
--------

After MBridge is enabled, RLinf imports and builds the Megatron-LM model through
``Megatron-Bridge`` instead of relying on the traditional Megatron checkpoint
conversion workflow.

The key configuration is different for reasoning/RL and SFT tasks.

For reasoning tasks:

.. code:: yaml

   actor:
     training_backend: megatron
     megatron:
       mbridge: True
       use_hf_ckpt: True
       ckpt_convertor:
         hf_model_path: /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B

When ``actor.megatron.mbridge`` is ``True`` and ``use_hf_ckpt`` is ``True``,
RLinf reads the model path from ``actor.megatron.ckpt_convertor.hf_model_path``
and lets MBridge build the Megatron model provider.

For SFT tasks:

.. code:: yaml

   actor:
     training_backend: megatron
     model:
       model_path: /path/to/Qwen2.5-VL-3B-Instruct
       megatron_checkpoint: null
     megatron:
       use_hf_ckpt: True
       mbridge: True

When ``actor.megatron.mbridge`` is ``True``, RLinf reads the model path from
``actor.model.model_path`` and lets MBridge build the Megatron model provider.

Quick Start
-----------

1. Export the MBridge paths before launching training:

.. code:: bash

   export PYTHONPATH=/path/to/Megatron-Bridge-0.3.0/src:$PYTHONPATH
   export PYTHONPATH=/path/to/Megatron-LM-b0cc2706ddc60d2aefd5fff346445b5c013036a8:$PYTHONPATH
   export CUDA_DEVICE_MAX_CONNECTIONS=1

2. Prepare the HuggingFace model and data directories:

.. code:: text

   # Reasoning tasks need:
   /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B
   /dataset/boba/AReaL-boba-106k.jsonl
   # SFT tasks need:
   /path/to/Qwen2.5-VL-3B-Instruct
   /path/to/Robo2VLM-1/data
   /path/to/Robo2VLM-1/eval_data

3. Update the model, tokenizer, and dataset paths in the config.

Path Differences
----------------

MBridge reads HuggingFace checkpoint paths from different config entries for
different training tasks:

- Reasoning / RL tasks usually read the HuggingFace model path from
  ``actor.megatron.ckpt_convertor.hf_model_path``;

- SFT tasks usually read the HuggingFace model path from
  ``actor.model.model_path``;

- the tokenizer path is still specified by ``actor.tokenizer.tokenizer_model``.
  We recommend keeping it consistent with the model directory.

Therefore, do not only copy ``mbridge: True`` when migrating configs. Also check
whether the model path is configured in the entry used by the current task type.

Reasoning task example:

.. code:: yaml

   actor:
     tokenizer:
       tokenizer_model: "/path/to/model/DeepSeek-R1-Distill-Qwen-1.5B"
     training_backend: megatron
     megatron:
       mbridge: True
       use_hf_ckpt: True
       ckpt_convertor:
         hf_model_path: /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B

   data:
     train_data_paths: ["/dataset/boba/AReaL-boba-106k.jsonl"]
     val_data_paths: ["/dataset/boba/AReaL-boba-106k.jsonl"]

SFT example:

.. code:: yaml

   actor:
     model:
       model_type: "qwen2.5_vl"
       model_path: "/path/to/Qwen2.5-VL-3B-Instruct"
       megatron_checkpoint: null

     tokenizer:
       tokenizer_model: "/path/to/Qwen2.5-VL-3B-Instruct"

     megatron:
       use_hf_ckpt: True
       mbridge: True

   data:
     train_data_paths: "/path/to/Robo2VLM-1/data"
     eval_data_paths: "/path/to/Robo2VLM-1/eval_data"

4. Launch the corresponding training script.

Start reasoning training from the repository root. Use the Python entrypoint so
the MBridge override is applied explicitly:

.. code:: bash

   python examples/reasoning/main_grpo.py \
       --config-path "$(pwd)/examples/reasoning/config/math" \
       --config-name qwen2.5-1.5b-grpo-megatron \
       actor.megatron.mbridge=True

Start VLM SFT training from the repository root:

.. code:: bash

   bash examples/sft/run_vlm_sft.sh qwen2_5_vl_megatron_sft_vlm

Checkpoint Loading
------------------

When Megatron-Bridge is used in RLinf, RLinf saves both checkpoint formats:

- HuggingFace checkpoint;
- Megatron checkpoint.

The checkpoint directory is organized as follows:

.. code:: text

   /path/to/logs/qwen2.5-1.5b-grpo-megatron/checkpoints/
   ├── global_step_10/
   │   └── actor/
   │       ├── hf_model/
   │       │   ├── model.safetensors
   │       │   └── tokenizer.json
   │       ├── iter_0000010/
   │       │   ├── mp_rank_00/
   │       │   │   ├── distrib_optim.pt
   │       │   │   └── model_optim_rng.pt
   │       │   └── mp_rank_01/
   │       │       ├── distrib_optim.pt
   │       │       └── model_optim_rng.pt
   │       └── latest_checkpointed_iteration.txt
   └── global_step_20/
       └── ...

The ``hf_model`` directory stores HuggingFace-format model weights and tokenizer
files. The ``iter_XXXXXXX`` directory stores Megatron model weights and optimizer
states. ``latest_checkpointed_iteration.txt`` records the latest checkpointed
iteration. In this example, ``global_step_10/`` and ``global_step_20/`` are two
different checkpoints for step 10 and step 20.

For resume training, you can load only the Megatron checkpoint. The
HuggingFace-format checkpoint is not required.

.. code:: yaml

   runner:
     resume_dir: /path/to/logs/qwen2.5-1.5b-grpo-megatron/checkpoints/global_step_10

Practical Notes
---------------

- Keep ``actor.model.megatron_checkpoint: null`` when ``use_hf_ckpt: True``.
- Set ``actor.megatron.use_hf_ckpt: False`` only when loading a prepared
  Megatron checkpoint.
- For Qwen3-VL models, keep ``actor.model.apply_rope_fusion: False``.
- For Qwen2.5 models, ``qkv_bias`` is forced on for model compatibility.
- For Qwen3 models, ``qk_layernorm`` is forced on for model compatibility.
- Make sure the tokenizer path matches the HuggingFace model directory.

Troubleshooting
---------------

``model.megatron_checkpoint is required if use_hf_ckpt is False``
  ``use_hf_ckpt`` is disabled, but no Megatron checkpoint path was provided.
  Set ``actor.megatron.use_hf_ckpt: True`` or provide ``runner.resume_dir``.
``model.megatron_checkpoint should be None if use_hf_ckpt is True``
  HuggingFace loading and Megatron checkpoint loading are both enabled. Set
  ``actor.model.megatron_checkpoint: null``.

Qwen3-VL fails with a ``deepstack_visual_indexes`` assertion
  The model's visual deepstack configuration does not match the current pipeline
  split. First try ``pipeline_model_parallel_size: 1``. If pipeline parallelism
  is required, make sure the first language pipeline stage has enough layers to
  contain all ``deepstack_visual_indexes``. If you are using a reduced-layer
  checkpoint, also verify that the visual deepstack configuration matches the
  number of language model layers.
