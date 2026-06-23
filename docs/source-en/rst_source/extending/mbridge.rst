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

MBridge currently uses RLinf's agentic environment. Install it with:

.. code:: bash

   bash requirements/install.sh agentic
   source .venv/bin/activate

In addition, update and install the following extra packages:

.. code:: bash

   uv pip install transformers==4.57.1 bitsandbytes

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
         hf_model_path: /path/to/huggingface_model

When ``actor.megatron.mbridge`` is ``True`` and ``use_hf_ckpt`` is ``True``,
RLinf reads the model path from ``actor.megatron.ckpt_convertor.hf_model_path``
and lets MBridge build the Megatron model provider.

For SFT tasks:

.. code:: yaml

   actor:
     training_backend: megatron
     model:
       model_path: /path/to/huggingface_model
       megatron_checkpoint: null
     megatron:
       use_hf_ckpt: True
       mbridge: True

When ``actor.megatron.mbridge`` is ``True``, RLinf reads the model path from
``actor.model.model_path`` and lets MBridge build the Megatron model provider.

Quick Start
-----------

1. Add Megatron-Bridge and the corresponding Megatron-LM version to
   ``PYTHONPATH``:

.. code:: bash

   export PYTHONPATH=/path/to/Megatron-Bridge/src:$PYTHONPATH
   export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
   export CUDA_DEVICE_MAX_CONNECTIONS=1

2. Prepare a HuggingFace model directory, for example:

.. code:: text

   /path/to/Qwen2.5-VL-3B-Instruct

3. Update the model and tokenizer paths in the config.

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
         hf_model_path: /path/to/huggingface_model

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

4. Launch the corresponding training script.

Start reasoning training from the repository root:

.. code:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron

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
