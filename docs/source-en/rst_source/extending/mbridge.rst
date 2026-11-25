LLM Model (Megatron-Bridge)
===========================

RLinf supports `Megatron-Bridge <https://github.com/NVIDIA/Megatron-Bridge>`__
through the Megatron-LM training backend. Start Megatron-LM training directly
from HuggingFace-format checkpoints, use model architectures supported by
Megatron-Bridge, and keep RLinf's training loop, data pipeline, logging, and
checkpoint workflow unchanged.

Use Megatron-Bridge when:

- the actor-side model is large and FSDP or FSDP2 becomes a performance bottleneck;
- the model architecture is not yet supported by RLinf's native Megatron-LM integration.

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Item
     - Details
   * - Requirements
     - Python 3.12 via ``bash requirements/install.sh agentic`` (installs ``megatron-bridge`` 0.5.0, ``nvidia-modelopt`` 0.45.0, and Megatron-LM ``core_r0.18.0``)
   * - Key YAML toggles
     - ``actor.training_backend: megatron``, ``actor.megatron.mbridge: True``, and ``actor.megatron.use_hf_ckpt: True``
   * - Supported tasks
     - Reasoning RL and VLM SFT
   * - Verification command
     - ``python -c "from megatron.bridge import AutoBridge; print('Megatron-Bridge OK')"``

Upstream resources:

- `Megatron-Bridge upstream repository <https://github.com/NVIDIA/Megatron-Bridge>`__
- `Megatron-Bridge version 0.5.0 used by RLinf <https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/v0.5.0>`__
- `Corresponding Megatron-LM branch core_r0.18.0 <https://github.com/NVIDIA/Megatron-LM/tree/core_r0.18.0>`__

Environment Setup
-----------------

MBridge uses RLinf's agentic environment. ``install.sh`` installs
Megatron-Bridge and the matching Megatron-LM together with it, so there is no
need to clone extra repositories or set ``PYTHONPATH``:

.. code-block:: bash

   bash requirements/install.sh agentic && source .venv/bin/activate
   python -c "from megatron.bridge import AutoBridge; print('Megatron-Bridge OK')"

This creates a Python 3.12 virtual environment, installs ``megatron-bridge``
0.5.0 and ``nvidia-modelopt`` 0.45.0, and clones Megatron-LM ``core_r0.18.0``
into the virtual environment directory, which ``.venv/bin/activate`` adds to
``PYTHONPATH``. To reuse an existing Megatron-LM checkout, set
``MEGATRON_PATH=<megatron_dir>`` before installing; ``install.sh`` reuses
that directory as is without switching branches, so verify it is on
``core_r0.18.0``.

.. note::

   Megatron-Bridge requires Python 3.12. If you pass a lower version with
   ``--python``, ``install.sh`` skips Megatron-Bridge with a WARNING and the
   import check above fails. The virtual environment in the reasoning image
   uses Python 3.11 and omits ``megatron.bridge``; install the environment with
   the commands above to use MBridge.

Download the model and dataset used by the reasoning example:

.. code-block:: bash

   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
       --local-dir ./DeepSeek-R1-Distill-Qwen-1.5B

   mkdir -p ./dataset/boba
   hf download inclusionAI/AReaL-boba-Data AReaL-boba-106k.jsonl \
       --repo-type dataset \
       --local-dir ./dataset/boba

Download the model and dataset used by the VLM SFT example:

.. code-block:: bash

   hf download Qwen/Qwen2.5-VL-3B-Instruct \
       --local-dir ./Qwen2.5-VL-3B-Instruct

   hf download keplerccc/Robo2VLM-1 \
       --repo-type dataset \
       --local-dir ./Robo2VLM-1

.. warning::

   The Robo2VLM download places training and evaluation files in the same
   directory, for example ``train-00000-of-00262.parquet`` and
   ``test-0000X-of-00003.parquet``. Move them into separate directories before
   training. Otherwise, RLinf reads the whole dataset as training data.

The example SFT config expects ``./Robo2VLM-1/data`` and
``./Robo2VLM-1/eval_data``. If you store the dataset elsewhere, update
``data.train_data_paths`` and ``data.eval_data_paths`` accordingly.

Enabling Megatron-Bridge on Supported Models
--------------------------------------------

After MBridge is enabled, RLinf imports and builds the Megatron-LM model through
Megatron-Bridge instead of running an offline Megatron checkpoint conversion
step.

Key Configuration
~~~~~~~~~~~~~~~~~

The model path key differs between reasoning RL and SFT tasks.

For reasoning tasks, set ``actor.megatron.mbridge`` and ``use_hf_ckpt`` to
``True``; RLinf reads the model path from
``actor.megatron.ckpt_convertor.hf_model_path``:

.. code-block:: yaml

   actor:
     training_backend: megatron
     megatron:
       mbridge: True
       use_hf_ckpt: True
       ckpt_convertor:
         hf_model_path: ./DeepSeek-R1-Distill-Qwen-1.5B

For SFT tasks, set ``actor.megatron.mbridge`` to ``True``; RLinf reads the
model path from ``actor.model.model_path``:

.. code-block:: yaml

   actor:
     training_backend: megatron
     model:
       model_path: ./Qwen2.5-VL-3B-Instruct
       megatron_checkpoint: null
     megatron:
       use_hf_ckpt: True
       mbridge: True

Quick Start
~~~~~~~~~~~

1. Activate the environment and set the CUDA connection environment variable before launching training:

.. code-block:: bash

   source .venv/bin/activate
   export CUDA_DEVICE_MAX_CONNECTIONS=1

2. Prepare the HuggingFace model and data directories:

.. code-block:: text

   # Reasoning tasks need:
   ./DeepSeek-R1-Distill-Qwen-1.5B
   ./dataset/boba/AReaL-boba-106k.jsonl
   # SFT tasks need:
   ./Qwen2.5-VL-3B-Instruct
   ./Robo2VLM-1/data
   ./Robo2VLM-1/eval_data

3. Update the model, tokenizer, and dataset paths in the config (see `Path Differences`_ below).

4. Launch the training script from the repository root.

For reasoning training, use the Python entrypoint so the MBridge override is
applied explicitly:

.. code-block:: bash

   python examples/reasoning/main_grpo.py \
       --config-path "$(pwd)/examples/reasoning/config/math" \
       --config-name qwen2.5-1.5b-grpo-megatron \
       +actor.megatron.mbridge=True

For VLM SFT training:

.. code-block:: bash

   bash examples/sft/run_vlm_sft.sh qwen2_5_vl_megatron_sft_vlm

Path Differences
~~~~~~~~~~~~~~~~

MBridge reads HuggingFace checkpoint paths from different config entries by task
type:

- Reasoning / RL tasks read the HuggingFace model path from
  ``actor.megatron.ckpt_convertor.hf_model_path``.
- SFT tasks read the HuggingFace model path from ``actor.model.model_path``.
- Both task types read the tokenizer path from
  ``actor.tokenizer.tokenizer_model``. Keep it identical to the model directory.

When migrating a config, check both ``mbridge: True`` and the task-specific
model path entry.

Reasoning task example:

.. code-block:: yaml

   actor:
     tokenizer:
       tokenizer_model: "./DeepSeek-R1-Distill-Qwen-1.5B"
     training_backend: megatron
     megatron:
       mbridge: True
       use_hf_ckpt: True
       ckpt_convertor:
         hf_model_path: ./DeepSeek-R1-Distill-Qwen-1.5B

   data:
     train_data_paths: ["./dataset/boba/AReaL-boba-106k.jsonl"]
     val_data_paths: ["./dataset/boba/AReaL-boba-106k.jsonl"]

SFT example:

.. code-block:: yaml

   actor:
     model:
       model_type: "qwen2.5_vl"
       model_path: "./Qwen2.5-VL-3B-Instruct"
       megatron_checkpoint: null

     tokenizer:
       tokenizer_model: "./Qwen2.5-VL-3B-Instruct"

     megatron:
       use_hf_ckpt: True
       mbridge: True

   data:
     train_data_paths: "./Robo2VLM-1/data"
     eval_data_paths: "./Robo2VLM-1/eval_data"

Checkpoint Loading
~~~~~~~~~~~~~~~~~~

With Megatron-Bridge enabled, RLinf saves both a HuggingFace checkpoint and a
Megatron checkpoint at each save step:

.. code-block:: text

   ./logs/qwen2.5-1.5b-grpo-megatron/checkpoints/
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

``hf_model`` stores HuggingFace-format weights and tokenizer files,
``iter_XXXXXXX`` stores Megatron model weights and optimizer states, and
``latest_checkpointed_iteration.txt`` records the latest checkpointed iteration.

To resume training, load only the Megatron checkpoint through
``runner.resume_dir``; the HuggingFace-format checkpoint is not required:

.. code-block:: yaml

   runner:
     resume_dir: ./logs/qwen2.5-1.5b-grpo-megatron/checkpoints/global_step_10

Troubleshooting & Version Notes
-------------------------------

Version and Model Notes
~~~~~~~~~~~~~~~~~~~~~~~

- RLinf pins ``megatron-bridge`` 0.5.0, ``nvidia-modelopt`` 0.45.0, and
  Megatron-LM ``core_r0.18.0`` on Python 3.12.
- Keep ``actor.model.megatron_checkpoint: null`` when ``use_hf_ckpt: True``.
- Set ``actor.megatron.use_hf_ckpt: False`` only when loading a prepared
  Megatron checkpoint.
- For Qwen3-VL models, keep ``actor.model.apply_rope_fusion: False``.
- For Qwen2.5 models, ``qkv_bias`` is forced on for model compatibility.
- For Qwen3 models, ``qk_layernorm`` is forced on for model compatibility.
- Keep ``actor.tokenizer.tokenizer_model`` aligned with the HuggingFace model
  directory.

Common Errors
~~~~~~~~~~~~~

``model.megatron_checkpoint is required if use_hf_ckpt is False``
  ``use_hf_ckpt`` is disabled, but no Megatron checkpoint path was provided.
  Set ``actor.megatron.use_hf_ckpt: True`` or provide ``runner.resume_dir``.

``model.megatron_checkpoint should be None if use_hf_ckpt is True``
  HuggingFace loading and Megatron checkpoint loading are both enabled. Set
  ``actor.model.megatron_checkpoint: null``.

Qwen3-VL fails with a ``deepstack_visual_indexes`` assertion
  The model's visual deepstack configuration does not match the pipeline split.
  First try ``pipeline_model_parallel_size: 1``. If pipeline parallelism is
  required, verify the first language pipeline stage has enough layers to
  contain all ``deepstack_visual_indexes``. For a reduced-layer checkpoint,
  also check that the visual deepstack configuration matches the number of
  language model layers.
