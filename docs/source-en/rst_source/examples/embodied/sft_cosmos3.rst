Cosmos3 Supervised Fine-Tuning (LIBERO)
========================================

Fine-tune the NVIDIA Cosmos3-Nano (OmniMoT video world model) via SFT to adapt it to the LIBERO action space, fitting the libero simulator.

Overview
----------------------------------------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Cosmos3-Nano · OmniMoT

   .. grid-item-card:: Method
      :text-align: center

      SFT on the Libero simulator

   .. grid-item-card:: Data
      :text-align: center

      LIBERO (LeRobot)

   .. grid-item-card:: Hardware
      :text-align: center

      8× A800 80GB GPU

| **You will:** install → prepare base model & data → train with ``run_vla_sft.sh`` → obtain a Cosmos3 model adapted for the libero simulator.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Cosmos3-Nano base weights (DCP) · Wan2.2 VAE · a downloaded LIBERO LeRobot dataset.

Installation
----------------------------------------

.. include:: _setup_common.rst

Install Cosmos3 (add ``--env libero`` if you also need simulation eval):

.. code-block:: bash

   unset PYTHONPATH
   # Users in China can add --use-mirror to speed up downloads.
   bash requirements/install.sh embodied --model cosmos3 --env libero
   source .venv/bin/activate

Cosmos3's data transforms and normalization stats depend on cosmos-framework; clone it and set the path before training:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cosmos-framework.git /path/to/cosmos-framework
   export COSMOS_FRAMEWORK_PATH=/path/to/cosmos-framework

Offline Cache: Qwen3-VL-8B-Instruct and Wan2.2 Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SFT and base checkpoint conversion pull three model files from Hugging Face. On offline machines, pre-download them into the HF cache and enable offline mode (``HF_HUB_OFFLINE=1``) so ``from_pretrained`` / ``hf download`` read directly from cache, otherwise workers hang on network retries:

- **Qwen3-VL-8B-Instruct** (tokenizer / VLM config): used at SFT startup and when ``convert_model_to_dcp`` rebuilds the model.
- **Wan2.2-TI2V-5B** and **Wan2.2-TI2V-5B-Diffusers**: downloaded only when ``convert_model_to_dcp`` rebuilds the model.

Download directly into the HF cache (recommended; cache layout is automatically correct):

.. code-block:: bash

   export HF_HOME=${HF_HOME:-~/.cache/huggingface}
   # Use a mirror in China: export HF_ENDPOINT=https://hf-mirror.com
   hf download Qwen/Qwen3-VL-8B-Instruct
   hf download Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e Wan2.2_VAE.pth
   hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include "vae/*"

Prepare Base Model
----------------------------------------

Cosmos3 SFT **warm-starts from the Cosmos3-Nano base model, training only the action heads**. ``actor.model.model_path`` must point to a **DCP-format** base weights directory, and ``wan_vae_path`` to the Wan2.2 VAE:

.. code-block:: yaml

   defaults:
     - model/cosmos3@actor.model      # pulls in examples/sft/config/model/cosmos3.yaml

   actor:
     model:
       model_path: /path/to/Cosmos3-Nano-DCP
       wan_vae_path: /path/to/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       load_to_device: false          # see warning below

Prepare Base DCP Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Download Cosmos3-Nano (diffusers format)**

Download ``nvidia/Cosmos3-Nano`` from Hugging Face. It is in **diffusers/safetensors format** (``model_index.json`` + ``transformer/*.safetensors`` shards + ``vae/`` + ``text_tokenizer/`` etc.), **not DCP** — SFT cannot use it directly yet; it must be converted to DCP in Step 2.

.. code-block:: bash

   # Use a mirror in China: export HF_ENDPOINT=https://hf-mirror.com
   hf download nvidia/Cosmos3-Nano \
     --local-dir /path/to/Cosmos3-Nano

.. note::

   The base model can be downloaded with ``--local-dir`` to a **plain directory**.

**Step 2: Convert to DCP**

SFT cannot load the base from a diffusers directory directly: the cosmos3 base loader ``_load_base_weights`` uses ``_is_safetensors_checkpoint`` to detect format — it only checks whether ``*.safetensors`` files exist at the **top level** of the path (non-recursive). The diffusers directory places shards in ``transformer/`` subdirectories with nothing at the top level, so the check returns ``False``, falling through to the DCP loading branch, but the diffusers directory is not DCP, so loading fails. Therefore you must first convert the diffusers weights to DCP via cosmos-framework's ``convert_model_to_dcp.py`` before using them as ``model_path``.

.. note::

   This step rebuilds the entire OmniMoTModel and downloads ``Qwen3-VL-8B-Instruct`` (tokenizer) and ``Wan2.2-TI2V-5B``'s ``Wan2.2_VAE.pth`` (VAE) via cosmos's ``hf_cli`` — on offline machines, pre-download both into the HF cache (see above), then run with ``HF_HUB_OFFLINE=1``.

.. code-block:: bash

   # Run in the training venv (ensures DCP metadata matches the training Python version)
   export HF_HOME=/path/to/hf-cache
   export HF_HUB_OFFLINE=1          # read from cache; requires the two models above pre-downloaded
   export TRANSFORMERS_OFFLINE=1
   python -m cosmos_framework.scripts.convert_model_to_dcp \
     --checkpoint-path /path/to/Cosmos3-Nano \
     --no-use-ema-weights \
     -o /path/to/Cosmos3-Nano-DCP

.. note::

   - ``--checkpoint-path`` accepts either a model name (triggers HF download) or a local diffusers directory; on offline machines, download first and point to the local path.
   - The DCP ``.metadata`` is pickle and **tied to the Python version it was saved with**, so ``convert_model_to_dcp`` must run in the **same venv** as training for the DCP to load (the ``-py311`` directory suffix means exactly this).

Prepare Data
----------------------------------------

Training data can be the **original** LeRobot v3 layout (containing ``meta/``, ``data/``) of LIBERO, specified via ``data.train_data_paths``.

The ``frame_wise_relative`` + ``rot6d`` + ``quantile_rot`` action transform **does not require pre-processing the dataset**: ``rlinf/data/datasets/cosmos3/dataloader.py`` calls cosmos-framework's experiment ``action_policy_libero_all_nano``, which converts the original 7-D actions to 10-D rot6d at load time and applies quantile normalization. The normalization stats file ``libero_native_frame_wise_relative_rot6d.json`` is computed once by cosmos-framework and referenced via ``action_stats_path`` on the eval side. **Training and eval must use the same recipe + the same stats file**, otherwise de-normalization will be misaligned.

.. code-block:: yaml

   data:
     train_data_paths: /path/to/LIBERO_LeRobot
     data_type: "libero_all"          # all LIBERO task suites; can also train just "libero_10"
     num_workers: 4
     prefetch_factor: 4
     val_ratio: 0.01

Launch Training
----------------------------------------

Run from the RLinf repo root:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_cosmos3

**What this command does:** reads ``examples/sft/config/libero_sft_cosmos3.yaml``, shards the Cosmos3 model with FSDP2 on each GPU and trains, saving checkpoints every ``save_interval`` steps to ``.../checkpoints/global_step_<N>/``, with logs to ``logs/<timestamp>-libero_sft_cosmos3/run_embodiment.log``.

.. warning::

   **Initialization must set** ``actor.model.load_to_device: false``. When Cosmos3 is built, ``net`` (bf16 ~27GB) + ``net_ema`` (fp32 ~54GB) ≈ 81GB; the default eager ``model.to(device)`` on a single 80GB GPU will OOM **before** FSDP2 ``fully_shard`` sharding runs. Setting it to ``false`` keeps the model on CPU and lets ``fully_shard`` shard it directly onto the GPU.

To resume from a checkpoint: set ``runner.resume_dir`` to a ``global_step_<N>`` directory and re-run.

Key Configuration
----------------------------------------

Most fields can be copied from ``libero_sft_cosmos3.yaml``. The ones that really matter:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field
     - Description
   * - ``actor.model.model_path``
     - Cosmos3-Nano base weights directory (DCP).
   * - ``actor.model.wan_vae_path``
     - Wan2.2 VAE weights (``Wan2.2_VAE.pth``).
   * - ``actor.model.load_to_device``
     - **Must be ``false``** (see warning above).
   * - ``actor.model.ema_enabled``
     - Cosmos3 power-EMA (rate=0.1); updates ``net_ema`` each optimizer step; occupies an fp32 copy of the network in memory; if still OOM, temporarily set ``false``.
   * - ``actor.micro_batch_size`` / ``global_batch_size``
     - Per-GPU micro batch and global batch (example 32 / 2048).
   * - ``actor.optim.lr`` / ``lr_warmup_steps``
     - Learning rate and warmup (example 5e-5 / 500). Action heads have an additional 5× LR multiplier (``lr_multipliers`` in ``model/cosmos3.yaml``).
   * - ``runner.max_steps`` / ``save_interval``
     - Training steps and checkpoint interval (example 5000 / 500).


Special Note: Action Representation — rot6d 10-D vs axis-angle 7-D
------------------------------------------------------------------------

Cosmos3 internally represents actions as **10-D rot6d**; the LIBERO environment uses **7-D axis-angle**. This is key to understanding the whole pipeline:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Representation
     - Dim
     - Composition
   * - ``raw_action_dim`` (model side)
     - 10
     - 3 translation + 6 rot6d + 1 gripper
   * - ``action_dim`` (env side)
     - 7
     - 3 translation + 3 axis-angle + 1 gripper

rot6d is used because axis-angle is discontinuous at ±π, making it hard for diffusion models to regress; rot6d (the first two columns of a rotation matrix) is continuous and easier to learn.

At training time, the cosmos data loader converts LIBERO's 7-D actions to 10-D rot6d online (see "Prepare Data" above); at inference time, the model's 10-D rot6d output is converted back to 7-D for the environment. See :doc:`SGLang Evaluation <../../evaluations/guides/cosmos3_sglang>`.


Checkpoint Conversion
----------------------------------------

The SFT output ``.../checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt`` is in FSDP2 sharded format and **cannot be used for eval directly**. Cosmos3 eval uses a diffusers component directory ``model_diffusers`` (containing ``model_index.json`` / ``transformer/`` / ``vae/`` / ``text_tokenizer/`` / ``scheduler/``). Converting from ``full_weights.pt`` to ``model_diffusers`` takes four steps (requires cosmos-framework):

.. code-block:: text

   1) full_weights.pt  --strip omni. prefix-->  model.safetensors
   2) model.safetensors  --to DCP directory-->   model_dcp
   3) model_dcp  --cosmos_framework.scripts.export_model + cosmos config-->  model_hf
   4) model_hf   --cosmos_framework.scripts.convert_model_to_diffusers-->  model_diffusers
      and change model_index.json _class_name to Cosmos3OmniDiffusersPipeline

Set the path variables first (used by all four steps below):

.. code-block:: bash

   # RLinf checkpoint produced by SFT (omni.net.* prefix)
   export SRC="/path/to/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt"
   # Conversion working directory (all four step outputs go here)
   export OUT="/path/to/converted"
   # cosmos_framework checkpoint conversion needs the LIBERO_ROOT env var
   export LIBERO_ROOT=/path/to/LIBERO_LeRobot_v3
   mkdir -p "$OUT"

**Step 1:** ``full_weights.pt`` → ``model.safetensors`` (strip the ``omni.`` prefix so cosmos recognizes ``net.*`` / ``net_ema.*``).

.. code-block:: bash

   python - <<'PY'
   import torch, os
   from safetensors.torch import save_file

   SRC  = os.environ["SRC"]
   OUTD = os.path.join(os.environ["OUT"], "model_safetensors")
   os.makedirs(OUTD, exist_ok=True)

   print("Loading full_weights.pt (~91GB)...")
   sd = torch.load(SRC, map_location="cpu", weights_only=False)
   print(f"Loaded {len(sd)} keys")

   # Strip "omni." prefix: omni.net.* -> net.*, omni.net_ema.* -> net_ema.*
   stripped = {(k[5:] if k.startswith("omni.") else k): v.contiguous() for k, v in sd.items()}
   print(f"Stripped to {len(stripped)} keys (e.g. {list(stripped.keys())[:3]})")

   dst = os.path.join(OUTD, "model.safetensors")
   save_file(stripped, dst)
   print(f"Saved {os.path.getsize(dst)/1e9:.1f} GB -> {dst}")
   PY

.. note::

   ``export SRC=... OUT=...`` before running (the code block reads these two env vars). ``weights_only=False`` because RLinf stores the state dict with metadata.

**Step 2:** ``model.safetensors`` → ``model_dcp`` (convert to a DCP directory; ``export_model`` only accepts DCP format).

.. code-block:: bash

   python - <<'PY'
   import os, math, torch
   from safetensors.torch import load_file
   import torch.distributed.checkpoint as dcp
   from torch.distributed.checkpoint.filesystem import FileSystemWriter
   from cosmos_framework.checkpoint.dcp import CustomSavePlanner

   SRC = os.path.join(os.environ["OUT"], "model_safetensors", "model.safetensors")
   OUT = os.path.join(os.environ["OUT"], "model_dcp", "model")
   os.makedirs(OUT, exist_ok=True)

   state_dict = load_file(SRC)
   print(f"Loaded {len(state_dict)} keys")

   # ~5GB per shard (aligned with transformers default max_shard_size)
   nbytes  = sum(v.numel() * v.element_size() for v in state_dict.values() if isinstance(v, torch.Tensor))
   nshards = max(1, math.ceil(nbytes / (5 * 1024**3)))
   writer  = FileSystemWriter(OUT, thread_count=nshards)
   dcp.save(state_dict=state_dict, storage_writer=writer, planner=CustomSavePlanner())
   print(f"Saved DCP -> {OUT}")
   PY

.. note::

   This step depends on ``cosmos_framework.checkpoint.dcp.CustomSavePlanner``, so it must be run in the cosmos-framework environment (same env as steps 3 and 4 below). The output ``$OUT/model_dcp/`` is the input to ``export_model --checkpoint-path`` in the next step.

**Steps 3 & 4:** DCP → HF → diffusers (cosmos-framework's built-in scripts).

.. code-block:: bash

   python -m cosmos_framework.scripts.export_model \
     --checkpoint-path "$OUT/model_dcp" \
     --config-file cosmos_framework/configs/base/config.py \
     --experiment action_policy_libero_all_nano \
     --no-use-ema-weights \
     -o "$OUT/model_hf"

   python -m cosmos_framework.scripts.convert_model_to_diffusers \
     --checkpoint-path "$OUT/model_hf" \
     -o "$OUT/model_diffusers"

   # convert_model_to_diffusers writes a _class_name that is not the standard diffusers format; change it to Cosmos3OmniDiffusersPipeline
   python -c "
   import json
   p = '$OUT/model_diffusers/model_index.json'
   d = json.load(open(p))
   if d.get('_class_name') != 'Cosmos3OmniDiffusersPipeline':
       d['_class_name'] = 'Cosmos3OmniDiffusersPipeline'
       json.dump(d, open(p, 'w'), indent=2)"

.. note::

   ``--config-file`` is cosmos-framework's model structure config; just use the relative path ``cosmos_framework/configs/base/config.py``.

Step 3 exports the DCP weights to HF format ``model_hf`` using the cosmos training config;

Step 4 splits the HF weights into a diffusers component directory ``model_diffusers``, and immediately changes ``model_index.json``'s ``_class_name`` to ``Cosmos3OmniDiffusersPipeline`` (standard diffusers checkpoint format).

.. warning::

   The conversion must be run in a **cosmos-framework environment**: ``transformers 4.57.x`` + ``diffusers 0.39.0`` (which has ``Cosmos3OmniTransformer``). Higher transformers versions crash at ``save_pretrained`` and their diffusers lacks the cosmos3 classes. The conversion also needs the Wan2.2 VAE (diffusers format) and an offline HF cache (``HF_HUB_OFFLINE=1``). Specific paths and versions depend on your cosmos-framework deployment.

The resulting ``model_diffusers`` is the ``rollout.model.model_path`` for :doc:`Cosmos3 SGLang Evaluation <../../evaluations/guides/cosmos3_sglang>`.

Visualization and Results
----------------------------------------

View training logs and curves:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

See :doc:`Training Metrics </rst_source/reference/metrics>` for metric definitions.

FAQ
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Fix
   * - OOM at initialization (even on a single 80GB GPU)
     - Set ``actor.model.load_to_device: false``.
   * - Loading reports action head shape mismatch
     - Confirm ``keys_to_skip_loading`` skips the base DROID 8-D action head (``model/cosmos3.yaml`` has this by default).
   * - HuggingFace download hangs
     - Offline + local cache: ``HF_HUB_OFFLINE=1``, ``TRANSFORMERS_OFFLINE=1``, and ``HF_HOME`` pointing to the local cache.
   * - Actions are wrong / loss is abnormal
     - Verify ``action_normalization`` (``quantile_rot``) and the stats file are the same source on both training and eval sides.
