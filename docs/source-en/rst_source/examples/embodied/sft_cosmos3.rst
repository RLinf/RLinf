Cosmos3 SFT
===========

Fine-tune NVIDIA Cosmos3-Nano (OmniMoT video world model) with SFT to adapt it to the LIBERO action space and simulator.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      Cosmos3-Nano · OmniMoT

   .. grid-item-card:: Methods
      :text-align: center

      FSDP2 SFT

   .. grid-item-card:: Data
      :text-align: center

      LIBERO (LeRobot)

   .. grid-item-card:: Hardware
      :text-align: center

      8× A800 80GB GPU

| **You'll do:** install → download and convert base weights to DCP → configure data paths → launch ``run_vla_sft.sh`` → convert checkpoint for evaluation.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Cosmos3-Nano base weights (DCP) · Wan2.2 VAE · LIBERO LeRobot v3 dataset.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO

      **1. Install**

      Install RLinf for Cosmos3 and clone ``cosmos-framework`` (required for data transforms, normalization stats, and DCP conversion):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         unset PYTHONPATH
         bash requirements/install.sh embodied --model cosmos3 --env libero && source .venv/bin/activate
         git clone https://github.com/NVIDIA/cosmos-framework.git ./cosmos-framework
         export COSMOS_FRAMEWORK_PATH="$PWD/cosmos-framework"

      **2. Download**

      Download the ``nvidia/Cosmos3-Nano`` diffusers checkpoint together with the companion Hugging Face assets cached under ``HF_HOME`` (used at SFT startup and during DCP conversion, including offline runs with ``HF_HUB_OFFLINE=1``):

      .. code-block:: bash

         # Base model (diffusers format; convert to DCP below) and Wan2.2 VAE
         hf download nvidia/Cosmos3-Nano --local-dir ./Cosmos3-Nano
         hf download Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e Wan2.2_VAE.pth --local-dir ./Wan2.2-TI2V-5B

         # Companion assets downloaded directly into the HF cache (required for offline mode)
         export HF_HOME=${HF_HOME:-~/.cache/huggingface}
         hf download Qwen/Qwen3-VL-8B-Instruct
         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include "vae/*"

      **3. Prepare Base Model and Data**

      Cosmos3 SFT warm-starts from a **DCP-format** Cosmos3-Nano base model and trains only the action heads. Diffusers directories store shards under ``transformer/`` rather than at the top level, so SFT cannot load them directly. Convert ``./Cosmos3-Nano`` to DCP inside the **same training venv** (DCP ``.metadata`` pickles are tied to the Python version):

      .. code-block:: bash

         export HF_HOME=${HF_HOME:-~/.cache/huggingface}
         export HF_HUB_OFFLINE=1
         export TRANSFORMERS_OFFLINE=1
         python -m cosmos_framework.scripts.convert_model_to_dcp \
           --checkpoint-path ./Cosmos3-Nano \
           --no-use-ema-weights \
           -o ./Cosmos3-Nano-DCP

      Training uses the original LIBERO LeRobot v3 layout (``meta/`` and ``data/``) with no offline preprocessing: the Cosmos3 data loader applies ``frame_wise_relative`` + ``rot6d`` + ``quantile_rot`` online via cosmos-framework's ``action_policy_libero_all_nano`` recipe. Set the model, VAE, and dataset paths in ``examples/sft/config/libero_sft_cosmos3.yaml``:

      .. code-block:: yaml

         actor:
           model:
             model_path: ./Cosmos3-Nano-DCP
             wan_vae_path: ./Wan2.2-TI2V-5B/Wan2.2_VAE.pth
             load_to_device: false

         data:
           train_data_paths: ./LIBERO_LeRobot
           data_type: "libero_all"          # all LIBERO suites; or "libero_10"

      .. warning::

         Keep ``actor.model.load_to_device: false``. At initialization, ``net`` (bf16 ~27 GB) + ``net_ema`` (fp32 ~54 GB) ≈ 81 GB, so eager GPU loading OOMs on an 80 GB GPU before FSDP2 sharding runs.

      **4. Launch**

      Start SFT from the repository root (checkpoints save every ``save_interval`` steps under ``checkpoints/global_step_<N>/``, and logs write to ``logs/YYYYMMDD-HHMMSS-libero_sft_cosmos3/run_embodiment.log``; set ``runner.resume_dir`` to a saved ``global_step_<N>`` directory to resume):

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_cosmos3

.. dropdown:: Key configuration

   Most fields can be kept from ``examples/sft/config/libero_sft_cosmos3.yaml``. Key settings to tune:

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
        - Must be ``false`` so FSDP2 shards directly from CPU onto GPUs.
      * - ``actor.model.ema_enabled``
        - Cosmos3 power-EMA (rate 0.1); holds an fp32 copy in memory (set ``false`` temporarily if still OOM).
      * - ``actor.micro_batch_size`` / ``global_batch_size``
        - Per-GPU micro batch and global batch (default 32 / 2048).
      * - ``actor.optim.lr`` / ``lr_warmup_steps``
        - Learning rate and warmup (default 5e-5 / 500); action heads have an additional 5× multiplier in ``model/cosmos3.yaml``.
      * - ``runner.max_steps`` / ``save_interval``
        - Training steps and checkpoint interval (default 5000 / 500).

.. dropdown:: Action representation (rot6d 10-D vs axis-angle 7-D)

   Cosmos3 represents actions internally as **10-D rot6d**, while LIBERO uses **7-D axis-angle** (rot6d is continuous, avoiding the ±π discontinuity of axis-angle):

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

   The Cosmos3 data loader converts 7-D LIBERO actions to 10-D rot6d online during training, and evaluation converts predictions back to 7-D. Training and evaluation must use the same recipe and normalization stats file (``libero_native_frame_wise_relative_rot6d.json`` via ``action_stats_path``); see :doc:`SGLang Evaluation <../../evaluations/guides/cosmos3_sglang>`.

.. dropdown:: Checkpoint conversion

   SFT writes FSDP2 checkpoints to ``checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``, which cannot be evaluated directly. Convert them into a diffusers directory ``model_diffusers`` with ``toolkits/cosmos3/convert_checkpoint.py`` (strips ``omni.`` prefixes, saves DCP, exports HF, and splits diffusers components; pass ``--use-ema-weights`` to keep EMA weights):

   .. code-block:: bash

      export SRC="./checkpoints/global_step_500/actor/model_state_dict/full_weights.pt"
      export OUT="./converted"
      export LIBERO_ROOT="./LIBERO_LeRobot"
      mkdir -p "$OUT"

      python toolkits/cosmos3/convert_checkpoint.py --src "$SRC" --out "$OUT"

   .. warning::

      Run conversion in a **cosmos-framework environment** with ``transformers 4.57.x`` and ``diffusers 0.39.0`` (which provides ``Cosmos3OmniTransformer``), Wan2.2 diffusers VAE, and an offline HF cache (``HF_HUB_OFFLINE=1``).

   Point ``rollout.model.model_path`` to the resulting ``model_diffusers`` directory in :doc:`Cosmos3 SGLang Evaluation <../../evaluations/guides/cosmos3_sglang>`.

Visualization and Results
-------------------------

Watch training logs and curves in TensorBoard (``tensorboard --logdir ./logs``); see :doc:`Training metrics <../../reference/metrics>` for metric definitions.

.. dropdown:: Troubleshooting

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - Symptom
        - Fix
      * - OOM at initialization (even on a single 80 GB GPU)
        - Set ``actor.model.load_to_device: false``.
      * - Loading reports action head shape mismatch
        - Confirm ``keys_to_skip_loading`` skips the base DROID 8-D action head (enabled by default in ``model/cosmos3.yaml``).
      * - Hugging Face download hangs
        - Use offline mode with local cache: ``HF_HUB_OFFLINE=1``, ``TRANSFORMERS_OFFLINE=1``, and ``HF_HOME`` pointing to the cache.
      * - Wrong actions or abnormal loss
        - Verify ``action_normalization`` (``quantile_rot``) and the stats file match between training and evaluation.
