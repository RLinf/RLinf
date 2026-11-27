Cosmos3 Supervised Fine-Tuning and NFT Training (LIBERO)
========================================================

Fine-tune NVIDIA Cosmos3-Nano or Cosmos3-Edge on LIBERO demonstrations with FSDP2, export the checkpoint to Diffusers format, and run Diffusion-NFT reinforcement learning with SGLang rollout.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Cosmos3-Nano · Cosmos3-Edge · OmniMoT

   .. grid-item-card:: Method
      :text-align: center

      SFT · Diffusion-NFT (GRPO)

   .. grid-item-card:: Data
      :text-align: center

      LIBERO (LeRobot v3)

   .. grid-item-card:: Hardware
      :text-align: center

      4–8× A800 / A100 80 GB GPU

| **You will:** install dependencies → prepare base weights and LIBERO data → run SFT → export ``model_diffusers`` → evaluate or continue with Edge NFT training.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Cosmos3 base checkpoint (DCP) · Wan2.2 VAE · LIBERO LeRobot v3 dataset.

Installation
------------

.. include:: _setup_common.rst

Install the Cosmos3 training environment (``--env libero`` adds the LIBERO simulator):

.. tab-set::

   .. tab-item:: Global

      .. code-block:: bash

         unset PYTHONPATH
         bash requirements/install.sh embodied --model cosmos3 --env libero
         source .venv/bin/activate

   .. tab-item:: China Mirror

      .. code-block:: bash

         unset PYTHONPATH
         bash requirements/install.sh embodied --model cosmos3 --env libero --use-mirror
         source .venv/bin/activate

Clone ``cosmos-framework`` and export its path so RLinf can load the Cosmos3 data transforms and normalization statistics:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cosmos-framework.git ./cosmos-framework
   export COSMOS_FRAMEWORK_PATH="$PWD/cosmos-framework"

Offline Cache for Nano SFT
~~~~~~~~~~~~~~~~~~~~~~~~~~

Nano SFT and base DCP conversion read ``Qwen3-VL-8B-Instruct`` and ``Wan2.2-TI2V-5B`` from the Hugging Face cache. On offline machines, pre-download them into ``HF_HOME`` and set ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` before running:

.. tab-set::

   .. tab-item:: Global

      .. code-block:: bash

         export HF_HOME=${HF_HOME:-~/.cache/huggingface}
         hf download Qwen/Qwen3-VL-8B-Instruct
         hf download Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e Wan2.2_VAE.pth
         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include "vae/*"

   .. tab-item:: China Mirror

      .. code-block:: bash

         export HF_HOME=${HF_HOME:-~/.cache/huggingface}
         export HF_ENDPOINT=https://hf-mirror.com
         hf download Qwen/Qwen3-VL-8B-Instruct
         hf download Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e Wan2.2_VAE.pth
         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include "vae/*"

Prepare Data
------------

Point ``data.train_data_paths`` at the raw LIBERO LeRobot v3 directory (containing ``libero_spatial``, ``libero_object``, ``libero_goal``, and ``libero_10`` for ``data_type: libero_all``). The Cosmos3 dataloader converts LIBERO's 7-D axis-angle actions into 10-D rot6d online and applies quantile normalization using ``libero_native_frame_wise_relative_rot6d.json``.

.. code-block:: yaml

   data:
     train_data_paths: ./data/LIBERO_LeRobot_v3
     data_type: "libero_all"   # or "libero_10" for a single suite

Training and evaluation must use the same ``action_normalization`` (``quantile_rot``) and the same ``action_stats_path`` file.

Action Representation
~~~~~~~~~~~~~~~~~~~~~

Cosmos3 predicts 10-D rot6d actions internally, while LIBERO expects 7-D axis-angle actions:

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Representation
     - Dim
     - Composition
   * - ``raw_action_dim`` (model)
     - 10
     - 3 translation + 6 rot6d + 1 gripper
   * - ``action_dim`` (env)
     - 7
     - 3 translation + 3 axis-angle + 1 gripper

The dataloader maps 7-D LIBERO actions to 10-D rot6d during training; evaluation and RL rollout convert the 10-D predictions back to 7-D before stepping LIBERO.

Cosmos3-Nano SFT
----------------

Nano SFT warm-starts from a DCP-format ``nvidia/Cosmos3-Nano`` base checkpoint. First download the Hugging Face Diffusers snapshot:

.. tab-set::

   .. tab-item:: Global

      .. code-block:: bash

         hf download nvidia/Cosmos3-Nano --local-dir ./assets/Cosmos3-Nano

   .. tab-item:: China Mirror

      .. code-block:: bash

         export HF_ENDPOINT=https://hf-mirror.com
         hf download nvidia/Cosmos3-Nano --local-dir ./assets/Cosmos3-Nano

Convert the Diffusers snapshot to DCP inside the training virtualenv so the pickled DCP metadata matches the training Python version:

.. code-block:: bash

   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   python -m cosmos_framework.scripts.convert_model_to_dcp \
     --checkpoint-path ./assets/Cosmos3-Nano \
     --no-use-ema-weights \
     -o ./assets/Cosmos3-Nano-DCP-py311

Edit ``examples/sft/config/libero_sft_cosmos3.yaml`` to set the local model, tokenizer, VAE, statistics, and dataset paths:

.. code-block:: yaml

   data:
     train_data_paths: ./data/LIBERO_LeRobot_v3

   actor:
     model:
       model_path: ./assets/Cosmos3-Nano-DCP-py311
       tokenizer_path: ./assets/Qwen3-VL-8B-Instruct-tokenizer
       wan_vae_path: ./assets/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       action_stats_path: ./cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json

Launch Nano SFT from the repository root:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_cosmos3

The script shards the model across GPUs with FSDP2, writes logs to ``logs/<timestamp>-libero_sft_cosmos3/run_embodiment.log``, and saves checkpoints every ``runner.save_interval`` steps under ``checkpoints/global_step_<N>/``. Set ``runner.resume_dir`` to resume from a saved step.

.. warning::

   Keep ``actor.model.load_to_device: false`` in SFT configs. Cosmos3 builds both the online network and the FP32 EMA replica on CPU before FSDP2 sharding; eager GPU placement OOMs on an 80 GB GPU.

Cosmos3-Edge SFT
----------------

Edge SFT uses ``examples/sft/config/libero_sft_cosmos3_edge.yaml`` with the Nemotron backbone and Edge processor. Download the complete Edge snapshot and Wan2.2 VAE:

.. tab-set::

   .. tab-item:: Global

      .. code-block:: bash

         hf download nvidia/Cosmos3-Edge \
           --revision 344d602b128d1bbdacb43b08d0a3626f46343e29 \
           --local-dir ./assets/Cosmos3-Edge
         hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
           --local-dir ./assets/Wan2.2-TI2V-5B

   .. tab-item:: China Mirror

      .. code-block:: bash

         export HF_ENDPOINT=https://hf-mirror.com
         hf download nvidia/Cosmos3-Edge \
           --revision 344d602b128d1bbdacb43b08d0a3626f46343e29 \
           --local-dir ./assets/Cosmos3-Edge
         hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
           --local-dir ./assets/Wan2.2-TI2V-5B

Convert the Edge snapshot to DCP in the same virtualenv (clear the VAE bucket and credentials overrides so the local VAE file is read directly):

.. code-block:: bash

   COSMOS_DEVICE=cpu python -m cosmos_framework.scripts.convert_model_to_dcp \
     --checkpoint-path "$PWD/assets/Cosmos3-Edge" \
     --config-file "$COSMOS_FRAMEWORK_PATH/cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml" \
     --output-path "$PWD/assets/Cosmos3-Edge-DCP-py311" \
     --experiment-overrides \
       'model.config.tokenizer.bucket_name=""' \
       'model.config.tokenizer.object_store_credential_path_pretrained=""' \
       "model.config.tokenizer.vae_path=$PWD/assets/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
       model.config.vlm_config.tokenizer.repository=null \
       "model.config.vlm_config.tokenizer.tokenizer_type=$PWD/assets/Cosmos3-Edge"

Keep the original ``./assets/Cosmos3-Edge`` directory for ``actor.model.tokenizer_path`` because it holds the Edge processor metadata and tokenizer.

Set the local paths in ``examples/sft/config/libero_sft_cosmos3_edge.yaml`` (``data_type: libero_all`` requires at least 4 GPUs):

.. code-block:: yaml

   data:
     train_data_paths: ./data/LIBERO_LeRobot_v3

   actor:
     model:
       model_path: ./assets/Cosmos3-Edge-DCP-py311
       tokenizer_path: ./assets/Cosmos3-Edge
       wan_vae_path: ./assets/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       action_stats_path: ./cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json

Launch Edge SFT from the repository root:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_cosmos3_edge

Key SFT Configuration
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field
     - Description
   * - ``actor.model.model_path``
     - Cosmos3-Nano or Cosmos3-Edge DCP base checkpoint directory.
   * - ``actor.model.tokenizer_path``
     - Qwen3-VL tokenizer directory (Nano) or full Edge snapshot directory (Edge).
   * - ``actor.model.wan_vae_path``
     - Path to ``Wan2.2_VAE.pth``.
   * - ``actor.model.load_to_device``
     - Must be ``false`` so FSDP2 shards directly from CPU.
   * - ``actor.model.ema_enabled``
     - Enables Cosmos3 power-EMA (rate 0.1); set ``false`` if GPU memory is tight.
   * - ``actor.micro_batch_size`` / ``global_batch_size``
     - Per-GPU micro-batch and global batch (Nano: 32 / 2048; Edge: 64 / 256).
   * - ``runner.max_steps`` / ``save_interval``
     - Total optimizer steps and checkpoint save frequency.

Checkpoint Conversion
---------------------

SFT saves ``checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``. SGLang evaluation and NFT training load a Diffusers component directory (``model_diffusers``). Export it using a matching Diffusers template directory (works for both Nano and Edge):

.. code-block:: bash

   python -m toolkits.cosmos3.convert_checkpoint \
     --src ./results/checkpoints/global_step_10000/actor/model_state_dict/full_weights.pt \
     --base-model-path ./assets/Cosmos3-Edge \
     --out ./assets/cosmos3-edge-policy-liberoall-10k

The converter checks every transformer tensor name and shape, writes the SFT weights into ``./assets/cosmos3-edge-policy-liberoall-10k/model_diffusers``, and copies the template's tokenizer, VAE, scheduler, and config. The output directory must be new and outside the template. By default it exports the online weights; pass ``--use-ema-weights`` to export the EMA weights instead.

.. note::

   If you do not have a matching Diffusers template and have a ``cosmos-framework`` export environment (``transformers 4.57.x`` + ``diffusers 0.39.0``), omit ``--base-model-path`` and set ``LIBERO_ROOT`` to run the four-step export path. See ``python -m toolkits.cosmos3.convert_checkpoint --help``.

.. _cosmos3-sglang-install:

Install SGLang Serving and RL Environment
-----------------------------------------

For Cosmos3-Edge NFT training and SGLang evaluation in a single environment, install Cosmos3, LIBERO, and the SGLang 0.5.19 Diffusion runtime (including the pinned ``FxxxxU/sglang@cf565fa`` Cosmos3 extension) together with ``--engine sglang``:

.. tab-set::

   .. tab-item:: Global

      .. code-block:: bash

         unset PYTHONPATH
         bash requirements/install.sh embodied --model cosmos3 --env libero --engine sglang --venv .venv-sglang
         source .venv-sglang/bin/activate

   .. tab-item:: China Mirror

      .. code-block:: bash

         unset PYTHONPATH
         bash requirements/install.sh embodied --model cosmos3 --env libero --engine sglang --venv .venv-sglang --use-mirror
         source .venv-sglang/bin/activate

.. note::

   If you only need standalone SGLang evaluation without actor training dependencies, omit ``--model cosmos3`` and run ``bash requirements/install.sh embodied --env libero --engine sglang --venv .venv-sglang``. To reuse a local checkout of ``FxxxxU/sglang`` instead of cloning into ``.venv-sglang/sglang-cosmos3``, export ``SGLANG_COSMOS3_PATH`` before running ``install.sh``.

Cosmos3-Edge NFT Training
-------------------------

After exporting ``model_diffusers``, train Cosmos3-Edge on LIBERO-Spatial with Diffusion-NFT and SGLang rollout. Edit the local asset paths in ``examples/embodiment/config/libero_spatial_nft_cosmos3_edge_sglang.yaml``:

.. code-block:: yaml

   actor:
     model:
       model_path: ./assets/cosmos3-edge-policy-liberoall-10k/model_diffusers
       wan_vae_path: ./assets/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       action_stats_path: ./cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json

Launch training from the repository root:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_nft_cosmos3_edge_sglang

Each rollout worker launches its own SGLang Diffusion server on its assigned GPUs during initialization and shuts it down when training ends. Patch weight synchronization initializes the full serving baseline on step 0 and transfers only changed parameter elements on subsequent steps. If GPU memory is tight when collocating actor and rollout on the same GPUs, set ``rollout.enable_offload: true`` and ``actor.enable_offload: true``.

After training, evaluate the exported checkpoint following :doc:`Cosmos3 SGLang Evaluation <../../evaluations/guides/cosmos3_sglang>`.

Visualization and Results
-------------------------

Monitor training metrics and loss curves with TensorBoard:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

See :doc:`Training Metrics </rst_source/reference/metrics>` for metric definitions.

FAQ
---

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Fix
   * - OOM during model initialization on an 80 GB GPU
     - Keep ``actor.model.load_to_device: false``; for collocated SGLang NFT training, also enable ``rollout.enable_offload: true`` and ``actor.enable_offload: true``.
   * - ``sglang version 0.0.0.dev... is not supported``
     - Reinstall ``../sglang-cosmos3/python`` with ``SETUPTOOLS_SCM_PRETEND_VERSION=0.5.19``.
   * - Action head shape mismatch during SFT warm-start
     - Confirm ``keys_to_skip_loading`` skips the base DROID 8-D action head (configured by default in ``examples/embodiment/config/model/cosmos3.yaml``).
   * - Hugging Face download hangs on an offline machine
     - Pre-download into ``HF_HOME`` and export ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``.
   * - Actions look wrong or success rate is near zero
     - Verify ``action_normalization`` (``quantile_rot``), ``action_stats_path``, ``num_inference_steps``, ``num_frames``, and ``size`` match between SFT and rollout.
