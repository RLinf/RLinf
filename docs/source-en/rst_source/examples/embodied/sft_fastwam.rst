FastWAM Evaluation and Supervised Fine-Tuning
==============================================

.. figure:: https://yuantianyuan01.github.io/FastWAM/static/images/teaser_main.png
   :align: center
   :width: 90%

   Fast-WAM keeps video co-training but generates actions without future-video
   denoising at evaluation time.

Run the released `FastWAM <https://github.com/yuantianyuan01/FastWAM>`__ model
on LIBERO or LIBERO-Plus, and supervised-fine-tune its world/action experts with
RLinf's FSDP SFT pipeline.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO · LIBERO-Plus

   .. grid-item-card:: Algorithms
      :text-align: center

      Evaluation · SFT

   .. grid-item-card:: Tasks
      :text-align: center

      LIBERO Spatial

   .. grid-item-card:: Hardware
      :text-align: center

      CUDA GPUs · multi-GPU SFT

| **You'll do:** install → download checkpoint and statistics → evaluate → prepare LeRobot data and text embeddings → launch SFT.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · CUDA GPUs · Hugging Face access for Wan2.2 components.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_fastwam_eval``
     - Batched action-only evaluation.
   * - LIBERO-Plus
     - Spatial perturbations
     - Same config with ``LIBERO_TYPE=plus``
     - Evaluate all or one perturbation family.
   * - Offline data
     - LIBERO LeRobot
     - ``libero_sft_fastwam``
     - Full-parameter FSDP SFT of the MoT experts.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Main and wrist RGB images plus the 8-dimensional LIBERO robot state.
   * - Prompt
     - The natural-language LIBERO instruction encoded by FastWAM's text encoder.
   * - Action
     - A 32-step, 7-dimensional action prediction; RLinf executes
       ``num_action_chunks`` steps before replanning.
   * - Training target
     - Video flow-matching and action flow-matching losses from FastWAM's
       ``training_loss``.

Installation
------------

.. include:: _setup_common.rst

For evaluation, install FastWAM and LIBERO together:

.. code-block:: bash

   bash requirements/install.sh embodied --model fastwam --env libero
   source .venv/bin/activate

For offline SFT without a simulator:

.. code-block:: bash

   bash requirements/install.sh embodied --model fastwam
   source .venv/bin/activate

The installer clones a pinned FastWAM revision, installs its non-Torch
dependencies, and uses RLinf's platform-aware Torch override to select Torch
2.7.1 by default (required by TorchCodec 0.5). An explicit ``--torch`` still
takes precedence. Set
``FASTWAM_PATH=/path/to/FastWAM`` before installation to reuse a checkout.

Use ``--env liberoplus`` for LIBERO-Plus. Its additional assets must also be
installed as described in :ref:`liberopro-plus-benchmark`.

Download the Model
------------------

For evaluation or fine-tuning from the released policy, download its
checkpoint and matching normalization statistics:

.. code-block:: bash

   huggingface-cli download yuanty/fastwam \
     libero_uncond_2cam224.pt \
     libero_uncond_2cam224_dataset_stats.json \
     --local-dir /workspace/checkpoints/fastwam

For the official base-model SFT recipe documented below, keep
``model_path: null``. The SFT config then loads the official Wan2.2 video DiT
and initializes ActionDiT from the official interpolated backbone payload.
Use ``dataset_stats_path`` for the released normalization statistics:

.. code-block:: yaml

   model_type: fastwam
   model_path: null
   dataset_stats_path: /workspace/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json

FastWAM and RLinf Configuration
-------------------------------

RLinf composes FastWAM's upstream YAML with OmegaConf without changing Hydra's
global state. The two configuration layers have separate responsibilities:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Layer
     - Responsibility
   * - ``model.fastwam.config_name``
     - Selects the upstream architecture, processor, dataset shape, scheduler,
       and training-loss defaults. RLinf uses ``sim_libero`` by default.
   * - ``model.fastwam.overrides``
     - Applies upstream-compatible dot-list overrides, such as
       ``model.load_text_encoder=false`` for cached-text SFT.
   * - RLinf model fields
     - ``model_path``, ``dataset_stats_path``, action chunking, sampling, and
       optional future-video visualization. These values take precedence over
       FastWAM's evaluation defaults.
   * - RLinf FSDP config
     - Owns mixed precision and gradient checkpointing. Keep the model preset at ``precision: bf16`` for SFT; FSDP2 casting is unset and
       the worker uses bf16 autocast, matching the upstream Accelerator path.

Use only ``model_path`` for the FastWAM checkpoint. ``checkpoint_path`` is not a
supported alias.

Evaluate
--------

The single ``libero_spatial_fastwam_eval.yaml`` config replaces separate small,
large, LIBERO-Plus, language-only, and future-video YAML files.

**Standard LIBERO smoke evaluation:**

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval

**Larger evaluation:** run 80 trajectories through eight reusable environment
processes and disable video recording:

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=8 \
     env.eval.max_steps_per_rollout_epoch=2400 \
     env.eval.video_cfg.save_video=false

**LIBERO-Plus:** select all perturbations or a single family with environment
variables; the YAML stays unchanged:

.. code-block:: bash

   LIBERO_TYPE=plus LIBERO_SUFFIX=all MUJOCO_GL=egl \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

   LIBERO_TYPE=plus LIBERO_SUFFIX=language MUJOCO_GL=egl \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=8 env.eval.video_cfg.save_video=false

**Future-video visualization:** action generation remains batched; optional
future imagination is generated only for the first sample and capped by
``max_video_saves``.

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=2 \
     env.eval.video_cfg.save_video=false \
     rollout.model.visualize_future_video=true \
     rollout.model.future_video_dir=/workspace/future_video_demo

Supervised Fine-Tuning
----------------------

Download the `FastWAM LIBERO dataset
<https://huggingface.co/datasets/yuanty/LIBERO-fastwam>`__ and precompute the T5
text embeddings with the upstream script:

.. code-block:: bash

   python "$FASTWAM_PATH/scripts/precompute_text_embeds.py" \
     task=libero_uncond_2cam224_1e-4 \
     'data.train.dataset_dirs=[/path/to/libero_spatial_no_noops_lerobot]' \
     data.train.text_embedding_cache_dir=/workspace/data/text_embeds_cache/libero \
     model.redirect_common_files=false

Update ``data.train_data_paths`` and ``data.text_embedding_cache_dir`` in
``examples/sft/config/libero_sft_fastwam.yaml``, then launch:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_fastwam

FastWAM's MoT accesses the video and action transformer blocks directly, so the
example intentionally uses whole-model FSDP2 wrapping. The full trainable MoT is
too large for ordinary single-GPU SFT; use multiple GPUs and tune the batch size
for available memory.
FastWAM SFT quick start
~~~~~~~~~~~~~~~~~~~~~~~~

After installing the environment, the following idempotent helper downloads
the Wan2.2 VAE, T5 encoder/tokenizer files, official Wan2.2 video DiT shards,
the official LIBERO archives, prepares the interpolated ActionDiT backbone,
and precomputes the text-embedding cache:

.. code-block:: bash

   source .venv/bin/activate
   tmux new -s fastwam-sft
   bash examples/sft/prepare_fastwam_sft.sh
   bash examples/sft/run_vla_sft.sh libero_sft_fastwam
   # Detach with Ctrl-b d; reattach with: tmux attach -t fastwam-sft

The helper uses repository-relative defaults and also prepares the official Wan2.2 video DiT and ActionDiT backbone. Override
FASTWAM_CHECKPOINT_DIR, DIFFSYNTH_MODEL_BASE_PATH, FASTWAM_DATASET_DIR, or
FASTWAM_TEXT_EMBEDDING_CACHE_DIR, or FASTWAM_ACTION_DIT_BACKBONE_PATH when assets already live elsewhere. Set
FASTWAM_DOWNLOAD_DATA=0 to skip the dataset archive download and point
FASTWAM_DATASET_DIR at an existing extraction.

The VAE is required even for SFT because the upstream training_loss encodes the
video observations. The official base-model SFT path also requires the Wan2.2
video DiT and the generated ActionDiT backbone: ``model_path`` remains null,
``skip_dit_load_from_pretrain=false``, and only the MoT plus proprio encoder are
trainable.

Official FastWAM versus RLinf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The official recipe uses preprocess_action_dit_backbone.py, cached T5
embeddings, and accelerate with DeepSpeed ZeRO-1 (the README's LIBERO example
uses eight GPUs). RLinf instead:

* reuses the upstream RobotVideoDataset, FastWAMProcessor, and training_loss;
* runs the generic RLinf train_vla_sft.py / FSDP2 worker rather than the
  upstream train_zero1.sh entrypoint;
* composes the upstream sim_libero config without changing Hydra global
  state, starts from the official Wan2.2 base plus the interpolated ActionDiT
  backbone, and trains only MoT plus the proprio encoder; and
* keeps precision: bf16 in the model preset and enables bf16 autocast around
  the loss, matching the upstream Accelerator path. FSDP2's own casting policy
  is unset to avoid a second mixed-precision path; the RLinf wrapper still
  aligns direct-call inputs (VAE video, text context, action, and proprio)
  with the active model dtype.

This is an intentional integration difference: RLinf SFT does not reproduce
the upstream optimizer/distributed launcher byte-for-byte, but it keeps the
upstream model loss, data transforms, masks, normalization, and text-cache
format compatible.
