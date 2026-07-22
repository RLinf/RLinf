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

Download the released LIBERO checkpoint and matching normalization statistics:

.. code-block:: bash

   hf download yuanty/fastwam \
     libero_uncond_2cam224.pt \
     libero_uncond_2cam224_dataset_stats.json \
     --local-dir /workspace/checkpoints/fastwam

Set both paths in ``examples/embodiment/config/model/fastwam.yaml`` and
``examples/sft/config/model/fastwam.yaml``:

.. code-block:: yaml

   model_type: fastwam
   model_path: /workspace/checkpoints/fastwam/libero_uncond_2cam224.pt
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
     - Owns mixed precision and gradient checkpointing. Keep the model preset at
       ``precision: fp32`` for SFT; FSDP applies bf16 forward/backward precision.

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
