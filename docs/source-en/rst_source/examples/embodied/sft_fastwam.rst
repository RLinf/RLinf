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

      Spatial · Object · Goal · Long

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
     - Spatial / Object / Goal / Long
     - Four ``libero_*_fastwam_eval`` configs
     - Evaluate all 500 initial states of one suite per run.
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
     --local-dir /your_path_to/fastwam

For the official base-model SFT recipe documented below, keep
``model_path: null``. The SFT config then loads the official Wan2.2 video DiT
and initializes ActionDiT from the official interpolated backbone payload.
Use ``dataset_stats_path`` for the released normalization statistics:

.. code-block:: yaml

   model_type: fastwam
   model_path: null
   dataset_stats_path: /your_path_to/fastwam/libero_uncond_2cam224_dataset_stats.json

FastWAM and RLinf Configuration
-------------------------------

The evaluation YAML inherits the shared
``examples/embodiment/config/model/fastwam.yaml`` preset. Before launching an
evaluation, edit the two paths in that preset:

.. code-block:: yaml

   model_path: /your_path_to/fastwam/libero_uncond_2cam224.pt  # https://huggingface.co/yuanty/fastwam
   dataset_stats_path: /your_path_to/fastwam/libero_uncond_2cam224_dataset_stats.json

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

.. note::

   The generic FSDP startup log may recommend fp32 model parameters for the
   optimizer policy used by ordinary models. That recommendation does not apply
   to FastWAM: its official SFT path uses bf16 model parameters plus bf16
   autocast, and FSDP2 must not add another parameter dtype cast. This generic
   message does not indicate a FastWAM configuration error; do not switch to
   fp32 merely to silence it.

Use only ``model_path`` for the FastWAM checkpoint. ``checkpoint_path`` is not a
supported alias.

Evaluate
--------

FastWAM provides one configuration for each LIBERO suite. After setting the
checkpoint and statistics paths in the shared model YAML, run the suite you
want to evaluate:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval
   bash evaluations/run_eval.sh libero libero_object_fastwam_eval
   bash evaluations/run_eval.sh libero libero_goal_fastwam_eval
   bash evaluations/run_eval.sh libero libero_10_fastwam_eval

Each config covers all 500 fixed initial states of its suite. It uses GPUs
``0-3`` and 20 environments (five per worker). Because
``ignore_terminations=True`` keeps every trajectory running until
``max_episode_steps``, each environment runs exactly 25 episodes:
``20 * 25 = 500``. This avoids starting 500 simulator/EGL contexts while still
using batched FastWAM inference.

Do not change the placement to ``env,rollout: all`` on an eight-GPU host. The
non-decoupled channel requires ``total_num_envs`` to be divisible by the
rollout world size, but no multiple of eight divides LIBERO's 500 trajectories.
Without padding or terminal-slot suppression, an eight-worker layout therefore
either omits states or exhausts some slots early. The provided configs use four
workers on both four- and eight-GPU hosts.

The suite-specific configs also fix the official FastWAM controller limits, so
no command-line horizon overrides are needed:

.. list-table::
   :header-rows: 1
   :widths: 18 42 20 20

   * - Suite
     - Config
     - ``max_episode_steps``
     - ``max_steps_per_rollout_epoch``
   * - Spatial
     - ``libero_spatial_fastwam_eval``
     - ``400``
     - ``10000``
   * - Object
     - ``libero_object_fastwam_eval``
     - ``400``
     - ``10000``
   * - Goal
     - ``libero_goal_fastwam_eval``
     - ``400``
     - ``10000``
   * - Long
     - ``libero_10_fastwam_eval``
     - ``700``
     - ``17500``

**LIBERO-Plus:** select all perturbations or a single family with environment
variables; the YAML stays unchanged:

.. code-block:: bash

   LIBERO_TYPE=plus LIBERO_SUFFIX=all \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

   LIBERO_TYPE=plus LIBERO_SUFFIX=language \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

**Future-video visualization:** action generation remains batched; optional
future imagination is generated only for the first sample and capped by
``max_video_saves``.

Set the following fields in the evaluation YAML and launch it with the same
single-suite command:

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 2
       video_cfg:
         save_video: false
   rollout:
     model:
       visualize_future_video: true
       future_video_dir: /your_path_to/future_video_demo

Supervised Fine-Tuning
----------------------

The RLinf recipe starts from the official Wan2.2 base model and interpolated
ActionDiT backbone, rather than continuing from the released FastWAM policy.
Asset preparation follows the same steps as the upstream FastWAM repository,
but the paths are written directly in the RLinf YAML files.

Download and extract the four LIBERO LeRobot suites:

.. code-block:: bash

   huggingface-cli download yuanty/LIBERO-fastwam \
     --repo-type dataset \
     --local-dir /your_path_to/LIBERO-fastwam

   for archive in /your_path_to/LIBERO-fastwam/*.tar.gz; do
     tar -xzf "$archive" -C /your_path_to/LIBERO-fastwam
   done

Download the Wan components and generate the ActionDiT backbone:

.. code-block:: bash

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
     --local-dir /your_path_to/checkpoints/Wan-AI/Wan2.2-TI2V-5B
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
     --include "google/umt5-xxl/*" \
     --local-dir /your_path_to/checkpoints/Wan-AI/Wan2.1-T2V-1.3B

   export DIFFSYNTH_MODEL_BASE_PATH=/your_path_to/checkpoints
   python .venv/FastWAM/scripts/preprocess_action_dit_backbone.py \
     --model-config .venv/FastWAM/configs/model/fastwam.yaml \
     --output /your_path_to/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
     --device cuda \
     --dtype bfloat16

Precompute the T5 text cache for the same four dataset directories:

.. code-block:: bash

   python .venv/FastWAM/scripts/precompute_text_embeds.py \
     task=libero_uncond_2cam224_1e-4 \
     "data.train.dataset_dirs=[/your_path_to/LIBERO-fastwam/libero_spatial_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_object_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_goal_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_10_no_noops_lerobot]" \
     "data.train.text_embedding_cache_dir=/your_path_to/text_embeds_cache/libero" \
     model.redirect_common_files=true

Finally, update the placeholders in
``examples/sft/config/libero_sft_fastwam.yaml`` and
``examples/sft/config/model/fastwam.yaml``. Each path has a nearby comment
linking to the corresponding Hugging Face repository or preparation command.

Launch SFT from the repository root:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_fastwam

Use tmux or another session manager if the training process must survive a
terminal disconnect.

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
