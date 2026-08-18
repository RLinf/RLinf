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

The smoke evaluation YAML inherits the shared
``examples/embodiment/config/model/fastwam.yaml`` preset. Do not edit that
preset: append the following two launch-time overrides to an evaluation
command:

.. code-block:: text

   rollout.model.model_path=/path/to/FASTWAM_CHECKPOINT
   rollout.model.dataset_stats_path=/path/to/FASTWAM_DATASET_STATS

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

FastWAM provides separate smoke and full-suite configurations.

**Standard LIBERO smoke evaluation** uses
``libero_spatial_fastwam_eval.yaml``. It is intended to validate a local
checkpoint on a small Spatial batch; pass the two paths without editing the
shared model preset:

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval \
     rollout.model.model_path=/path/to/FASTWAM_CHECKPOINT \
     rollout.model.dataset_stats_path=/path/to/FASTWAM_DATASET_STATS

**Full-suite evaluation** uses
``evaluations/libero/libero_fastwam_full_eval.yaml``. Before running it,
replace that file's ``rollout.model.model_path`` and
``rollout.model.dataset_stats_path`` placeholders with local files; they are
not environment variables and are not downloaded automatically. Then run:

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_fastwam_full_eval

**LIBERO-Plus:** select all perturbations or a single family with environment
variables; the YAML stays unchanged:

.. code-block:: bash

   LIBERO_TYPE=plus LIBERO_SUFFIX=all MUJOCO_GL=egl \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval \
     rollout.model.model_path=/path/to/FASTWAM_CHECKPOINT \
     rollout.model.dataset_stats_path=/path/to/FASTWAM_DATASET_STATS

   LIBERO_TYPE=plus LIBERO_SUFFIX=language MUJOCO_GL=egl \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=8 env.eval.video_cfg.save_video=false \
     rollout.model.model_path=/path/to/FASTWAM_CHECKPOINT \
     rollout.model.dataset_stats_path=/path/to/FASTWAM_DATASET_STATS

**Future-video visualization:** action generation remains batched; optional
future imagination is generated only for the first sample and capped by
``max_video_saves``.

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=2 \
     env.eval.video_cfg.save_video=false \
     rollout.model.visualize_future_video=true \
     rollout.model.future_video_dir=/workspace/future_video_demo \
     rollout.model.model_path=/path/to/FASTWAM_CHECKPOINT \
     rollout.model.dataset_stats_path=/path/to/FASTWAM_DATASET_STATS

Supervised Fine-Tuning
----------------------

This page contains all commands required to prepare SFT, so no separate helper
script is needed. The configuration starts from the official Wan2.2 base model
and interpolated ActionDiT backbone rather than continuing from the released
FastWAM checkpoint. The code below downloads required weights and dataset stats,
prepares ActionDiT, downloads and extracts all four LIBERO suites, and
precomputes T5 text embeddings. Activate the environment, then paste it into a
terminal from the repository root; use tmux for long downloads and training:

.. code-block:: bash

   #!/usr/bin/env bash
   set -euo pipefail

   REPO_PATH="$(git rev-parse --show-toplevel)"
   PYTHON_BIN="${REPO_PATH}/.venv/bin/python"
   FASTWAM_PATH="${FASTWAM_PATH:-${REPO_PATH}/.venv/FastWAM}"
   CHECKPOINT_DIR="${FASTWAM_CHECKPOINT_DIR:-${REPO_PATH}/checkpoints/fastwam_release}"
   MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_PATH}/checkpoints}"
   DATASET_ROOT="${FASTWAM_DATASET_ROOT:-${REPO_PATH}/data/libero_mujoco3.3.2}"
   if [[ -n "${FASTWAM_DATASET_DIR:-}" ]]; then
       # Keep a one-suite override for smoke/debug runs.
       DATASET_DIRS=("${FASTWAM_DATASET_DIR}")
   else
       # Match the official task/libero_uncond_2cam224_1e-4.yaml.
       DATASET_DIRS=(
           "${DATASET_ROOT}/libero_spatial_no_noops_lerobot"
           "${DATASET_ROOT}/libero_object_no_noops_lerobot"
           "${DATASET_ROOT}/libero_goal_no_noops_lerobot"
           "${DATASET_ROOT}/libero_10_no_noops_lerobot"
       )
   fi
   DATASET_DIR="${DATASET_DIRS[0]}"
   TEXT_CACHE_DIR="${FASTWAM_TEXT_EMBEDDING_CACHE_DIR:-${DATASET_ROOT}/text_embeds_cache/libero}"
   DATASET_DIRS_HYDRA=$(IFS=,; echo "${DATASET_DIRS[*]}")
   ACTION_DIT_BACKBONE_PATH="${FASTWAM_ACTION_DIT_BACKBONE_PATH:-${MODEL_BASE_PATH}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
   DOWNLOAD_DATA="${FASTWAM_DOWNLOAD_DATA:-1}"

   export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"

   if command -v hf >/dev/null 2>&1; then
       HF_BIN=hf
   elif command -v huggingface-cli >/dev/null 2>&1; then
       HF_BIN=huggingface-cli
   else
       echo "The Hugging Face CLI is required. Activate .venv first." >&2
       exit 1
   fi
   if [ ! -x "${PYTHON_BIN}" ]; then
       echo "RLinf venv not found at ${PYTHON_BIN}; run requirements/install.sh first." >&2
       exit 1
   fi

   mkdir -p "${CHECKPOINT_DIR}" "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B" \
       "${MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B" "${DATASET_ROOT}"

   echo "[FastWAM] Downloading the released LIBERO checkpoint and stats..."
   "${HF_BIN}" download yuanty/fastwam \
       libero_uncond_2cam224.pt \
       libero_uncond_2cam224_dataset_stats.json \
       --local-dir "${CHECKPOINT_DIR}"

   echo "[FastWAM] Downloading the VAE and T5 weights used by RLinf..."
   "${HF_BIN}" download Wan-AI/Wan2.2-TI2V-5B \
       Wan2.2_VAE.pth \
       models_t5_umt5-xxl-enc-bf16.pth \
       --local-dir "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B"
   "${HF_BIN}" download Wan-AI/Wan2.1-T2V-1.3B \
       --include "google/umt5-xxl/*" \
       --local-dir "${MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B"

   echo "[FastWAM] Downloading the official Wan2.2 video DiT for base-model SFT..."
   if ! find "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B" -maxdepth 1 -type f \
       -name 'diffusion_pytorch_model*.safetensors' -print -quit | grep -q .; then
       "${HF_BIN}" download Wan-AI/Wan2.2-TI2V-5B \
           --include "diffusion_pytorch_model*.safetensors" \
           --local-dir "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B"
   fi

   if [ ! -s "${ACTION_DIT_BACKBONE_PATH}" ]; then
       echo "[FastWAM] Preprocessing the official ActionDiT backbone..."
       mkdir -p "$(dirname -- "${ACTION_DIT_BACKBONE_PATH}")"
       "${PYTHON_BIN}" "${FASTWAM_PATH}/scripts/preprocess_action_dit_backbone.py" \
           --model-config "${FASTWAM_PATH}/configs/model/fastwam.yaml" \
           --output "${ACTION_DIT_BACKBONE_PATH}" \
           --device "${FASTWAM_ACTION_DIT_DEVICE:-cuda}" \
           --dtype bfloat16
   fi

   if [ "${DOWNLOAD_DATA}" = "1" ]; then
       echo "[FastWAM] Downloading the official LIBERO LeRobot archive set..."
       "${HF_BIN}" download --repo-type dataset yuanty/LIBERO-fastwam \
           --local-dir "${DATASET_ROOT}"
       shopt -s nullglob
       archives=("${DATASET_ROOT}"/*.tar.gz)
       if [ "${#archives[@]}" -eq 0 ]; then
           echo "No LIBERO tar.gz archives found in ${DATASET_ROOT}." >&2
           exit 1
       fi
       for archive in "${archives[@]}"; do
           echo "[FastWAM] Extracting ${archive}"
           tar -xzf "${archive}" -C "${DATASET_ROOT}"
       done
   fi

   for dataset_dir in "${DATASET_DIRS[@]}"; do
       if [ ! -f "${dataset_dir}/meta/tasks.jsonl" ]; then
           echo "Expected LIBERO dataset at ${dataset_dir} was not found." >&2
           echo "Extract the official archive set or set FASTWAM_DATASET_DIR for a one-suite run." >&2
           exit 1
       fi
   done

   mkdir -p "${TEXT_CACHE_DIR}"
   export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"
   echo "[FastWAM] Precomputing T5 text embeddings..."
   "${PYTHON_BIN}" "${FASTWAM_PATH}/scripts/precompute_text_embeds.py" \
       task=libero_uncond_2cam224_1e-4 \
       "data.train.dataset_dirs=[${DATASET_DIRS_HYDRA}]" \
       "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}" \
       +overwrite=false \
       model.redirect_common_files=true

   cat <<EOF
   FastWAM SFT assets are ready.
     FASTWAM_CHECKPOINT_DIR=${CHECKPOINT_DIR}
     DIFFSYNTH_MODEL_BASE_PATH=${MODEL_BASE_PATH}
     FASTWAM_DATASET_ROOT=${DATASET_ROOT}
     FASTWAM_DATASET_DIRS=${DATASET_DIRS_HYDRA}
     FASTWAM_DATASET_DIR=${DATASET_DIR}
     FASTWAM_TEXT_EMBEDDING_CACHE_DIR=${TEXT_CACHE_DIR}
     FASTWAM_ACTION_DIT_BACKBONE_PATH=${ACTION_DIT_BACKBONE_PATH}
   EOF

After preparation, start training:

.. code-block:: bash

   tmux new -s fastwam-sft
   source .venv/bin/activate
   bash examples/sft/run_vla_sft.sh libero_sft_fastwam
   # Detach with Ctrl-b d; reattach with: tmux attach -t fastwam-sft

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
