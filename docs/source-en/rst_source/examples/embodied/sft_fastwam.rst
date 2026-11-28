FastWAM SFT & Evaluation
========================

.. figure:: https://yuantianyuan01.github.io/FastWAM/static/images/teaser_main.png
   :align: center
   :width: 90%

   Fast-WAM predicts robot actions with a video-diffusion world-action model.

RLinf supports FSDP supervised fine-tuning of FastWAM on the four LIBERO
LeRobot datasets and evaluation on the corresponding suites. The policy uses
main-camera and wrist-camera RGB images, the 8-dimensional robot state, and the
language instruction. It predicts 32 action steps and executes 10 steps before
replanning.

Overview
--------

Fine-tune FastWAM on the four LIBERO suites with FSDP SFT and run closed-loop evaluation on LIBERO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      FastWAM

   .. grid-item-card:: Methods
      :text-align: center

      FSDP SFT · Evaluation

   .. grid-item-card:: Data
      :text-align: center

      LIBERO (LeRobot)

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA CUDA · :ref:`Huawei Ascend CANN <sft-fastwam-hardware>`

| **You'll do:** install → download LIBERO data & Wan backbones → preprocess embeddings → launch ``run_vla_sft.sh`` or ``run_eval.sh``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · LIBERO LeRobot datasets and Wan2.2 / Wan2.1 checkpoints (steps below).

.. _sft-fastwam-hardware:

Setup and Run
-------------

.. include:: _setup_common.rst

The SFT and Evaluation tabs cover NVIDIA CUDA. Use the Huawei Ascend CANN tab for NPU setup and launch steps.

.. tab-set::

   .. tab-item:: SFT (LIBERO)
      :sync: fastwam-sft

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model fastwam && source .venv/bin/activate

      **2. Download Data and Wan Components**

      Download the LIBERO LeRobot datasets, normalization stats, and Wan2.2 / Wan2.1 components:

      .. code-block:: bash

         hf download yuanty/LIBERO-fastwam --repo-type dataset --local-dir ./LIBERO-fastwam
         for archive in ./LIBERO-fastwam/*.tar.gz; do
           tar -xzf "$archive" -C ./LIBERO-fastwam
         done
         hf download yuanty/fastwam libero_uncond_2cam224_dataset_stats.json --local-dir ./fastwam
         hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan-AI/Wan2.2-TI2V-5B
         hf download Wan-AI/Wan2.1-T2V-1.3B --include "google/umt5-xxl/*" --local-dir ./checkpoints/Wan-AI/Wan2.1-T2V-1.3B

      **3. Preprocess ActionDiT Backbone and T5 Embeddings**

      .. code-block:: bash

         export DIFFSYNTH_MODEL_BASE_PATH=./checkpoints
         python .venv/FastWAM/scripts/preprocess_action_dit_backbone.py \
           --model-config .venv/FastWAM/configs/model/fastwam.yaml \
           --output ./fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
           --device cuda \
           --dtype bfloat16

         python .venv/FastWAM/scripts/precompute_text_embeds.py \
           task=libero_uncond_2cam224_1e-4 \
           "data.train.dataset_dirs=[./LIBERO-fastwam/libero_spatial_no_noops_lerobot,./LIBERO-fastwam/libero_object_no_noops_lerobot,./LIBERO-fastwam/libero_goal_no_noops_lerobot,./LIBERO-fastwam/libero_10_no_noops_lerobot]" \
           "data.train.text_embedding_cache_dir=./text_embeds_cache/libero" \
           model.redirect_common_files=true

      **4. Set paths** in ``examples/sft/config/libero_sft_fastwam.yaml`` and ``examples/sft/config/model/fastwam.yaml``:

      .. code-block:: yaml

         # examples/sft/config/libero_sft_fastwam.yaml
         data:
           train_data_paths:
             - ./LIBERO-fastwam/libero_spatial_no_noops_lerobot
             - ./LIBERO-fastwam/libero_object_no_noops_lerobot
             - ./LIBERO-fastwam/libero_goal_no_noops_lerobot
             - ./LIBERO-fastwam/libero_10_no_noops_lerobot
           text_embedding_cache_dir: ./text_embeds_cache/libero

      .. code-block:: yaml

         # examples/sft/config/model/fastwam.yaml
         dataset_stats_path: ./fastwam/libero_uncond_2cam224_dataset_stats.json
         fastwam:
           overrides:
             - model.action_dit_pretrained_path=./fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

      **5. Launch SFT:**

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_fastwam

   .. tab-item:: Evaluation (LIBERO)
      :sync: fastwam-evaluation

      **1. Install with LIBERO and download the released checkpoint and stats:**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model fastwam --env libero && source .venv/bin/activate
         hf download yuanty/fastwam libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json --local-dir ./fastwam

      **2. Set paths** in ``examples/embodiment/config/model/fastwam.yaml``:

      .. code-block:: yaml

         model_path: ./fastwam/libero_uncond_2cam224.pt
         dataset_stats_path: ./fastwam/libero_uncond_2cam224_dataset_stats.json

      **3. Run one command per suite** (uses four GPUs so 20 environment slots × 25 episodes = 500 initial states divide evenly across four workers):

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval
         bash evaluations/run_eval.sh libero libero_object_fastwam_eval
         bash evaluations/run_eval.sh libero libero_goal_fastwam_eval
         bash evaluations/run_eval.sh libero libero_10_fastwam_eval

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

   .. tab-item:: Huawei Ascend CANN
      :sync: ascend

      FastWAM supports SFT and LIBERO evaluation on Ascend with the same model weights, datasets, and text embedding cache.

      **1. Install**

      Use the Ascend LIBERO container or a host with CANN and the NPU driver installed.

      .. include:: _ascend_libero.rst

      Create the FastWAM environment from the RLinf checkout inside the container or on the Ascend host:

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model fastwam --env libero
         source .venv/bin/activate

      For SFT only, omit ``--env libero``.

      **2. Prepare Model and Data**

      For SFT, complete steps 2–4 in the SFT (LIBERO) tab to download the resources, preprocess them, and set the paths. For ActionDiT preprocessing, use ``--device cpu``. The text embedding script also uses the CPU when CUDA is unavailable.

      Keep ``DIFFSYNTH_MODEL_BASE_PATH`` pointing to the directory containing ``Wan-AI``. When loading the original Wan ``.pth`` weights, set ``model.redirect_common_files=false`` in the text embedding command and in the model's ``fastwam.overrides`` list.

      **3. Launch SFT**

      After setting the data and model paths, update ``examples/sft/config/libero_sft_fastwam.yaml`` for eight NPUs:

      .. code-block:: yaml

         cluster:
           component_placement:
             actor: 0-7
         actor:
           micro_batch_size: 1
           global_batch_size: 8
           fsdp_config:
             gradient_checkpointing: true

      Adjust placement for your available NPUs. Keep the global batch size divisible by the micro batch size times the number of NPUs.

      .. code-block:: bash

         export DIFFSYNTH_MODEL_BASE_PATH=./checkpoints
         bash examples/sft/run_vla_sft.sh libero_sft_fastwam

      **4. Evaluate on LIBERO**

      Download the released checkpoint and normalization stats and set their paths as shown in the Evaluation (LIBERO) tab. The evaluation configs use four devices by default. Adjust ``cluster.component_placement`` for your available NPUs before launching.

      Install the OSMesa runtime in the evaluation environment, then enable CPU rendering and launch:

      .. code-block:: bash

         export DIFFSYNTH_MODEL_BASE_PATH=./checkpoints
         export MUJOCO_GL=osmesa
         export PYOPENGL_PLATFORM=osmesa
         bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

      Use the other suite configs listed in the Evaluation (LIBERO) tab to evaluate Object, Goal, or Long.

      NPU runs automatically enable fused RoPE and RMSNorm.

.. dropdown:: Advanced configuration (optional)

   By default, ``examples/sft/config/model/fastwam.yaml`` initializes from the base model (``model_path: null``) and trains the MoT experts and proprio encoder (``freeze_non_dit: true``):

   .. code-block:: yaml

      actor:
        model:
          model_path: null
          freeze_non_dit: true

Visualization and Results
-------------------------

Launch TensorBoard with ``tensorboard --logdir ./logs --port 6006`` to track training loss during SFT and ``env/success_once`` during evaluation. Pass ``env.eval.video_cfg.save_video=True`` to record evaluation videos; see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and :doc:`LIBERO evaluation guide <../../evaluations/guides/libero>` for suite evaluation options.
