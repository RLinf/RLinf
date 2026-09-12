FastWAM Supervised Fine-Tuning and Evaluation
==============================================

.. figure:: https://yuantianyuan01.github.io/FastWAM/static/images/teaser_main.png
   :align: center
   :width: 90%

   Fast-WAM predicts robot actions with a video-diffusion world-action model.

RLinf supports FSDP supervised fine-tuning of FastWAM on the four LIBERO
LeRobot datasets and evaluation on the corresponding suites. The policy uses
main-camera and wrist-camera RGB images, the 8-dimensional robot state, and the
language instruction. It predicts 32 action steps and executes 10 steps before
replanning.

Installation
------------

.. include:: _setup_common.rst

Install FastWAM for SFT:

.. code-block:: bash

   bash requirements/install.sh embodied --model fastwam
   source .venv/bin/activate

Install FastWAM with LIBERO for evaluation:

.. code-block:: bash

   bash requirements/install.sh embodied --model fastwam --env libero
   source .venv/bin/activate

Supervised Fine-Tuning
----------------------

Download and extract the four LIBERO LeRobot suites:

.. code-block:: bash

   huggingface-cli download yuanty/LIBERO-fastwam \
     --repo-type dataset \
     --local-dir /your_path_to/LIBERO-fastwam

   for archive in /your_path_to/LIBERO-fastwam/*.tar.gz; do
     tar -xzf "$archive" -C /your_path_to/LIBERO-fastwam
   done

Download the normalization statistics:

.. code-block:: bash

   huggingface-cli download yuanty/fastwam \
     libero_uncond_2cam224_dataset_stats.json \
     --local-dir /your_path_to/fastwam

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

Precompute the T5 embeddings:

.. code-block:: bash

   python .venv/FastWAM/scripts/precompute_text_embeds.py \
     task=libero_uncond_2cam224_1e-4 \
     "data.train.dataset_dirs=[/your_path_to/LIBERO-fastwam/libero_spatial_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_object_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_goal_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_10_no_noops_lerobot]" \
     "data.train.text_embedding_cache_dir=/your_path_to/text_embeds_cache/libero" \
     model.redirect_common_files=true

Set the paths in ``examples/sft/config/libero_sft_fastwam.yaml`` and
``examples/sft/config/model/fastwam.yaml``. The key fields are:

.. code-block:: yaml

   data:
     # Four LIBERO suites in LeRobot format.
     train_data_paths:
       - /your_path_to/LIBERO-fastwam/libero_spatial_no_noops_lerobot
       - /your_path_to/LIBERO-fastwam/libero_object_no_noops_lerobot
       - /your_path_to/LIBERO-fastwam/libero_goal_no_noops_lerobot
       - /your_path_to/LIBERO-fastwam/libero_10_no_noops_lerobot
     text_embedding_cache_dir: /your_path_to/text_embeds_cache/libero  # Precomputed T5 embeddings

   actor:
     model:
       model_path: null  # Initialize from the base model.
       dataset_stats_path: /your_path_to/fastwam/libero_uncond_2cam224_dataset_stats.json
       freeze_non_dit: true  # Train the MoT experts and proprio encoder.
       fastwam:
         overrides:
           # Preprocessed ActionDiT backbone.
           - model.action_dit_pretrained_path=/your_path_to/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

Launch SFT:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_fastwam

Evaluation
----------

Download the released checkpoint:

.. code-block:: bash

   huggingface-cli download yuanty/fastwam \
     libero_uncond_2cam224.pt \
     --local-dir /your_path_to/fastwam

Set the paths in ``examples/embodiment/config/model/fastwam.yaml``:

.. code-block:: yaml

   model_path: /your_path_to/fastwam/libero_uncond_2cam224.pt  # FastWAM checkpoint
   dataset_stats_path: /your_path_to/fastwam/libero_uncond_2cam224_dataset_stats.json  # Normalization statistics

Run one command for each suite:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval
   bash evaluations/run_eval.sh libero libero_object_fastwam_eval
   bash evaluations/run_eval.sh libero libero_goal_fastwam_eval
   bash evaluations/run_eval.sh libero libero_10_fastwam_eval

The configs use four GPUs because the 20 environment slots divide evenly across
four workers. Each slot runs 25 episodes, covering all ``20 * 25 = 500`` initial
states; the same 20 slots cannot be divided evenly across eight workers.

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
