GR00T-N1.7 Supervised Fine-Tuning
=================================

Fine-tune GR00T-N1.7 on LIBERO Spatial demonstrations with FSDP, save resumable checkpoints, and evaluate the trained policy on LIBERO Spatial.

Overview
--------

The policy predicts 16-step action chunks from main-camera and wrist-camera images, robot state, and a task instruction.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      GR00T-N1.7-3B

   .. grid-item-card:: Methods
      :text-align: center

      Full-parameter SFT

   .. grid-item-card:: Data
      :text-align: center

      Physical Intelligence LIBERO

   .. grid-item-card:: Hardware
      :text-align: center

      8 NVIDIA GPUs · FSDP

| **You'll do:** install → download model and data → launch ``run_vla_sft.sh`` → inspect ``train/loss`` and evaluate.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Hugging Face access to ``nvidia/Cosmos-Reason2-2B`` · 8 NVIDIA GPUs.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Suite
     - Demonstrations
     - Config
   * - LIBERO Spatial (10 tasks)
     - 432 episodes, 52,970 timesteps at 10 Hz
     - ``libero_sft_gr00t_n1d7``

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Training Data
   * - Observation
     - Main-camera and wrist-camera RGB images; 8-dimensional end-effector and gripper state
   * - Action
     - 16 steps of 7-dimensional Cartesian displacement, rotation increment, and gripper control
   * - Prompt
     - Task instruction from the demonstration episode

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO Spatial

      **1. Install**

      Clone the repository, install the ``gr00t_n1d7`` environment, and activate it:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
         source .venv/bin/activate

      **2. Download**

      Accept the access terms for `Cosmos-Reason2-2B <https://huggingface.co/nvidia/Cosmos-Reason2-2B>`_ and authenticate with ``hf auth login``, then download the base model, backbone configuration and tokenizer, and `Physical Intelligence LIBERO <https://huggingface.co/datasets/physical-intelligence/libero>`_ dataset:

      .. code-block:: bash

         hf auth login
         hf download nvidia/GR00T-N1.7-3B --local-dir ./checkpoints/GR00T-N1.7-3B
         hf download nvidia/Cosmos-Reason2-2B --include '*.json' '*.txt' --local-dir ./checkpoints/Cosmos-Reason2-2B
         hf download physical-intelligence/libero --repo-type dataset --local-dir ./datasets/physical-intelligence-libero

      ``GR00T-N1.7-3B`` provides the pretrained weights, while ``Cosmos-Reason2-2B`` provides the backbone configuration and tokenizer. The dataset contains 1,693 episodes across 40 tasks with camera images embedded in Parquet files; ``examples/sft/config/libero_sft_gr00t_n1d7.yaml`` selects the 10 Spatial tasks, converts gripper actions to the GR00T convention, and caches normalization statistics.

      **3. Launch**

      Run single-node FSDP training from the repository root:

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_gr00t_n1d7 \
           data.train_data_paths="$PWD/datasets/physical-intelligence-libero" \
           actor.model.model_path="$PWD/checkpoints/GR00T-N1.7-3B" \
           actor.model.backbone_model_path="$PWD/checkpoints/Cosmos-Reason2-2B"

      The launcher writes logs and TensorBoard events to ``logs/<timestamp>-libero_sft_gr00t_n1d7/``, and saves checkpoints under ``libero_sft_gr00t_n1d7/checkpoints/global_step_<N>/``. To resume training, pass ``runner.resume_dir="$PWD/logs/.../checkpoints/global_step_<N>"`` with the same dataset, GPU allocation, batch sizes, and worker count (see :doc:`../../guides/resume`; stream replay does not reproduce random image augmentations bit for bit).

      .. dropdown:: Advanced configuration (optional)

         Override these keys in ``examples/sft/config/libero_sft_gr00t_n1d7.yaml`` or on the command line when adjusting hardware allocation or training length:

         .. list-table::
            :header-rows: 1
            :widths: 34 66

            * - Key
              - Effect
            * - ``cluster.component_placement.actor``
              - GPU ranks for FSDP training (default ``0-7``).
            * - ``actor.micro_batch_size`` / ``actor.global_batch_size``
              - Per-GPU micro-batch size and global batch size (defaults ``1`` and ``32``); ``global_batch_size`` must be divisible by ``num_gpus * micro_batch_size``.
            * - ``runner.max_steps`` / ``runner.save_interval``
              - Total training steps and checkpoint frequency (keep ``runner.max_epochs: -1``).
            * - ``actor.optim.total_training_steps``
              - Cosine learning-rate schedule horizon in steps.
            * - ``data.statistics_path``
              - Stem for hashed per-dataset normalization caches; statistics are also saved in each checkpoint.

.. dropdown:: Evaluate an SFT checkpoint (optional)

   Evaluate a saved ``global_step_<N>`` checkpoint directly on LIBERO Spatial with :doc:`../../evaluations/guides/libero` (or continue to RL post-training in :doc:`gr00t`):

   .. code-block:: bash

      bash evaluations/run_eval.sh libero libero_spatial_gr00t_n1d7_eval \
        rollout.model.model_path="$PWD/logs/<run_dir>/libero_sft_gr00t_n1d7/checkpoints/global_step_3000" \
        rollout.model.backbone_model_path="$PWD/checkpoints/Cosmos-Reason2-2B"

   ``evaluations/libero/libero_spatial_gr00t_n1d7_eval.yaml`` disables the value head, sets ``env.eval.ignore_terminations=False``, and runs all 500 trials across 40 parallel environments.

Visualization and Results
-------------------------

Start TensorBoard to monitor ``train/loss`` and checkpoint progress (metric definitions are in :doc:`../../reference/metrics`):

.. code-block:: bash

   tensorboard --logdir ./logs

Measured LIBERO Spatial evaluation before and after 3,000 steps of FSDP SFT (8× NVIDIA A800, global batch size 64):

.. list-table::
   :header-rows: 1
   :widths: 38 32 30

   * - Model
     - Setup
     - LIBERO Spatial (``success_once``)
   * - Base (``nvidia/GR00T-N1.7-3B``)
     - 500 trials
     - ``0/500`` (0.0%)
   * - SFT (step 3,000)
     - 8× A800 · GBS 64 · 500 trials
     - ``367/500`` (73.4%)
