π₀.₅ Short Memory on RMBench
==================================

Fine-tune a RoboTwin π₀.₅ checkpoint on public Aloha demonstrations, then
evaluate it in the `RMBench <https://rmbench.github.io/>`__ simulator. RMBench
tests whether a policy can use earlier observations to complete a task, such
as picking up an object it saw before its view was blocked.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models

      π₀.₅ with Short Memory

   .. grid-item-card:: Algorithms

      Supervised fine-tuning

   .. grid-item-card:: Tasks

      Aloha ``demo_clean``

   .. grid-item-card:: Hardware

      NVIDIA GPUs

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Data
     - Coverage
   * - Public ``demo_clean`` release
     - 500 demonstrations across 10 task directories
   * - Training split
     - 450 episodes, 298,032 frames
   * - Validation split
     - 50 episodes, 32,509 frames

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Specification
   * - Observation
     - Head and both wrist RGB views, current Aloha joint and gripper positions,
       and a language instruction.
   * - History
     - Eight frames per camera, including the current frame, sampled every
       53 steps at 15 Hz. The window spans about 24.7 seconds.
   * - Action
     - A chunk of 32 targets. Each target has six joint positions and one gripper
       position per arm. The training transform uses joint deltas and absolute
       gripper targets.

See :doc:`Short Memory <pi0/short_memory>` for how the model uses the earlier frames.

Setup and Run
-------------

See :doc:`Installation </rst_source/start/installation>` for system requirements.

.. tab-set::

   .. tab-item:: NVIDIA CUDA

      **1. Install**

      From the RLinf root, install the OpenPI training environment and check out the
      RMBench revision used by this recipe. Keep that checkout for evaluation.

      .. code-block:: bash

         bash requirements/install.sh embodied --model openpi --env rmbench
         source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RMBench.git
         git -C RMBench checkout 7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7

      **2. Prepare the Demonstrations**

      Download the pinned `RMBench demonstration release <https://huggingface.co/datasets/TianxingChen/RMBench/tree/5d8ff21038fba32ea549e9eee8497aaba9aab5cf>`__.
      Convert the HDF5 demonstrations to LeRobot v2.1. The converter splits each
      task into train and validation episodes and computes normalization statistics
      from the training split.

      .. code-block:: bash

         bash RMBench/scripts/_download_data.sh --revision 5d8ff21038fba32ea549e9eee8497aaba9aab5cf
         python -m toolkits.rmbench.prepare_data --dataset-root RMBench/data/demo_clean \
           --output-dir rmbench_lerobot

      The converter writes ``rmbench_lerobot/train`` and ``rmbench_lerobot/val``.
      The camera frames are embedded in LeRobot Parquet files. Allow space for
      the converted dataset alongside the downloads. Use the generated
      ``rmbench_lerobot/norm_stats.json`` for training and evaluation.

      **3. Download the Model**

      Start from the `RoboTwin π₀.₅ checkpoint <https://huggingface.co/motus-robotics/pi0.5_robotwin2/tree/effc6e1a99bd4d0edf9447e8ba313def67649cfe>`__.

      .. code-block:: bash

         hf download motus-robotics/pi0.5_robotwin2 model.safetensors \
           --revision effc6e1a99bd4d0edf9447e8ba313def67649cfe --local-dir checkpoints/rmbench_base

      **4. Launch Training**

      The recipe fine-tunes all model weights for 11,642 optimizer steps with a
      global batch size of 128, about five passes over the training split. The
      reference batch layout uses eight 80 GB GPUs and a micro-batch size of eight.
      If you change the GPU count, adjust ``actor.micro_batch_size`` so the global
      batch remains divisible by the number of GPUs times the micro-batch size.

      From the RLinf root, launch training with the dataset, checkpoint, and
      normalization statistics you prepared above.

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh rmbench_sft_openpi_pi05 \
           data.train_data_paths="$PWD/rmbench_lerobot/train" \
           actor.model.model_path="$PWD/checkpoints/rmbench_base" \
           actor.model.openpi_data.norm_stats_path="$PWD/rmbench_lerobot/norm_stats.json"

      For SFT with current camera frames only, set
      ``actor.model.openpi.short_memory: false`` in the training config.

Visualization and Results
-------------------------

Open ``logs/<run>/`` in TensorBoard to follow the training loss. For evaluation,
use the checkpoint directory
``logs/<run>/rmbench_pi05_short_memory_sft/checkpoints/global_step_<N>/``
and the training ``norm_stats.json``. See the
:doc:`RMBench evaluation guide <../../evaluations/guides/rmbench>` to run the simulator.
