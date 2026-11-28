π₀.₅ Short Memory on RMBench
============================

Fine-tune a RoboTwin π₀.₅ checkpoint on public Aloha demonstrations, then evaluate it in the `RMBench <https://rmbench.github.io/>`__ simulator. RMBench tests whether a policy can use earlier observations to complete a task, such as picking up an object it saw before its view was blocked.

Overview
--------

The policy receives eight historical RGB frames per camera (head and both wrists) plus proprioception and a language instruction to predict 32-step Aloha joint and gripper targets.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ with Short Memory

   .. grid-item-card:: Algorithms
      :text-align: center

      SFT

   .. grid-item-card:: Tasks
      :text-align: center

      Aloha ``demo_clean``

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA GPUs

| **You'll do:** install deps → convert demos → download base weights → launch SFT → monitor ``train/loss``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · NVIDIA GPUs (8 × 80 GB GPUs for the default SFT batch layout).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

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
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Head and both wrist RGB views, current Aloha joint and gripper positions, and a language instruction.
   * - History
     - Eight frames per camera, including the current frame, sampled every 53 steps at 15 Hz (~24.7 s window).
   * - Action
     - 32-step chunk of 14-dim Aloha targets (6 joints + 1 gripper per arm); training uses joint deltas and absolute gripper targets.

See :doc:`Short Memory <pi0/short_memory>` for how the model uses the earlier frames.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. Install**

      Install the OpenPI environment into ``.venv`` (used for data preparation, SFT, and inference; evaluation adds a separate ``.rmbench-venv`` for the SAPIEN/CuRobo simulator worker and its asset download), then check out the pinned RMBench commit:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env rmbench && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RMBench.git
         git -C RMBench checkout 7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7

      **2. Prepare the Demonstrations**

      Download the pinned `RMBench demonstration release <https://huggingface.co/datasets/TianxingChen/RMBench/tree/5d8ff21038fba32ea549e9eee8497aaba9aab5cf>`__ and convert HDF5 episodes into LeRobot v2.1 (splits each task into train/val and writes ``rmbench_lerobot/norm_stats.json`` from the training split):

      .. code-block:: bash

         bash RMBench/scripts/_download_data.sh --revision 5d8ff21038fba32ea549e9eee8497aaba9aab5cf
         python -m toolkits.rmbench.prepare_data --dataset-root RMBench/data/demo_clean \
           --output-dir rmbench_lerobot

      Camera frames are embedded in the output Parquet files under ``rmbench_lerobot/train`` and ``rmbench_lerobot/val``, so keep disk space for both the downloads and the converted dataset.

      **3. Download the Model**

      Download the starting `RoboTwin π₀.₅ checkpoint <https://huggingface.co/motus-robotics/pi0.5_robotwin2/tree/effc6e1a99bd4d0edf9447e8ba313def67649cfe>`__:

      .. code-block:: bash

         hf download motus-robotics/pi0.5_robotwin2 model.safetensors \
           --revision effc6e1a99bd4d0edf9447e8ba313def67649cfe --local-dir ./checkpoints/rmbench_base

      **4. Launch Training**

      Runs full-parameter SFT for 11,642 optimizer steps with a global batch size of 128 (~5 epochs on 8 × 80 GB GPUs, micro-batch size 8). When changing the GPU count, adjust ``actor.micro_batch_size`` so ``global_batch_size`` remains divisible by ``GPUs × micro_batch_size``:

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh rmbench_sft_openpi_pi05 \
           data.train_data_paths="$PWD/rmbench_lerobot/train" \
           actor.model.model_path="$PWD/checkpoints/rmbench_base" \
           actor.model.openpi_data.norm_stats_path="$PWD/rmbench_lerobot/norm_stats.json"

      For current-frame SFT without Short Memory, append ``actor.model.openpi.short_memory=false`` to the launch command.

Visualization and Results
-------------------------

Track training loss with ``tensorboard --logdir ./logs --port 6006`` (see :doc:`Training Metrics </rst_source/reference/metrics>` for metric definitions and Weights & Biases / SwanLab setup).

Checkpoints are saved every 2,329 steps up to step 11,642 under ``logs/<run>/rmbench_pi05_short_memory_sft/checkpoints/global_step_<N>/``. Pass a saved checkpoint directory and ``rmbench_lerobot/norm_stats.json`` to the :doc:`RMBench evaluation guide <../../evaluations/guides/rmbench>` to measure closed-loop ``eval/success_once`` in the simulator.
