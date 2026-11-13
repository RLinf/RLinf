GR00T N1.7 Supervised Fine-Tuning
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gr00t.png
   :align: center
   :width: 90%

   The GR00T vision-language-action model family.

Train GR00T N1.7 on RoboCasa365 BowlAndCup demonstrations using independent random observation-action windows. This recipe covers dataset preparation, the native H40 Flow Matching loss, and supervised fine-tuning with the shared VLA FSDP training loop.

Overview
--------

Each training sample pairs the current observation with 40 action targets. Batch sizes count independent windows.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      GR00T-N1.7-3B with Cosmos-Reason2-2B

   .. grid-item-card:: Methods
      :text-align: center

      Native Flow Matching SFT with independent random windows

   .. grid-item-card:: Data
      :text-align: center

      RoboCasa365 BowlAndCup in GR00T-compatible LeRobot format

   .. grid-item-card:: Hardware
      :text-align: center

      Eight NVIDIA GPUs; an earlier revision of the H40 recipe completed a two-step check on A800 GPUs.

Tasks
~~~~~

The supplied configuration targets a single manipulation task and its demonstration dataset.

.. list-table::
   :header-rows: 1

   * - Task
     - Demonstrations
     - Configuration
   * - BowlAndCup
     - Place the cup in the bowl, move the bowl to the target counter, and withdraw the arm
     - ``robocasa_bowlandcup_sft_gr00t``

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Each window supplies the current observation and the following 40 action targets; future observations are not policy inputs.

.. list-table::
   :header-rows: 1

   * - Field
     - Content
   * - Observation
     - Three RGB cameras, resized to 256 × 256 with a 230 × 230 training crop by the processor, plus base pose, relative end-effector pose, and gripper state
   * - Action
     - 12 real RoboCasa action values per step; native model tensors are padded to 40 × 132
   * - Reward
     - Not used for supervised training
   * - Prompt
     - ``annotation.human.task_description``

Prepare the Dataset
-------------------

Point ``data.train_data_paths`` at one LeRobot dataset root containing its parquet data, videos, and ``meta/`` directory. The loader requires ``info.json``, ``episodes.jsonl``, ``tasks.jsonl``, ``stats.json``, ``modality.json``, and ``embodiment.json`` under ``meta/``. The processor uses this dataset's normalization statistics.

The expected camera keys are ``robot0_agentview_left``, ``robot0_agentview_right``, and ``robot0_eye_in_hand``. State and action modality ordering follows the RoboCasa365 schema in ``rlinf/data/datasets/gr00t_n1d7/window_dataset.py``. Convert other schemas before using this example; renaming the dataset path does not change the modality mapping.

Every sampled window must have all 40 real action targets available. For episode length :math:`L`, legal start indices end at :math:`L-40`; episodes shorter than 40 steps are rejected. This requirement is separate from padding the model's action dimensions.

Set ``data.expected_episodes`` to an integer to check a known dataset size, or leave it ``null``.

Keep parquet files, videos, metadata, and sampling settings unchanged across the original and resumed runs to replay the same samples. Sampler restoration checks size, seed, rank, and world size; these checks do not verify dataset content integrity.

Sampling and Loss
-----------------

The sampler shuffles all legal window starts across episodes. Each window is an independent model input.

.. list-table::
   :header-rows: 1

   * - Setting
     - Value
   * - Supervised horizon
     - 40
   * - Window selection
     - Shuffle all legal individual starts across episodes
   * - Global / per-GPU micro batch
     - 128 / 16 windows
   * - Supervised windows per update on eight GPUs
     - 128

Random-window sampling visits every legal window once globally per sampling epoch. Longer episodes contribute more windows. The distributed sampler continues across epoch boundaries without dropping the tail. Sampling order and augmentation seeds are derived deterministically from the configured seed and window identity.

The native head computes masked, elementwise Flow Matching squared errors. For each window, the loss sums these errors and divides by the number of valid action values, :math:`40\times12=480`. All 40 time steps contribute; padded action dimensions have zero supervision weight. The native collator batches the targets as :math:`[B,40,132]`.

With masked error :math:`E_{b,h,a}` and supervision mask :math:`M_{b,h,a}`, the loss averages the per-window means:

.. math::

   L = \frac{1}{B}\sum_{b=1}^{B}
       \frac{\sum_{h,a} E_{b,h,a}}{\sum_{h,a} M_{b,h,a}}.

The shared VLA worker handles backward, gradient accumulation, and one AdamW update per global batch. With the example's eight GPUs and 16 windows per GPU, each update uses 128 windows.

Download the Model
------------------

Prepare local copies of the pretrained GR00T model and its Cosmos backbone, including their configuration and processor files. Model construction loads the GR00T state dictionary strictly and requires the native H40, 132-dimensional padded action head.

.. code-block:: bash

   hf download nvidia/GR00T-N1.7-3B --local-dir /path/to/GR00T-N1.7-3B
   hf download nvidia/Cosmos-Reason2-2B --local-dir /path/to/Cosmos-Reason2-2B

Installation
------------

Install the GR00T N1.7 dependencies from the repository root, then activate the environment. The existing installer accepts the ``maniskill_libero`` environment for this model; the offline SFT loader uses the prepared dataset rather than starting that simulator. This installation combination still needs end-to-end validation for the SFT recipe.

.. code-block:: bash

   bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
   source .venv/bin/activate

Pin the installed Isaac-GR00T checkout to the revision targeted by this adapter. The following path is the installer's default; if ``GR00T_PATH`` selected an existing checkout, use that path instead.

.. code-block:: bash

   git -C .venv/gr00t checkout 23ace64f17aa5015259b8609d371eb61a357c776
   uv pip install -e .venv/gr00t --no-deps

Run It
------

Launch the random-window example after replacing the three asset paths. ``native_sft=true`` selects the native GR00T training adapter through the shared VLA SFT entry point.

.. code-block:: bash

   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/train_vla_sft.py \
     --config-path config \
     --config-name robocasa_bowlandcup_sft_gr00t \
     data.train_data_paths=/path/to/BowlAndCup/lerobot \
     actor.model.model_path=/path/to/GR00T-N1.7-3B \
     actor.model.backbone_model_path=/path/to/Cosmos-Reason2-2B \
     actor.model.dependency_commit=23ace64f17aa5015259b8609d371eb61a357c776

The example uses FP32 master parameters, BF16 FSDP parameter computation, FP32 gradient reduction, AdamW, 200 warmup steps, and a 30,000-step cosine schedule from :math:`2\times10^{-5}` to :math:`5\times10^{-6}`. Additional AMP autocast is disabled.

Use ``runner.max_steps > 0`` and ``runner.max_epochs <= 0``; the dataloader rejects other limits. Its nominal length supports the shared runner's epoch bookkeeping, while the sampler iterates continuously across sampling epochs. ``StatefulDataLoader`` retains the sampler cursor for checkpoint restore; the nominal length does not truncate iteration.

Checkpoints are saved every 5,000 steps through the shared VLA worker's existing FSDP save/load path for model, optimizer, scheduler, and RNG state. The worker also saves ``StatefulDataLoader`` state in ``data.pt`` and per-rank RNG state in ``rng.pt``. On resume, ``SFTRunner`` restores the global step from the ``global_step_<N>`` directory name. Keep the complete checkpoint directory for training resume; atomic completion across the directory is not guaranteed.

During dataloader setup, rank 0 exports modality definitions, normalization statistics, and processor keyword arguments once to ``<log_path>/<experiment_name>/processor_bundle.json``, using ``runner.logger.log_path`` and ``runner.logger.experiment_name``. An existing bundle must match; a mismatch stops setup. This file belongs to the run directory, not each checkpoint. Training resume reconstructs the processor from the same model and data configuration; it does not restore this bundle. Compatibility with checkpoints from earlier experimental implementations remains unverified.

Visualization and Results
-------------------------

The example writes TensorBoard logs to the configured ``runner.logger.log_path``. Inspect loss, gradient norm, and learning rate with the shared :doc:`Training Metrics <../../reference/metrics>` guide.

The current random-window H40 implementation passes 34 unit tests. On eight A800 80GB GPUs, a real-data check completed four optimizer steps, saving checkpoints at steps 2 and 4. A second run loaded the step-2 checkpoint and continued to step 4; both runs exited successfully. At step 4, their dataloader state, RNG state, scheduler state, and Adam step counters matched. Model tensors and Adam moments differed, so bitwise continuation has not been established.

Checkpoint evaluation used the same eight scene seeds (7–14), with three policy-seed trials per scene, giving 24 episodes per checkpoint. Episodes used OSMesa rendering and a maximum of 2,400 environment steps, with H40 action predictions and eight actions executed per chunk. Success means completing the full task under the environment's native success criterion.

.. list-table:: BowlAndCup Checkpoint Evaluation
   :header-rows: 1

   * - Training Step
     - Successful Episodes
     - Success Rate
   * - 5,000
     - 0 / 24
     - 0.00%
   * - 10,000
     - 2 / 24
     - 8.33%
   * - 15,000
     - 5 / 24
     - 20.83%

These results were produced with an external evaluation harness that is not included in this contribution. Compatibility of these checkpoints with the standalone evaluation entry points remains unverified. For general rollout setup, see :doc:`Evaluation <../../evaluations/index>`.
