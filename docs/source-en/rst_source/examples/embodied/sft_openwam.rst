OpenWAM Supervised Fine-Tuning
==============================

Fine-tune the released OpenWAM LIBERO checkpoint on demonstrations, then export it for LIBERO evaluation.

Overview
--------

This recipe fine-tunes video and action prediction together with FSDP2.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenWAM Alpha

   .. grid-item-card:: Methods
      :text-align: center

      SFT · FSDP2

   .. grid-item-card:: Data
      :text-align: center

      OpenWAM LIBERO

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

**Prerequisites:** :doc:`Installation </rst_source/start/installation>` and eight GPUs with at least 80 GB each. The recipe runs on 8 × A800 80 GB with bf16 computation, fp32 master weights, one sample per GPU, and a global batch of eight.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO

      **1. Install**

      Run from the RLinf repository root:

      .. code-block:: bash

         bash requirements/install.sh embodied --model openwam --env libero
         source .venv/bin/activate

      **2. Download**

      Download the checkpoint and its matching LeRobot v3 dataset:

      .. code-block:: bash

         hf download OpenWAM/OpenWAM-Alpha-Sim-LIBERO --local-dir ./OpenWAM-Alpha-Sim-LIBERO
         hf download OpenWAM/LIBERO --repo-type dataset --local-dir ./LIBERO

      Keep the dataset directory intact, including its videos and metadata.

      **3. Launch**

      Train for the default 30,000 steps and save logs and checkpoints under ``./logs/openwam-sft``:

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_openwam \
           actor.model.model_path=./OpenWAM-Alpha-Sim-LIBERO \
           data.train_data_paths=./LIBERO \
           runner.logger.log_path=./logs/openwam-sft

      Checkpoints are saved every 1,000 steps in ``./logs/openwam-sft/libero_sft_openwam/checkpoints/global_step_<N>/``.

      .. dropdown:: Adjust Training

         To change the GPU count, set ``cluster.component_placement.actor`` and use a global batch divisible by that count. Check memory use before a long run. Earlier H200 measurements reached 55.5/83.7 GiB peak allocated/reserved memory per GPU on eight GPUs, and 72.3/113.3 GiB on four GPUs, with the same batch settings.

         For a shorter run, set both ``runner.max_steps`` and ``actor.optim.total_training_steps`` to the desired duration, and choose ``actor.optim.lr_warmup_steps`` below it. For example, a 1,000-step run can use 100 warmup steps. Append these overrides to the launch command above.

Export and Evaluate
-------------------

Export the saved step-1,000 checkpoint into a directory that OpenWAM can load:

.. code-block:: bash

   python toolkits/openwam/export_checkpoint.py \
     --rlinf-checkpoint ./logs/openwam-sft/libero_sft_openwam/checkpoints/global_step_1000 \
     --source-checkpoint ./OpenWAM-Alpha-Sim-LIBERO \
     --output ./openwam-libero-step1000

Use the same source checkpoint as training so the export retains the matching configuration and tokenizer. The export also includes the normalization statistics saved during training. Add ``--verify cuda`` to check loading on an available GPU.

Evaluate the exported model with the OpenWAM protocol:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openwam_eval \
     rollout.model.model_path=./openwam-libero-step1000

The default evaluation uses four GPUs and three episodes. Use :ref:`Full OpenWAM Coverage <openwam-libero-full-coverage>` to evaluate all tasks and initial states in each suite.

Visualization and Results
-------------------------

Open TensorBoard to inspect total, video, and action loss:

.. code-block:: bash

   tensorboard --logdir ./logs/openwam-sft/tensorboard

Training reports ``train/loss``, ``train/loss_video``, and ``train/loss_action``. Compare LIBERO success rates before and after fine-tuning using the same tasks, initial states, and evaluation settings.

The run below fine-tuned the released LIBERO checkpoint for 1,000 steps on eight A800 80 GB GPUs. It used a global batch of eight, learning rate 1e-6, 100 warmup steps, and cosine decay over 1,000 steps. Training took 49.9 minutes; exporting step 1,000 and reloading it on a GPU both passed.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/3a760f24bdc20480a8feba57ecb659ae4111648f/pic/openwam/libero-sft-1000-loss.png
   :align: center
   :width: 100%

   Total, video, and action training loss. Pale lines show raw values; orange lines show a 50-step moving mean. Logged steps 0–999 correspond to 1,000 optimizer updates.

.. list-table:: Training Loss Means
   :header-rows: 1

   * - Loss
     - First 100 Steps
     - Last 100 Steps
   * - Total
     - 0.137325
     - 0.063728
   * - Video
     - 0.062742
     - 0.049344
   * - Action
     - 0.074584
     - 0.014385

The following **subset evaluation** uses 100 episodes per suite and checkpoint. Each checkpoint runs initial states 0–9 for each of the ten tasks, using identical task/state pairs without repeats within a run. The 800 episodes use seed 0, 30 settling steps, ten executed actions per prediction, and episode budgets of 600 steps (700 for Long).

.. list-table:: LIBERO Success Rate
   :header-rows: 1

   * - Suite
     - Released Checkpoint
     - Exported Step 1,000
   * - Spatial
     - 99/100 (99%)
     - 100/100 (100%)
   * - Object
     - 100/100 (100%)
     - 99/100 (99%)
   * - Goal
     - 100/100 (100%)
     - 95/100 (95%)
   * - Long
     - 98/100 (98%)
     - 98/100 (98%)
   * - All Four
     - 397/400 (99.25%)
     - 392/400 (98.00%)

Training loss decreased, while the observed success rate decreased by 1.25 percentage points. These results verify the training, export, and evaluation workflow; they do not show a benchmark improvement. The run used training data without a held-out validation split and covers a 1,000-step schedule, not the default 30,000 steps or the full 500-state evaluation per suite.

.. dropdown:: Reproduce These Results

   Append ``runner.max_steps=1000``, ``actor.optim.total_training_steps=1000``, and ``actor.optim.lr_warmup_steps=100`` to the training command. Export step 1,000 as above. Use :ref:`Full OpenWAM Coverage <openwam-libero-full-coverage>` with a rollout budget of 15,000 steps for Spatial/Object/Goal or 17,500 for Long to evaluate the same 100-state subsets.

   The run used RLinf revision ``0a671691``, OpenWAM revision ``a8992b613``, model revision ``110b07e4``, and dataset revision ``bcb2eaf1``. Runtime: Python 3.11.14 and PyTorch 2.11.0+cu128, without flash-attn. See the `experiment evidence <https://github.com/RLinf/misc/tree/3a760f24bdc20480a8feba57ecb659ae4111648f/pic/openwam>`_ for raw loss values, per-episode outcomes, and full revisions.

.. dropdown:: Validation and Resume

   Set ``data.val_data_paths`` to held-out demonstrations and choose ``runner.val_check_interval``. Use the same normalization statistics as training. If the validation dataset calls its split ``train``, set ``data.openwam_val_split: train``.

   Validation reports ``eval/loss``, ``eval/loss_video``, and ``eval/loss_action``.

   To resume, set ``runner.resume_dir`` to the absolute path of a saved ``global_step_<N>`` directory containing ``actor/``. The recipe requires saved data-stream and random-number state for exact resume. See :doc:`../../guides/resume`.

.. dropdown:: Other Data Sources

   Match the dataset to ``dataloader.type`` in the source checkpoint's ``config.yaml``, including image fields, robot state, action representation, and normalization. Use ``libero_sft_openwam`` for one data source or ``mixture_sft_openwam`` for a checkpoint configured for ``mixture``.

   For mixture training, ``data.train_data_paths`` is a root containing one ``<name>`` subdirectory per enabled source. To override a source's location, set ``data.openwam.datasets.<name>.dataset_dir`` and retain that source's normalization statistics.
