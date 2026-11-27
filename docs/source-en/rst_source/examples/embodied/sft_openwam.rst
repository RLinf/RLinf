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

      Clone RLinf, install OpenWAM and LIBERO, and activate the environment:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
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

         Eight A800 80 GB GPUs support this recipe. Four-GPU runs need more memory per GPU, such as H200. Set ``cluster.component_placement.actor`` to the selected GPUs and keep ``actor.global_batch_size`` divisible by the GPU count.

         For a shorter run, set both ``runner.max_steps`` and ``actor.optim.total_training_steps`` to the desired duration, and choose ``actor.optim.lr_warmup_steps`` below it. Append these overrides to the launch command above.

Export and Evaluate
-------------------

For example, export the step-1,000 checkpoint into a directory that OpenWAM can load:

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

Training reports ``train/loss``, ``train/loss_video``, and ``train/loss_action``. See also the general :doc:`Training metrics <../../reference/metrics>` reference and :doc:`LIBERO Evaluation <../../evaluations/guides/libero>` for standalone evaluation.

This reference run fine-tuned the released LIBERO checkpoint for 1,000 steps on eight A800 80 GB GPUs in about 50 minutes, using global batch 8, learning rate 1e-6, 100 warmup steps, and cosine decay.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/3a760f24bdc20480a8feba57ecb659ae4111648f/pic/openwam/libero-sft-1000-loss.png
   :align: center
   :width: 100%

   Total, video, and action training loss. Pale lines show raw values; orange lines show a 50-step moving mean. Logged steps 0–999 correspond to 1,000 optimizer updates.

After 1,000 training steps, both checkpoints were evaluated on all 500 task/initial-state pairs per suite: ten tasks with 50 initial states each. Both runs use the same pairs and the OpenWAM evaluation presets.

.. list-table:: LIBERO Success Rate
   :header-rows: 1

   * - Suite
     - Released Checkpoint
     - Exported Step 1,000
   * - Spatial
     - 496/500 (99.2%)
     - 489/500 (97.8%)
   * - Object
     - 497/500 (99.4%)
     - 497/500 (99.4%)
   * - Goal
     - 496/500 (99.2%)
     - 477/500 (95.4%)
   * - Long
     - 495/500 (99.0%)
     - 482/500 (96.4%)
   * - All Four
     - 1984/2000 (99.20%)
     - 1945/2000 (97.25%)

Training loss decreased, while overall success fell by 1.95 percentage points. This is a reference for the 1,000-step workflow; training used no held-out validation split.

.. dropdown:: Reproduce These Results

   Append ``runner.max_steps=1000``, ``actor.optim.total_training_steps=1000``, and ``actor.optim.lr_warmup_steps=100`` to the training command, then export step 1,000. Use :ref:`Full OpenWAM Coverage <openwam-libero-full-coverage>` with ``env.eval.max_steps_per_rollout_epoch=75000`` for Spatial/Object/Goal or ``env.eval.max_steps_per_rollout_epoch=87500`` for Long.

.. dropdown:: Validation and Resume

   Set ``data.val_data_paths`` to held-out demonstrations and choose ``runner.val_check_interval``. Use the same normalization statistics as training. If the validation dataset calls its split ``train``, set ``data.openwam_val_split=train``.

   Validation reports ``eval/loss``, ``eval/loss_video``, and ``eval/loss_action``.

   To resume, set ``runner.resume_dir`` to the absolute path of a saved ``global_step_<N>`` directory containing ``actor/``. The recipe requires saved data-stream and random-number state for exact resume. See :doc:`../../guides/resume`.

.. dropdown:: Other Data Sources

   Match the dataset to ``dataloader.type`` in the source checkpoint's ``config.yaml``, including image fields, robot state, action representation, and normalization. Use ``libero_sft_openwam`` for one data source or ``mixture_sft_openwam`` for a checkpoint configured for ``mixture``.

   For mixture training, ``data.train_data_paths`` is a root containing one ``<name>`` subdirectory per enabled source. To override a source's location, set ``data.openwam.datasets.<name>.dataset_dir`` and retain that source's normalization statistics.
