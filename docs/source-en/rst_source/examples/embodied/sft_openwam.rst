OpenWAM Supervised Fine-Tuning
==============================

Fine-tune the released OpenWAM LIBERO checkpoint with FSDP on OpenWAM's LIBERO dataset, then export it for evaluation.

Use the released OpenWAM RoboTwin checkpoints for evaluation. RoboTwin SFT has not been validated in RLinf; use the LIBERO recipe below for SFT and the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>` for RoboTwin evaluation.

Installation
------------

From the RLinf repository root, install the OpenWAM and LIBERO environment and activate it. If the environment is already installed, activate it directly:

.. code-block:: bash

   bash requirements/install.sh embodied --model openwam --env libero
   source .venv/bin/activate

Supervised Fine-Tuning
----------------------

Prepare the Model Weights
~~~~~~~~~~~~~~~~~~~~~~~~~

Download the released OpenWAM LIBERO checkpoint:

.. code-block:: bash

   hf download OpenWAM/OpenWAM-Alpha-Sim-LIBERO --local-dir /path/to/OpenWAM-Alpha-Sim-LIBERO

The downloaded directory should contain the weights, model configuration, normalization statistics, and tokenizer. This checkpoint includes the required model components, so a separate Wan base model download is unnecessary. Set ``actor.model.model_path`` to the entire directory, rather than an individual weights file:

.. code-block:: text

   /path/to/OpenWAM-Alpha-Sim-LIBERO/
   ├── checkpoint_step_10690.safetensors
   ├── config.yaml
   ├── normalization_stats.npy
   └── tokenizer/
       └── google/umt5-xxl/

Prepare the Dataset
~~~~~~~~~~~~~~~~~~~

Download the OpenWAM LIBERO dataset paired with this checkpoint:

.. code-block:: bash

   hf download OpenWAM/LIBERO --repo-type dataset --local-dir /path/to/LIBERO

The dataset is already in LeRobot v3 format and needs no unpacking or additional conversion. Keep the complete ``data/``, ``meta/``, and ``videos/`` directories. The dataset normalization statistics are stored in ``meta/libero_normalization_stats.npy``:

.. code-block:: text

   /path/to/LIBERO/
   ├── data/
   ├── meta/
   │   ├── info.json
   │   ├── tasks.parquet
   │   ├── episodes/
   │   └── libero_normalization_stats.npy
   └── videos/
       ├── observation.images.image/
       └── observation.images.image2/

Set ``data.train_data_paths`` to the dataset root containing these three directories. Training uses the LIBERO data configuration from the checkpoint. For custom data, keep the image fields, robot state, action representation, and normalization conventions consistent with that configuration.

Configure Training
~~~~~~~~~~~~~~~~~~

Set the downloaded checkpoint path in ``examples/sft/config/model/openwam.yaml`` and retain the training precision and loading settings:

.. code-block:: yaml

   model_path: /path/to/OpenWAM-Alpha-Sim-LIBERO
   precision: fp32
   load_to_device: false

Set the dataset path in ``examples/sft/config/libero_sft_openwam.yaml``. The key fields are shown below; the remaining fields can keep their recipe defaults:

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-7

   data:
     train_data_paths: /path/to/LIBERO

   runner:
     max_steps: 30000
     save_interval: 1000

   actor:
     micro_batch_size: 1
     global_batch_size: 8
     optim:
       lr: 1.0e-6
       total_training_steps: 30000

The recipe defaults to eight GPUs, FSDP2, and bf16 mixed-precision computation while retaining fp32 master weights. When changing the GPU count, update ``cluster.component_placement.actor`` and ensure that ``actor.global_batch_size`` is divisible by the actor GPU count. When changing the training duration, also update ``actor.optim.total_training_steps``.

.. note::

   H200 measurements used LIBERO data, ``micro_batch_size: 1``, and ``global_batch_size: 8``. Peak allocated/reserved memory per GPU was 55.5/83.7 GiB with eight GPUs and 72.3/113.3 GiB with four GPUs.

Run It
------

Run the following command from the RLinf repository root:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openwam

Training logs are saved under ``logs/<timestamp>-libero_sft_openwam/``. Read ``run_embodiment.log`` for training logs, or use TensorBoard to inspect ``train/loss``, ``train/loss_video``, and ``train/loss_action``. Replace RUN_DIR below with the actual run directory:

.. code-block:: bash

   RUN_DIR="/absolute/path/to/RLinf/logs/<timestamp>-libero_sft_openwam"
   tensorboard --logdir "$RUN_DIR/tensorboard"

By default, checkpoints are saved every 1000 steps under ``$RUN_DIR/libero_sft_openwam/checkpoints/global_step_<N>/``.

Export and Evaluate
-------------------

Training saves RLinf checkpoints. Export a checkpoint to an OpenWAM checkpoint directory before using it for inference or evaluation. Using RUN_DIR from the previous step, export the checkpoint saved at step 1000:

.. code-block:: bash

   python toolkits/openwam/export_checkpoint.py \
     --rlinf-checkpoint "$RUN_DIR/libero_sft_openwam/checkpoints/global_step_1000" \
     --source-checkpoint /path/to/OpenWAM-Alpha-Sim-LIBERO \
     --output /path/to/openwam-libero-step1000

Set ``--source-checkpoint`` to the checkpoint directory used to initialize training. This preserves the matching configuration, tokenizer, and normalization files. Add ``--verify cuda`` to check that the exported checkpoint loads on a GPU.

Set ``rollout.model.model_path`` in the LIBERO evaluation configuration to ``/path/to/openwam-libero-step1000``, then follow the OpenWAM instructions in :doc:`../../evaluations/guides/libero`. For RoboTwin checkpoints, see :doc:`../../evaluations/guides/robotwin`.

Optional Configuration
----------------------

Validation and Resuming Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable validation during training, point ``data.val_data_paths`` to a separate validation dataset and set ``runner.val_check_interval``. The OpenWAM LIBERO dataset downloaded above only has a ``train`` split; do not reuse the training data as the validation set. If the separate validation dataset also only has a ``train`` split, set ``data.openwam_val_split: train`` and ensure that it uses the same normalization statistics as training. Validation logs ``eval/loss``, ``eval/loss_video``, and ``eval/loss_action``.

To resume training, set ``runner.resume_dir`` to the absolute path of ``$RUN_DIR/libero_sft_openwam/checkpoints/global_step_<N>`` in the training configuration. Use the directory containing ``actor/``. The recipe sets ``runner.strict_resume: true``, so resuming fails if the checkpoint lacks data-stream or random-number state. See :doc:`../../guides/resume` for more details.

Other Data Sources and Mixture Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For another OpenWAM checkpoint, first check that ``dataloader.type`` in its ``config.yaml`` matches the dataset format. For a single data source, use ``libero_sft_openwam.yaml`` with the corresponding weight and data paths. Use ``mixture_sft_openwam.yaml`` only when the checkpoint is configured for ``mixture``.

For mixture training, ``data.train_data_paths`` takes one root directory, with each enabled source loaded from ``<root>/<name>`` by default. To use a different location for a source, set ``data.openwam.datasets.<name>.dataset_dir`` and retain that source's matching normalization statistics.
