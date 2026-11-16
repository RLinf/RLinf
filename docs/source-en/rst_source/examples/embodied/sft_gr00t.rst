GR00T-N1.7 Supervised Fine-Tuning
=================================

Fine-tune GR00T-N1.7 on LIBERO demonstrations with RLinf, starting from the
NVIDIA 3B checkpoint. This recipe covers dataset preparation, full-parameter
training, and checkpoint recovery.

Overview
--------

Use the main-camera and wrist-camera images, robot state, and task instruction
to learn LIBERO action chunks.

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

      LIBERO Spatial · LeRobot

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA CUDA · FSDP

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Dataset
     - Task / Suite
     - Config
   * - LIBERO LeRobot
     - Spatial
     - ``libero_sft_gr00t_n1d7``

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - Observation
     - Main-camera and wrist-camera RGB images; end-effector position, orientation, and gripper state
   * - Action
     - 16 steps of LIBERO Cartesian displacement, rotation increment, and gripper control
   * - Prompt
     - The demonstration's task instruction

Installation
------------

.. include:: _setup_common.rst

Install GR00T-N1.7 and the supported simulation dependencies:

.. code-block:: bash

   bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
   source .venv/bin/activate

The installer places the upstream GR00T checkout at ``.venv/gr00t``. If you reuse
an existing checkout with ``GR00T_PATH``, copy ``examples/LIBERO/modality.json``
from that checkout instead.

Download the Model and Data
---------------------------

Accept the access terms for `Cosmos-Reason2-2B <https://huggingface.co/nvidia/Cosmos-Reason2-2B>`_
and authenticate your Hugging Face account before downloading the backbone
configuration and tokenizer. The GR00T checkpoint contains the backbone weights.

Download the model assets with the Hugging Face CLI:

.. code-block:: bash

   hf auth login
   hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/GR00T-N1.7-3B
   hf download nvidia/Cosmos-Reason2-2B \
     --include '*.json' '*.txt' --local-dir checkpoints/Cosmos-Reason2-2B

Download the Spatial demonstrations and copy the upstream modality description
into the dataset metadata:

.. code-block:: bash

   hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
     --repo-type dataset --local-dir datasets/libero_spatial
   cp .venv/gr00t/examples/LIBERO/modality.json datasets/libero_spatial/meta/modality.json

Keep the dataset's ``meta/``, ``data/``, and ``videos/`` directories together.
The training processor uses the dataset statistics for normalization and saves
them with each checkpoint. The action and state transformations follow the
upstream GR00T LIBERO configuration.

Run It
------

Set the model and dataset paths in
``examples/sft/config/libero_sft_gr00t_n1d7.yaml``:

.. code-block:: yaml

   data:
     train_data_paths:
       - /absolute/path/to/datasets/libero_spatial
   actor:
     model:
       model_path: /absolute/path/to/checkpoints/GR00T-N1.7-3B
       backbone_model_path: /absolute/path/to/checkpoints/Cosmos-Reason2-2B

The example places eight actors on one node. Set ``cluster.component_placement.actor``
for your available GPUs, then choose ``actor.global_batch_size`` and
``actor.micro_batch_size`` so that gradient accumulation is an integer.
The model loads in FP32. FSDP uses BF16 for the backbone and attention blocks
and FP32 for the projection layers. The visual encoder, language model, and
action head are trainable; gradient checkpointing is disabled.

Launch the training entry point:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_gr00t_n1d7

Set the update budget with ``runner.max_steps`` and match
``actor.optim.total_training_steps`` to your learning-rate schedule. The example
uses 20,000 updates and saves every 1,000 updates. These are example settings,
not a measured LIBERO convergence budget.

The upstream processor pads actions to the checkpoint's action space and masks
unused action dimensions and timesteps in its flow-matching loss.
The data stream samples indefinitely, so use an update budget with
``runner.max_epochs: -1``.

Visualization and Results
-------------------------

Inspect the training loss in TensorBoard under ``runner.logger.log_path``.
Each saved actor checkpoint includes model and optimizer state, the training
processor, normalization statistics, and dataloader recovery state. The actor
directory contains ``processor_config.json``, ``statistics.json``,
``embodiment_id.json``, and ``config.json``.
For checkpoint recovery, set ``runner.resume_dir`` to the saved ``checkpoints/global_step_N`` directory
and follow :doc:`../../guides/resume`.

Resume with the same training configuration and a saved update directory:

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/sft"
   export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
   python examples/sft/train_vla_sft.py --config-name libero_sft_gr00t_n1d7 \
     runner.resume_dir=/absolute/path/to/checkpoints/global_step_1000

The streaming dataloader resumes by fast-forwarding its iterator. Its random
image augmentation state is not an exact replay guarantee. Keep the dataset,
worker count, and GPU topology fixed when resuming.

Training loss measures action fitting. Evaluate closed-loop task success
separately. See :doc:`gr00t` for the LIBERO setup; exporting this SFT checkpoint
for closed-loop evaluation requires a separate compatibility check.
