GR00T-N1.7 Supervised Fine-Tuning
=================================

Fine-tune GR00T-N1.7 on LIBERO Spatial demonstrations and save a checkpoint you can resume. This recipe starts from NVIDIA's 3B model and trains the vision encoder, language model, and action head with FSDP.

Overview
--------

The policy learns 16-step action chunks from two camera views, robot state, and a task instruction.

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

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Suite
     - Demonstrations
     - Config
   * - LIBERO Spatial: 10 tasks
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
     - The demonstration's task instruction

Installation
------------

.. include:: _setup_common.rst

Install the GR00T-N1.7 environment, then activate it:

.. code-block:: bash

   bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
   source .venv/bin/activate

You can also activate an existing environment installed with these model and environment options. Run the remaining commands from the RLinf repository root in that environment. The example configuration uses one node with eight NVIDIA GPUs; adjust GPU placement and batch sizes together for another allocation.

Download the Model and Data
---------------------------

Accept the access terms for `Cosmos-Reason2-2B <https://huggingface.co/nvidia/Cosmos-Reason2-2B>`_, then download the GR00T weights and the backbone configuration and tokenizer:

.. code-block:: bash

   hf auth login
   hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/GR00T-N1.7-3B
   hf download nvidia/Cosmos-Reason2-2B \
     --include '*.json' '*.txt' --local-dir checkpoints/Cosmos-Reason2-2B

The GR00T checkpoint contains the backbone weights. The Cosmos directory supplies the configuration and tokenizer used to load them.

Download `Physical Intelligence LIBERO <https://huggingface.co/datasets/physical-intelligence/libero>`_:

.. code-block:: bash

   hf download physical-intelligence/libero \
     --repo-type dataset --local-dir datasets/physical-intelligence-libero

The full dataset contains 1,693 episodes across 40 tasks, with both camera images embedded in Parquet files. Keep ``meta/`` and ``data/`` together. You can reuse an existing download by setting its path below. The example selects the 10 Spatial tasks through ``data.task_names``, converts gripper actions to the GR00T convention, and computes normalization statistics from those selected episodes.

Run It
------

Edit ``examples/sft/config/libero_sft_gr00t_n1d7.yaml``. Set the dataset and model paths, then choose GPU placement, batch sizes, and the update budget:

.. code-block:: yaml

   cluster:
     component_placement:
       actor: 0-7
   data:
     train_data_paths:
       - /absolute/path/to/datasets/physical-intelligence-libero
     format: physical_libero
     statistics_path: ./data_statistics/libero_spatial.json
   runner:
     max_steps: 20000
     save_interval: 1000
   actor:
     micro_batch_size: 1
     global_batch_size: 32
     model:
       model_path: /absolute/path/to/checkpoints/GR00T-N1.7-3B
       backbone_model_path: /absolute/path/to/checkpoints/Cosmos-Reason2-2B
     optim:
       total_training_steps: 20000

Keep the Spatial task list supplied in the configuration. ``data.statistics_path`` is a writable cache file for the selected data; normalization statistics are also saved with each checkpoint.

The global batch size must be divisible by the number of GPUs times the microbatch size. Keep ``runner.max_epochs: -1``: training samples continuously and stops at ``runner.max_steps``. Set ``actor.optim.total_training_steps`` to the intended learning-rate schedule length. The 20,000-update budget is an example, not a measured convergence target.

Launch a single-node run with the edited configuration. Start a local Ray instance so the run does not join another cluster on the machine:

.. code-block:: bash

   export RLINF_NODE_RANK=0
   export RAY_ADDRESS=local
   bash examples/sft/run_vla_sft.sh libero_sft_gr00t_n1d7

The launcher writes the console log and TensorBoard events to ``logs/<timestamp>-libero_sft_gr00t_n1d7/``. Watch ``train/loss`` and the global update count; checkpoints appear under that run directory as ``libero_sft_gr00t_n1d7/checkpoints/global_step_N/``.

Resume from a Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~

Set ``runner.resume_dir`` in the same YAML to the absolute path of a saved ``global_step_N`` directory, then rerun the launch command above. Keep the dataset, GPU topology, batch sizes, and dataloader worker count unchanged. ``runner.max_steps`` is the total update count to reach, including the updates already completed.

For a short save-and-resume check, set ``runner.max_steps: 10``, ``runner.save_interval: 10``, ``actor.optim.lr_warmup_steps: 0``, and ``actor.optim.total_training_steps: 20`` for the first launch. After it finishes, set ``runner.max_steps: 20`` and ``runner.resume_dir`` to its ``global_step_10`` directory, then launch again. The second run should continue at update 10 and save ``global_step_20``. See :doc:`../../guides/resume` for checkpoint recovery details.

.. note::

   The dataloader restores its position by replaying the stream. Random image augmentation is not guaranteed to replay bit for bit.

Visualization and Results
-------------------------

Open TensorBoard to inspect the run:

.. code-block:: bash

   tensorboard --logdir logs

Use ``train/loss`` to track action fitting and check that the saved update directories match your budget. Each checkpoint contains model and optimizer state, training normalization and processor settings, and dataloader recovery state. Metric definitions are in :doc:`../../reference/metrics`.

Closed-loop evaluation also requires a model directory containing the matching processor settings. See :doc:`../../evaluations/index` for the evaluation workflow and :doc:`gr00t` for LIBERO setup.

.. warning::

   Evaluation must use both the trained weights and the processor settings saved with them. Setting ``runner.ckpt_path`` loads weights only; the processor comes from ``rollout.model.model_path``. An SFT ``actor/`` directory is a training checkpoint, not a complete Hugging Face model directory. Prepare a model directory with the matching processor before evaluating.
