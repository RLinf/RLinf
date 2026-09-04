OpenPI_RLinf Supervised Fine-Tuning
=================================================

This page explains how to run **supervised fine-tuning (SFT)** of the
self-contained **OpenPI_RLinf Pi0.5** flow-matching VLA on the
**BEHAVIOR-1K** task with RLinf. The model is a pure-PyTorch reimplementation
of the Pi0.5 architecture (dual-expert Gemma + SigLIP with a flow-matching
action head), registered in RLinf under ``model_type: openpi_rlinf``. SFT is
typically the first stage before reinforcement learning: the model imitates
high-quality demonstrations so that RL can continue optimization from a strong
prior. This page also describes use of the same JAX-aligned implementation for
**Pi0 + RoboTwin**.

For SFT and eval-only deployment of PyTorch OpenPI Pi0.5 on a real dual-Franka
setup, see :doc:`the Dual-Franka PyTorch OpenPI guide <dual_franka_openpi_pytorch>`.


Contents
--------

- The OpenPI_RLinf SFT flow and its configuration
- The FSDP optimizer and mixed-precision contract
- BEHAVIOR streaming-loader fields and norm-stat/tokenizer handling
- Launching training and converting checkpoints for evaluation
- The official OpenPI/LeRobot data path, training, and evaluation for Pi0 RoboTwin


What it is
----------

The ``openpi_rlinf`` model is a self-contained PyTorch port of the Pi0.5
flow-matching VLA. **It is worth emphasizing that** the PyTorch implementation
shipped in the official OpenPI repository is not numerically aligned with its
JAX reference, whereas this port is numerically aligned with the JAX
implementation. Unlike the JAX/LeRobot-backed OpenPI path (see
:doc:`sft_openpi`), it builds the model shape directly from a small set of
config fields (no ``config.json`` is read at construction time) and is wired
for BEHAVIOR-1K out of the box. During SFT, the policy predicts 32-step,
23-dimensional action chunks for the dual-arm R1 Pro robot from BEHAVIOR
demonstrations using the flow-matching denoising objective.


Pi0.5 + BEHAVIOR-1K
-------------------

Configuration
~~~~~~~~~~~~~

The example is split into a reusable, path-free **model template** and an
**experiment config** that supplies filesystem paths:

- Experiment config: ``examples/sft/config/behavior_pi05_vla.yaml``
- Model template: ``examples/sft/config/model/pi0_5_rlinf.yaml``

The experiment config imports the model template through Hydra ``defaults``:

.. code:: yaml

   defaults:
     - model/pi0_5_rlinf@actor.model
     - hybrid_engines/fsdp@actor.fsdp_config
     - override hydra/job_logging: stdout

Precision contract
~~~~~~~~~~~~~~~~~~

The OpenPI_RLinf SFT configuration deliberately separates the **load dtype**
from the **compute dtype**:

- The model template sets ``actor.model.precision`` to ``fp32`` (in
  ``pi0_5_rlinf.yaml``). fp32 weights are loaded as the **FSDP optimizer
  master**, preventing small warmup-LR updates from being lost to bf16 rounding.
- FSDP ``MixedPrecision`` computes in bf16 while keeping gradient all-reduce
  and buffers in fp32:

  .. code:: yaml

     actor:
       fsdp_config:
         gradient_checkpointing: True
         mixed_precision:
           param_dtype: bf16     # FSDP compute dtype
           reduce_dtype: fp32    # gradient all-reduce stays fp32

  ``param_dtype`` is the FSDP **compute** dtype and is explicitly set to bf16,
  rather than interpolated from ``actor.model.precision``. The load-dtype
  selector and compute dtype are independent, so an fp32-master load still
  computes in bf16.
- Gradient checkpointing is enabled on the dual-expert Gemma + SigLIP backbone
  through ``actor.fsdp_config.gradient_checkpointing: True`` to reduce
  activation memory.
- The learning-rate schedule uses the reference warmup + cosine decay through
  ``actor.optim.lr_scheduler: openpi_cosine``. Warmup begins at
  ``peak / (warmup + 1)`` and the schedule cosine-decays to ``min_lr`` over
  ``total_training_steps``.

Streaming data loader
~~~~~~~~~~~~~~~~~~~~~

The BEHAVIOR streaming loader reads all parameters directly from the ``data:``
section; there are no hidden defaults:

.. code:: yaml

   data:
     train_data_paths: /path/to/2025-challenge-demos
     behavior_dataset_root: /path/to/2025-challenge-demos
     repo_id: "behavior-1k/2025-challenge-demos"
     modalities: ["rgb"]
     num_workers: 8
     fine_grained_level: 0
     tolerance_s: 1.0e-4
     tasks: ["turning_on_radio"]
     use_skill: false
     task_subtasks:
       turning_on_radio:
         - "move to radio"
         - "pick up radio from coffee table"
         - "press radio"
         - "place radio on coffee table"

Key data fields:

- ``train_data_paths`` / ``behavior_dataset_root``: root of the BEHAVIOR
  dataset (the latter defaults to the former).
- ``repo_id``: BEHAVIOR demonstration repo id
  (``behavior-1k/2025-challenge-demos``).
- ``modalities``: input modalities consumed by the loader, such as ``["rgb"]``.
- ``num_workers``: number of data-loader worker processes.
- ``fine_grained_level`` and ``tolerance_s``: time-alignment controls for the
  streaming reader.
- ``tasks``: the BEHAVIOR task or tasks to train on.
- ``use_skill``: when ``false``, train on the main-task text; when ``true``,
  train on the per-frame REFERENCE skill text selected from ``task_subtasks``.
- ``task_subtasks``: ordered per-task skill labels used to build the
  index-to-label mapping when ``use_skill: true``.

Norm stats and tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~

Normalization-statistics paths are configured under ``actor.model.openpi``:

.. code:: yaml

   actor:
     model:
       model_path: /path/to/pi05_base_pytorch_new
       openpi:
         assets_dir: /path/to/assets
         asset_id: "behavior-1k/2025-challenge-demos"

- ``assets_dir``: directory holding quantile-normalization statistics.
- ``asset_id``: sub-path under ``assets_dir`` for the task statistics.

Norm stats resolve at ``{assets_dir}/{asset_id}/norm_stats.json``. The
PaliGemma tokenizer is loaded from the base-model configuration by OpenPI's
``ModelTransformFactory`` when constructing the input transform, so the
``openpi_rlinf`` SFT YAML does not need a separate SentencePiece tokenizer
path.

Filesystem paths
~~~~~~~~~~~~~~~~

All filesystem paths are written as ``/path/to/...`` placeholders in
``examples/sft/config/behavior_pi05_vla.yaml``. Replace them with your staged
resources:

- ``data.train_data_paths`` / ``data.behavior_dataset_root``: root of the
  BEHAVIOR streaming dataset.
- ``actor.model.model_path``: new-format **fp32 base checkpoint** loaded by the
  trainer.
- ``actor.model.openpi.assets_dir``: normalization-statistics directory.


Pi0 + RoboTwin
--------------

Pi0 RoboTwin uses the official OpenPI/LeRobot map-style data loader:

- Experiment config: ``examples/sft/config/robotwin_sft_openpi_rlinf.yaml``
- Model template: ``examples/sft/config/model/pi0_rlinf.yaml``
- OpenPI data config: ``pi0_aloha_robotwin``
- ``adjust_bottle`` dataset: `RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data <https://huggingface.co/datasets/RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data>`_
- HF-format SFT checkpoint for ``adjust_bottle``: `RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle <https://huggingface.co/RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle>`_

Replace the dataset, base checkpoint, and task-specific normalization-statistics
paths with your local paths:

.. code:: yaml

   data:
     train_data_paths: /path/to/robotwin-data
     num_workers: 4

   actor:
     model:
       model_path: /path/to/pi0_base_pytorch_new
       num_action_chunks: 50
       action_dim: 14
       openpi:
         config_name: "pi0_aloha_robotwin"
         assets_dir: ${actor.model.model_path}
         asset_id: "physical-intelligence/robotwin/adjust_bottle"
         num_images_in_input: 3
       openpi_data:
         norm_stats_path: ${actor.model.openpi.assets_dir}/${actor.model.openpi.asset_id}/norm_stats.json

RoboTwin uses 14-dimensional ALOHA actions and three input images. OpenPI pads
actions to the 32-dimensional model action space. Set ``asset_id`` to the
statistics directory for the selected task, such as ``adjust_bottle`` above.
``openpi_data.norm_stats_path`` explicitly selects that task's
``norm_stats.json``, ensuring SFT and evaluation use the same statistics.

The Pi0 RoboTwin recipe also uses fp32 master weights, bf16 FSDP computation,
and fp32 gradient reduction. Its experiment YAML defaults to
``actor.optim.lr_scheduler: openpi_cosine``. This schedule starts warmup at
``peak / (warmup + 1)`` and reproduces the RoboTwin JAX reference learning-rate
curve.


Launch scripts
--------------

Run the SFT helper from the repository root:

.. code:: bash

   # Pi0.5 + BEHAVIOR-1K
   bash examples/sft/run_vla_sft.sh behavior_pi05_vla

   # Pi0 + RoboTwin
   bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_rlinf

The script forwards the config name to the SFT entry point and writes logs and
checkpoints under the configured ``runner.logger.log_path``. Checkpoints are
saved every ``runner.save_interval`` steps under
``.../checkpoints/global_step_<N>/``.


Converting checkpoints for evaluation
-------------------------------------

Use the OpenPI checkpoint converter to convert an SFT checkpoint to the bare
``Pi0`` layout expected by the evaluation loader:

.. code:: bash

   # Pi0.5 + BEHAVIOR-1K
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft_to_openpi_rlinf \
       --config-name pi05_behavior \
       --dtype bf16 \
       --ckpt              /path/to/logs/.../checkpoints/global_step_30000 \
       --input-norm-stats  /path/to/norm_stats.json \
       --output-model      /path/to/pi05_sft_pytorch_new \
       --output-norm-stats /path/to/pi05_sft_pytorch_new/physical-intelligence/behavior/norm_stats.json

   # Pi0 + RoboTwin
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft_to_openpi_rlinf \
       --config-name pi0_aloha_robotwin \
       --dtype fp32 \
       --ckpt              /path/to/logs/.../checkpoints/global_step_30000 \
       --input-norm-stats  /path/to/pi0_base_pytorch_new/physical-intelligence/robotwin/adjust_bottle/norm_stats.json \
       --output-model      /path/to/pi0_robotwin_sft_hf \
       --output-norm-stats /path/to/pi0_robotwin_sft_hf/physical-intelligence/robotwin/adjust_bottle/norm_stats.json \
       --reference-model   /path/to/pi0_base_pytorch_new

``sft_to_openpi_rlinf`` strips wrapper/FSDP key prefixes, selects the Pi0 or
Pi0.5 model shape using ``--config-name``, copies normalization statistics, and
writes floating-point tensors using ``--dtype {fp32,bf16}``. For RoboTwin, the
converted directory can be placed in ``rollout.model.model_path`` in
``evaluations/robotwin/robotwin_adjust_bottle_openpi_rlinf_eval.yaml``;
``openpi_data.norm_stats_path`` in the evaluation config must select the same
task statistics.

The converted checkpoint can be evaluated on
:doc:`BEHAVIOR-1K <../../evaluations/guides/behavior>` or
:doc:`RoboTwin <../../evaluations/guides/robotwin>`. For other conversion modes
and full argument details, see
``rlinf/utils/ckpt_convertor/openpi/README.md``.
