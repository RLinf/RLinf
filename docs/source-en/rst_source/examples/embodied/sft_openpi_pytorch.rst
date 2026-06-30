Supervised Fine-Tuning with PyTorch OpenPI (Pi0.5) on BEHAVIOR
==============================================================

This page explains how to run **supervised fine-tuning (SFT)** of the
self-contained **PyTorch OpenPI Pi0.5** flow-matching VLA on the
**BEHAVIOR-1K** task with the RLinf framework. The model is a pure-PyTorch
re-implementation of the Pi0.5 architecture (dual-expert Gemma + SigLIP with a
flow-matching action head), registered in RLinf under
``model_type: openpi_pytorch``. SFT is typically the first stage before
reinforcement learning: the model imitates high-quality demonstrations so that
RL can continue optimization from a strong prior.

Contents
----------

- What the PyTorch OpenPI SFT flow is and how it is configured
- The precision contract used by the FSDP optimizer and mixed-precision compute
- The streaming BEHAVIOR data-loader keys and norm-stats / tokenizer settings
- How to launch training and convert the resulting checkpoints for evaluation


What it is
----------

The ``openpi_pytorch`` model is a self-contained PyTorch port of the Pi0.5
flow-matching VLA. Unlike the JAX/LeRobot-backed OpenPI path (see
:doc:`sft_openpi`), it builds the model shape directly from a small set of
config fields (no ``config.json`` is read at construction time) and is wired for
BEHAVIOR-1K out of the box. During SFT the policy is trained to predict the
32-step, 23-dim action chunk for the dual-arm R1 Pro robot from the BEHAVIOR
demonstrations, using the flow-matching denoising objective.


Configuration
-------------

The example is split into a reusable, path-free **model template** and an
**experiment config** that supplies the filesystem paths:

- Experiment config: ``examples/sft/config/behavior_pi05_vla.yaml``
- Model template: ``examples/sft/config/model/pi0_5_pytorch.yaml``

The experiment config pulls in the model template through Hydra ``defaults``:

.. code:: yaml

   defaults:
     - model/pi0_5_pytorch@actor.model
     - training_backend/fsdp@actor.fsdp_config
     - override hydra/job_logging: stdout

Precision contract
~~~~~~~~~~~~~~~~~~~

The PyTorch OpenPI SFT recipe deliberately separates the **load dtype** from the
**compute dtype**:

- The model template sets ``actor.model.precision: fp32`` (in
  ``pi0_5_pytorch.yaml``). The fp32 weights are loaded as the **FSDP optimizer
  master**, so warmup-LR updates are not lost to bf16 rounding.
- FSDP ``MixedPrecision`` computes in bf16 while keeping the gradient all-reduce
  and buffers in fp32:

  .. code:: yaml

     actor:
       fsdp_config:
         gradient_checkpointing: True
         mixed_precision:
           param_dtype: bf16     # FSDP compute dtype
           reduce_dtype: fp32    # grad all-reduce stays fp32
           buffer_dtype: fp32

  ``param_dtype`` is the FSDP **compute** dtype and is set explicitly to bf16
  rather than being interpolated from ``actor.model.precision``: the load-dtype
  selector and the compute dtype are independent knobs, so an fp32-master load
  still computes in bf16.
- Gradient checkpointing is enabled
  (``actor.fsdp_config.gradient_checkpointing: True``) on the dual-expert Gemma +
  SigLIP backbone to reduce activation memory.
- The learning-rate schedule is a reference-exact warmup + cosine decay,
  selected with ``actor.optim.lr_scheduler: openpi_cosine`` (warmup starts at
  ``peak / (warmup + 1)`` and cosine-decays to ``min_lr`` over
  ``total_training_steps``).

Streaming data loader
~~~~~~~~~~~~~~~~~~~~~~~

The BEHAVIOR streaming loader reads all of its parameters directly from the
``data:`` section (there are no hidden defaults):

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
- ``modalities``: input modalities consumed by the loader (e.g. ``["rgb"]``).
- ``num_workers``: number of data-loader worker processes.
- ``fine_grained_level`` and ``tolerance_s``: time-alignment controls for the
  streaming reader.
- ``tasks``: the BEHAVIOR task(s) to train on.
- ``use_skill``: when ``false``, train on the main-task text; when ``true``,
  train on the per-frame REFERENCE skill text selected from ``task_subtasks``.
- ``task_subtasks``: per-task ordered skill labels used to build the
  index-to-label mapping when ``use_skill: true``.

Norm stats and tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

The normalization statistics and PaliGemma tokenizer live under
``actor.model.openpi``:

.. code:: yaml

   actor:
     model:
       model_path: /path/to/pi05_base_pytorch_new
       openpi:
         assets_dir: /path/to/assets
         asset_id: "behavior-1k/2025-challenge-demos"
         paligemma_tokenizer: /path/to/paligemma_tokenizer/paligemma_tokenizer.model

- ``assets_dir``: directory holding the quantile-normalization stats.
- ``asset_id``: sub-path under ``assets_dir`` for this task's stats.
- ``paligemma_tokenizer``: the PaliGemma SentencePiece tokenizer model
  (resolved from YAML, not hardcoded in code).

The norm stats are resolved at ``{assets_dir}/{asset_id}/norm_stats.json``.

Filesystem paths
~~~~~~~~~~~~~~~~~

All filesystem paths are set directly in the config as ``/path/to/...``
placeholders. Edit them in ``examples/sft/config/behavior_pi05_vla.yaml`` to
point at your own staged assets:

- ``data.train_data_paths`` / ``data.behavior_dataset_root``: root of the
  BEHAVIOR streaming dataset.
- ``actor.model.model_path``: the new-format **fp32 base checkpoint** the
  trainer loads.
- ``actor.model.openpi.assets_dir``: the norm-stats directory.
- ``actor.model.openpi.paligemma_tokenizer``: the PaliGemma SentencePiece
  tokenizer model.


Launch scripts
----------------

Run the SFT helper with the BEHAVIOR Pi0.5 config name:

.. code:: bash

   # return to repo root
   bash examples/sft/run_vla_sft.sh behavior_pi05_vla

The script forwards the config name to the SFT entry point and writes logs and
checkpoints under the configured ``runner.logger.log_path``. Checkpoints are
saved every ``runner.save_interval`` steps under
``.../checkpoints/global_step_<N>/``.


Converting checkpoints for evaluation
-------------------------------------

An SFT-trained checkpoint can be converted into the bare new-format ``Pi0``
layout (the layout the evaluation loader expects) with the OpenPI checkpoint
convertor:

.. code:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2new \
       --ckpt              /path/to/logs/.../checkpoints/global_step_30000 \
       --input-norm-stats  /path/to/norm_stats.json \
       --output-model      /path/to/pi05_sft_pytorch_new \
       --output-norm-stats /path/to/pi05_sft_pytorch_new/physical-intelligence/behavior/norm_stats.json

The ``sft2new`` mode strips the wrapper/FSDP key prefixes, casts floating-point
tensors to bf16 (the new-format eval loader validates that every checkpoint
tensor is bf16), and copies the norm-stats file verbatim. See the convertor
package README at ``rlinf/utils/ckpt_convertor/openpi/README.md`` for the other
conversion modes and full flag reference. The converted checkpoint can then be
used to evaluate on BEHAVIOR; see :doc:`behavior` for the eval config and launch
command.
