JAX-Aligned PyTorch OpenPI Supervised Fine-Tuning
=================================================

This page documents supervised fine-tuning (SFT) for RLinf's self-contained,
JAX-aligned PyTorch implementation of OpenPI. It supports the ``Pi0`` and
``Pi0.5`` flow-matching VLA variants, registered as
``model_type: openpi_pytorch``. The implementation follows the OpenPI JAX
reference architecture and precision behavior while using PyTorch and FSDP for
training.

The currently maintained SFT recipes are:

- **Pi0 on RoboTwin**
- **Pi0.5 on BEHAVIOR-1K**

There is currently no maintained Pi0-on-BEHAVIOR SFT configuration. Use one of
the recipes below rather than mixing a model template and a dataset config that
are not listed together.


Available recipes
-----------------

Each recipe is an experiment configuration under ``examples/sft/config/``.
The configuration imports its matching path-free model template through Hydra
and supplies the local dataset, checkpoint, and normalization-statistics paths.

.. list-table::
   :header-rows: 1
   :widths: 18 18 32 32

   * - Model
     - Dataset
     - Experiment configuration
     - Model template / OpenPI config
   * - Pi0
     - RoboTwin
     - ``robotwin_sft_openpi_pytorch.yaml``
     - ``model/pi0_pytorch.yaml`` / ``pi0_aloha_robotwin``
   * - Pi0.5
     - BEHAVIOR-1K
     - ``behavior_pi05_vla.yaml``
     - ``model/pi0_5_pytorch.yaml`` / ``pi05_behavior``

Both recipes load fp32 master weights and use FSDP mixed precision with
bf16 parameter computation and fp32 gradient reduction/buffers. This keeps the
reference-aligned optimizer behavior while reducing activation and compute
memory use. Gradient checkpointing is enabled in the supplied configs.


Prepare a recipe
----------------

Start from the experiment configuration in the table and replace every
``/path/to/...`` placeholder. The model checkpoint must be in the RLinf PyTorch
layout (``model.safetensors`` plus ``config.json``), with
the matching ``norm_stats.json`` available at the configured asset location.
Use the ``jax2rlinf_pytorch`` checkpoint-converter mode if you are starting from an
OpenPI JAX checkpoint; see
``rlinf/utils/ckpt_convertor/openpi/README.md`` for the full conversion flow.

RoboTwin
~~~~~~~~

The RoboTwin recipe uses the LeRobot-format RoboTwin dataset with these data
settings:

.. code:: yaml

   data:
     train_data_paths: /path/to/robotwin-data
     num_workers: 4
     tolerance_s: 1.0e-4

RoboTwin uses 14-dimensional ALOHA actions and three input images. The model
config pads the actions to OpenPI's 32-dimensional model action space and sets
``num_action_chunks: 50``. Set the model and assets paths in the selected
recipe, for example:

.. code:: yaml

   actor:
     model:
       model_path: /path/to/pi0_base_rlinf_pytorch
       openpi:
         assets_dir: ${actor.model.model_path}
         asset_id: "physical-intelligence/robotwin"
         num_images_in_input: 3

Use the Pi0 checkpoint with ``robotwin_sft_openpi_pytorch.yaml``.

BEHAVIOR-1K
~~~~~~~~~~~

``behavior_pi05_vla.yaml`` uses the Pi0.5 streaming BEHAVIOR loader. It trains
on 32-step, 23-dimensional dual-arm R1 Pro action chunks with the
flow-matching denoising objective. Configure the dataset root, task selection,
and matching Pi0.5 assets under ``data`` and ``actor.model.openpi``:

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

   actor:
     model:
       model_path: /path/to/pi05_base_rlinf_pytorch
       openpi:
         assets_dir: /path/to/assets
         asset_id: "behavior-1k/2025-challenge-demos"

``train_data_paths`` and ``behavior_dataset_root`` identify the local BEHAVIOR
dataset. ``tasks`` selects the task or tasks to train. With ``use_skill:
false``, training uses the main-task text; with ``true``, it uses the per-frame
REFERENCE skill text specified by ``task_subtasks``. When skill training is
enabled, use the explicit ordered labels that correspond to the selected task.


Launch training
---------------

From the repository root, launch the recipe that matches the desired model and
dataset:

.. code:: bash

   # Pi0 on RoboTwin
   bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pytorch

   # Pi0.5 on BEHAVIOR-1K
   bash examples/sft/run_vla_sft.sh behavior_pi05_vla

The helper sets the SFT config path, records the run command, and writes logs
and checkpoints under ``logs/<timestamp>-<config-name>``. Checkpoints are saved
according to ``runner.save_interval`` in
``checkpoints/global_step_<N>/``.


Convert an SFT checkpoint
-------------------------

All SFT checkpoints use ``sft2rlinf_pytorch``. ``--config-name`` selects the
matching Pi0/Pi0.5 architecture; ``--dtype fp32`` preserves SFT master weights.

For RoboTwin Pi0:

.. code:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2rlinf_pytorch \
       --config-name pi0_aloha_robotwin \
       --dtype fp32 \
       --ckpt /path/to/checkpoints/global_step_30000 \
       --input-norm-stats /path/to/pi0_base_rlinf_pytorch/physical-intelligence/robotwin/norm_stats.json \
       --output-model /path/to/pi0_robotwin_sft_rlinf_pytorch \
       --output-norm-stats /path/to/pi0_robotwin_sft_rlinf_pytorch/physical-intelligence/robotwin/norm_stats.json \
       --reference-model /path/to/pi0_base_rlinf_pytorch

For Pi0.5 on BEHAVIOR-1K:

.. code:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2rlinf_pytorch \
       --config-name pi05_behavior \
       --dtype fp32 \
       --ckpt /path/to/checkpoints/global_step_30000 \
       --input-norm-stats /path/to/norm_stats.json \
       --output-model /path/to/pi05_behavior_sft_rlinf_pytorch \
       --output-norm-stats /path/to/pi05_behavior_sft_rlinf_pytorch/physical-intelligence/behavior/norm_stats.json

The selected ``--config-name`` preserves the RoboTwin or BEHAVIOR architecture
in the output configuration. See the converter README for every option and the
matching evaluation configuration.
