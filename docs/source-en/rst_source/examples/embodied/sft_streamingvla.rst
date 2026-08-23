StreamingVLA Supervised Fine-Tuning
===================================

This page describes RLinf's **training-only** StreamingVLA integration for
LIBERO demonstrations. The model is registered as ``model_type:
streamingvla`` and implements the Streaming Flow Policy (SFP) objective on a
Pi0.5 backbone. The integration supports full-parameter SFT through FSDP2. It
does not implement action generation, rollout, simulator evaluation, or
reinforcement learning.

The StreamingVLA implementation is isolated under
``rlinf/models/embodiment/streamingvla``. It does not patch RLinf's existing
``openpi`` or ``openpi_rlinf`` models, the installed Transformers package, or
the shared normalization and optimizer implementations.


Install dependencies
--------------------

From the repository root, create an environment with the StreamingVLA and
LIBERO dependencies:

.. code:: bash

   bash requirements/install.sh embodied --model streamingvla --env libero
   source .venv/bin/activate

Unlike the ``--model openpi`` installation, this model does not copy a
replacement package over the installed Transformers files. Its compatible
Gemma, SigLIP, preprocessing, and SFP code is imported from the private
StreamingVLA namespace.


Prepare data and weights
------------------------

Convert the public LIBERO RLDS suites into a LeRobot dataset with the required
start-of-step cumulative ``action_states`` field:

.. code:: bash

   python toolkits/lerobot/convert_libero_data_to_lerobot.py \
       --data-dir /path/to/modified_libero_rlds \
       --repo-name your_hf_username/libero_streamingvla

If that exact output dataset already exists, the converter fails safely. Pass
``--overwrite`` only when you intend to replace it.

The recipe expects a map-style LeRobot LIBERO dataset. Every episode sample
must contain the real ``action_states`` array in addition to the normal image,
wrist image, robot state, action, and task fields. The loader checks the first
actual record and fails before training if ``action_states`` is absent; metadata
alone is not accepted as evidence that the field exists.

The local transforms map 7-dimensional LIBERO states, action states, and action
deltas into the model's 32-dimensional action space. They also apply the
zero-centred linear quantile formula

.. math::

   x / (\max(|q_{0.01}|, |q_{0.99}|) + 10^{-6}).

Place ``norm_stats.json`` at
``<assets_dir>/<asset_id>/norm_stats.json``. The statistics must include the
``state``, ``actions``, and ``action_states`` entries used by the dataset.
Generate it with the StreamingVLA-specific tool:

.. code:: bash

   python toolkits/lerobot/calculate_streamingvla_norm_stats.py \
       --repo-id /path/to/libero_lerobot_dataset \
       --assets-dir /path/to/dataset-parent \
       --asset-id libero_lerobot_dataset

The SFP training convention intentionally copies the complete ``actions``
statistics to ``action_states``; the two entries in ``norm_stats.json`` are
therefore identical.

Set ``actor.model.model_path`` to a Pi0.5 PyTorch checkpoint directory that
contains ``model.safetensors``. Legacy OpenPI base weights and RLinf
StreamingVLA checkpoints are loaded through a strict private loader: missing,
unexpected, or shape-mismatched tensors stop initialization.


Configure the recipe
--------------------

The checked-in files contain only portable placeholders:

- Experiment: ``examples/sft/config/libero_sft_streamingvla.yaml``
- Model template: ``examples/sft/config/model/streamingvla.yaml``

At minimum, replace these values in the experiment config:

.. code:: yaml

   data:
     train_data_paths: /path/to/libero_lerobot_dataset

   actor:
     model:
       model_path: /path/to/pi05_libero_pytorch
       streamingvla:
         data:
           repo_id: ${data.train_data_paths}
           assets:
             assets_dir: /path/to/dataset-parent
             asset_id: libero_lerobot_dataset

The public StreamingVLA contract is fixed to ``use_sfp: true``,
``use_action_states: true``, horizon 10, action dimension 32, ``sigma: 0.16``,
and ``noise_decay: 4.0``. The default recipe uses seed 42, global/micro batches
16/4, a 10,000-step warmup, 100,000 optimizer steps, and checkpoints every
5,000 steps.


Launch training
---------------

Run the existing VLA SFT helper from the repository root:

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_streamingvla

The default logger is TensorBoard. To explicitly enable Weights & Biases, log
in through the normal W&B environment and override only the backend and run
name:

.. code:: bash

   export EMBODIED_PATH="$(pwd)/examples/sft"
   python examples/sft/train_vla_sft.py \
       --config-path examples/sft/config \
       --config-name libero_sft_streamingvla \
       'runner.logger.logger_backends=[wandb]' \
       runner.logger.experiment_name=streamingvla_libero_sft

RLinf writes checkpoints below
``<runner.logger.log_path>/checkpoints/global_step_<N>/``. Resume only from a
complete RLinf checkpoint directory using ``runner.resume_dir``. The
integration intentionally raises ``NotImplementedError`` if an inference,
rollout, or non-SFT forward path is requested.
