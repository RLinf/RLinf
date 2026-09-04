StreamingVLA Supervised Fine-Tuning
===================================

StreamingVLA uses Streaming Flow Policy (SFP) to learn an action-space velocity
field from LIBERO demonstrations. This guide prepares a public LIBERO dataset,
computes the required normalization statistics, configures a Pi0.5 checkpoint,
and launches full-parameter FSDP2 training in RLinf.


Install dependencies
--------------------

From the RLinf repository root, create the StreamingVLA LIBERO environment:

.. code:: bash

   bash requirements/install.sh embodied --model streamingvla --env libero
   source .venv/bin/activate


Download and convert LIBERO
---------------------------

Download the public `OpenVLA modified LIBERO RLDS dataset
<https://huggingface.co/datasets/openvla/modified_libero_rlds>`_:

.. code:: bash

   hf download openvla/modified_libero_rlds \
       --repo-type dataset \
       --local-dir /data/libero-rlds

Choose where LeRobot datasets will be stored, then convert all four LIBERO
suites:

.. code:: bash

   export HF_LEROBOT_HOME=/data/lerobot

   python toolkits/lerobot/convert_libero_data_to_lerobot.py \
       --data-dir /data/libero-rlds \
       --repo-name local/libero_streamingvla

The converted dataset is written to
``$HF_LEROBOT_HOME/local/libero_streamingvla``. The converter adds the
``action_states`` field required by SFP to every episode. Keep
``HF_LEROBOT_HOME`` set to the same value when computing statistics and
training.


Compute normalization statistics
--------------------------------

Create an asset root and compute the statistics from the converted dataset:

.. code:: bash

   mkdir -p /data/streamingvla/assets

   python toolkits/lerobot/calculate_streamingvla_norm_stats.py \
       --repo-id local/libero_streamingvla \
       --assets-dir /data/streamingvla/assets \
       --asset-id libero_streamingvla

This example uses the following names:

- ``repo_id`` is the LeRobot dataset identifier below ``HF_LEROBOT_HOME``.
- ``assets_dir`` is the root directory for OpenPI-style data assets.
- ``asset_id`` is the subdirectory that identifies this dataset's assets.

The command writes
``/data/streamingvla/assets/libero_streamingvla/norm_stats.json``. The file
contains ``state``, ``actions``, and ``action_states`` statistics. SFP uses the
same normalization for ``actions`` and ``action_states``, so the tool copies
the complete ``actions`` entry to ``action_states``.


Prepare the Pi0.5 checkpoint
----------------------------

StreamingVLA fine-tuning starts from the official ``pi05_libero`` checkpoint.
The RLinf loader expects a PyTorch checkpoint directory containing
``model.safetensors``. In an `OpenPI
<https://github.com/Physical-Intelligence/openpi>`_ checkout, download the
official checkpoint and convert it:

.. code:: bash

   python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"

   python examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero" \
       --config_name pi05_libero \
       --output_path /data/checkpoints/pi05_libero_pytorch

After conversion, verify that
``/data/checkpoints/pi05_libero_pytorch/model.safetensors`` exists.


Configure and launch training
-----------------------------

Edit ``examples/sft/config/libero_sft_streamingvla.yaml`` to use the paths and
identifiers created above:

.. code:: yaml

   data:
     train_data_paths: local/libero_streamingvla

   actor:
     model:
       model_path: /data/checkpoints/pi05_libero_pytorch
       streamingvla:
         data:
           repo_id: ${data.train_data_paths}
           assets:
             assets_dir: /data/streamingvla/assets
             asset_id: libero_streamingvla

The supplied recipe uses ``use_sfp: true``, ``use_action_states: true``, action
horizon 10, model action dimension 32, ``sigma: 0.16``, and
``noise_decay: 4.0``. It runs 100,000 optimizer steps and saves a checkpoint
every 5,000 steps.

Launch training from the RLinf repository root:

.. code:: bash

   source .venv/bin/activate
   export HF_LEROBOT_HOME=/data/lerobot
   bash examples/sft/run_vla_sft.sh libero_sft_streamingvla

Training metrics are written through the logger configured in the YAML file.
Checkpoints are stored below
``<runner.logger.log_path>/checkpoints/global_step_<N>/``. To resume, set
``runner.resume_dir`` to a complete ``global_step_<N>`` checkpoint directory.
