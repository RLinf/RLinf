Psi0 RL
=======

Run PPO post-training for Psi0 on SIMPLE humanoid tasks, starting from the
official OpenOven checkpoint. For task checkpoints and evaluation commands, see
the :doc:`SIMPLE evaluation guide </rst_source/evaluations/guides/simple>`.

`Psi0 <https://github.com/physical-superintelligence-lab/Psi0>`__ combines a
Qwen3-VL-2B vision-language model (System-2), a flow-matching action expert
(System-1), and a whole-body controller (System-0). RLinf freezes System-2 during
PPO, and updates the action expert and an added value head.

Overview
--------

Fine-tune the official OpenOven checkpoint with PPO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      SIMPLE

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      G1 OpenOven Teleop task

   .. grid-item-card:: Hardware
      :text-align: center

      4 GPUs

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The adapter pairs the head-camera image with robot state and the task instruction,
then sends the predicted commands to SIMPLE's whole-body controller.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Head-camera RGB image in ``main_images`` and 32-D proprioceptive state
       in ``states``. The checkpoint's transforms pad and normalize the state.
   * - Action
     - 36-D commands for upper-body joints, waist, base height, and navigation.
       Each plan has 30 actions: execute 24 and retain 6 for the next plan's RTC conditioning.
   * - Reward
     - SIMPLE task reward and success signal.
   * - Instruction
     - Natural-language task instruction in ``task_descriptions``.

.. _psi0-setup-env:
.. _psi0-qwen-metadata:
.. _simple-sim-assets:
.. _simple-link-assets:
.. _psi0-model-download:
.. _psi0-ppo-training:

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: SIMPLE

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model psi0 --env simple && source .venv/bin/activate

      **2a. Qwen3-VL Metadata & Shared SIMPLE Assets**

      Download the Qwen3-VL metadata snapshot (loaded by model ID from ``HF_HOME``; do not pass ``--local-dir``) and the shared `USC-PSI-Lab/SIMPLE <https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE>`_ simulation assets, then symlink them to ``${VIRTUAL_ENV}/SIMPLE/data``:

      .. code-block:: bash

         export HF_HOME=./hf-cache
         export SIMPLE_ASSETS_DIR=./simple_assets

         hf download Qwen/Qwen3-VL-2B-Instruct \
           chat_template.json config.json generation_config.json merges.txt \
           preprocessor_config.json tokenizer.json tokenizer_config.json \
           video_preprocessor_config.json vocab.json

         bash requirements/embodied/download_assets.sh --assets simple --dir "${SIMPLE_ASSETS_DIR}"
         rm -rf "${VIRTUAL_ENV}/SIMPLE/data"
         ln -s "$(realpath "${SIMPLE_ASSETS_DIR}")" "${VIRTUAL_ENV}/SIMPLE/data"

      **2b. OpenOven Checkpoint & Reset States**

      Download the official OpenOven checkpoint and task reset states:

      .. code-block:: bash

         export PSI0_ARTIFACT_ROOT=./psi0-simple-artifacts
         export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
         export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604

         hf download USC-PSI-Lab/psi-model \
           "${PSI0_RUN_REL}/argv.txt" "${PSI0_RUN_REL}/run_config.json" \
           "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
           --revision d34a91932d25c45ef211582315b9224c7dc8ace9 --local-dir ./psi0-simple-artifacts/psi-model
         export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"

         hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
           --repo-type dataset --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 \
           --local-dir ./psi0-simple-artifacts/psi-data
         export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
         mkdir -p "${SIMPLE_RESET_DIR}"
         unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" -d "${SIMPLE_RESET_DIR}"

      **2c. HSSD Scene**

      Download and extract the HSSD scene into ``SIMPLE_ASSETS_DIR``:

      .. code-block:: bash

         export SIMPLE_ASSETS_DIR=./simple_assets
         hf download USC-PSI-Lab/SIMPLE scenes_hssd_102344280.zip \
           --repo-type dataset --local-dir ./simple_assets
         unzip -q "${SIMPLE_ASSETS_DIR}/scenes_hssd_102344280.zip" -d "${SIMPLE_ASSETS_DIR}"
         rm "${SIMPLE_ASSETS_DIR}/scenes_hssd_102344280.zip"

      **3. Launch**

      .. list-table::
         :header-rows: 1
         :widths: 24 50 26

         * - Recipe
           - Config
           - Command suffix
         * - Psi0 + PPO (OpenOven)
           - ``examples/embodiment/config/simple_openoven_ppo_psi0.yaml``
           - ``simple_openoven_ppo_psi0``

      In ``examples/embodiment/config/simple_openoven_ppo_psi0.yaml``, set ``actor.model.model_path`` to ``PSI0_RUN_DIR`` (rollout inherits it automatically) and ``env.train.reset_dataset.path`` to ``SIMPLE_RESET_DIR``:

      .. code-block:: yaml

         env:
           train:
             reset_dataset:
               path: ./psi0-simple-artifacts/reset-states/G1WholebodyOpenOvenTeleop-v0
         actor:
           model:
             model_path: ./psi0-simple-artifacts/psi-model/psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604

      Launch OpenOven training:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh simple_openoven_ppo_psi0

.. note::

   Keep ``HF_HOME`` exported to the same cache directory before training or evaluation. ``VIRTUAL_ENV`` defaults to ``.venv`` (or ``/opt/venv/psi0`` in prebuilt containers). If you already have the shared SIMPLE assets, point ``SIMPLE_ASSETS_DIR`` to that directory and skip ``download_assets.sh``.

The configuration uses stochastic-transition sampling for PPO, keeps System-2
frozen, and trains the action expert and value head. It runs four environments
on four GPUs with DR0 and a 600-step horizon. Each iteration collects 32
trajectories over 8 rollout epochs and performs 4 optimizer steps with
``global_batch_size=200`` and ``update_epoch=1``.

.. dropdown:: Advanced configuration (optional)

   Keep the following RTC and pipeline settings unchanged. The RTC defaults are in
   ``examples/embodiment/config/model/psi0.yaml``. Each rollout worker runs one
   environment, so keep ``rollout.pipeline_stage_num`` at ``1``.

   .. code-block:: yaml

      actor:
        model:
          num_action_chunks: 24
          psi0:
            plan_horizon: 30
            overlap_horizon: 6
      rollout:
        pipeline_stage_num: 1

Visualization and Results
-------------------------

Follow training in TensorBoard and the terminal log:

.. code-block:: text

   logs/<timestamp>-simple_openoven_ppo_psi0/run_embodiment.log

See :doc:`Training Metrics </rst_source/reference/metrics>` for metric definitions.
To evaluate saved weights, follow
:ref:`Evaluate a Trained Checkpoint <simple-trained-checkpoint>`.
