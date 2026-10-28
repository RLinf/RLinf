Psi0
====

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

   .. grid-item-card:: Environment
      :text-align: center

      SIMPLE

   .. grid-item-card:: Algorithm
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      8 humanoid Teleop tasks

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

Installation
------------

Set up Psi0 + SIMPLE in five steps. Steps 1–4 are shared by all SIMPLE tasks;
step 5 downloads the resources of one task.

.. _psi0-setup-env:

1. Set Up the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

From the RLinf repository root, install and activate the Psi0 + SIMPLE runtime:

.. code-block:: bash

   bash requirements/install.sh embodied --model psi0 --env simple
   source .venv/bin/activate

Alternatively, build the Docker image and switch to its Psi0 environment:

.. code-block:: bash

   docker build -f docker/Dockerfile --build-arg BUILD_TARGET=embodied-simple -t rlinf:embodied-simple .
   # Inside the container:
   source switch_env psi0

Both pin
`Psi0 <https://github.com/physical-superintelligence-lab/Psi0/tree/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06>`_
and
`SIMPLE <https://github.com/physical-superintelligence-lab/SIMPLE/tree/5e3d6f84e85343e34e9bca8d157f0d7813231185>`_,
and neither includes the simulation assets.
See :doc:`Installation </rst_source/start/installation>` for system prerequisites.

.. _psi0-qwen-metadata:

2. Download the Qwen Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The joint checkpoint already contains the Qwen3-VL weights, but the loader reads
the Qwen configuration, tokenizer, and processor by model ID:

.. code-block:: bash

   export HF_HOME=/path/to/hf-cache
   hf download Qwen/Qwen3-VL-2B-Instruct \
     chat_template.json config.json generation_config.json merges.txt \
     preprocessor_config.json tokenizer.json tokenizer_config.json \
     video_preprocessor_config.json vocab.json

.. note::

   The loader needs the raw Hugging Face snapshot of these files. Set ``HF_HOME`` when downloading them instead of using ``--local-dir``, and export the same ``HF_HOME`` before training or evaluation.

.. _simple-sim-assets:

3. Download the Simulation Assets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the assets shared by the SIMPLE tasks (G1 robot, GraspNet objects,
articulated objects, and vMaterials) from
`USC-PSI-Lab/SIMPLE <https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE>`_.
If you already have them, set ``SIMPLE_ASSETS_DIR`` to that directory and skip
the download.

.. code-block:: bash

   export SIMPLE_ASSETS_DIR=/path/to/simple_assets
   bash requirements/embodied/download_assets.sh --assets simple --dir "${SIMPLE_ASSETS_DIR}"

The script unpacks each package into ``SIMPLE_ASSETS_DIR``, removes the zip,
and skips packages that are already unpacked. If you want to use a mirror, pass
the ``--use-mirror`` flag.

.. _simple-link-assets:

4. Link the Assets
~~~~~~~~~~~~~~~~~~

SIMPLE reads assets from ``data/`` in its checkout, which the installer places
at ``${VIRTUAL_ENV}/SIMPLE``. In the checkout, ``data/`` only holds SIMPLE's
README and download helper. Replace it with a link to the assets:

.. code-block:: bash

   rm -rf "${VIRTUAL_ENV}/SIMPLE/data"
   ln -s "${SIMPLE_ASSETS_DIR}" "${VIRTUAL_ENV}/SIMPLE/data"

If you installed with ``SIMPLE_PATH``, link ``${SIMPLE_PATH}/data`` instead.

.. _psi0-model-download:

5. Download the Task Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the official OpenOven checkpoint with its run configuration, the
OpenOven reset data, and the OpenOven HSSD scene:

.. code-block:: bash

   export PSI0_ARTIFACT_ROOT=/path/to/psi0-simple-artifacts
   export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604

   hf download USC-PSI-Lab/psi-model \
     "${PSI0_RUN_REL}/argv.txt" \
     "${PSI0_RUN_REL}/run_config.json" \
     "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
     --revision d34a91932d25c45ef211582315b9224c7dc8ace9 \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-model"
   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"

   hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
     --repo-type dataset \
     --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-data"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
   mkdir -p "${SIMPLE_RESET_DIR}"
   unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" \
     -d "${SIMPLE_RESET_DIR}"

   hf download USC-PSI-Lab/SIMPLE scenes_hssd_102344280.zip \
     --repo-type dataset \
     --local-dir "${SIMPLE_ASSETS_DIR}"
   unzip -q "${SIMPLE_ASSETS_DIR}/scenes_hssd_102344280.zip" -d "${SIMPLE_ASSETS_DIR}"
   rm "${SIMPLE_ASSETS_DIR}/scenes_hssd_102344280.zip"

``PSI0_RUN_DIR`` is the run directory. Its configuration supplies the model
architecture and preprocessing transforms; the joint checkpoint contains both
System-2 and System-1 weights. The training configuration reads
``PSI0_RUN_DIR`` for both actor and rollout models, and ``SIMPLE_RESET_DIR``
for reset data. ``scenes_hssd_102344280`` is the OpenOven scene
``hssd:scene1``. If it is missing, SIMPLE downloads it on the first rollout.

.. _psi0-ppo-training:

Run It
------

With the asset paths exported, launch the OpenOven training configuration:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh simple_openoven_ppo_psi0

The configuration uses stochastic-transition sampling for PPO, keeps System-2
frozen, and trains the action expert and value head. It runs four environments
on four GPUs with DR0 and a 600-step horizon. Each iteration collects 32
trajectories over 8 rollout epochs and performs 4 optimizer steps with
``global_batch_size=200`` and ``update_epoch=1``.

To train another Teleop task, set ``PSI0_RUN_DIR`` and ``SIMPLE_RESET_DIR`` to its
assets from the :ref:`SIMPLE task table <simple-tasks>`. In
``examples/embodiment/config/simple_openoven_ppo_psi0.yaml``, update
``env.train.init_params.task_id``, ``env.train.max_episode_steps``, and
``env.train.max_steps_per_rollout_epoch``. Choose ``actor.global_batch_size`` to
divide the number of sampled chunks per update. The launcher reads these
settings from YAML.

.. note::

   Keep ``actor.model.psi0.plan_horizon`` / ``actor.model.num_action_chunks`` /
   ``actor.model.psi0.overlap_horizon`` at ``30`` / ``24`` / ``6``. Their defaults
   are in ``examples/embodiment/config/model/psi0.yaml``. Each rollout worker
   runs one environment, so keep ``rollout.pipeline_stage_num`` at ``1``.

Visualization and Results
-------------------------

Follow training in TensorBoard and the terminal log:

.. code-block:: text

   logs/<timestamp>-simple_openoven_ppo_psi0/run_embodiment.log

See :doc:`Training Metrics </rst_source/reference/metrics>` for metric definitions.
To evaluate saved weights, follow
:ref:`Evaluate a Trained Checkpoint <simple-trained-checkpoint>`.
