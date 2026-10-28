SIMPLE Evaluation
=================

Evaluate Psi0 on eight SIMPLE Teleop tasks through RLinf. Each task uses its
matching checkpoint and reset states. For PPO post-training, see the
:doc:`Psi0 guide </rst_source/examples/embodied/psi0>`.

Scope and Reproducibility
-------------------------

The installer pins the
`Psi0 model code <https://github.com/physical-superintelligence-lab/Psi0/tree/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06>`_
and
`SIMPLE environment code <https://github.com/physical-superintelligence-lab/SIMPLE/tree/5e3d6f84e85343e34e9bca8d157f0d7813231185>`_.
The ``--revision`` arguments below pin the model weights and reset data to
specific repository snapshots. These pins prevent code or asset updates from
changing the setup.

The current scope is the eight Teleop tasks with task-specific checkpoints in
the pinned Psi0 release. ``G1WholebodyBendPickMP-v0`` and
``G1WholebodyTabletopGraspMP-v0`` use different state, action, and System-0
protocols and are not supported yet.

Install the Runtime
-------------------

Install the Psi0 and SIMPLE dependencies into the default virtual environment:

.. code-block:: bash

   bash requirements/install.sh embodied \
     --model psi0 \
     --env simple
   source .venv/bin/activate

.. _simple-sim-assets:

Simulation Assets
~~~~~~~~~~~~~~~~~

Before the installer finishes, it downloads the OpenOven simulation assets from
`USC-PSI-Lab/SIMPLE <https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE>`_,
unpacks them into ``.venv/SIMPLE/data``, and removes the zip files. A package
is skipped when its unpacked directory is already present:

- ``robots/g1``
- ``assets/graspnet``
- ``assets/articulated``
- ``scenes/hssd/102344280`` (the task's ``hssd:scene1``)
- ``vMaterials_2``

Assets stored inside the virtual environment are deleted with ``.venv``. Set
``SIMPLE_ASSETS_PATH`` before installation to keep them outside it. The
installer copies any files already in ``.venv/SIMPLE/data`` into that directory
without overwriting existing files, then replaces ``.venv/SIMPLE/data`` with a
symlink. Later installs only recreate the symlink and skip packages that are
already unpacked.

.. code-block:: bash

   export SIMPLE_ASSETS_PATH=/path/to/simple_assets
   bash requirements/install.sh embodied \
     --model psi0 \
     --env simple

Without this variable, assets stay in ``.venv/SIMPLE/data``. To fetch only the
missing packages without reinstalling the environment:

.. code-block:: bash

   bash requirements/embodied/download_assets.sh --assets simple --dir .venv/SIMPLE

``install.sh --use-mirror`` forwards ``HF_ENDPOINT`` to this download. If you
run ``download_assets.sh`` on its own and want to use a mirror, pass the
``--use-mirror`` flag.

.. _simple-tasks:

Select a Task and Download Assets
---------------------------------

The table lists the eight valid Teleop substitutions in the
`pinned Psi0 checkpoint release <https://huggingface.co/USC-PSI-Lab/psi-model/tree/d34a91932d25c45ef211582315b9224c7dc8ace9/psi0/simple-checkpoints>`_.
``Official result`` is the Psi0 success count over 10 episodes at each of
levels 0, 1, and 2 in the
`pinned SIMPLE results <https://github.com/physical-superintelligence-lab/SIMPLE/blob/5e3d6f84e85343e34e9bca8d157f0d7813231185/README.md#-simulation-benchmarking-results>`_.

.. list-table::
   :header-rows: 1
   :widths: 24 48 12 16

   * - ``SIMPLE_TASK``
     - Run name
     - Horizon
     - Official result
   * - ``G1WholebodyXMovePickTeleop-v0``
     - ``g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205``
     - 800
     - 10/10/6
   * - ``G1WholebodyHandoverTeleop-v0``
     - ``g1wholebodyhandover-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604071507``
     - 800
     - 7/7/10
   * - ``G1WholebodyLocomotionPickBetweenTablesTeleop-v0``
     - ``g1wholebodylocomotionpickbetweentablesteleop-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604081126``
     - 1200
     - 7/5/6
   * - ``G1WholebodyXMoveBendPickTeleop-v0``
     - ``g1wholebodyxmovebendpickteleop-v0.simple.flow1000.cosine.lr1.0e-04.b112.gpus7.2604100422``
     - 800
     - 10/9/9
   * - ``G1WholebodyCloseDoorTeleop-v0``
     - ``g1wholebodyclosedoorteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605070100``
     - 1000
     - 10/10/10
   * - ``G1WholebodyOpenOvenTeleop-v0``
     - ``g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604``
     - 1000
     - 7/5/4
   * - ``G1WholebodyOpenFaucetTeleop-v0``
     - ``g1wholebodyopenfaucetteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605081439``
     - 1000
     - 3/3/4
   * - ``G1WholebodyPickAndPlaceAndHugContainerTeleop-v0``
     - ``g1wholebodypickandplaceandhugcontainerteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604280201``
     - 1000
     - 7/6/3

.. _simple-reset-data:

Task-Specific Assets
~~~~~~~~~~~~~~~~~~~~

The commands below use OpenOven. All tasks share the asset root and
``HF_HOME``, so set them once. When switching tasks, change only the annotated
``SIMPLE_TASK`` and run name. Both values must come from the same table row.
The remaining commands derive the matching paths from these values.

.. code-block:: bash

   # Set once. Reuse for every SIMPLE task.
   export PSI0_ARTIFACT_ROOT=/path/to/psi0-simple-artifacts
   export HF_HOME="${PSI0_ARTIFACT_ROOT}/hf-cache"

   # Replace these two values together for each task.
   export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0  # Replace the task ID.
   # Replace only the run name after psi0/simple-checkpoints/.
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604

   mkdir -p "${PSI0_ARTIFACT_ROOT}"
   hf download USC-PSI-Lab/psi-model \
     "${PSI0_RUN_REL}/argv.txt" \
     "${PSI0_RUN_REL}/run_config.json" \
     "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
     --revision d34a91932d25c45ef211582315b9224c7dc8ace9 \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-model"

   hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
     --repo-type dataset \
     --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-data"

   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
   mkdir -p "${SIMPLE_RESET_DIR}"
   unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" \
     -d "${SIMPLE_RESET_DIR}"

Shared Qwen Metadata
~~~~~~~~~~~~~~~~~~~~

The joint Psi0 checkpoint contains both the System-2 ``vlm_model`` weights and
the System-1 ``action_header`` weights. The pinned loader still resolves the
Qwen configuration, tokenizer, and processor by model ID. Download the following
shared metadata once per ``HF_HOME``:

.. code-block:: bash

   hf download Qwen/Qwen3-VL-2B-Instruct \
     chat_template.json config.json generation_config.json merges.txt \
     preprocessor_config.json tokenizer.json tokenizer_config.json \
     video_preprocessor_config.json vocab.json

The upstream loader resolves ``main`` by default. This command writes metadata
to the standard cache under ``HF_HOME`` for offline reuse. Downloading only a
commit SHA does not create the cache's ``main`` reference, and an arbitrary
``--local-dir`` cannot replace the standard cache. ``main`` can change; strict
pinning also requires the loader to use the same revision.

Run Evaluation
--------------

``evaluations/simple/simple_psi0_eval.yaml`` defaults to OpenOven, one episode,
and DR0. Set the local asset paths to run it:

.. code-block:: bash

   bash evaluations/run_eval.sh simple simple_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}"

Evaluation reports ``eval/success``; logs and videos are saved under
``logs/<timestamp>-simple_psi0_eval/``.

For a full evaluation, run 10 episodes at each of the three DR levels:

.. code-block:: bash

   for SIMPLE_DR_LEVEL in 0 1 2; do
     bash evaluations/run_eval.sh simple simple_psi0_eval \
       rollout.model.model_path="${PSI0_RUN_DIR}" \
       env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
       env.eval.reset_dataset.dr_level="${SIMPLE_DR_LEVEL}" \
       env.eval.reset_dataset.num_episodes=10 \
       env.eval.rollout_epoch=10
   done

Switch to Another Task
~~~~~~~~~~~~~~~~~~~~~~

Reuse the same config for another Teleop task. For XMovePick, update all four
variables; exported paths do not change automatically:

.. code-block:: bash

   export SIMPLE_TASK=G1WholebodyXMovePickTeleop-v0
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205
   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"

If these assets are not ready, rerun the model and reset-data download commands
with the new variables. Then evaluate:

.. code-block:: bash

   bash evaluations/run_eval.sh simple simple_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
     env.eval.init_params.task_id="simple/${SIMPLE_TASK}" \
     env.eval.max_episode_steps=800 \
     env.eval.max_steps_per_rollout_epoch=816

The checkpoint, reset data, and ``task_id`` must match. In the three-level loop,
also add ``env.eval.init_params.task_id="simple/${SIMPLE_TASK}"`` and keep both
asset-path and step-limit overrides.
``max_steps_per_rollout_epoch`` is the smallest multiple of 24 that is not
smaller than the task horizon.

.. _simple-task-horizons:

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Task
     - ``max_episode_steps``
     - ``max_steps_per_rollout_epoch``
   * - XMovePick, Handover, XMoveBendPick
     - 800
     - 816
   * - CloseDoor, OpenOven, OpenFaucet, PickAndPlaceAndHugContainer
     - 1000
     - 1008
   * - LocomotionPickBetweenTables
     - 1200
     - 1200

All eight tasks use ``mujoco_isaac``, 50 Hz rendering, 200 Hz physics, at most
300 stabilization steps, and the 30/24/6 RTC protocol. The official result has
only 10 episodes per level. Report each level separately; neither a one-episode
smoke nor an aggregate count establishes strict reproduction.

.. _simple-trained-checkpoint:

Evaluate a Trained Checkpoint
-----------------------------

Use the original run directory to build the model and processor, then load the
saved PPO weights with ``runner.ckpt_path``. Set
``rollout.model.add_value_head=true`` to restore the saved value-head parameters.

For an OpenOven checkpoint, select the matching assets and evaluate at the
official 1,000-step horizon:

.. code-block:: bash

   export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604
   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
   export PSI0_PPO_CHECKPOINT=/absolute/path/to/global_step_20

   bash evaluations/run_eval.sh simple simple_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     rollout.model.add_value_head=true \
     runner.ckpt_path="${PSI0_PPO_CHECKPOINT}/actor/model_state_dict/full_weights.pt" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
     env.eval.reset_dataset.num_episodes=10 \
     env.eval.rollout_epoch=10
