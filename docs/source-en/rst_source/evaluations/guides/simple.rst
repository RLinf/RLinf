Psi0 + SIMPLE Evaluation and Training
=====================================

This guide evaluates Psi0 on eight SIMPLE Teleop tasks through RLinf and runs
four-GPU PPO post-training on OpenOven. Every task requires its matching
checkpoint and reset states; these assets are not interchangeable.

Scope and Reproducibility
-------------------------

The installer pins the
`Psi0 model code <https://github.com/physical-superintelligence-lab/Psi0/tree/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06>`_
and
`SIMPLE environment code <https://github.com/physical-superintelligence-lab/SIMPLE/tree/5e3d6f84e85343e34e9bca8d157f0d7813231185>`_.
The ``--revision`` arguments below pin the model weights and reset data to
specific repository snapshots. These pins prevent code or asset updates from
changing the setup; no separate version-selection step is needed.

The current scope is the eight Teleop tasks with task-specific checkpoints in
the pinned Psi0 release. ``G1WholebodyBendPickMP-v0`` and
``G1WholebodyTabletopGraspMP-v0`` use different state, action, and System-0
protocols and are not supported yet.

Install the Runtime
-------------------

.. code-block:: bash

   bash requirements/install.sh embodied \
     --venv .venv-psi0-simple \
     --model psi0 \
     --env simple
   source .venv-psi0-simple/bin/activate

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

Task-Specific Assets
~~~~~~~~~~~~~~~~~~~~

The commands below use OpenOven. All tasks share the asset root and
``HF_HOME``, so set them once. When switching tasks, change only the annotated
``SIMPLE_TASK`` and run name. Both values must come from the same table row.
The remaining commands derive the matching paths from these values.

.. code-block:: bash

   # Set once. Reuse for every SIMPLE task.
   export PSI0_ARTIFACT_ROOT=/mnt/public2/yangtingyuan/RLinf_yty/checkpoint/psi0-simple-artifacts
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
the System-1 ``action_header`` weights, so it does not download Qwen model
weights separately. The pinned loader still resolves the Qwen configuration,
tokenizer, and processor by model ID. Download the following shared metadata
only once per ``HF_HOME``; do not repeat it when switching tasks:

.. code-block:: bash

   hf download Qwen/Qwen3-VL-2B-Instruct \
     chat_template.json config.json generation_config.json merges.txt \
     preprocessor_config.json tokenizer.json tokenizer_config.json \
     video_preprocessor_config.json vocab.json

The upstream loader resolves ``main`` by default. Downloading only a commit SHA
does not create the cache's ``main`` reference, so an offline load may fail even
when that snapshot exists. The command follows the loader's default and uses
the standard cache under ``HF_HOME``; an arbitrary ``--local-dir`` is not a
replacement. Once cached, these files can be reused offline. Unlike the pinned
Psi0 and SIMPLE assets, this metadata is not strictly version-locked: ``main``
can change, and strict pinning would also require the loader to use that revision.

Run Evaluation
--------------

The repository provides direct configs for CloseDoor, OpenFaucet, and
OpenOven:

.. code-block:: text

   evaluations/simple/simple_closedoor_psi0_eval.yaml
   evaluations/simple/simple_openfaucet_psi0_eval.yaml
   evaluations/simple/simple_openoven_psi0_eval.yaml

OpenOven
~~~~~~~~

The OpenOven config supplies the task, horizon, episode start, and video
settings, so the minimal smoke command only needs local asset path overrides:

.. code-block:: bash

   bash evaluations/run_eval.sh simple simple_openoven_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}"

For a full evaluation, run 10 episodes at each of the three DR levels:

.. code-block:: bash

   for SIMPLE_DR_LEVEL in 0 1 2; do
     bash evaluations/run_eval.sh simple simple_openoven_psi0_eval \
       rollout.model.model_path="${PSI0_RUN_DIR}" \
       env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
       env.eval.reset_dataset.dr_level="${SIMPLE_DR_LEVEL}" \
       env.eval.reset_dataset.num_episodes=10 \
       env.eval.rollout_epoch=10
   done

Switch to Another Task
~~~~~~~~~~~~~~~~~~~~~~

You can reuse the OpenOven config for another Teleop task. If its assets are
already downloaded and extracted, skip downloading them again, but reset all
four variables below. Changing the task or run name does not automatically
update previously exported paths. For XMovePick:

.. code-block:: bash

   export SIMPLE_TASK=G1WholebodyXMovePickTeleop-v0
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205
   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"

   bash evaluations/run_eval.sh simple simple_openoven_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
     env.eval.init_params.task_id="simple/${SIMPLE_TASK}" \
     env.eval.max_episode_steps=800 \
     env.eval.max_steps_per_rollout_epoch=816

For each task switch, update the four variables together and choose both step
limits from the table. The checkpoint, reset data, and ``task_id`` must match.
For the full three-level loop, retain both path overrides and add ``task_id``
and both step limits. ``max_steps_per_rollout_epoch`` is the smallest multiple
of 24 that is not smaller than the task horizon.

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

Run PPO Post-Training on OpenOven
----------------------------------

Set ``PSI0_RUN_DIR`` and ``SIMPLE_RESET_DIR`` to OpenOven before starting.
The current training config uses DR0, a 600-step limit, and four environments
on four GPUs. Each iteration collects 32 trajectories over 8 rollout epochs;
``global_batch_size=200`` and ``update_epoch=1`` give 4 optimizer steps.

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh simple_openoven_ppo_psi0

To switch to another Teleop task, update ``PSI0_RUN_DIR`` and ``SIMPLE_RESET_DIR``
as described in the download section. This config reads both variables for the
actor and rollout model paths and the training reset path; no repeated overrides
are needed. In ``examples/embodiment/config/simple_openoven_ppo_psi0.yaml``, also
change ``env.train.init_params.task_id``, ``env.train.max_episode_steps``, and
``env.train.max_steps_per_rollout_epoch`` using the task IDs and horizons in the
evaluation section. After changing the horizon, check that
``actor.global_batch_size`` divides the number of sampled chunks per update.
Edit these task fields in the YAML rather than appending them to the bash command.

To evaluate the trained checkpoint, use the same task, reset data, DR level,
and horizon. The command below uses the OpenOven evaluation config with the
official 1,000-step horizon.

The value head estimates future returns during PPO training, and its parameters
are saved in ``full_weights.pt``. ``add_value_head=true`` creates the matching
module so the strict checkpoint load can restore those parameters. It does not
train a new critic, and evaluation does not use it to generate actions.
``PSI0_RUN_DIR`` still points to the original task run directory to construct
the model and processor; ``runner.ckpt_path`` then loads the trained weights.

.. code-block:: bash

   export PSI0_PPO_CHECKPOINT=/absolute/path/to/global_step_20

   bash evaluations/run_eval.sh simple simple_openoven_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     rollout.model.add_value_head=true \
     runner.ckpt_path="${PSI0_PPO_CHECKPOINT}/actor/model_state_dict/full_weights.pt" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
     env.eval.reset_dataset.num_episodes=10 \
     env.eval.rollout_epoch=10
