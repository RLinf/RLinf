SIMPLE Evaluation
=================

Overview
--------

Evaluate Psi0 on eight SIMPLE whole-body humanoid Teleop tasks through RLinf, each using its matching checkpoint, reset states, and HSSD scene. All eight tasks run ``mujoco_isaac`` at 50 Hz rendering / 200 Hz physics with the 30/24/6 RTC protocol and report ``eval/success``.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      SIMPLE (8 G1 whole-body Teleop tasks)

   .. grid-item-card:: Models
      :text-align: center

      Psi0 (task-specific & PPO checkpoints)

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/simple/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`Psi0 Guide </rst_source/examples/embodied/psi0>`

.. note::

   **Scope:** Covers the eight Teleop tasks with task-specific checkpoints in the pinned Psi0 release. ``G1WholebodyBendPickMP-v0`` and ``G1WholebodyTabletopGraspMP-v0`` use different state, action, and System-0 protocols and are not supported.

.. _simple-reset-data:

Setup and Run
-------------

Prepare the shared prerequisites from the :doc:`Psi0 Guide </rst_source/examples/embodied/psi0>` first: :ref:`environment setup <psi0-setup-env>`, :ref:`Qwen3-VL metadata <psi0-qwen-metadata>` (cached via ``HF_HOME``, without ``--local-dir``), :ref:`SIMPLE simulation assets <simple-sim-assets>` in ``SIMPLE_ASSETS_DIR``, and :ref:`linking the assets <simple-link-assets>` to ``${VIRTUAL_ENV}/SIMPLE/data``.

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: 1-Episode Quick Check (DR0)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model psi0 --env simple && source .venv/bin/activate

      **2. Download Checkpoint and Reset States**

      Download the OpenOven checkpoint and reset states (requires ``HF_HOME`` from the Psi0 Guide):

      .. code-block:: bash

         export PSI0_ARTIFACT_ROOT=./psi0-simple-artifacts
         export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
         export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604
         export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
         export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"

         hf download USC-PSI-Lab/psi-model \
           "${PSI0_RUN_REL}/argv.txt" "${PSI0_RUN_REL}/run_config.json" "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
           --revision d34a91932d25c45ef211582315b9224c7dc8ace9 --local-dir ./psi0-simple-artifacts/psi-model
         hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
           --repo-type dataset --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 --local-dir ./psi0-simple-artifacts/psi-data
         mkdir -p "${SIMPLE_RESET_DIR}" && unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" -d "${SIMPLE_RESET_DIR}"

      **3. Download HSSD Scene** (requires ``SIMPLE_ASSETS_DIR`` from the Psi0 Guide)

      .. code-block:: bash

         export SIMPLE_SCENE=102344280
         hf download USC-PSI-Lab/SIMPLE "scenes_hssd_${SIMPLE_SCENE}.zip" \
           --repo-type dataset --local-dir "${SIMPLE_ASSETS_DIR}"
         unzip -o -q "${SIMPLE_ASSETS_DIR}/scenes_hssd_${SIMPLE_SCENE}.zip" -d "${SIMPLE_ASSETS_DIR}"
         rm "${SIMPLE_ASSETS_DIR}/scenes_hssd_${SIMPLE_SCENE}.zip"

      **4. Evaluate**

      .. code-block:: bash

         bash evaluations/run_eval.sh simple simple_psi0_eval \
           rollout.model.model_path="${PSI0_RUN_DIR}" \
           env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}"

   .. tab-item:: Full 3-Level DR (30 Episodes)

      Run 10 episodes at each of the three domain-randomization (DR) levels and report each level separately.

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model psi0 --env simple && source .venv/bin/activate

      **2. Download Checkpoint and Reset States**

      .. code-block:: bash

         export PSI0_ARTIFACT_ROOT=./psi0-simple-artifacts
         export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
         export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604
         export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
         export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"

         hf download USC-PSI-Lab/psi-model \
           "${PSI0_RUN_REL}/argv.txt" "${PSI0_RUN_REL}/run_config.json" "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
           --revision d34a91932d25c45ef211582315b9224c7dc8ace9 --local-dir ./psi0-simple-artifacts/psi-model
         hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
           --repo-type dataset --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 --local-dir ./psi0-simple-artifacts/psi-data
         mkdir -p "${SIMPLE_RESET_DIR}" && unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" -d "${SIMPLE_RESET_DIR}"

      **3. Download HSSD Scene**

      .. code-block:: bash

         export SIMPLE_SCENE=102344280
         hf download USC-PSI-Lab/SIMPLE "scenes_hssd_${SIMPLE_SCENE}.zip" \
           --repo-type dataset --local-dir "${SIMPLE_ASSETS_DIR}"
         unzip -o -q "${SIMPLE_ASSETS_DIR}/scenes_hssd_${SIMPLE_SCENE}.zip" -d "${SIMPLE_ASSETS_DIR}"
         rm "${SIMPLE_ASSETS_DIR}/scenes_hssd_${SIMPLE_SCENE}.zip"

      **4. Evaluate**

      .. code-block:: bash

         for SIMPLE_DR_LEVEL in 0 1 2; do
           bash evaluations/run_eval.sh simple simple_psi0_eval \
             rollout.model.model_path="${PSI0_RUN_DIR}" \
             env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
             env.eval.reset_dataset.dr_level="${SIMPLE_DR_LEVEL}" \
             env.eval.reset_dataset.num_episodes=10 \
             env.eval.rollout_epoch=10
         done

``evaluations/simple/simple_psi0_eval.yaml`` defaults to OpenOven (1 episode, DR0). All tasks share ``PSI0_ARTIFACT_ROOT``; when switching tasks, update ``SIMPLE_TASK``, ``PSI0_RUN_REL``, and ``SIMPLE_SCENE`` together from :ref:`the task table <simple-tasks>`.

.. _simple-tasks:

.. dropdown:: Supported Tasks and Switching Tasks

   The eight valid Teleop substitutions in the `pinned Psi0 checkpoint release <https://huggingface.co/USC-PSI-Lab/psi-model/tree/d34a91932d25c45ef211582315b9224c7dc8ace9/psi0/simple-checkpoints>`_ (``Official result`` is success count over 10 episodes at DR levels 0/1/2 from the `pinned SIMPLE results <https://github.com/physical-superintelligence-lab/SIMPLE/blob/5e3d6f84e85343e34e9bca8d157f0d7813231185/README.md#-simulation-benchmarking-results>`_):

   .. list-table::
      :header-rows: 1
      :widths: 22 40 16 10 12

      * - ``SIMPLE_TASK``
        - Run name
        - ``SIMPLE_SCENE``
        - Horizon
        - Official result
      * - ``G1WholebodyXMovePickTeleop-v0``
        - ``g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205``
        - ``107734119_175999932``
        - 800
        - 10/10/6
      * - ``G1WholebodyHandoverTeleop-v0``
        - ``g1wholebodyhandover-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604071507``
        - ``107734119_175999932``
        - 800
        - 7/7/10
      * - ``G1WholebodyLocomotionPickBetweenTablesTeleop-v0``
        - ``g1wholebodylocomotionpickbetweentablesteleop-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604081126``
        - ``102344022``
        - 1200
        - 7/5/6
      * - ``G1WholebodyXMoveBendPickTeleop-v0``
        - ``g1wholebodyxmovebendpickteleop-v0.simple.flow1000.cosine.lr1.0e-04.b112.gpus7.2604100422``
        - ``107734119_175999932``
        - 800
        - 10/9/9
      * - ``G1WholebodyCloseDoorTeleop-v0``
        - ``g1wholebodyclosedoorteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605070100``
        - ``107734119_175999932``
        - 1000
        - 10/10/10
      * - ``G1WholebodyOpenOvenTeleop-v0``
        - ``g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604``
        - ``102344280``
        - 1000
        - 7/5/4
      * - ``G1WholebodyOpenFaucetTeleop-v0``
        - ``g1wholebodyopenfaucetteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605081439``
        - ``107733960_175999701``
        - 1000
        - 3/3/4
      * - ``G1WholebodyPickAndPlaceAndHugContainerTeleop-v0``
        - ``g1wholebodypickandplaceandhugcontainerteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604280201``
        - ``102344307``
        - 1000
        - 7/6/3

   **Switch to another task**

   Update all five variables (exported paths do not update automatically), rerun the download commands if those assets are missing, and override ``task_id`` and step limits:

   .. code-block:: bash

      export SIMPLE_TASK=G1WholebodyXMovePickTeleop-v0
      export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205
      export SIMPLE_SCENE=107734119_175999932
      export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
      export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"

      bash evaluations/run_eval.sh simple simple_psi0_eval \
        rollout.model.model_path="${PSI0_RUN_DIR}" \
        env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
        env.eval.init_params.task_id="simple/${SIMPLE_TASK}" \
        env.eval.max_episode_steps=800 \
        env.eval.max_steps_per_rollout_epoch=816

   ``max_steps_per_rollout_epoch`` is the smallest multiple of 24 not smaller than the task horizon:

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

.. _simple-trained-checkpoint:

.. dropdown:: Evaluate a Trained Checkpoint

   Build the model and processor from the original run directory, load saved PPO weights via ``runner.ckpt_path``, and set ``rollout.model.add_value_head=true`` to restore value-head parameters:

   .. code-block:: bash

      export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
      export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604
      export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
      export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
      export PSI0_PPO_CHECKPOINT=./logs/psi0_ppo/checkpoints/global_step_20

      bash evaluations/run_eval.sh simple simple_psi0_eval \
        rollout.model.model_path="${PSI0_RUN_DIR}" \
        rollout.model.add_value_head=true \
        runner.ckpt_path="${PSI0_PPO_CHECKPOINT}/actor/model_state_dict/full_weights.pt" \
        env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
        env.eval.reset_dataset.num_episodes=10 \
        env.eval.rollout_epoch=10

Results and Troubleshooting
---------------------------

Evaluation reports ``eval/success``; logs and videos go to ``logs/<timestamp>-simple_psi0_eval/`` (see :doc:`../reference/results`). All eight tasks use ``mujoco_isaac``, 50 Hz rendering, 200 Hz physics, at most 300 stabilization steps, and 30/24/6 RTC.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Qwen metadata fails to load
     - Psi0 requires the raw Hugging Face snapshot layout; set ``HF_HOME`` when downloading (without ``--local-dir``) and export the same ``HF_HOME`` before evaluation.
   * - Mismatched task or missing HSSD scene
     - Update ``SIMPLE_TASK``, ``PSI0_RUN_REL``, ``SIMPLE_SCENE``, ``PSI0_RUN_DIR``, and ``SIMPLE_RESET_DIR`` together and pass ``env.eval.init_params.task_id="simple/${SIMPLE_TASK}"``.
   * - Step-count validation error
     - Set ``env.eval.max_steps_per_rollout_epoch`` to the smallest multiple of 24 covering ``max_episode_steps`` (``816`` for 800, ``1008`` for 1000, ``1200`` for 1200).
   * - PPO checkpoint missing value head keys
     - Pass ``rollout.model.add_value_head=true`` alongside ``runner.ckpt_path``.
