RoboCasa365 RL
==============

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

RoboCasa365 is the benchmark-native kitchen manipulation integration in RLinf.
Unlike the legacy :doc:`RoboCasa <robocasa>` recipe, ``robocasa365`` selects
tasks through RoboCasa's official dataset registry (``split`` + ``task_soup``)
and exposes split and task-soup metadata at runtime. You'll use RLinf to
PPO-fine-tune and evaluate OpenPI π₀ on official RoboCasa365 benchmark splits.

Overview
--------

Train and evaluate OpenPI π₀ on RoboCasa365 benchmark splits selected via the official dataset registry.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      OpenDrawer · atomic_seen · composite_unseen

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install → download RoboCasa365 assets + π₀ checkpoint → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · RoboCasa365 kitchen assets · an OpenPI π₀ checkpoint.

Tasks
~~~~~

The default recipe trains ``OpenDrawer`` under ``split=pretrain`` and ``task_soup=atomic_seen``; evaluation supports any official task soup such as ``atomic_seen``, ``composite_seen``, or ``composite_unseen``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task / Slice
     - Description
   * - ``OpenDrawer`` (``pretrain`` / ``atomic_seen``)
     - Single-task PPO training with ``task_filter=["OpenDrawer"]`` and ``task_mode=atomic``.
   * - ``atomic_seen`` / ``composite_unseen``
     - Official RoboCasa365 task soups selected via ``split`` and ``task_soup`` for evaluation.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Multi-view RGB images (``robot0_agentview_left_image`` and ``robot0_eye_in_hand_image`` by default) plus a configurable proprioceptive state assembled from ``observation.state_layout``.
   * - Action
     - Configurable 12-dim RoboCasa mobile-manipulation action schema for ``PandaOmron``; the wrapper can disable base control and map OpenPI action slices before stepping.
   * - Reward
     - Sparse task-completion reward.
   * - Prompt
     - Natural-language instruction and runtime split/task-soup metadata from the RoboCasa365 environment.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robocasa365 && source .venv/bin/activate

      **2a. Download Assets & Checkpoint**

      Download the RoboCasa365 scene archives into ``$ROBOCASA_ASSETS_PATH/_downloads`` and the official RoboCasa365 OpenPI π₀ checkpoint:

      .. code-block:: bash

         export ROBOCASA_ASSETS_PATH="$PWD/robocasa_assets"

         hf download robocasa/robocasa-assets --repo-type dataset \
           --include textures.zip generative_textures.zip fixtures.zip objaverse.zip aigen_objs.zip \
           --local-dir "$ROBOCASA_ASSETS_PATH/_downloads"
         hf download nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF --repo-type dataset \
           --include "fixtures_lightwheel/*.zip" \
           --local-dir "$ROBOCASA_ASSETS_PATH/_downloads"
         hf download nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF --repo-type dataset \
           --include "objects_lightwheel/*.zip" \
           --local-dir "$ROBOCASA_ASSETS_PATH/_downloads"
         hf download RLinf/RLinf-pi0-robocasa-pretrain-human300 \
           --local-dir ./RLinf-pi0-robocasa-pretrain-human300

      **2b. Extract Assets**

      Unpack the downloaded archives into the layout expected under ``$ROBOCASA_ASSETS_PATH``:

      .. code-block:: bash

         export ROBOCASA_ASSETS_PATH="$PWD/robocasa_assets"
         mkdir -p "$ROBOCASA_ASSETS_PATH"/{fixtures,objects/lightwheel}

         unzip -q "$ROBOCASA_ASSETS_PATH/_downloads/textures.zip" -d "$ROBOCASA_ASSETS_PATH"
         unzip -q "$ROBOCASA_ASSETS_PATH/_downloads/generative_textures.zip" -d "$ROBOCASA_ASSETS_PATH"
         unzip -q "$ROBOCASA_ASSETS_PATH/_downloads/objaverse.zip" -d "$ROBOCASA_ASSETS_PATH/objects"
         unzip -q "$ROBOCASA_ASSETS_PATH/_downloads/aigen_objs.zip" -d "$ROBOCASA_ASSETS_PATH/objects"

         find "$ROBOCASA_ASSETS_PATH/_downloads/fixtures_lightwheel" -name "*.zip" \
           -exec unzip -q {} -d "$ROBOCASA_ASSETS_PATH/fixtures" \;
         find "$ROBOCASA_ASSETS_PATH/_downloads/objects_lightwheel" -name "*.zip" \
           -exec unzip -q {} -d "$ROBOCASA_ASSETS_PATH/objects/lightwheel" \;

      **3. Launch**

      Shipped RoboCasa365 configs:

      .. list-table::
         :header-rows: 1
         :widths: 26 48 26

         * - Workflow
           - Config
           - Launcher
         * - PPO Training (OpenDrawer)
           - ``examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml``
           - ``examples/embodiment/run_embodiment.sh``
         * - Evaluation
           - ``evaluations/robocasa365/robocasa365_eval_openpi.yaml``
           - ``evaluations/run_eval.sh``
         * - Short E2E Test
           - ``tests/e2e_tests/embodied/robocasa365_ppo_openpi.yaml``
           - ``tests/e2e_tests/embodied/run.sh``

      .. include:: _model_path.rst

      Launch PPO training on ``OpenDrawer``:

      .. code-block:: bash

         export ROBOCASA_ASSETS_PATH="$PWD/robocasa_assets"
         bash examples/embodiment/run_embodiment.sh robocasa365_opendrawer_ppo_openpi actor.model.model_path=$PWD/RLinf-pi0-robocasa-pretrain-human300

      .. dropdown:: Advanced configuration (optional)

         - **Environment config fields** (``examples/embodiment/config/env/robocasa365.yaml``):

           - ``task_source``: keep ``dataset_registry`` for RoboCasa365.
           - ``dataset_source``: data source registered by RoboCasa, typically ``human``.
           - ``split`` / ``task_soup``: official benchmark split (``pretrain``, ``target``) and task soup (``atomic_seen``, ``composite_seen``, ``composite_unseen``). RLinf's ``task_soup`` maps to RoboCasa's ``task_set`` argument in ``get_ds_soup`` (see `RoboCasa dataset usage <https://robocasa.ai/docs/build/html/datasets/using_datasets.html>`__).
           - ``task_filter`` / ``task_mode``: optional include/exclude list and ``atomic`` / ``composite`` guardrail.
           - ``observation`` / ``action_space``: camera keys, ``state_layout`` mapping, and OpenPI action slice mapping.

         - **OpenDrawer batch divisibility**: check these constraints when changing GPU placement, rollout size, or actor batch size in ``examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml``:

           .. code-block:: text

              chunk_steps = env.train.max_steps_per_rollout_epoch / actor.model.num_action_chunks
              rollout_size_per_actor_rank = (
                  env.train.total_num_envs * env.train.rollout_epoch / actor_world_size
              ) * chunk_steps

              rollout_size_per_actor_rank % (actor.global_batch_size / actor_world_size) == 0
              actor.global_batch_size % (actor.micro_batch_size * actor_world_size) == 0

         - **Standalone evaluation**: evaluate ``pretrain`` / ``atomic_seen`` (or override ``env.eval.split``, ``env.eval.task_soup``, and ``env.eval.task_mode`` for another slice):

           .. code-block:: bash

              bash evaluations/run_eval.sh robocasa365 robocasa365_eval_openpi \
                env.eval.task_soup=composite_unseen \
                env.eval.task_mode=composite

         - **End-to-end test**: run the shortened PPO e2e config:

           .. code-block:: bash

              bash tests/e2e_tests/embodied/run.sh robocasa365_ppo_openpi

Visualization and Results
-------------------------

Monitor ``env/success_once`` with ``tensorboard --logdir ./logs --port 6006``, enable rollout video recording with ``env.eval.video_cfg.save_video=True``, and see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.
