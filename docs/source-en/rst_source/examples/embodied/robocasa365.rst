RL with RoboCasa365 Benchmark
====================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document describes the benchmark-native RoboCasa365 integration in RLinf.
Unlike the legacy :doc:`RoboCasa <robocasa>` recipe, this setup keeps the original
RoboCasa task wrapper untouched and adds a separate ``robocasa365`` environment that
selects tasks through the official RoboCasa dataset registry.

The main goal is to train and evaluate vision-language-action models on the
official RoboCasa365 benchmark splits while keeping the legacy RoboCasa recipe
stable. Compared with the single-task RoboCasa examples, this recipe focuses on:

1. **Benchmark split control**: select ``pretrain`` or ``target`` tasks through the official registry.
2. **Task-soup evaluation**: evaluate benchmark slices such as ``atomic_seen`` or ``composite_unseen``.
3. **Task metadata at runtime**: expose split and task-soup metadata in environment observations.

Environment
-----------

**RoboCasa365 Benchmark**

- **Environment**: RoboCasa365 kitchen simulation benchmark
- **Selection API**: official RoboCasa dataset registry via ``split`` + ``task_soup``
- **Robot**: mobile Panda configuration (``PandaOmron`` by default)
- **Observation**: multi-view RGB images + configurable proprioceptive state extractor
- **Action Space**: configurable RoboCasa mobile-manipulation action schema

The default RLinf recipe uses:

- ``split=pretrain`` for training
- ``split=pretrain`` for the lightweight evaluation config
- ``task_soup=atomic_seen`` as the first benchmark slice

You can switch to other official task soups, for example ``composite_seen`` or
``composite_unseen``, by editing the YAML config.

**Observation Structure**

- **Main camera image**: ``robot0_agentview_left_image`` by default
- **Wrist camera image**: ``robot0_eye_in_hand_image`` by default
- **Proprioceptive state**: configurable state vector assembled from the keys in
  ``observation.state_layout``

**Action Structure**

The default OpenPI recipe uses a 12-dimensional action schema. The wrapper can
disable base control and map valid OpenPI action slices before stepping the
underlying RoboCasa environment.

Configuration
-------------

The benchmark-specific environment config lives in:

.. code:: bash

   examples/embodiment/config/env/robocasa365.yaml

Key fields:

- ``task_source``: should stay ``dataset_registry`` for RoboCasa365
- ``dataset_source``: data source registered by RoboCasa, typically ``human``
- ``split``: benchmark split such as ``pretrain`` or ``target``
- ``task_soup``: official task soup name such as ``atomic_seen``
- ``task_filter``: optional include / exclude filter for narrowing the selected tasks
- ``task_mode``: optional ``atomic`` or ``composite`` guardrail
- ``observation``: camera keys and state-layout mapping used by RLinf
- ``action_space``: action schema and OpenPI slice mapping used before env stepping

Dependency Installation
-----------------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Prepare RoboCasa Assets
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   export ROBOCASA_ASSETS_PATH=/path/to/robocasa_assets
   mkdir -p "$ROBOCASA_ASSETS_PATH"/{_downloads,objects}

   hf download robocasa/robocasa-assets \
      --repo-type dataset \
      --include textures.zip generative_textures.zip fixtures.zip objaverse.zip aigen_objs.zip \
      --local-dir "$ROBOCASA_ASSETS_PATH/_downloads"
   hf download nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF \
      --repo-type dataset \
      --include "fixtures_lightwheel/*.zip" \
      --local-dir "$ROBOCASA_ASSETS_PATH/_downloads"
   hf download nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF \
      --repo-type dataset \
      --include "objects_lightwheel/*.zip" \
      --local-dir "$ROBOCASA_ASSETS_PATH/_downloads"

   for item in \
   textures.zip:textures \
   generative_textures.zip:generative_textures \
   objaverse.zip:objects/objaverse \
   aigen_objs.zip:objects/aigen_objs; do

   zip_name="${item%%:*}"
   target_rel="${item#*:}"
   parent_rel="$(dirname "$target_rel")"

   if [[ "$parent_rel" == "." ]]; then
      unzip_dir="$ROBOCASA_ASSETS_PATH"
   else
      unzip_dir="$ROBOCASA_ASSETS_PATH/$parent_rel"
   fi

   mkdir -p "$unzip_dir"

   unzip -q "$ROBOCASA_ASSETS_PATH/_downloads/$zip_name" \
      -d "$unzip_dir"
   done

   mkdir -p "$ROBOCASA_ASSETS_PATH/fixtures"
   find "$ROBOCASA_ASSETS_PATH/_downloads/fixtures_lightwheel" -name "*.zip" \
   -exec unzip -q {} -d "$ROBOCASA_ASSETS_PATH/fixtures" \;

   mkdir -p "$ROBOCASA_ASSETS_PATH/objects/lightwheel"
   find "$ROBOCASA_ASSETS_PATH/_downloads/objects_lightwheel" -name "*.zip" \
   -exec unzip -q {} -d "$ROBOCASA_ASSETS_PATH/objects/lightwheel" \;

3. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   export ROBOCASA_ASSETS_PATH=/path/to/robocasa_assets
   bash requirements/install.sh embodied --model openpi --env robocasa365
   source .venv/bin/activate

Dataset Selection
-----------------

RLinf now delegates task selection to RoboCasa's official registry. The following
command pattern from the RoboCasa docs is what the RLinf wrapper mirrors internally:

.. code:: python

   from robocasa.utils.dataset_registry import get_ds_soup

   task_names = get_ds_soup(
       task_set="atomic_seen",
       split="pretrain",
       source="human",
   )

RoboCasa names this argument ``task_set``; it corresponds to RLinf's
``task_soup`` YAML field.

Useful references:

- RoboCasa dataset usage:
  https://robocasa.ai/docs/build/html/datasets/using_datasets.html

Model Checkpoint
----------------

Download the official RoboCasa365 Pi0 checkpoint:

.. code:: bash

   hf download RLinf/pi0-robocasa-pretrain-human300 \
     --local-dir /path/to/pi0-robocasa-pretrain-human300

Point the config to the downloaded directory:

.. code:: yaml

   rollout:
     model:
       model_path: "/path/to/pi0-robocasa-pretrain-human300"

   actor:
     model:
       model_path: "/path/to/pi0-robocasa-pretrain-human300"

Training
--------

The OpenDrawer training recipe is:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh robocasa365_opendrawer_ppo_openpi

This config trains a single RoboCasa365 task with OpenPI and PPO:

- ``env.train.split=pretrain``
- ``env.train.task_soup=atomic_seen``
- ``env.train.task_filter=["OpenDrawer"]``
- ``env.train.task_mode=atomic``

OpenDrawer Batch Divisibility

Use this check when changing GPU placement, rollout size, or actor batch size in
``examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml``.

.. code:: text

   chunk_steps = env.train.max_steps_per_rollout_epoch / actor.model.num_action_chunks
   rollout_size_per_actor_rank = (
       env.train.total_num_envs * env.train.rollout_epoch / actor_world_size
   ) * chunk_steps

   rollout_size_per_actor_rank % (actor.global_batch_size / actor_world_size) == 0
   actor.global_batch_size % (actor.micro_batch_size * actor_world_size) == 0

End-to-End Test
----------------

The shortened PPO e2e config is ``robocasa365_ppo_openpi``:

.. code:: bash

   bash tests/e2e_tests/embodied/run.sh robocasa365_ppo_openpi

Evaluation
----------

The lightweight evaluation config is ``robocasa365_eval_openpi``:

.. code:: bash

   bash evaluations/run_eval.sh robocasa365_eval_openpi

This config evaluates on:

- ``env.eval.split=pretrain``
- ``env.eval.task_soup=atomic_seen``
- ``env.eval.task_mode=atomic``
- no ``env.eval.task_filter`` by default, so the selected pretrain slice is not narrowed

To evaluate a different benchmark slice, override the YAML fields directly:

.. code:: yaml

   env:
     eval:
       split: target
       task_soup: composite_unseen
       task_mode: composite

For example:

.. code:: bash

   bash evaluations/run_eval.sh robocasa365_eval_openpi \
      env.eval.task_soup=composite_unseen \
      env.eval.task_mode=composite
      