RL with RoboCasa365 Benchmark
====================================

This document describes the benchmark-native RoboCasa365 integration in RLinf.
Unlike the legacy :doc:`RoboCasa <robocasa>` recipe, this setup keeps the original
RoboCasa task wrapper untouched and adds a separate ``robocasa365`` environment that
selects tasks through the official RoboCasa dataset registry.

Environment
-----------

**RoboCasa365 Benchmark**

- **Environment**: RoboCasa365 kitchen simulation benchmark
- **Selection API**: official RoboCasa dataset registry via ``split`` + ``task_soup``
- **Robot**: mobile Panda configuration (``PandaMobile`` by default)
- **Observation**: multi-view RGB images + configurable proprioceptive state extractor
- **Action Space**: configurable RoboCasa mobile-manipulation action schema

The default RLinf recipe uses:

- ``split=pretrain`` for training
- ``split=target`` for evaluation
- ``task_soup=atomic_seen`` as the first benchmark slice

You can switch to other official task soups, for example ``composite_seen`` or
``composite_unseen``, by editing the YAML config.

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

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Use the RoboCasa dependency set:

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env robocasa
   source .venv/bin/activate

3. Download RoboCasa Assets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   python -m robocasa.scripts.download_kitchen_assets

Dataset Selection
-----------------

RLinf now delegates task selection to RoboCasa's official registry. The following
command pattern from the RoboCasa docs is what the RLinf wrapper mirrors internally:

.. code:: python

   from robocasa.utils.dataset_registry import get_ds_soup

   task_names = get_ds_soup(
       task_soup="atomic_seen",
       split="target",
       source="human",
   )

Useful references:

- RoboCasa dataset usage:
  https://robocasa.ai/docs/build/html/datasets/using_datasets.html
- RoboCasa dataset-registry API:
  https://robocasa.ai/docs/build/html/modules/robocasa.utils.dataset_registry.html

Model Checkpoint
----------------

The RoboCasa365 recipe uses a separate OpenPI config name, ``pi0_robocasa365``,
but it intentionally reuses the RoboCasa modality transform path. Provide your own
Pi0 checkpoint path in the config:

.. code:: yaml

   rollout:
     model:
       model_path: "/path/to/model/pi0_robocasa365"

   actor:
     model:
       model_path: "/path/to/model/pi0_robocasa365"

Training
--------

The benchmark-aligned training recipe is:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh robocasa365_grpo_openpi

This config trains on:

- ``env.train.split=pretrain``
- ``env.train.task_soup=atomic_seen``

Evaluation
----------

The benchmark-aligned evaluation recipe is:

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh robocasa365_eval_openpi

This config evaluates on:

- ``env.eval.split=target``
- ``env.eval.task_soup=atomic_seen``

To evaluate a different benchmark slice, change the YAML fields directly:

.. code:: yaml

   env:
     eval:
       split: target
       task_soup: composite_unseen
       task_mode: composite

Data Collection Notes
---------------------

When data collection is enabled, RLinf now preserves RoboCasa365 task metadata in
LeRobot exports:

- canonical task string
- benchmark selection
- split
- task soup
- task mode
- dataset source

This metadata is written into ``meta/tasks.jsonl`` and ``meta/episodes.jsonl`` so
offline analysis can distinguish benchmark subsets without parsing directory names.

Notes
-----

- The legacy :doc:`RoboCasa <robocasa>` page and ``robocasa`` env remain unchanged.
- ``robocasa365`` is a separate env type and folder on purpose, to keep old recipes stable.
- The first RLinf integration targets OpenPI / Pi0. Additional model-specific recipes can be added later on top of the same env.
