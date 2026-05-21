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
- **Robot**: mobile Panda configuration (``PandaMobile`` by default)
- **Observation**: multi-view RGB images + configurable proprioceptive state extractor
- **Action Space**: configurable RoboCasa mobile-manipulation action schema

The default RLinf recipe uses:

- ``split=pretrain`` for training
- ``split=target`` for evaluation
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

**Note:** Since NVIDIA's HuggingFace dataset now stores Lightwheel assets as
multiple zip files under ``fixtures_lightwheel/`` and ``objects_lightwheel/``,
the upstream RoboCasa downloader cannot handle them automatically. Download and
extract those zips into the corresponding RoboCasa asset directories manually.
Locate the RoboCasa asset root with:

.. code:: bash

   python -c "import os, robocasa; print(os.path.join(robocasa.__path__[0], 'models', 'assets'))"

Then extract the files to the following paths:

- ``fixtures_lightwheel/*.zip`` → ``<robocasa_assets>/fixtures/``
- ``objects_lightwheel/*.zip`` → ``<robocasa_assets>/objects/lightwheel/``

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

An official RoboCasa365 Pi0 checkpoint is available at:
https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi0/pi0_robocasa_pretrain_human300

Download the checkpoint locally and point both rollout and actor model paths to
that directory:

.. code:: yaml

   rollout:
     model:
       model_path: "/path/to/pi0_robocasa_pretrain_human300"

   actor:
     model:
       model_path: "/path/to/pi0_robocasa_pretrain_human300"

Training
--------

The benchmark-aligned training recipe is:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh robocasa365_grpo_openpi

This config trains on:

- ``env.train.split=pretrain``
- ``env.train.task_soup=atomic_seen``
- ``env.train.task_mode=atomic``

Evaluation
----------

The lightweight evaluation config is ``robocasa365_eval_openpi``:

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh robocasa365_eval_openpi

This config evaluates on:

- ``env.eval.split=target``
- ``env.eval.task_soup=atomic_seen``
- ``env.eval.task_mode=atomic``

To evaluate a different benchmark slice, override the YAML fields directly:

.. code:: yaml

   env:
     eval:
       split: target
       task_soup: composite_unseen
       task_mode: composite

For example:

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh robocasa365_eval_openpi \
      env.eval.task_soup=composite_unseen \
      env.eval.task_mode=composite

Notes
-----

- The legacy :doc:`RoboCasa <robocasa>` page and ``robocasa`` env remain unchanged.
- ``robocasa365`` is a separate env type and folder on purpose, to keep old recipes stable.
- The first RLinf integration targets OpenPI / Pi0. Additional model-specific recipes can be added later on top of the same env.
