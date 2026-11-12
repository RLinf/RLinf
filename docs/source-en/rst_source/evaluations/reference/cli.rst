CLI Reference
=============

Activate your virtual environment from the repository root, then use ``evaluations/run_eval.sh``.

Option 1: Explicit Benchmark
----------------------------

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh <benchmark> <config_name> [hydra_overrides...]

Examples:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval
   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval

Option 2: Auto-Infer Benchmark
------------------------------

When the config name starts with ``libero_``, ``robotwin_``, ``behavior_``, etc., the benchmark can be omitted:

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval

Option 3: Hydra Overrides on the Command Line
---------------------------------------------

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model/RLinf-Pi05-SFT \
     env.eval.total_num_envs=128 \
     env.eval.max_steps_per_rollout_epoch=960 \
     runner.ckpt_path=/path/to/checkpoint.pt

Supported Benchmark Prefixes
----------------------------

``run_eval.sh`` infers the benchmark from the config name prefix:

- ``libero_*`` → libero
- ``robotwin_*`` → robotwin
- ``behavior_*`` → behavior
- ``realworld_*`` → realworld
- ``maniskill_*`` → maniskill
- ``polaris_*`` → polaris
- ``supermimic_*`` → supermimic; see `SuperMimic Evaluation`_ below.

SuperMimic Evaluation
---------------------

SuperMimic configs run through ``eval_embodied_agent.py`` like the other
benchmarks. Pass the checkpoint's ``full_weights.pt`` as ``runner.ckpt_path``;
``runner.eval_export.output_dir`` chooses where the result tables go. For the
main training recipe, run:

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=/data/runs/gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt \
       env.eval.action_noise_std=0 \
       runner.eval_export.output_dir=/data/runs/gencoord/eval_2000

A MaxCoord checkpoint also needs ``actor.model.representation=maxcoord``.
The protocol is in the Evaluate section of
:doc:`SuperMimic <../../examples/embodied/supermimic>`.

Config Fallback
---------------

If ``evaluations/<benchmark>/<config>.yaml`` does not exist, the script falls back to ``examples/embodiment/config/`` with the same config name.

Full launch examples per benchmark:

- :doc:`../guides/libero`
- :doc:`../guides/robotwin`
- :doc:`../guides/behavior`
- :doc:`../guides/maniskill_ood`
- :doc:`../guides/realworld`
- :doc:`../guides/polaris`
- :doc:`SuperMimic <../../examples/embodied/supermimic>` (the Evaluate section)

Direct Python Invocation
------------------------

You can also call the main evaluation program directly:

.. code-block:: bash

   python evaluations/eval_embodied_agent.py \
     --config-path evaluations/libero/ \
     --config-name libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

``run_eval.sh`` wraps this with path setup, log directories, and environment variable exports.
