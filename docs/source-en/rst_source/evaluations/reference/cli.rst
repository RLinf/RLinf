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

The arguments below apply to the two SuperMimic configs,
``supermimic_lafan1_eval`` (motion tracking) and ``supermimic_push_recovery``
(standing push recovery). The full commands are in
:doc:`../../examples/embodied/supermimic_eval`.

.. list-table::
   :header-rows: 1
   :widths: 38 22 40

   * - Argument
     - Default
     - Meaning
   * - ``runner.ckpt_path``
     - none, required
     - The ``actor/model_state_dict/full_weights.pt`` file of a checkpoint
   * - ``runner.eval_export.output_dir``
     - ``<log_path>/<experiment_name>/eval_results``
     - Folder for ``summary.json``, ``per_clip.csv`` or ``per_trial.csv``,
       and ``config.yaml``
   * - ``runner.logger.log_path``
     - ``../results/eval``
     - Evaluation logs. Calling Python directly uses this YAML default.
       ``evaluations/run_eval.sh`` sets ``logs/<timestamp>-<config>/``
   * - ``actor.model.representation``
     - ``gencoord``
     - ``maxcoord`` for a checkpoint trained with MaxCoord
   * - ``cluster.component_placement``
     - every GPU
     - Pass ``'~cluster.component_placement'`` and
       ``'+cluster.component_placement={actor:0,env:0,rollout:0}'`` to run on
       GPU 0
   * - ``env.eval.num_clips``
     - ``40``
     - Clips to load from ``SUPERMIMIC_MOTION_PATH``; ``null`` loads all
   * - ``env.eval.total_num_envs``
     - ``40``, push: ``16``
     - Environments; each clip or trial takes one
   * - ``env.eval.rollout_epoch``
     - ``1``
     - Evaluation rounds; ``total_num_envs`` times this must cover every
       clip or trial
   * - ``env.eval.max_steps_per_rollout_epoch``
     - ``13656``, push: ``300``
     - Control steps per round; must cover the longest clip or trial
   * - ``env.eval.push_recovery.speeds``
     - ``[0.5, 1.0, 2.0, 4.0]``
     - Push speeds in m/s
   * - ``env.eval.push_recovery.trials``
     - ``4``
     - Trials per speed
   * - ``env.eval.push_recovery.push_mode``
     - ``add``
     - ``add`` adds the push to the root's horizontal velocity; ``set``
       replaces it
   * - ``env.eval.push_recovery.direction``
     - ``random``
     - ``random``, or a horizontal direction ``[x, y]``
   * - ``env.eval.push_recovery.frame``
     - ``world``
     - ``world`` or ``heading``, the frame of ``direction``

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
- :doc:`SuperMimic <../../examples/embodied/supermimic_eval>`

Direct Python Invocation
------------------------

You can also call the main evaluation program directly:

.. code-block:: bash

   python evaluations/eval_embodied_agent.py \
     --config-path evaluations/libero/ \
     --config-name libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

``run_eval.sh`` wraps this with path setup, log directories, and environment variable exports.
