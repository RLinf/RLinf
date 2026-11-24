CLI
===

Activate your virtual environment from the repository root, then launch evaluations with ``evaluations/run_eval.sh``. Supported call forms:

- **Explicit benchmark:** ``bash evaluations/run_eval.sh <benchmark> <config_name> [hydra_overrides...]``
- **Auto-inferred benchmark:** ``bash evaluations/run_eval.sh <config_name> [hydra_overrides...]``
- **ManiSkill OOD batch mode:** ``bash evaluations/run_eval.sh mani-ood [config_name]`` (requires ``EVAL_NAME``, ``CKPT_PATH``, ``TOTAL_NUM_ENVS``, and ``EVAL_ROLLOUT_EPOCH``)

Explicit Benchmark
------------------

Pass the benchmark folder name under ``evaluations/`` followed by the YAML config name (without ``.yaml``). This form loads configs from ``evaluations/<benchmark>/``, or falls back to eval-ready configs in ``examples/embodiment/config/`` that define ``rollout.model``:

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval
   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval
   bash evaluations/run_eval.sh robolab robolab_pi05_droid_jointpos
   bash evaluations/run_eval.sh simple simple_psi0_eval

Auto-Infer Benchmark
--------------------

When the config name starts with a recognized benchmark prefix, omit the benchmark argument and run ``<config_name> [hydra_overrides...]`` directly:

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval

Hydra Overrides
---------------

Pass Hydra key-value overrides after the config name to change model paths, checkpoints, or parallel scale without editing YAML files:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT \
     env.eval.total_num_envs=128 \
     env.eval.max_steps_per_rollout_epoch=960 \
     runner.ckpt_path=./checkpoints/full_weights.pt

Supported Benchmark Prefixes
----------------------------

``evaluations/run_eval.sh`` infers the benchmark from the config prefix (both ``_`` and ``-`` separators match):

.. list-table::
   :header-rows: 1
   :widths: 36 24 40

   * - Config prefix
     - Inferred benchmark
     - Notes
   * - ``behavior_*`` / ``behavior-*``
     - ``behavior``
     - BEHAVIOR-1K
   * - ``calvin_*`` / ``calvin-*``
     - ``calvin``
     - CALVIN (via config fallback)
   * - ``geniesim_*`` / ``geniesim-*``
     - ``geniesim``
     - GenieSim
   * - ``libero_*`` / ``libero-*``
     - ``libero``
     - Sets ``ROBOT_PLATFORM`` and ``LIBERO_TYPE``
   * - ``maniskill_*`` / ``maniskill-*``
     - ``maniskill``
     - Also supports ``mani-ood`` batch mode
   * - ``metaworld_*`` / ``metaworld-*``
     - ``metaworld``
     - MetaWorld (via config fallback)
   * - ``polaris_*`` / ``polaris-*``
     - ``polaris``
     - PolaRiS
   * - ``realworld_*`` / ``realworld-*``
     - ``realworld``
     - Skips simulator environment setup
   * - ``robocasa365_*`` / ``robocasa365-*``
     - ``robocasa365``
     - Sets ``NUMBA_DISABLE_JIT=1`` by default
   * - ``robocasa_*`` / ``robocasa-*``
     - ``robocasa``
     - RoboCasa (via config fallback)
   * - ``robodojo_*`` / ``robodojo-*``
     - ``robodojo``
     - RoboDojo
   * - ``robolab_*`` / ``robolab-*``
     - ``robolab``
     - RoboLab
   * - ``robotwin_*`` / ``robotwin-*``
     - ``robotwin``
     - RoboTwin
   * - ``roboverse_*`` / ``roboverse-*``
     - ``roboverse``
     - RoboVerse (via config fallback)
   * - ``simple_*`` / ``simple-*``
     - ``simple``
     - SIMPLE

Benchmark Environment Variables
-------------------------------

``run_eval.sh`` reads these optional environment variables automatically and configures ``PYTHONPATH`` when needed:

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Benchmark
     - Environment Variable(s)
     - Notes
   * - RoboTwin
     - ``ROBOTWIN_PATH=/path/to/RoboTwin``, ``ROBOT_PLATFORM=ALOHA``
     - Path to cloned RoboTwin checkout
   * - BEHAVIOR-1K
     - ``OMNIGIBSON_DATA_PATH``
     - See :doc:`../../examples/embodied/behavior`
   * - DreamZero
     - ``DREAMZERO_PATH=/path/to/DreamZero``
     - Path to DreamZero checkout
   * - PolaRiS
     - ``POLARIS_DATA_PATH=/path/to/PolaRiS-Hub``
     - Path to downloaded PolaRiS-Hub assets

Config Fallback
---------------

If ``evaluations/<benchmark>/<config>.yaml`` does not exist, ``run_eval.sh`` falls back to ``examples/embodiment/config/<config>.yaml``. Fallback only works for eval-ready configs that define ``rollout.model`` (with ``model_type`` and ``num_action_chunks``); training-only configs that define only ``actor.model`` fail validation and must be adapted per :doc:`configuration`. Both ``<benchmark> <config_name> [hydra_overrides...]`` (when ``evaluations/<benchmark>/`` exists) and ``<config_name> [hydra_overrides...]`` (with a recognized prefix) support this fallback.

Full launch examples per benchmark:

- :doc:`../guides/behavior`
- :doc:`../guides/libero`
- :doc:`../guides/maniskill_ood`
- :doc:`../guides/polaris`
- :doc:`../guides/realworld`
- :doc:`../guides/robolab`
- :doc:`../guides/robotwin`
- :doc:`../guides/simple`
- :doc:`../guides/cosmos3_sglang`
- :doc:`../guides/dreamzero_sglang`

Direct Python Invocation
------------------------

You can also invoke ``evaluations/eval_embodied_agent.py`` directly (``run_eval.sh`` wraps this with path setup, log directories, and simulator env exports):

.. code-block:: bash

   python evaluations/eval_embodied_agent.py \
     --config-path evaluations/libero/ \
     --config-name libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT
