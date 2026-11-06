Training Configuration
======================

RLinf recipes are Hydra YAML configs. Use this page as the shared reference for
configuration ownership; example pages should link here instead of repeating long key
tables.

Where Configs Live
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Workload
     - Config location
     - Launcher
   * - Embodied RL
     - ``examples/embodiment/config/*.yaml``
     - ``bash examples/embodiment/run_embodiment.sh <config_name>``
   * - Reasoning RL
     - ``examples/reasoning/config/**.yaml``
     - ``bash examples/reasoning/run_main_grpo_math.sh <config_name>``
   * - Agent workflows
     - ``examples/agent/**/config/*.yaml``
     - The launcher under the matching ``examples/agent/<recipe>/`` directory
   * - SFT
     - ``examples/sft/**`` and recipe-specific config directories
     - The recipe's ``run_*_sft.sh`` launcher
   * - Evaluation
     - ``evaluations/<benchmark>/*.yaml``
     - ``bash evaluations/run_eval.sh <benchmark> <config_name>``

Common Sections
---------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Section
     - Purpose
   * - ``cluster``
     - Node count, node groups, and component placement for actor, rollout, env, reward, or agent workers.
   * - ``actor``
     - Training backend, model path, optimizer, batch sizes, offload, checkpointing, and loss settings.
   * - ``rollout``
     - Inference engine, sampling parameters, model path, and rollout batch sizing.
   * - ``env``
     - Training/evaluation environment type, task selection, assets, video settings, and episode controls.
   * - ``runner``
     - Task type, logging, checkpoint cadence, validation cadence, and resume behavior.
   * - ``algorithm``
     - Advantage and loss selections such as PPO, GRPO, SAC, IQL, or DAgger-specific settings.
   * - ``data``
     - Dataset paths, prompt/answer fields, preprocessing, train/validation splits, and SFT data options.

Edit a Recipe
-------------

1. Start from the recipe's named config in ``examples/`` or ``evaluations/``.
2. Set local paths such as ``rollout.model.model_path``, ``actor.model.model_path``,
   dataset paths, and environment asset paths.
3. Keep hardware-specific placement in ``cluster``. For multi-node runs, set
   ``cluster.num_nodes`` and start Ray on every node before launching the recipe.
4. Put logs and checkpoints under ``runner.logger.log_path`` so TensorBoard, videos, and
   checkpoints stay together.

Further Reading
---------------

- :doc:`Basic configuration <../guides/basic_config>`
- :doc:`Embodied configuration <../guides/embodiment_config>`
- :doc:`Agentic configuration <../guides/agentic_config>`
- :doc:`Placement <../concepts/placement>`
- :doc:`Execution modes <../concepts/execution_modes>`
- :doc:`Training metrics <metrics>`
- :doc:`Evaluation configuration <../evaluations/reference/configuration>`
