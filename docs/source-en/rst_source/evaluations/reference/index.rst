Reference
=========

This section documents the evaluation reference material: config structure, CLI usage, supported models, and how to read outputs.

.. list-table::
   :header-rows: 1

   * - Page
     - What you get
   * - :doc:`Configuration <configuration>`
     - Hydra YAML layout under ``evaluations/<benchmark>/``, including ``runner``, ``env``, and ``rollout`` fields required for ``embodied_eval``.
   * - :doc:`CLI <cli>`
     - How to launch evaluations with ``run_eval.sh``, pass Hydra overrides, and auto-infer the benchmark from config names.
   * - :doc:`Models <models>`
     - VLA models with example configs in ``evaluations/`` today (OpenPI, OpenVLA-OFT, StarVLA, DreamZero, LingBotVLA) and how to set ``model_path``.
   * - :doc:`Results <results>`
     - Where logs and rollout videos are written, terminal metrics such as ``eval/success_once``, and TensorBoard usage.

.. toctree::
   :hidden:
   :maxdepth: 1

   configuration
   cli
   models
   results
