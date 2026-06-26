Benchmark Guides
================

This section provides end-to-end evaluation workflows organized by benchmark. Each guide covers environment setup, example configs, step-by-step commands, and advanced usage.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Benchmark
     - Description
     - Guide
   * - RealWorld
     - Franka real-robot evaluation and deployment
     - :doc:`realworld`
   * - SO101
     - Evaluate post-SFT π₀ on the SO101 6-DOF arm
     - :doc:`so101_eval`
   * - BEHAVIOR-1K
     - Large-scale household scene simulation
     - :doc:`behavior`
   * - LIBERO
     - Robotic manipulation benchmark with Spatial / Object / Goal / Long / 90 suites
     - :doc:`libero`
   * - ManiSkill OOD
     - ManiSkill out-of-distribution generalization evaluation
     - :doc:`maniskill_ood`
   * - PolaRiS
     - Tabletop manipulation simulation platform
     - :doc:`polaris`
   * - RoboTwin
     - Bimanual manipulation simulation with multiple tasks
     - :doc:`robotwin`

.. note::

   Benchmarks such as IsaacLab and MetaWorld do not yet have example configs under ``evaluations/``. For evaluation, refer to the corresponding training docs in :doc:`../../examples/simulators_index` and use the config fallback mechanism with YAMLs under ``examples/embodiment/config/``.

.. toctree::
   :hidden:
   :maxdepth: 1

   realworld
   so101_eval
   behavior
   libero
   maniskill_ood
   polaris
   robotwin
