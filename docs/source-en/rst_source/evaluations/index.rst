Evaluation
==========

RLinf provides a unified embodied evaluation entry point. It runs parallel rollouts in simulation or on real robots and reports task-level metrics such as success rate. This module covers environment setup, a quick first evaluation, and end-to-end workflows per benchmark.

**Supported Benchmarks**

The table below lists benchmarks that have example configs under ``evaluations/`` and can be launched directly with ``run_eval.sh``.

.. list-table::
   :header-rows: 1
   :widths: 18 28 34

   * - Benchmark
     - Task / env preset
     - Example config
   * - RealWorld
     - ``realworld_franka_sft_env``, ``realworld_bin_relocation``
     - ``realworld/realworld_eval.yaml``, ``realworld/realworld_pnp_eval.yaml``, ``realworld/realworld_pnp_eval_dreamzero.yaml``
   * - BEHAVIOR-1K
     - ``behavior_r1pro``
     - ``behavior/behavior_openpi_pi05_eval.yaml``
   * - LIBERO
     - ``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``
     - ``libero/libero_spatial_openpi_pi05_eval.yaml``, etc.
   * - ManiSkill OOD
     - ``maniskill_ood_template`` (out-of-distribution generalization)
     - ``maniskill/maniskill_ood_openvlaoft_eval.yaml``
   * - PolaRiS
     - ``polaris_droid_tapeintocontainer``, ``polaris_droid_movelattecup``, etc.
     - ``polaris/polaris_tapeintocontainer_openpi_pi05_eval.yaml``, ``polaris/polaris_movelattecup_openpi_eval.yaml``
   * - RoboTwin
     - ``robotwin_place_empty_cup``, ``robotwin_adjust_bottle``, ``robotwin_place_shoe``, ``robotwin_click_bell``
     - ``robotwin/robotwin_place_empty_cup_openvlaoft_eval.yaml``, etc.

**LIBERO variants:** Standard LIBERO, LIBERO-PRO, and LIBERO-PLUS are supported via environment variables (see :doc:`guides/libero`).

**Config fallback:** If ``evaluations/<benchmark>/<config>.yaml`` does not exist, ``run_eval.sh`` falls back to ``examples/embodiment/config/`` with the same config name, so training configs can be reused for evaluation.

Get Started
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - What you get
   * - :doc:`Overview <get_started/overview>`
     - Evaluation architecture and the ``evaluations/`` layout.
   * - :doc:`Installation <get_started/installation>`
     - Environment setup and benchmark-specific variables.
   * - :doc:`Quick Tour <get_started/quick_tour>`
     - Run your first LIBERO Spatial evaluation in ~5 minutes.

Guides
------

End-to-end evaluation workflows per benchmark (setup → config → launch → results):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Benchmark
     - Workflow
   * - :doc:`RealWorld <guides/realworld>`
     - Franka real-robot evaluation and deployment.
   * - :doc:`BEHAVIOR-1K <guides/behavior>`
     - BEHAVIOR-1K evaluation.
   * - :doc:`LIBERO <guides/libero>`
     - LIBERO / LIBERO-PRO / LIBERO-PLUS.
   * - :doc:`ManiSkill OOD <guides/maniskill_ood>`
     - ManiSkill out-of-distribution evaluation.
   * - :doc:`PolaRiS <guides/polaris>`
     - PolaRiS tabletop manipulation.
   * - :doc:`RoboTwin <guides/robotwin>`
     - RoboTwin bimanual manipulation.

Reference
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - What you get
   * - :doc:`Configuration <reference/configuration>`
     - Hydra YAML structure and required fields.
   * - :doc:`CLI <reference/cli>`
     - ``run_eval.sh`` usage and Hydra overrides.
   * - :doc:`Models <reference/models>`
     - Supported models and example configs.
   * - :doc:`Results <reference/results>`
     - Logs, metrics, and video output.

Related Documentation
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Resource
     - Where
   * - Per-benchmark setup and training examples
     - :doc:`../examples/simulators_index`
   * - Installation details
     - :doc:`../start/installation`
   * - Math reasoning LLM evaluation (non-embodied)
     - `LLMEvalKit <https://github.com/RLinf/LLMEvalKit>`_
   * - Model-specific standalone eval scripts (outside the unified entry)
     - ``toolkits/standalone_eval_scripts/``

.. toctree::
   :hidden:
   :maxdepth: 2

   get_started/index
   guides/index
   reference/index
