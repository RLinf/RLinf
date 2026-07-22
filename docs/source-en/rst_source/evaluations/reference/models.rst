Supported Models
================

Eval configs reference model presets from ``examples/embodiment/config/model/`` via ``defaults``, and override fields such as ``model_path`` under ``rollout.model``. Models with example configs in ``evaluations/`` today:

.. list-table::
   :header-rows: 1
   :widths: 20 18 42

   * - Model
     - ``model_type``
     - Example config
   * - π₀ / π₀.₅ (OpenPI)
     - ``openpi``
     - ``libero_spatial_openpi_pi05_eval``, ``libero_goal_openpi_eval``, ``robotwin_adjust_bottle_openpi_eval``, etc.
   * - OpenVLA-OFT
     - ``openvla_oft``
     - ``libero_10_openvlaoft_eval``, ``robotwin_place_empty_cup_openvlaoft_eval``, ``maniskill_ood_openvlaoft_eval``, etc.
   * - StarVLA
     - ``starvla``
     - ``libero_spatial_starvla_eval``
   * - DreamZero
     - ``dreamzero``
     - ``libero_spatial_dreamzero_eval``, ``realworld_pnp_eval_dreamzero``
   * - FastWAM
     - ``fastwam``
     - ``libero_spatial_fastwam_eval``
   * - LingBotVLA
     - ``lingbotvla``
     - ``robotwin_click_bell_lingbotvla_eval``, ``robotwin_place_shoe_lingbotvla_eval``

Model Path
----------

Set ``rollout.model.model_path`` in YAML or on the command line to a local model directory. For ``.pt`` checkpoints from RL training, also set ``runner.ckpt_path``.

Standalone Eval Scripts
-----------------------

Some models also provide standalone eval scripts under ``toolkits/standalone_eval_scripts/``. These follow specific evaluation protocols but do not support parallel environment acceleration. Prefer ``evaluations/run_eval.sh`` when possible.
