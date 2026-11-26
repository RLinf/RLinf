Supported Models
================

Eval configs compose model presets from ``examples/embodiment/config/model/`` via ``defaults`` and override fields such as ``rollout.model.model_path``. Models with ready-to-run configs under ``evaluations/``:

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Model
     - ``model_type``
     - Example configs
   * - π₀ / π₀.₅ (OpenPI)
     - ``openpi``
     - ``libero_spatial_openpi_pi05_eval``, ``libero_10_apxinf_pi05_eval`` (ApxInf), ``behavior_openpi_pi05_eval``, ``robolab_pi05_droid_jointpos``, ``robotwin_adjust_bottle_openpi_eval``, ``realworld_pnp_eval``, etc.
   * - OpenVLA-OFT
     - ``openvla_oft``
     - ``libero_10_openvlaoft_eval``, ``robotwin_place_empty_cup_openvlaoft_eval``, ``maniskill_ood_openvlaoft_eval``, etc.
   * - StarVLA
     - ``starvla``
     - ``libero_spatial_starvla_eval``
   * - DreamZero
     - ``dreamzero``
     - ``libero_spatial_dreamzero_eval``, ``libero_spatial_dreamzero_eval_sglang``, ``realworld_pnp_eval_dreamzero``
   * - Cosmos3
     - ``cosmos3``
     - ``libero_spatial_cosmos3_eval_sglang``
   * - Psi0
     - ``psi0``
     - ``simple_psi0_eval``
   * - Evo-1
     - ``evo1``
     - ``libero_spatial_evo1_eval``
   * - FastWAM
     - ``fastwam``
     - ``libero_spatial_fastwam_eval``, ``libero_object_fastwam_eval``, ``libero_goal_fastwam_eval``, ``libero_10_fastwam_eval``
   * - MolmoAct2
     - ``molmoact2``
     - ``libero_spatial_molmoact2_eval``, ``libero_object_molmoact2_eval``, ``libero_goal_molmoact2_eval``, ``libero_10_molmoact2_eval``
   * - LingbotVLA 1.0
     - ``lingbotvla``
     - ``robotwin_click_bell_lingbotvla_eval``, ``robotwin_place_shoe_lingbotvla_eval``
   * - LingbotVLA 2.0
     - ``lingbotvlav2``
     - ``robotwin_click_bell_lingbotvlav2_eval``
   * - π₀-FAST (LeRobot)
     - ``pi0_fast``
     - ``libero_10_pi0_fast_eval``

Rollout Backends
----------------

Evaluation supports four rollout backends selected by ``rollout.rollout_backend``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Backend
     - Usage
   * - ``huggingface``
     - Default in-process PyTorch rollout used by most embodied evaluation configs.
   * - ``sglang``
     - Standalone SGLang embodied server group for world-action models such as Cosmos3 (:doc:`../guides/cosmos3_sglang`) and DreamZero (:doc:`../guides/dreamzero_sglang`).
   * - ``apxinf``
     - In-process Rust/CUDA inference backend for low-latency OpenPI π₀.₅ evaluation (``libero_10_apxinf_pi05_eval``; see :doc:`../../guides/apxinf`).
   * - ``phyai``
     - Accelerated PhyAI rollout engine for OpenPI π₀.₅ on LIBERO (install with ``--engine phyai``; see :doc:`../../guides/rollout-engines/phyai`).

Model Path
----------

Set ``rollout.model.model_path`` in YAML or on the command line to either a local model directory or a Hugging Face ``owner/repo`` ID (downloaded into your local Hugging Face cache automatically on first run). For ``.pt`` checkpoints from RL or SFT training, also set ``runner.ckpt_path``.

Standalone Eval Scripts
-----------------------

Some models also ship protocol-specific standalone scripts under ``toolkits/standalone_eval_scripts/``. Those scripts do not use RLinf's parallel environment workers; prefer ``evaluations/run_eval.sh`` whenever available.
