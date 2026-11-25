Evaluation
==========

RLinf provides a unified embodied evaluation entry point. It runs parallel rollouts in simulation or on real robots and reports task-level metrics such as success rate.

Get Started
-----------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Quick Tour (5 min)
      :link: get_started/quick_tour
      :link-type: doc

      Run your first LIBERO-Spatial evaluation end to end.

   .. grid-item-card:: Installation
      :link: ../start/installation
      :link-type: doc

      Install embodied environments and model dependencies.

   .. grid-item-card:: Benchmark Guides
      :link: guides/index
      :link-type: doc

      Step-by-step evaluation guides for every supported benchmark.

Supported Benchmarks
--------------------

The matrix below covers benchmarks with ready-to-run configs under ``evaluations/`` launched via ``evaluations/run_eval.sh``.

.. list-table::
   :header-rows: 1
   :widths: 18 28 34 20

   * - Benchmark
     - Tasks / Presets
     - Example Config
     - Training Guide
   * - :doc:`LIBERO <guides/libero>`
     - ``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10`` (plus Pro / Plus)
     - ``libero/libero_spatial_openpi_pi05_eval.yaml``, etc.
     - :doc:`LIBERO <../examples/embodied/libero>`
   * - :doc:`ManiSkill OOD <guides/maniskill_ood>`
     - ``maniskill_ood_template`` (13 OOD scenes + in-distribution)
     - ``maniskill/maniskill_ood_openvlaoft_eval.yaml``
     - :doc:`ManiSkill <../examples/embodied/maniskill>`
   * - :doc:`BEHAVIOR-1K <guides/behavior>`
     - ``behavior_r1pro`` (50 household tasks)
     - ``behavior/behavior_openpi_pi05_eval.yaml``
     - :doc:`BEHAVIOR-1K <../examples/embodied/behavior>`
   * - :doc:`RoboTwin <guides/robotwin>`
     - ``robotwin_place_empty_cup``, ``robotwin_adjust_bottle``, ``robotwin_place_shoe``, ``robotwin_click_bell``
     - ``robotwin/robotwin_place_empty_cup_openvlaoft_eval.yaml``, etc.
     - :doc:`RoboTwin <../examples/embodied/robotwin>`
   * - :doc:`PolaRiS <guides/polaris>`
     - ``polaris_droid_tapeintocontainer``, ``polaris_droid_movelattecup``, etc.
     - ``polaris/polaris_tapeintocontainer_openpi_pi05_eval.yaml``, ``polaris/polaris_movelattecup_openpi_eval.yaml``
     - :doc:`PolaRiS <../examples/embodied/polaris>`
   * - :doc:`RoboLab <guides/robolab>`
     - 120 DROID joint-position manipulation tasks
     - ``robolab/robolab_pi05_droid_jointpos.yaml``
     - :doc:`RoboLab <../examples/embodied/robolab>`
   * - RoboCasa365
     - ``robocasa365`` pretrain task slices
     - ``robocasa365/robocasa365_eval_openpi.yaml``
     - :doc:`RoboCasa365 <../examples/embodied/robocasa365>`
   * - DexJoCo
     - ``dexjoco_water_plant``
     - ``dexjoco/dexjoco_openpi_pi05_eval.yaml``, ``dexjoco/dexjoco_cnn_eval.yaml``
     - :doc:`DexJoCo <../examples/embodied/dexjoco>`
   * - RoboDojo
     - Isaac Sim manipulation tasks
     - ``robodojo/robodojo_put_bottles_into_dustbin_pi05_eval.yaml``
     - :doc:`RoboDojo <../examples/embodied/robodojo>`
   * - GenieSim
     - Server-backed humanoid manipulation tasks
     - ``geniesim/geniesim_pick_block_color_eval.yaml``
     - :doc:`GenieSim <../examples/embodied/geniesim>`
   * - :doc:`SIMPLE <guides/simple>`
     - 8 G1 whole-body Teleop tasks (``G1WholebodyOpenOvenTeleop-v0``, etc.)
     - ``simple/simple_psi0_eval.yaml``
     - :doc:`Psi0 <../examples/embodied/psi0>`
   * - :doc:`SuperMimic <guides/supermimic>`
     - LAFAN1 reference motion; GenCoord and MaxCoord; standing push recovery
     - ``supermimic/supermimic_lafan1_eval.yaml``, ``supermimic/supermimic_push_recovery.yaml``
     - :doc:`SuperMimic <../examples/embodied/supermimic>`
   * - :doc:`RealWorld <guides/realworld>`
     - ``realworld_franka_sft_env``, ``realworld_bin_relocation``, Dual Franka
     - ``realworld/realworld_pnp_eval.yaml``, ``realworld/realworld_eval.yaml``, ``realworld/realworld_pnp_eval_dreamzero.yaml``
     - :doc:`Franka <../examples/embodied/franka>`

SGLang Serving Guides
---------------------

Dedicated guides for world-action models served through RLinf's SGLang embodied backend on LIBERO:

.. list-table::
   :header-rows: 1
   :widths: 22 26 32 20

   * - Model Guide
     - Benchmark
     - Example Config
     - Training Guide
   * - :doc:`Cosmos3 (SGLang) <guides/cosmos3_sglang>`
     - LIBERO-Spatial (SGLang embodied server)
     - ``libero/libero_spatial_cosmos3_eval_sglang.yaml``
     - :doc:`Cosmos3 SFT <../examples/embodied/sft_cosmos3>`
   * - :doc:`DreamZero (SGLang) <guides/dreamzero_sglang>`
     - LIBERO-Spatial (SGLang embodied server)
     - ``libero/libero_spatial_dreamzero_eval_sglang.yaml``
     - :doc:`DreamZero SFT <../examples/embodied/sft_dreamzero>`

.. note::

   **Config fallback:** If ``evaluations/<benchmark>/<config>.yaml`` does not exist, ``run_eval.sh`` falls back to ``examples/embodiment/config/`` for eval-ready configs that define ``rollout.model`` (with ``model_type`` and ``num_action_chunks``). To evaluate a training-only config that defines only ``actor.model``, rebind ``@rollout.model`` per :doc:`reference/configuration`. Standard LIBERO, LIBERO-PRO, and LIBERO-PLUS are switched via environment variables (see :doc:`guides/libero`).

Evaluation Architecture
-----------------------

The evaluation runner coordinates env workers and rollout workers over channels to run parallel rollouts under ``env.eval``, printing and logging metrics such as ``eval/success_once`` and ``eval/return``.

1. **Load config:** Hydra reads the YAML under ``evaluations/<benchmark>/`` and composes environment and model presets from ``examples/embodiment/config/`` via ``defaults``.
2. **Launch workers:** Env and rollout workers start on hardware according to ``cluster.component_placement``.
3. **Run parallel rollouts:** Env workers reset environments and send observations; rollout workers return action chunks until episodes finish.
4. **Aggregate metrics:** Task-level metrics such as ``eval/success_once`` and ``eval/return`` are computed and logged.

.. dropdown:: Directory Layout

   All evaluation entry points and benchmark configs live under ``evaluations/`` at the repository root:

   .. list-table::
      :header-rows: 1
      :widths: 36 64

      * - Path
        - Contents
      * - ``evaluations/eval_embodied_agent.py``
        - Main embodied evaluation program.
      * - ``evaluations/run_eval.sh``
        - Unified launcher with benchmark inference, log setup, and simulator env setup.
      * - ``evaluations/behavior/``
        - BEHAVIOR-1K household evaluation configs.
      * - ``evaluations/geniesim/``
        - GenieSim server-backed evaluation configs.
      * - ``evaluations/libero/``
        - LIBERO evaluation configs (Spatial, Object, Goal, 10, Mem, RTC, ApxInf, SGLang) and custom init states.
      * - ``evaluations/maniskill/``
        - ManiSkill out-of-distribution (OOD) evaluation configs.
      * - ``evaluations/polaris/``
        - PolaRiS tabletop manipulation evaluation configs.
      * - ``evaluations/realworld/``
        - Franka single-arm, Dual Franka RTC, and Dual YAM real-robot evaluation configs.
      * - ``evaluations/robocasa365/``
        - RoboCasa365 pretrain task-slice evaluation configs.
      * - ``evaluations/robodojo/``
        - RoboDojo Isaac Sim evaluation configs.
      * - ``evaluations/robolab/``
        - RoboLab 120-task DROID joint-position evaluation configs.
      * - ``evaluations/robotwin/``
        - RoboTwin bimanual manipulation evaluation configs.
      * - ``evaluations/simple/``
        - SIMPLE whole-body humanoid evaluation configs for Psi0.
      * - ``evaluations/supermimic/``
        - Unitree G1 motion tracking and standing push recovery configs.

Reference
---------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Configuration
      :link: reference/configuration
      :link-type: doc

      Hydra YAML structure and required fields.

   .. grid-item-card:: CLI
      :link: reference/cli
      :link-type: doc

      ``run_eval.sh`` usage and Hydra overrides.

   .. grid-item-card:: Models
      :link: reference/models
      :link-type: doc

      Supported models and example configs.

   .. grid-item-card:: Results
      :link: reference/results
      :link-type: doc

      Logs, metrics, and video output.

For full evaluation CLI flags, configuration schema, supported models, and log formats, see :doc:`Eval Reference <reference/index>`. For non-embodied math reasoning LLM evaluation, use `LLMEvalKit <https://github.com/RLinf/LLMEvalKit>`_. For full environment setup options, see :doc:`../start/installation` (model-specific standalone scripts live under ``toolkits/standalone_eval_scripts/``).

.. toctree::
   :hidden:
   :maxdepth: 2

   Quick Tour <get_started/quick_tour>
   Benchmarks <guides/index>
   Eval Reference <reference/index>
