Evaluation Architecture & Layout
================================

RLinf provides a unified embodied evaluation entry point. It runs parallel rollouts in simulation or on real robots and reports task-level metrics such as success rate.

Directory Layout
----------------

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

Evaluation Architecture
-----------------------

The evaluation runner coordinates env workers and rollout workers over channels to run parallel rollouts under ``env.eval``. Metrics such as ``eval/success_once`` and ``eval/return`` print to the terminal and write to log files.

1. **Load config:** Hydra reads the YAML under ``evaluations/<benchmark>/`` and composes environment and model presets from ``examples/embodiment/config/`` via ``defaults``.
2. **Launch workers:** Env and rollout workers start on hardware according to ``cluster.component_placement``.
3. **Run parallel rollouts:** Env workers reset environments and send observations; rollout workers return action chunks until episodes finish.
4. **Aggregate metrics:** Task-level metrics such as ``eval/success_once`` and ``eval/return`` are computed and logged.

Next Steps
----------

- Install dependencies: :doc:`installation`
- Run your first evaluation: :doc:`quick_tour`
- Browse benchmark guides: :doc:`../guides/index`
