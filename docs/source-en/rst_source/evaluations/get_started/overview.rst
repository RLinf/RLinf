Overview
========

RLinf provides a unified embodied evaluation entry point. It runs parallel rollouts in simulation or on real robots and reports task-level metrics such as success rate.

Directory Layout
----------------

All evaluation code and configs live under ``evaluations/`` at the repository root:

- ``evaluations/eval_embodied_agent.py``: Main evaluation program
- ``evaluations/run_eval.sh``: One-click launcher
- ``evaluations/libero/``: LIBERO eval configs
- ``evaluations/robotwin/``: RoboTwin eval configs
- ``evaluations/behavior/``: BEHAVIOR-1K eval configs
- ``evaluations/maniskill/``: ManiSkill OOD eval configs
- ``evaluations/realworld/``: Real-robot eval configs
- ``evaluations/polaris/``: PolaRiS eval configs

Evaluation Architecture
-----------------------

The evaluation runner coordinates environment workers and rollout workers over channels to run parallel evaluation under ``env.eval``. Metrics such as ``eval/success_once`` and ``eval/return`` are printed to the terminal and written to log files.

Typical data flow:

1. **Config loading** — Hydra reads YAML from ``evaluations/<benchmark>/`` and references environment and model presets from ``examples/embodiment/config/`` via ``defaults``.
2. **Worker launch** — Env and Rollout workers start on GPUs according to ``cluster.component_placement``.
3. **Parallel rollout** — Env workers reset environments and return observations; Rollout workers generate actions from the model; loop until episodes finish.
4. **Metric aggregation** — Task-level metrics such as ``success_once`` and ``return`` are computed and logged.

Next Steps
----------

- Install the environment: :doc:`installation`
- Quick first run: :doc:`quick_tour`
- Per-benchmark deep dives: :doc:`../guides/index`
