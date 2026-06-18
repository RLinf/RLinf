Overview
========

RLinf provides a unified embodied evaluation entry point. It runs parallel rollouts in simulation or on real robots and reports task-level metrics such as success rate.

Directory Layout
----------------

All evaluation code and configs live under ``evaluations/`` at the repository root:

.. code-block:: text

   evaluations/
   ├── eval_embodied_agent.py   # Main evaluation program
   ├── run_eval.sh              # One-click launcher
   ├── libero/                  # LIBERO eval configs
   ├── robotwin/                # RoboTwin eval configs
   ├── behavior/                # BEHAVIOR-1K eval configs
   ├── maniskill/               # ManiSkill OOD eval configs
   ├── realworld/               # Real-robot eval configs
   └── polaris/                 # PolaRiS eval configs

Evaluation Architecture
-----------------------

Evaluation is driven by ``EmbodiedEvalRunner``: the environment worker and rollout worker communicate through Channels and run parallel evaluation under ``env.eval``. Metrics such as ``eval/success_once`` and ``eval/return`` are printed to the terminal and written to log files.

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
