LADDER Evaluation
=================

`LADDER <https://ladderrobotics.ai>`__ (`paper <https://arxiv.org/abs/2610.18053>`__, `code <https://github.com/insait-institute/ladder>`__) is a long-horizon robot manipulation benchmark spanning **145 tasks** (1 to 10 sequential subgoals) across three simulation suites:

- ``simpler`` (**SimplerEnv-Long**): 54 WidowX tasks on SAPIEN 3 / ManiSkill 3 (24 parallel environments/task).
- ``robot_arena`` (**RobotArena-∞-Long**): 47 WidowX tasks on Genesis (8 parallel environments/task).
- ``robolab`` (**RoboLab-Long**): 44 Franka DROID tasks on Isaac Sim 6 / Isaac Lab 3 (8 parallel environments/task).

RLinf evaluates both binary task completion (``eval/success_once``) and subgoal completion progress (``eval/subgoal_progress``).

Environment Setup
-----------------

Clone the official `insait-institute/ladder <https://github.com/insait-institute/ladder>`__ repository and point ``LADDER_ROOT`` to it:

.. code-block:: bash

   git clone https://github.com/insait-institute/ladder.git ~/ladder
   export LADDER_ROOT=~/ladder

Example Configs
---------------

.. list-table::
   :header-rows: 1
   :widths: 42 28 30

   * - Config file
     - Suite / Task preset
     - Model
   * - ``ladder/ladder_simpler_starvla_eval.yaml``
     - SimplerEnv-Long (``all1_plate``)
     - StarVLA (Qwen3-VL GR00T)
   * - ``ladder/ladder_robot_arena_starvla_eval.yaml``
     - RobotArena-∞-Long (``scene54_1goal``)
     - StarVLA (Qwen3-VL GR00T)
   * - ``ladder/ladder_robolab_openpi_pi05_eval.yaml``
     - RoboLab-Long (``comp_bin_1``)
     - OpenPI π₀.₅

End-to-End Workflow
-------------------

**Step 1: Set environment variables**

.. code-block:: bash

   source .venv/bin/activate
   export LADDER_ROOT=/path/to/ladder

**Step 2: Launch evaluation**

.. code-block:: bash

   bash evaluations/run_eval.sh ladder ladder_simpler_starvla_eval \
       rollout.model.model_path=/path/to/StarVLA-Qwen3VL-GR00T-Bridge-RT-1 \
       env.eval.init_params.task_name=all1_plate

Or switch suites / tasks via Hydra overrides:

.. code-block:: bash

   # RobotArena-∞-Long
   bash evaluations/run_eval.sh ladder ladder_robot_arena_starvla_eval \
       rollout.model.model_path=/path/to/StarVLA-Qwen3VL-GR00T-Bridge-RT-1 \
       env.eval.init_params.task_name=scene54_1goal

   # RoboLab-Long
   bash evaluations/run_eval.sh ladder ladder_robolab_openpi_pi05_eval \
       rollout.model.model_path=/path/to/openpi-pi05-droid \
       env.eval.init_params.task_name=comp_bin_1

To evaluate with oracle subgoal instructions instead of the high-level goal instruction, pass ``env.eval.init_params.prompt_mode=oracle_hlc``.

**Step 3: Check results**

The terminal prints ``eval/success_once`` (full task success rate) and ``eval/subgoal_progress`` (mean fraction of subgoals completed in ``[0.0, 1.0]``).
