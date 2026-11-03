LADDER 评测
===========

`LADDER <https://ladderrobotics.ai>`__ （`论文 <https://arxiv.org/abs/2610.18053>`__，`代码 <https://github.com/insait-institute/ladder>`__）是长程机器人操作评测基准，包含三大仿真套件共 **145 个任务**（每个任务含 1 至 10 个子目标）：

- ``simpler`` （**SimplerEnv-Long**）：基于 SAPIEN 3 / ManiSkill 3 的 54 个 WidowX 任务（每任务 24 个并行环境）。
- ``robot_arena`` （**RobotArena-∞-Long**）：基于 Genesis 的 47 个 WidowX 任务（每任务 8 个并行环境）。
- ``robolab`` （**RoboLab-Long**）：基于 Isaac Sim 6 / Isaac Lab 3 的 44 个 Franka DROID 任务（每任务 8 个并行环境）。

RLinf 同时输出完整任务成功率（``eval/success_once``）与子目标完成进度（``eval/subgoal_progress``）。

环境准备
--------

克隆官方 `insait-institute/ladder <https://github.com/insait-institute/ladder>`__ 仓库并设置 ``LADDER_ROOT``：

.. code-block:: bash

   git clone https://github.com/insait-institute/ladder.git ~/ladder
   export LADDER_ROOT=~/ladder

示例配置
--------

.. list-table::
   :header-rows: 1
   :widths: 42 28 30

   * - 配置文件
     - 套件 / 默认任务
     - 模型
   * - ``ladder/ladder_simpler_starvla_eval.yaml``
     - SimplerEnv-Long (``all1_plate``)
     - StarVLA (Qwen3-VL GR00T)
   * - ``ladder/ladder_robot_arena_starvla_eval.yaml``
     - RobotArena-∞-Long (``scene54_1goal``)
     - StarVLA (Qwen3-VL GR00T)
   * - ``ladder/ladder_robolab_openpi_pi05_eval.yaml``
     - RoboLab-Long (``comp_bin_1``)
     - OpenPI π₀.₅

完整流程
--------

**第 1 步：设置环境变量**

.. code-block:: bash

   source .venv/bin/activate
   export LADDER_ROOT=/path/to/ladder

**第 2 步：启动评测**

.. code-block:: bash

   bash evaluations/run_eval.sh ladder ladder_simpler_starvla_eval \
       rollout.model.model_path=/path/to/StarVLA-Qwen3VL-GR00T-Bridge-RT-1 \
       env.eval.init_params.task_name=all1_plate

或通过 Hydra 参数切换套件与任务：

.. code-block:: bash

   # RobotArena-∞-Long
   bash evaluations/run_eval.sh ladder ladder_robot_arena_starvla_eval \
       rollout.model.model_path=/path/to/StarVLA-Qwen3VL-GR00T-Bridge-RT-1 \
       env.eval.init_params.task_name=scene54_1goal

   # RoboLab-Long
   bash evaluations/run_eval.sh ladder ladder_robolab_openpi_pi05_eval \
       rollout.model.model_path=/path/to/openpi-pi05-droid \
       env.eval.init_params.task_name=comp_bin_1

如需使用子目标级别的指令（``oracle_hlc``），可传入 ``env.eval.init_params.prompt_mode=oracle_hlc``。

**第 3 步：查看结果**

终端输出 ``eval/success_once`` （完整任务成功率）与 ``eval/subgoal_progress`` （平均子目标完成度，范围 ``[0.0, 1.0]``）。
