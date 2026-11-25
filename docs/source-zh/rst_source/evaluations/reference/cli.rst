CLI 参考
========

在仓库根目录激活虚拟环境后，使用 ``evaluations/run_eval.sh`` 启动评测。支持以下三种调用形式：

- **显式指定 benchmark：** ``bash evaluations/run_eval.sh <benchmark> <config_name> [hydra_overrides...]``
- **自动推断 benchmark：** ``bash evaluations/run_eval.sh <config_name> [hydra_overrides...]``
- **ManiSkill OOD 批量模式：** ``bash evaluations/run_eval.sh mani-ood [config_name]``\ （需先导出 ``EVAL_NAME``、``CKPT_PATH``、``TOTAL_NUM_ENVS`` 与 ``EVAL_ROLLOUT_EPOCH``）

显式指定 benchmark
------------------

依次传入 ``evaluations/`` 下的 benchmark 目录名与 YAML 配置名（不带 ``.yaml`` 后缀）。该形式加载 ``evaluations/<benchmark>/`` 下的评测配置，或回退加载 ``examples/embodiment/config/`` 中已定义 ``rollout.model`` 的评测就绪配置：

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval
   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval
   bash evaluations/run_eval.sh robolab robolab_pi05_droid_jointpos
   bash evaluations/run_eval.sh simple simple_psi0_eval

自动推断 benchmark
------------------

配置名以受支持的 benchmark 前缀开头时，可省略 benchmark 参数，直接按 ``<config_name> [hydra_overrides...]`` 调用：

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval

Hydra 参数覆盖
--------------

在配置名后追加 Hydra 键值覆盖项，无需修改 YAML 即可替换模型路径、权重或并行规模：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT \
     env.eval.total_num_envs=128 \
     env.eval.max_steps_per_rollout_epoch=960 \
     runner.ckpt_path=./checkpoints/full_weights.pt

支持的 benchmark 前缀
---------------------

``evaluations/run_eval.sh`` 根据配置名前缀自动推断 benchmark（下划线 ``_`` 与连字符 ``-`` 均匹配）：

.. list-table::
   :header-rows: 1
   :widths: 36 24 40

   * - 配置前缀
     - 推断出的 benchmark
     - 说明
   * - ``behavior_*`` / ``behavior-*``
     - ``behavior``
     - BEHAVIOR-1K
   * - ``calvin_*`` / ``calvin-*``
     - ``calvin``
     - CALVIN（配置回退）
   * - ``geniesim_*`` / ``geniesim-*``
     - ``geniesim``
     - GenieSim
   * - ``libero_*`` / ``libero-*``
     - ``libero``
     - 自动导出 ``ROBOT_PLATFORM`` 与 ``LIBERO_TYPE``
   * - ``maniskill_*`` / ``maniskill-*``
     - ``maniskill``
     - 另支持 ``mani-ood`` 批量模式
   * - ``metaworld_*`` / ``metaworld-*``
     - ``metaworld``
     - MetaWorld（配置回退）
   * - ``polaris_*`` / ``polaris-*``
     - ``polaris``
     - PolaRiS
   * - ``realworld_*`` / ``realworld-*``
     - ``realworld``
     - 跳过仿真环境初始化
   * - ``robocasa365_*`` / ``robocasa365-*``
     - ``robocasa365``
     - 默认导出 ``NUMBA_DISABLE_JIT=1``
   * - ``robocasa_*`` / ``robocasa-*``
     - ``robocasa``
     - RoboCasa（配置回退）
   * - ``robodojo_*`` / ``robodojo-*``
     - ``robodojo``
     - RoboDojo
   * - ``robolab_*`` / ``robolab-*``
     - ``robolab``
     - RoboLab
   * - ``robotwin_*`` / ``robotwin-*``
     - ``robotwin``
     - RoboTwin
   * - ``roboverse_*`` / ``roboverse-*``
     - ``roboverse``
     - RoboVerse（配置回退）
   * - ``simple_*`` / ``simple-*``
     - ``simple``
     - SIMPLE

Benchmark 环境变量
------------------

``run_eval.sh`` 会自动读取下列可选环境变量，并按需配置 ``PYTHONPATH``：

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Benchmark
     - 环境变量
     - 说明
   * - RoboTwin
     - ``ROBOTWIN_PATH=/path/to/RoboTwin``、``ROBOT_PLATFORM=ALOHA``
     - RoboTwin 仓库路径
   * - BEHAVIOR-1K
     - ``OMNIGIBSON_DATA_PATH``
     - 详见 :doc:`../../examples/embodied/behavior`
   * - DreamZero
     - ``DREAMZERO_PATH=/path/to/DreamZero``
     - DreamZero 仓库路径
   * - PolaRiS
     - ``POLARIS_DATA_PATH=/path/to/PolaRiS-Hub``
     - 下载的 PolaRiS-Hub 资产路径

配置回退
--------

若 ``evaluations/<benchmark>/<config>.yaml`` 不存在，``run_eval.sh`` 会回退查找 ``examples/embodiment/config/<config>.yaml``。配置回退仅适用于已定义 ``rollout.model``\ （含 ``model_type`` 与 ``num_action_chunks``）的评测就绪配置；只定义了 ``actor.model`` 的训练配置无法通过校验，须按 :doc:`configuration` 改写。显式指定 ``<benchmark> <config_name> [hydra_overrides...]``\ （只要 ``evaluations/<benchmark>/`` 目录存在）与自动推断 ``<config_name> [hydra_overrides...]``\ （使用受支持前缀）两种形式均支持该回退。

各 benchmark 的完整启动示例见对应指南：

- :doc:`../guides/behavior`
- :doc:`../guides/libero`
- :doc:`../guides/maniskill_ood`
- :doc:`../guides/polaris`
- :doc:`../guides/realworld`
- :doc:`../guides/robolab`
- :doc:`../guides/robotwin`
- :doc:`../guides/simple`
- :doc:`../guides/cosmos3_sglang`
- :doc:`../guides/dreamzero_sglang`

直接调用 Python
---------------

也可直接调用 ``evaluations/eval_embodied_agent.py``\ （``run_eval.sh`` 在此基础上封装了路径、日志目录与仿真环境变量导出）：

.. code-block:: bash

   python evaluations/eval_embodied_agent.py \
     --config-path evaluations/libero/ \
     --config-name libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT
