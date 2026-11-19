CLI 参考
========

进入仓库根目录并激活虚拟环境后，使用 ``evaluations/run_eval.sh`` 启动评测。

方式一：显式指定 benchmark
--------------------------

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh <benchmark> <config_name> [hydra_overrides...]

示例：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval
   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval

方式二：自动推断 benchmark
--------------------------

配置名以 ``libero_``、``robotwin_``、``behavior_`` 等前缀开头时，可省略 benchmark：

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval

方式三：命令行覆盖 Hydra 参数
------------------------------

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model/RLinf-Pi05-SFT \
     env.eval.total_num_envs=128 \
     env.eval.max_steps_per_rollout_epoch=960 \
     runner.ckpt_path=/path/to/checkpoint.pt

支持的 benchmark 前缀
---------------------

``run_eval.sh`` 根据配置名前缀自动推断 benchmark：

- ``libero_*`` → libero
- ``robotwin_*`` → robotwin
- ``behavior_*`` → behavior
- ``realworld_*`` → realworld
- ``maniskill_*`` → maniskill
- ``polaris_*`` → polaris
- ``supermimic_*`` → supermimic，见下面的 `SuperMimic 评测`_。

SuperMimic 评测
---------------

下表的参数适用于两份 SuperMimic 配置：``supermimic_lafan1_eval``\ （动作跟踪）和 ``supermimic_push_recovery``\ （站立推力恢复）。完整命令见 :doc:`../../examples/embodied/supermimic_eval`。

.. list-table::
   :header-rows: 1
   :widths: 38 22 40

   * - 参数
     - 默认值
     - 含义
   * - ``runner.ckpt_path``
     - 无，必填
     - checkpoint 里的 ``actor/model_state_dict/full_weights.pt``
   * - ``runner.eval_export.output_dir``
     - ``<log_path>/<experiment_name>/eval_results``
     - 存放 ``summary.json``、``per_clip.csv`` 或 ``per_trial.csv`` 以及 ``config.yaml`` 的目录
   * - ``runner.logger.log_path``
     - ``../results/eval``
     - 评测日志目录
   * - ``actor.model.representation``
     - ``gencoord``
     - 用 MaxCoord 训练的 checkpoint 要设为 ``maxcoord``
   * - ``cluster.component_placement``
     - 所有 GPU
     - 传入 ``'~cluster.component_placement'`` 和 ``'+cluster.component_placement={actor:0,env:0,rollout:0}'``，只用 0 号 GPU
   * - ``env.eval.num_clips``
     - ``40``
     - 从 ``SUPERMIMIC_MOTION_PATH`` 加载的片段数，``null`` 加载全部
   * - ``env.eval.total_num_envs``
     - ``40``，推力恢复为 ``16``
     - 环境数，每条片段或每次试验占一个
   * - ``env.eval.rollout_epoch``
     - ``1``
     - 评测轮数，和 ``total_num_envs`` 的乘积要覆盖全部片段或试验
   * - ``env.eval.max_steps_per_rollout_epoch``
     - ``13656``，推力恢复为 ``300``
     - 每轮的控制步数，要覆盖最长的片段或试验
   * - ``env.eval.push_recovery.speeds``
     - ``[0.5, 1.0, 2.0, 4.0]``
     - 推动速度，单位 m/s
   * - ``env.eval.push_recovery.trials``
     - ``4``
     - 每种速度的试验次数
   * - ``env.eval.push_recovery.push_mode``
     - ``add``
     - ``add`` 在根节点水平速度上叠加推动速度，``set`` 直接替换
   * - ``env.eval.push_recovery.direction``
     - ``random``
     - ``random``，或水平方向 ``[x, y]``
   * - ``env.eval.push_recovery.frame``
     - ``world``
     - ``direction`` 所在的坐标系，``world`` 或 ``heading``

配置回退
--------

若 ``evaluations/<benchmark>/<config>.yaml`` 不存在，脚本会回退到 ``examples/embodiment/config/`` 下同名配置。这在复用训练 YAML 做评测时非常方便。

各 benchmark 的完整启动示例见对应指南：

- :doc:`../guides/libero`
- :doc:`../guides/robotwin`
- :doc:`../guides/behavior`
- :doc:`../guides/maniskill_ood`
- :doc:`../guides/realworld`
- :doc:`../guides/polaris`
- :doc:`SuperMimic <../../examples/embodied/supermimic_eval>`

直接调用 Python
---------------

也可直接调用评测主程序：

.. code-block:: bash

   python evaluations/eval_embodied_agent.py \
     --config-path evaluations/libero/ \
     --config-name libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

``run_eval.sh`` 在此基础上封装了路径设置、日志目录与环境变量导出。
