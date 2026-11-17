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

SuperMimic 配置和其他 benchmark 一样通过 ``eval_embodied_agent.py`` 运行。用 ``runner.ckpt_path`` 指定检查点中的 ``full_weights.pt``，``runner.eval_export.output_dir`` 指定结果表格的目录。评测 SuperMimic 指南里那次训练的第 6000 次更新（沿用其中的 ``RUN_DIR``）：

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval_6000 \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor:0,env:0,rollout:0}'

最后两行让评测只用 0 号 GPU，不加时会用上机器上的所有 GPU。MaxCoord 检查点再加上 ``actor.model.representation=maxcoord``。评测协议、输出文件和推力恢复见 :doc:`../../examples/embodied/supermimic_eval`。

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
