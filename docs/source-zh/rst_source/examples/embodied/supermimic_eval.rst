SuperMimic 评测
===============

本页说明如何脱离训练流程，单独评测训练好的 SuperMimic checkpoint，包括 40 条 LAFAN1 片段上的动作跟踪和站立推力恢复，并解释评测写出的文件和指标。

安装
----

沿用 :doc:`supermimic` 里的环境、片段和那次训练。所有命令都在 RLinf 仓库根目录执行，并且已经激活这个环境；``RUN_DIR`` 是那里设置的运行目录。

运行
----

``evaluations/run_eval.sh`` 看到 ``supermimic_`` 开头的配置名，就读取 ``evaluations/supermimic/<配置名>.yaml``。``runner.ckpt_path`` 是 checkpoint 里的 ``actor/model_state_dict/full_weights.pt``；checkpoint 的其余部分只在续训时需要。MaxCoord 运行要改用 ``supermimic_lafan1_maxcoord`` 下的 checkpoint 目录，并加上 ``actor.model.representation=maxcoord``。

动作跟踪
~~~~~~~~

下面评测主指南里那次训练的第 6000 次更新：

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval_6000 \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor:0,env:0,rollout:0}'

最后两行让评测只用 0 号 GPU。不加时会沿用训练配置的 placement，用上机器上的所有 GPU。把两个路径里的 ``6000`` 换成 2000 或 4000，就评测对应的 checkpoint。每条片段从第 0 帧播放一遍，直接执行 policy 动作。40 条片段在 40 个环境里并行播放，评测约 7 分钟。

站立推力恢复
~~~~~~~~~~~~

机器人保持 mjlab 的屈膝站立姿态，1 秒后沿随机水平方向推一下：在根当前的水平速度上叠加 0.5、1、2 或 4 m/s，每种 4 次。跌倒沿用训练的失败判据。论文 §4.3 从 T-pose 出发，这个配置没有实现那个姿态。

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_push_recovery \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/push_6000 \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor:0,env:0,rollout:0}'

可视化与结果
------------

结果写到 ``runner.eval_export.output_dir``，不设置时写到 ``<log_path>/<experiment_name>/eval_results/``。目录里有 ``summary.json``、这次评测的 ``config.yaml`` 和一张表：动作跟踪写 ``per_clip.csv``，每条片段一行；推力恢复写 ``per_trial.csv``，每次试验一行。同样的指标以 ``eval/`` 为前缀写进日志。

``summary.json`` 里的动作跟踪指标：

* ``normalized_return``：每条片段实际拿到的奖励，除以它最多能拿到的奖励，即整条片段在 50 Hz 下每步 0.1。摔倒后没播放的步仍算在分母里。最终取 40 条片段的不加权平均。
* ``fail_rate``：机器人摔倒的片段比例。``clip_end_rate`` 是播到最后一帧的比例。
* ``fail_step`` 和 ``fail_time_s``：摔倒发生的时刻，只对摔倒的片段取平均。
* ``body_pos_err``：跟踪部位相对骨盆的位置误差，单位米，在每条片段实际播放的步上取平均。

主指南那次训练的已报告数字见 :doc:`supermimic` 的“可视化与结果”。
