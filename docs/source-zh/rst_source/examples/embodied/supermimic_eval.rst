SuperMimic 评测
===============

本页单独评测一个训练好的 SuperMimic 检查点。环境安装和片段准备见 :doc:`supermimic`；所有命令都在 RLinf 仓库根目录执行，并且已经激活该环境。

运行评测
--------

评测走 ``evaluations/eval_embodied_agent.py``。``evaluations/run_eval.sh`` 看到 ``supermimic_`` 开头的配置名，就读取 ``evaluations/supermimic/<配置名>.yaml``。``runner.ckpt_path`` 是检查点里的 ``full_weights.pt``。

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=/data/runs/gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=/data/runs/gencoord/eval_2000

MaxCoord 加上 ``actor.model.representation=maxcoord``，并换上对应的检查点路径。把上面命令里的 ``global_step_2000`` 和输出目录换成 ``global_step_4000``，就是第 4000 次更新的评测。``env.eval.eval_tasks`` 只能是 ``[tracking]`` 或 ``[push_recovery]``。40 条 LAFAN1 片段由 ``env.eval.total_num_envs=40`` 一次放完；``max_steps_per_rollout_epoch=13656`` 覆盖最长片段（30 fps 下 8194 帧，重采样到 50 Hz）。

输出
----

``runner.eval_export.enabled`` 在评测配置里默认打开。结果写到 ``runner.eval_export.output_dir``，不设置时写到 ``<log_path>/<experiment_name>/eval_results/``。目录里有 ``summary.json``、``per_clip.csv`` 和这次评测的 ``config.yaml``。同样的指标以 ``eval/`` 为前缀写进日志。

动作跟踪
--------

``supermimic_lafan1_eval`` 把每条片段从第 0 帧播放一次，直接执行策略动作。片段 :math:`i` 的归一化回报是实际奖励之和除以 :math:`0.1 \times T_i`，其中 :math:`T_i` 是片段在 50 Hz 上的帧数，0.1 是单步奖励上限。提前失败时没播放的步仍留在分母里。``normalized_return`` 是 40 条片段的不加权平均。``fail_rate`` 是摔倒片段的比例，``clip_end_rate`` 是播到片段末帧的比例。``fail_step`` 和 ``fail_time_s`` 只对摔倒的片段取平均。

站立姿态推力恢复
----------------

``supermimic_push_recovery`` 加载跟踪检查点，以 mjlab 的屈膝站立姿态作为静止参考，在 1 秒后把根的水平速度改成 0.5、1、2 或 4 m/s，每种 4 次。跌倒沿用训练的失败判据。论文 §4.3 写的是 T-pose，这里没有实现那个姿态。MaxCoord 检查点同样要加 ``actor.model.representation=maxcoord``。
