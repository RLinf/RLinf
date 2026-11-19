SuperMimic 评测
===============

本页用两种方式评测训练好的 SuperMimic checkpoint：跟踪 40 条 LAFAN1 片段，以及站立推力恢复。本页还说明每种评测写出的文件和指标。

安装
----

沿用 :doc:`supermimic` 里的环境和片段，并先在终端里执行那一页的设置代码。下面的命令用到设置代码里的 ``RUN_DIR``，评测那次训练写出的 checkpoint。

运行
----

``evaluations/run_eval.sh`` 从 ``evaluations/supermimic/`` 读取配置。``runner.ckpt_path`` 指向 checkpoint 里的 ``full_weights.pt``，``runner.eval_export.output_dir`` 指定结果目录。每条命令的最后两行让评测只用 0 号 GPU，不加时会占用机器上的所有 GPU。评测 MaxCoord checkpoint 时，改用 ``supermimic_lafan1_maxcoord`` 下的目录，并加上 ``actor.model.representation=maxcoord``。全部参数见 :doc:`CLI 参考 <../../evaluations/reference/cli>`。

动作跟踪
~~~~~~~~

40 条片段各占一个环境，从第一帧播放一遍，policy 动作不加噪声。评测第 6000 次更新：

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval_6000 \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor:0,env:0,rollout:0}'

要评测第 2000 或 4000 次更新，把两个路径里的 ``6000`` 换掉即可。

站立推力恢复
~~~~~~~~~~~~

机器人保持 mjlab 的屈膝站立姿态。1 秒后沿随机水平方向推一下，在根节点（骨盆）的水平速度上叠加 0.5、1、2 或 4 m/s。每种速度各试 4 次，每次试验在受推后最多持续 5 秒。这是站立版本，论文 §4.3 从 T-pose 出发的设置不在这个配置里。

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_push_recovery \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/push_6000 \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor:0,env:0,rollout:0}'

可视化与结果
------------

评测结束时，日志最后一行是 ``Wrote SuperMimic evaluation results to`` 加输出目录。目录里有：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 文件
     - 内容
   * - ``summary.json``
     - 全部片段或全部试验的汇总指标
   * - ``per_clip.csv``
     - 动作跟踪：每条片段一行
   * - ``per_trial.csv``
     - 推力恢复：每次试验一行，含推动速度、方向和结果
   * - ``config.yaml``
     - 评测实际使用的完整配置

同样的指标也以 ``eval/`` 为前缀写进日志。

动作跟踪指标：

* ``normalized_return``：每条片段实际拿到的奖励除以满分，满分是片段的控制步数乘以 0.1。摔倒后没播完的步也算进分母。最终对所有片段取平均。
* ``fail_rate``：机器人摔倒的片段比例。``clip_end_rate`` 是播到最后一帧的比例。
* ``fail_step`` 和 ``fail_time_s``：摔倒发生的时刻，只对摔倒的片段取平均。
* ``body_pos_err``：跟踪部位相对骨盆的位置误差，单位米，在每条片段实际播放的步上取平均。

推力恢复指标，先对全部试验统计，再按每种速度各统计一次：

* ``recovery_rate``：机器人重新站稳的试验比例。站稳指根节点的水平速度低于 0.15 m/s、身体保持直立，并持续 0.5 秒。
* ``fall_rate``：机器人摔倒的试验比例。
* ``timeout_rate``：5 秒内既没站稳也没摔倒的试验比例。
* ``recovery_time_mean_s`` 和 ``recovery_time_median_s``：只统计恢复成功的试验，从受推到 0.5 秒站稳区间结束的时间。
