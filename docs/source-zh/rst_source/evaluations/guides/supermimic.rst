SuperMimic 评测
===============

使用 ``evaluations/run_eval.sh`` 在 MuJoCo Warp 中评测训练好的 SuperMimic checkpoint，支持\ **LAFAN1 动作跟踪**\ 和\ **站立推力恢复**\ 两项协议，并导出 JSON 与 CSV 汇总结果。

相关文档：:doc:`SuperMimic 训练示例 <../../examples/embodied/supermimic>` · :doc:`G1 动作跟踪环境 <../../examples/embodied/g1_mimic>` · :doc:`动作数据准备 <../../examples/embodied/supermimic_motion>`。

**前置条件：** :doc:`安装 </rst_source/start/installation>` · 训练好的 SuperMimic checkpoint · 按 :doc:`../../examples/embodied/supermimic` 创建的 ``.venv-supermimic`` 虚拟环境。

评测配置
--------

``evaluations/supermimic/`` 目录下提供两份预置配置：

.. list-table::
   :header-rows: 1
   :widths: 26 18 12 16 14 14

   * - 配置文件
     - 任务
     - 环境数
     - 最大步数
     - 需要动作片段
     - 明细输出
   * - ``supermimic_lafan1_eval.yaml``
     - ``tracking``
     - ``40``
     - ``13656``
     - 是（LAFAN1 或自定义）
     - ``per_clip.csv``
   * - ``supermimic_push_recovery.yaml``
     - ``push_recovery``
     - ``16``
     - ``300``
     - 否（内置姿态）
     - ``per_trial.csv``

两份配置默认在 0 号 GPU 上以 ``rollout.model.representation: gencoord`` 运行，并关闭域随机化、重置噪声与动作噪声。训练保存的权重路径为 ``<log_path>/<experiment_name>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``。

安装与运行
----------

在仓库根目录激活 ``.venv-supermimic`` 后，按需选择以下评测协议。评测 MaxCoord checkpoint 时需追加 ``rollout.model.representation=maxcoord``\ （checkpoint 只能加载到训练时对应的状态表示中）。

.. tab-set::

   .. tab-item:: 动作跟踪

      动作跟踪评测为每条参考片段分配一个环境，从第 0 帧开始确定性执行 policy 动作，直到片段播完或机器人摔倒。

      .. code-block:: bash

         source .venv-supermimic/bin/activate
         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/lafan1_g1

         bash evaluations/run_eval.sh supermimic_lafan1_eval \
             runner.ckpt_path=./logs/<timestamp>-supermimic_lafan1_gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
             runner.eval_export.output_dir=./eval_results/supermimic_6000

      * **自定义或超过 40 条片段：** 设置 ``env.eval.num_clips=null``，同时保证 ``env.eval.total_num_envs * env.eval.rollout_epoch`` 不少于片段总数；如果有片段超过 ``13656`` 个控制步（50 Hz 下 273.12 秒），需同步调大 ``env.eval.max_steps_per_rollout_epoch``。
      * **切换 GPU：** 用 ``cluster.component_placement`` 指定卡号，例如 ``'~cluster.component_placement' '+cluster.component_placement={env:1,rollout:1}'``。

   .. tab-item:: 站立推力恢复

      站立推力恢复使用 mjlab 内置屈膝站立姿态测试抗扰能力（无需动作片段）。机器人先站稳 1 秒，随后在随机水平方向施加根节点速度冲量（``0.5``、``1.0``、``2.0``、``4.0`` m/s，每种速度 4 次共 16 次试验），受推后继续运行最多 5 秒。

      .. note::

         这项评测采用 mjlab 内置屈膝站立关键帧，而非 SuperMimic 论文第 4.3 节的 T-pose。

      .. code-block:: bash

         source .venv-supermimic/bin/activate
         export MUJOCO_GL=disabled

         bash evaluations/run_eval.sh supermimic_push_recovery \
             runner.ckpt_path=./logs/<timestamp>-supermimic_lafan1_gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
             runner.eval_export.output_dir=./eval_results/supermimic_push_6000

      .. dropdown:: 自定义推力协议

         可在命令行或 YAML 中覆盖 ``env.eval.push_recovery.*`` 调整测试参数：

         .. list-table::
            :header-rows: 1
            :widths: 36 22 42

            * - 配置项
              - 默认值
              - 说明
            * - ``env.eval.push_recovery.speeds``
              - ``[0.5, 1.0, 2.0, 4.0]``
              - 水平推动速度列表（m/s）
            * - ``env.eval.push_recovery.trials``
              - ``4``
              - 每种速度的试验次数（需要满足 ``total_num_envs * rollout_epoch >= len(speeds) * trials``）
            * - ``env.eval.push_recovery.push_mode``
              - ``add``
              - ``add`` 在根节点当前水平速度上叠加推力；``set`` 直接替换水平速度
            * - ``env.eval.push_recovery.direction``
              - ``random``
              - ``random`` 或指定水平方向向量 ``[x, y]``
            * - ``env.eval.push_recovery.frame``
              - ``world``
              - ``direction`` 所在坐标系：``world`` 或 ``heading``

输出文件与指标
--------------

评测结束后，日志（位于 ``logs/<timestamp>-<config>/``）末尾会打印 ``Wrote evaluation results to <output_dir>``，所有汇总指标也会以 ``eval/`` 为前缀写入日志。

输出目录结构
~~~~~~~~~~~~

未指定 ``runner.eval_export.output_dir`` 时，结果默认写入 ``<log_path>/<experiment_name>/eval_results/``：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 文件
     - 内容
   * - ``summary.json``
     - 全部片段或试验的汇总指标（推力恢复还包含分速度统计）
   * - ``per_clip.csv``
     - 仅动作跟踪：每条参考片段一行指标
   * - ``per_trial.csv``
     - 仅推力恢复：每次试验一行，含速度、方向、结果与恢复耗时
   * - ``config.yaml``
     - 本次评测实际生效的完整 Hydra 配置

动作跟踪指标
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 指标
     - 定义
   * - ``normalized_return``
     - 每条片段实得奖励除以全片满分（``0.1 × 片段总步数``）后的均值；摔倒后未播放的步数计入分母。
   * - ``fail_rate`` / ``clip_end_rate``
     - 摔倒片段占比 / 完整播到最后一帧的片段占比。
   * - ``fail_step`` / ``fail_time_s``
     - 摔倒发生的平均步数与时间（秒），仅统计摔倒片段。
   * - ``body_pos_err``
     - 实际播放步数上，14 个跟踪部位相对骨盆的平均位置误差（m）。

推力恢复指标
~~~~~~~~~~~~

同时报告总体结果与各推动速度分项结果。受推前 1 秒等待期内摔倒、超出关节限位或出现非有限值的试验记为 ``initialization_failed``；前三项比率以实际受推试验数为分母：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 定义
   * - ``recovery_rate``
     - 受推后重新站稳的试验占比：根节点水平速度 ≤ ``0.15 m/s``、骨盆倾斜误差 ≤ ``0.35``、躯干朝向误差 ≤ ``0.5 rad``，且连续保持 ``0.5 s``。
   * - ``fall_rate``
     - 受推后摔倒的试验占比。
   * - ``timeout_rate``
     - 受推后 5 秒内既未站稳也未摔倒的试验占比。
   * - ``initialization_failed_rate``
     - 全部试验中在受推前初始化失败的占比。
   * - ``recovery_time_mean_s`` / ``recovery_time_median_s``
     - 从受推到 0.5 秒站稳窗口结束的时间均值与中位数（秒，仅统计恢复成功的试验）。
