日志与结果
==========

``evaluations/run_eval.sh`` 会为每次评测创建独立运行目录，保存终端日志、rollout 视频与可选实验记录。

输出路径
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 输出项
     - 路径
   * - 默认运行日志
     - ``logs/<时间戳>-<config_name>/eval_embodiment.log``
   * - 指标汇总表
     - ``<log_path>/metrics.log``\ （与评测结束时输出到终端的表格内容一致）
   * - ManiSkill OOD 批量日志
     - ``logs/eval/<EVAL_NAME>/<时间戳>-<env_id>-<obj_set>/run_ppo.log``
   * - Rollout 视频
     - ``<log_path>/video/eval/``\ （开启 ``env.eval.video_cfg.save_video: True`` 时生成；``log_path`` 由 ``runner.logger.log_path`` 与 ``run_eval.sh`` 共同决定）
   * - TensorBoard / 日志后端
     - ``<log_path>/``\ （在 ``runner.logger.logger_backends`` 中启用 ``tensorboard``、``wandb`` 或 ``swanlab`` 时写入）

终端输出示例
------------

评测结束时，RLinf 会在终端打印如下汇总表，并同步追加到 ``<log_path>/metrics.log``：

.. code-block:: text

   ╭────────────────────────────── Metric Table ──────────────────────────────╮
   │ Global Step:    1/1 │ Progress: ████████████████████ │ 100.0%            │
   │ Elapsed: 03:45      │ ETA: 00:00                     │ Step Time: 225.0s │
   ├────────────────────────────── Evaluation ────────────────────────────────┤
   │ episode_len=142.3   │ num_trajectories=500           │ success_once=0.96 │
   ╰──────────────────────────────────────────────────────────────────────────╯

核心指标
--------

评测会在终端与日志中输出任务级汇总指标，具体字段随 benchmark 略有差异，完整指标定义见 :doc:`../../reference/metrics`：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 指标
     - 含义
   * - ``eval/success_once``
     - 回合内至少成功一次的比例（LIBERO、BEHAVIOR-1K、RoboTwin、PolaRiS 与 RoboLab 的主指标）。
   * - ``eval/success_at_end``
     - 回合结束时刻仍处于成功状态的比例。
   * - ``eval/success``
     - SIMPLE 等 benchmark 输出的回合成功率。
   * - ``eval/return``
     - 平均回合累计回报。
   * - ``eval/num_trajectories``
     - 纳入本次汇总的总轨迹数。
   * - ``eval/episode_len``
     - 平均回合步数。
