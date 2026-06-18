日志与结果
==========

日志目录
--------

- **默认：** ``logs/<时间戳>-<config_name>/eval_embodiment.log``
- **ManiSkill OOD 批量模式：** ``logs/eval/<EVAL_NAME>/<时间戳>-<env_id>-<obj_set>/run_ppo.log``

终端指标
--------

评测过程中，终端与日志会输出以下典型指标：

- ``eval/success_once`` — 任务成功率（episode 内至少成功一次）
- ``eval/return`` — 累计回报

视频输出
--------

若 ``env.eval.video_cfg.save_video: True``，rollout 视频保存在：

.. code-block:: text

   <log_path>/video/eval/

其中 ``log_path`` 由 ``runner.logger.log_path`` 与 ``run_eval.sh`` 传入的时间戳目录共同决定。

TensorBoard
-----------

若配置中启用了 ``tensorboard`` 等 logger backend，可在 ``runner.logger.log_path`` 下查看更详细的曲线与统计。
