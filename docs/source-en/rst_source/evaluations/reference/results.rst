Logs and Results
================

Log Directories
---------------

- **Default:** ``logs/<timestamp>-<config_name>/eval_embodiment.log``
- **ManiSkill OOD batch mode:** ``logs/eval/<EVAL_NAME>/<timestamp>-<env_id>-<obj_set>/run_ppo.log``

Terminal Metrics
----------------

During evaluation, the terminal and logs report metrics such as:

- ``eval/success_once`` — Task success rate (at least one success per episode)
- ``eval/return`` — Cumulative return

Video Output
------------

When ``env.eval.video_cfg.save_video: True``, rollout videos are saved under:

.. code-block:: text

   <log_path>/video/eval/

``log_path`` is determined by ``runner.logger.log_path`` and the timestamp directory passed by ``run_eval.sh``.

TensorBoard
-----------

When logger backends such as ``tensorboard`` are enabled in the config, detailed curves and statistics are available under ``runner.logger.log_path``.
