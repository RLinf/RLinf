Logs & Results
==============

Evaluation writes terminal logs, rollout videos, and optional experiment-tracker events under the run directory created by ``evaluations/run_eval.sh``.

Output Paths
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Artifact
     - Path
   * - Default run log
     - ``logs/<timestamp>-<config_name>/eval_embodiment.log``
   * - Metrics summary table
     - ``<log_path>/metrics.log`` (same formatted table printed to stdout at the end of evaluation)
   * - ManiSkill OOD batch log
     - ``logs/eval/<EVAL_NAME>/<timestamp>-<env_id>-<obj_set>/run_ppo.log``
   * - Rollout videos
     - ``<log_path>/video/eval/`` (when ``env.eval.video_cfg.save_video: True``; ``log_path`` is set by ``runner.logger.log_path`` and ``run_eval.sh``)
   * - TensorBoard / tracker logs
     - ``<log_path>/`` (when ``tensorboard``, ``wandb``, or ``swanlab`` is enabled in ``runner.logger.logger_backends``)

Sample Terminal Output
----------------------

At the end of an evaluation run, RLinf prints a summary table and appends it to ``<log_path>/metrics.log``:

.. code-block:: text

   ╭────────────────────────────── Metric Table ──────────────────────────────╮
   │ Global Step:    1/1 │ Progress: ████████████████████ │ 100.0%            │
   │ Elapsed: 03:45      │ ETA: 00:00                     │ Step Time: 225.0s │
   ├────────────────────────────── Evaluation ────────────────────────────────┤
   │ episode_len=142.3   │ num_trajectories=500           │ success_once=0.96 │
   ╰──────────────────────────────────────────────────────────────────────────╯

Key Metrics
-----------

Evaluation prints and logs task-level summary metrics. Exact fields vary slightly by benchmark; see :doc:`../../reference/metrics` for full definitions:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Meaning
   * - ``eval/success_once``
     - Fraction of episodes that succeeded at least once (primary metric on LIBERO, BEHAVIOR-1K, RoboTwin, PolaRiS, and RoboLab).
   * - ``eval/success_at_end``
     - Fraction of episodes still in a success state at the final step.
   * - ``eval/success``
     - Episode success rate reported on benchmarks such as SIMPLE.
   * - ``eval/return``
     - Mean cumulative episode return.
   * - ``eval/num_trajectories``
     - Total completed trajectories aggregated into the summary.
   * - ``eval/episode_len``
     - Mean episode length in environment steps.
