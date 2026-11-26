SuperMimic Evaluation
=====================

Evaluate a trained SuperMimic checkpoint on the Unitree G1 in MuJoCo Warp across two protocols: **LAFAN1 motion tracking** and **standing push recovery**. Both run through ``evaluations/run_eval.sh`` and export JSON and CSV summaries.

Related docs: :doc:`SuperMimic training recipe <../../examples/embodied/supermimic>` · :doc:`G1 Motion Tracking Environment <../../examples/embodied/g1_mimic>` · :doc:`Motion data preparation <../../examples/embodied/supermimic_motion>`.

**Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a trained SuperMimic checkpoint · the ``.venv-supermimic`` environment from :doc:`../../examples/embodied/supermimic`.

Evaluation Configs
------------------

Two preset configs live under ``evaluations/supermimic/``:

.. list-table::
   :header-rows: 1
   :widths: 26 18 12 16 14 14

   * - Config
     - Task
     - Envs
     - Max Steps
     - Clips Needed
     - Per-Item Export
   * - ``supermimic_lafan1_eval.yaml``
     - ``tracking``
     - ``40``
     - ``13656``
     - Yes (LAFAN1 or custom)
     - ``per_clip.csv``
   * - ``supermimic_push_recovery.yaml``
     - ``push_recovery``
     - ``16``
     - ``300``
     - No (built-in pose)
     - ``per_trial.csv``

Both configs default to ``rollout.model.representation: gencoord`` on GPU 0 and disable domain randomization, reset noise, and action noise. Training writes checkpoint weights to ``<log_path>/<experiment_name>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``.

Setup and Run
-------------

Activate ``.venv-supermimic`` from the repository root and choose an evaluation protocol below. For MaxCoord checkpoints, append ``rollout.model.representation=maxcoord`` (checkpoints load only into the representation they were trained with).

.. tab-set::

   .. tab-item:: Motion Tracking

      Motion-tracking evaluation assigns one environment per reference clip, starts each clip at frame 0, and runs deterministically until the clip finishes or the robot falls.

      .. code-block:: bash

         source .venv-supermimic/bin/activate
         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/lafan1_g1

         bash evaluations/run_eval.sh supermimic_lafan1_eval \
             runner.ckpt_path=./logs/<timestamp>-supermimic_lafan1_gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
             runner.eval_export.output_dir=./eval_results/supermimic_6000

      * **Custom or >40 clips:** Set ``env.eval.num_clips=null`` and ensure ``env.eval.total_num_envs * env.eval.rollout_epoch`` is at least the total clip count. If any clip exceeds ``13656`` control steps (273.12 s at 50 Hz), raise ``env.eval.max_steps_per_rollout_epoch`` as well.
      * **Select another GPU:** Override ``cluster.component_placement``, for example ``'~cluster.component_placement' '+cluster.component_placement={env:1,rollout:1}'``.

   .. tab-item:: Standing Push Recovery

      Standing push recovery tests disturbance rejection from mjlab's built-in knees-bent standing pose (no motion clips required). The robot holds the pose for 1 second, receives a root velocity impulse in a random horizontal direction at one of four speeds (``0.5``, ``1.0``, ``2.0``, or ``4.0`` m/s, 4 trials per speed = 16 trials), and runs for up to 5 seconds post-push.

      .. note::

         This protocol uses mjlab's knees-bent standing keyframe rather than the T-pose mentioned in Section 4.3 of the SuperMimic paper.

      .. code-block:: bash

         source .venv-supermimic/bin/activate
         export MUJOCO_GL=disabled

         bash evaluations/run_eval.sh supermimic_push_recovery \
             runner.ckpt_path=./logs/<timestamp>-supermimic_lafan1_gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
             runner.eval_export.output_dir=./eval_results/supermimic_push_6000

      .. dropdown:: Customize Push Protocol

         Override ``env.eval.push_recovery.*`` on the command line or in YAML to change the test grid:

         .. list-table::
            :header-rows: 1
            :widths: 36 22 42

            * - Config Key
              - Default
              - Meaning
            * - ``env.eval.push_recovery.speeds``
              - ``[0.5, 1.0, 2.0, 4.0]``
              - Horizontal push speeds (m/s)
            * - ``env.eval.push_recovery.trials``
              - ``4``
              - Number of trials per speed (ensure ``total_num_envs * rollout_epoch >= len(speeds) * trials``)
            * - ``env.eval.push_recovery.push_mode``
              - ``add``
              - ``add`` adds the impulse to current root horizontal velocity; ``set`` replaces it
            * - ``env.eval.push_recovery.direction``
              - ``random``
              - ``random`` or a fixed horizontal vector ``[x, y]``
            * - ``env.eval.push_recovery.frame``
              - ``world``
              - Reference frame for ``direction``: ``world`` or ``heading``

Output Files and Metrics
------------------------

When evaluation finishes, the log under ``logs/<timestamp>-<config>/`` prints ``Wrote evaluation results to <output_dir>`` and logs the same summary metrics with an ``eval/`` prefix.

Output Directory
~~~~~~~~~~~~~~~~

If ``runner.eval_export.output_dir`` is omitted, files go to ``<log_path>/<experiment_name>/eval_results/``:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - File
     - Contents
   * - ``summary.json``
     - Aggregate metrics over all clips or trials (and per-speed breakdown for push recovery)
   * - ``per_clip.csv``
     - Motion tracking only: one row of metrics per reference clip
   * - ``per_trial.csv``
     - Push recovery only: one row per trial with speed, push direction, outcome, and recovery time
   * - ``config.yaml``
     - Resolved Hydra config used for the evaluation run

Motion-Tracking Metrics
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Definition
   * - ``normalized_return``
     - Per-clip reward divided by the full-clip maximum (``0.1 × clip_steps``), averaged across clips. Steps missed after a fall reduce the score.
   * - ``fail_rate`` / ``clip_end_rate``
     - Fraction of clips that ended in a fall vs. reached the final frame.
   * - ``fail_step`` / ``fail_time_s``
     - Mean failure step and time (s), averaged over failed clips only.
   * - ``body_pos_err``
     - Mean pelvis-relative position error across the 14 tracked bodies (m) over played steps.

Push-Recovery Metrics
~~~~~~~~~~~~~~~~~~~~~

Reported overall and per push speed. Trials where the robot falls, violates joint limits, or produces non-finite states during the 1 s pre-push wait count as ``initialization_failed``. The first three rates use pushed trials as their denominator:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Definition
   * - ``recovery_rate``
     - Share of pushed trials that settled: root horizontal speed ≤ ``0.15 m/s``, pelvis tilt error ≤ ``0.35``, and torso orientation error ≤ ``0.5 rad``, held continuously for ``0.5 s``.
   * - ``fall_rate``
     - Share of pushed trials that ended in a fall.
   * - ``timeout_rate``
     - Share of pushed trials that neither settled nor fell within 5 s.
   * - ``initialization_failed_rate``
     - Share of all trials that failed before the push was applied.
   * - ``recovery_time_mean_s`` / ``recovery_time_median_s``
     - Mean and median time (s) from the push to the end of the 0.5 s settled window (recovered trials only).
