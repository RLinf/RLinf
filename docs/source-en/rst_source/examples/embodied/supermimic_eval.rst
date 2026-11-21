SuperMimic Evaluation
=====================

This page evaluates a trained SuperMimic checkpoint in two ways, tracking the
40 LAFAN1 clips and standing push recovery, and explains the files and
metrics each evaluation writes.

Installation
------------

Use the environment and the clips from :doc:`supermimic`, and run its
setup block in the shell first. The commands below read ``RUN_DIR`` from
that block and evaluate the checkpoints of its training run.

Run It
------

``evaluations/run_eval.sh`` picks the config from
``evaluations/supermimic/``. ``runner.ckpt_path`` points at the
``full_weights.pt`` inside a checkpoint, and ``runner.eval_export.output_dir``
names the folder for the results. Evaluation runs on GPU 0. For a
MaxCoord checkpoint, use its ``supermimic_lafan1_maxcoord`` folder and add
``actor.model.representation=maxcoord``. Every argument is listed in the
:doc:`CLI reference <../../evaluations/reference/cli>`.

Motion Tracking
~~~~~~~~~~~~~~~

Each of the 40 clips plays once from its first frame in its own environment,
and the robot applies the policy's actions without noise. Evaluate update
6000:

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval_6000

To evaluate update 2000 or 4000, replace ``6000`` in both paths.

Standing Push Recovery
~~~~~~~~~~~~~~~~~~~~~~

The robot holds mjlab's knees-bent standing pose. After one second, a push
in a random horizontal direction adds 0.5, 1, 2, or 4 m/s to the horizontal
velocity of its root, the pelvis, with four trials per speed. The trial then runs for up to
five seconds. This is a standing variant: the T-pose start of Section 4.3 of
the paper is not part of this config.

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_push_recovery \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/push_6000

Visualization and Results
-------------------------

Each evaluation ends its log with ``Wrote SuperMimic evaluation results to``
and the output folder, which holds:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``summary.json``
     - The metrics over all clips or trials
   * - ``per_clip.csv``
     - Motion tracking: one row per clip
   * - ``per_trial.csv``
     - Push recovery: one row per trial, with its speed, direction, and
       outcome
   * - ``config.yaml``
     - The resolved evaluation config

The same metrics go to the log with an ``eval/`` prefix.

Motion-tracking metrics:

* ``normalized_return``: for each clip, the reward it collected divided by
  the most it could collect, which is 0.1 per control step over the whole
  clip. Steps lost to a fall count against it. The value is the mean over
  the clips.
* ``fail_rate``: the share of clips in which the robot fell.
  ``clip_end_rate`` is the share that reached the last frame.
* ``fail_step`` and ``fail_time_s``: when the falls happened, averaged over
  the clips that fell.
* ``body_pos_err``: the position error of the tracked bodies relative to the
  pelvis, in meters, averaged over the steps each clip played.

Push-recovery metrics, over all trials and again for each speed:

* ``recovery_rate``: the share of trials in which the robot settled again,
  with its root slower than 0.15 m/s horizontally and its body upright for
  0.5 s.
* ``fall_rate``: the share of trials in which the robot fell.
* ``timeout_rate``: the share that neither settled nor fell within five
  seconds.
* ``recovery_time_mean_s`` and ``recovery_time_median_s``: for the trials
  that recovered, the time from the push to the end of that 0.5 s settled
  stretch.
