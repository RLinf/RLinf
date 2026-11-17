SuperMimic Evaluation
=====================

This page evaluates a trained SuperMimic checkpoint separately from
training, for motion tracking on the 40 LAFAN1 clips or for standing push
recovery, and explains the files and metrics the evaluation writes.

Installation
------------

Use the environment, the clips, and the training run from :doc:`supermimic`.
Every command runs from the RLinf repository root with that environment
activated, and ``RUN_DIR`` is the run directory set there.

Run It
------

``evaluations/run_eval.sh`` reads ``evaluations/supermimic/<config>.yaml``
when the config name starts with ``supermimic_``. Point
``runner.ckpt_path`` at ``actor/model_state_dict/full_weights.pt`` inside a
checkpoint; the rest of the checkpoint is only needed to resume training. For a MaxCoord run, use its
``supermimic_lafan1_maxcoord`` checkpoint folder and add
``actor.model.representation=maxcoord``.

Motion Tracking
~~~~~~~~~~~~~~~

This evaluates update 6000 of the run in the main guide:

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval_6000 \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor:0,env:0,rollout:0}'

The last two lines run the evaluation on GPU 0. Without them it inherits
the training config's placement and uses every GPU on the machine. Replace
``6000`` in both paths to evaluate another checkpoint, such as
update 2000 or 4000. Each clip plays once from frame 0 and the policy action
is applied as is. The 40 clips run in parallel in 40 environments, and the
evaluation takes about 7 minutes.

Standing Push Recovery
~~~~~~~~~~~~~~~~~~~~~~

The robot holds mjlab's knees-bent standing pose. After one second, a
horizontal push of 0.5, 1, 2, or 4 m/s in a random direction is added to
its root's current horizontal velocity, four trials each.
A fall is detected with the same failure test as in training. The paper's Section 4.3 protocol starts from
a T-pose, which this config does not implement.

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_push_recovery \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/push_6000 \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor:0,env:0,rollout:0}'

Visualization and Results
-------------------------

Results go to ``runner.eval_export.output_dir``, or to
``<log_path>/<experiment_name>/eval_results/`` when it is unset. The folder
holds ``summary.json``, the resolved ``config.yaml``, and one table:
``per_clip.csv`` for motion tracking, with one row per clip, or
``per_trial.csv`` for push recovery, with one row per trial.
The same metrics are logged with an ``eval/`` prefix.

The motion-tracking metrics in ``summary.json`` are:

* ``normalized_return``: for each clip, the reward actually collected divided
  by the most it could collect, which is 0.1 per step over the whole clip at
  50 Hz. Steps lost to a fall stay in the denominator. The value is the
  unweighted mean over the 40 clips.
* ``fail_rate``: the fraction of clips in which the robot fell.
  ``clip_end_rate`` is the fraction that reached the last frame.
* ``fail_step`` and ``fail_time_s``: when the falls happened, averaged over
  the clips that fell.
* ``body_pos_err``: the position error of the tracked bodies relative to the
  pelvis, in meters, averaged over the steps each clip played.

The numbers reported for the main guide's run are in
:doc:`supermimic`, under Visualization and Results.
