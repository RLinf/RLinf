SuperMimic Evaluation
=====================

This page evaluates a trained SuperMimic checkpoint on its own. Install the
environment and prepare the clips as in :doc:`supermimic`; every command
runs from the RLinf repository root with that environment activated.

Run an Evaluation
-----------------

Evaluation uses ``evaluations/eval_embodied_agent.py``. ``evaluations/run_eval.sh``
reads ``evaluations/supermimic/<config>.yaml`` when the config name starts
with ``supermimic_``. ``runner.ckpt_path`` is the checkpoint's
``full_weights.pt``.

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=/data/runs/gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=/data/runs/gencoord/eval_2000

Add ``actor.model.representation=maxcoord`` for a MaxCoord checkpoint and
point ``runner.ckpt_path`` at that run. Replacing ``global_step_2000`` and
the output directory with ``global_step_4000`` evaluates update 4000.
``env.eval.eval_tasks`` is ``[tracking]`` or ``[push_recovery]``. The 40
LAFAN1 clips fit in ``env.eval.total_num_envs=40``.
``max_steps_per_rollout_epoch=13656`` covers the longest clip (8194 frames
at 30 fps, resampled to 50 Hz).

Output
------

``runner.eval_export.enabled`` is on in the evaluation configs. Results go
to ``runner.eval_export.output_dir``, or to
``<log_path>/<experiment_name>/eval_results/`` when that is unset. The
directory contains ``summary.json``, ``per_clip.csv``, and the resolved
``config.yaml``. The same metrics are logged with an ``eval/`` prefix.

Motion Tracking
---------------

``supermimic_lafan1_eval`` plays each clip once from frame 0 and applies the
policy action as is. Clip :math:`i`'s normalized return is the sum of the
rewards actually executed, divided by :math:`0.1 \times T_i`, where
:math:`T_i` is the clip length on the 50 Hz grid and 0.1 is the per-step
reward ceiling. Steps that were never played because the clip failed stay in
the denominator. ``normalized_return`` is the unweighted mean over the 40
clips. ``fail_rate`` is the fraction of clips that fell, and
``clip_end_rate`` is the fraction that reached the last frame. ``fail_step``
and ``fail_time_s`` are averaged only over the clips that fell.

Standing Push Recovery
----------------------

``supermimic_push_recovery`` loads a tracking checkpoint, holds mjlab's
knees-bent standing pose, and after one second sets the root's horizontal
velocity to 0.5, 1, 2, or 4 m/s, four trials each. A fall uses the training
failure test. The paper's Section 4.3 protocol is a T-pose; this config does
not implement that pose. A MaxCoord checkpoint still needs
``actor.model.representation=maxcoord``.
