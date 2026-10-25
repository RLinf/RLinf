SuperMimic
==========

SuperMimic trains a humanoid tracking policy by backpropagating through a
learned world model for 32 imagined steps. It does not use a PPO likelihood
ratio.

Launch
------

The only entry point is ``examples/embodiment/train_supermimic.py``.

.. code-block:: bash

   cd examples/embodiment
   # 8 GB smoke run
   bash run_supermimic.sh supermimic_smoke
   # paper observation (93-d) and 680-d goal; needs a larger GPU and mjlab
   export SUPERMIMIC_MOTION_PATH=/path/to/lafan1_g1_clips
   bash run_supermimic.sh supermimic_lafan1
   # 689-d goal: the same 93-d observation plus a 9-d anchor on the goal
   bash run_supermimic.sh supermimic_lafan1_anchor
   bash run_supermimic.sh supermimic_lafan1_goal730
   bash run_supermimic.sh supermimic_lafan1_obs99
   bash run_supermimic.sh supermimic_velocity
   export SUPERMIMIC_BONES_PATH=/path/to/bones_seed
   bash run_supermimic.sh supermimic_bones_seed

Dependencies
------------

``supermimic_smoke`` shrinks the number of environments and the batches. It
uses the same mjlab / MuJoCo Warp backend, ``k=8``, ``H=32``, and network
widths as the paper run. An empty ``SUPERMIMIC_MOTION_PATH`` is an error.
The analytic integrator remains a CPU test double and is not this smoke run.
The Ray object store is limited to 512 MiB because the replay never enters
it. The memory monitor keeps Ray's own threshold. Only
``run_supermimic.sh supermimic_smoke`` sets
``RAY_memory_usage_threshold=0.99`` when the user has not set it.
The stack checked on this machine is PyTorch ``2.14.0+cu126``, Warp
``1.18.0+cu12``, and mjlab commit ``8bb2bc9``, pinned in
``requirements/embodied/envs/supermimic.txt``. That file does not change
the torch pin used by other algorithms. Do not add LAFAN1 or BONES-SEED to
the repo. Each ``clip_*.pt`` must carry 50 Hz (or 30 Hz to be resampled),
the 29 G1 joint names, the default pose ``q0``, and a ``[T, 71]`` state.
The dynamics MLP uses the Table A.4 widths 2048, 2048, 2048.
Resets use the 50% / 40% / 10% mixture over 1 s bins. A short episode masks
its reward log by length before the return is accumulated, then clears it.
``only_eval`` walks every selected clip once from frame 0, in batches of
``num_envs``. Normalized return divides the undiscounted reward sum by
``0.1`` times the clip's full scheduled length. Early failure does not shrink
that denominator. ``fail_step`` is a zero-based index. ``fail_time_s`` and
``episode_length_s`` are seconds. The reward root is ``torso_link``. Failure
uses the pelvis. Evaluation also reports the free root, which is the pelvis
on G1. The 9-d anchor is recomputed every step. Timers call
``cuda.synchronize``. The recorded times are for the smoke batch, not for
4096 environments. Resume creates a new trainer, warms the new replay, and
does not treat the restored ``sim_vector_steps`` as a warm buffer.

One imagination produces ``s_0..s_H``, ``g_0..g_H``, rewards, and termination
probabilities. The critic regresses ``V(s_t, g_t)`` to ``R_t`` on that same
trajectory for ``t = 0..H-1``.

Checkpoints
-----------

``save_interval <= 0`` skips periodic checkpoints, but the last update is
always written, and only once if it is also a periodic step. The file stores
the world model, policy, critic, optimizers, schedules, and counters. With
velocity tracking it also stores that encoder and critic. It does not store
the replay, the simulator, or the reset bins. Resume creates a new trainer
and repeats the warmup collection.

Sensor noise is uniform, Table A.1: joint position ``U[-0.01, 0.01]``,
joint velocity ``U[-0.5, 0.5]``, base angular velocity ``U[-0.2, 0.2]``,
projected gravity ``U[-0.05, 0.05]``. The previous action is not noised.
Each control step samples noise for the new frame only. Reset repeats that
one noisy frame across the empty history. That cache is an implementation
choice. Evaluation turns sensor noise off. Imagination noise is exogenous
and does not detach the state. The root rotation loss is
``||Log(R_pred R_target^T)||^2``. Other state scales are unchanged.

The 730-d goal follows zita-ch: frames ``t+1`` through ``t+10``, last frame
held, 73 values per frame, joint velocity scaled by 0.1. It is not appended
to the 680-d goal. The 99-d observation appends
``Rot6D(R_robot^T R_reference)`` with no sensor noise on those six values.
Velocity training is the Appendix G RL interface, not distillation. Half of
the environments and half of the imagined batch are motion tracking, half are
velocity. The shared decoder is frozen on the velocity update, and the
alignment loss does not update the motion encoder. BONES-SEED applies the
Appendix E name and feasibility filters. The library stays on CPU and a query
copies only the gathered rows. Without the real dataset the loader errors; a
synthetic fixture is not a dataset acceptance. The name list is the terms
Appendix E prints, not SONIC's private list.

A clip of T frames is controlled at times ``0 .. T-1``, so the scheduled
length is T control steps, not T-1. Early failure does not shrink that
denominator. ``fail_step`` is a zero-based index. ``fail_time_s`` and
``episode_length_s`` are seconds.

This machine's ``jrwsdjango`` env also has conflicts outside SuperMimic, such
as TaH asking for ``torch==2.6.0`` while this stack is ``2.14.0+cu126``. The
requirements cap numpy below 2.4; the installed numpy is 2.4.6. Those are
left as environment limits.

Termination versus truncation
-----------------------------

A pelvis height or tilt failure is a world-model positive and an eval
failure. A 10 s timeout and a clip end are not failures. Timeout bootstraps
real collect returns with the critic; clip-end does not. An 8-step window
never crosses an episode boundary. Velocity tracking uses its own fall test:
pelvis height under 0.35 m, or ``1 + projected_gravity_z > 0.7``. It does not
use the motion reference height.
