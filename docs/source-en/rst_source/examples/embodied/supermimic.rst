SuperMimic: G1 Motion Tracking with a Differentiable World Model
================================================================

This guide trains and evaluates a Unitree G1 whole-body motion-tracking policy
with SuperMimic in MuJoCo Warp: install the environment, prepare reference
motion, pick a config, train on one or more GPUs, resume, and evaluate. The
algorithm, its equations, and where the implementation departs from the paper
are in :doc:`../../reference/algorithms/supermimic`.

Overview
--------

* **Simulator:** MuJoCo Warp through `mjlab <https://github.com/mujocolab/mjlab>`_
  (commit ``8bb2bc9``), G1 29-dof, 50 Hz control with four 5 ms physics steps.
* **Task:** track reference motion clips (LAFAN1 or BONES-SEED retargeted to
  the G1); optionally track velocity commands with a second goal encoder.
* **Observation:** the last 10 frames of joint positions relative to the
  default pose, joint velocities, base angular velocity, projected gravity,
  and the previous action (93 values per frame), with uniform sensor noise.
* **Action:** 29 joint setpoints :math:`q_{sp} = \alpha a + q_0` for low-gain
  PD control.
* **Goal:** 10 reference frames of 68 values (680-d), or one of the variants
  in `Configurations`_.
* **Training:** world-model fitting, 32-step backpropagation through the world
  model, and a GAE critic, all in one process per GPU.

Installation
------------

SuperMimic needs an NVIDIA GPU (MuJoCo Warp). Create a dedicated virtual
environment with RLinf's installer:

.. code-block:: bash

   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

The installer selects torch 2.14.0 for this environment only, because mjlab
requires ``torch>=2.14.0``; RLinf's project-wide torch pin is unchanged. When
the installed torch is a CUDA 12 build, it also installs the CUDA 12 build of
``warp-lang`` 1.18.0 (the PyPI wheel is a CUDA 13 build). The SuperMimic
packages are listed in ``requirements/embodied/envs/supermimic.txt``.
Installing that file with ``pip`` into an existing RLinf environment upgrades
its torch, so keep SuperMimic in its own environment. ``docker/Dockerfile``
has an ``embodied-supermimic-image`` target that runs the same install.

The G1 model, joint order, actuator gains, and the standing keyframe come from
mjlab. Every reference clip carries the default pose ``q0`` it was converted
with, and the simulator refuses clips whose ``q0`` differs from mjlab's
default joint pose, because the goal and observation subtract ``q0`` while the
PD target adds mjlab's.

Prepare reference motion
------------------------

Motion data is not part of the repository. Each ``clip_*.pt`` holds ``fps``
(50, or 30 to be resampled), the 29 G1 joint names, ``q0``, and a
``[T, 71]`` state: root position, root quaternion (wxyz), joint positions,
world-frame root linear velocity, root-frame angular velocity, and joint
velocities. A 30 Hz clip is resampled onto the 50 Hz control grid, and its
velocities are recomputed with central differences.

**LAFAN1 (the paper's 40 clips).** The clips are the ``g1/`` folder of the
Hugging Face dataset ``lvhaidong/LAFAN1_Retargeting_Dataset``. The script pins
revision ``ce1572906efe6157840e8474d5a0d7aa87481e74``, checks every CSV's size
and SHA-256 against it, and writes ``manifest.json``. The dataset card declares no license; the retargets
derive from Ubisoft La Forge LAFAN1, released under CC BY-NC-ND 4.0
(non-commercial, no derivatives). Read those terms before passing
``--accept-license``.

.. code-block:: bash

   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1

The 40 clips are 2.45 h of motion. With ``manifest.json`` present, the loader
refuses a clip whose SHA-256 differs or that the manifest does not list.

**BONES-SEED (experimental).** The loader expects clips that are already G1
retargets in the format above, plus a ``name`` used by the Appendix E name
filter. RLinf does not ship a converter from the BONES-SEED release, because
that format has not been checked against real data. The feasibility filter
needs per-frame geometry, which this tool computes with ``mj_geomDistance``
and writes into a copy of each clip:

.. code-block:: bash

   python toolkits/supermimic/bones_feasibility.py --clips /data/bones_g1 \
       --out /data/bones_g1_filtered
   export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered

The BONES-SEED configs load every clip that passes the filters
(``env.num_clips: null``) and log how many each filter removed. The paper's
142,220 / 129,667 / 129,514 counts belong to its release; no full BONES-SEED
dump has been run through this pipeline.

Configurations
--------------

All configs are in ``examples/embodiment/config/``. Batch sizes are global:
with several GPUs each rank takes an equal share.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Config
     - What it trains
   * - ``supermimic_lafan1``
     - The paper setting on LAFAN1: 680-d goal, 93-d observation, 4096 envs,
       10 x 2048 world-model windows and 4096 imagined rollouts per update.
   * - ``supermimic_lafan1_anchor``
     - 689-d goal: the 680-d goal plus the reference torso pose in the robot
       torso frame (CuboiLeo's anchor).
   * - ``supermimic_lafan1_goal730``
     - zita-ch's 730-d goal (73 values for each of the next 10 frames).
   * - ``supermimic_lafan1_obs99``
     - 99-d observation: the 93-d frame plus the root orientation relative to
       the reference.
   * - ``supermimic_velocity``
     - Motion tracking plus a velocity-command encoder (Appendix G), half of
       the environments each.
   * - ``supermimic_bones_seed``
     - The paper setting on filtered BONES-SEED.
   * - ``supermimic_bones_seed_2gpu``
     - The paper's two-GPU recipe on BONES-SEED (see `Multiple GPUs`_).
   * - ``supermimic_smoke``
     - A small run for one 8 GB GPU: 8 envs, small batches, 3 updates, same
       networks, ``k``, ``H``, and reward as the paper setting.

Two experimental options change the termination head and are off by default:
``algorithm.termination_head: analytic`` and
``algorithm.termination_bias_init``. The algorithm page explains when they
matter. The default ``algorithm.seed`` is 0; rank ``r`` uses ``seed + r``.

Train on one GPU
----------------

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1

``run_supermimic.sh <config> [hydra overrides...]`` passes everything after
the config name to Hydra and, unless ``runner.logger.log_path`` is given,
writes to ``logs/<time>-<config>``. For a quick check on a small GPU:

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_smoke env.num_clips=40

The smoke launch, and only it, sets ``RAY_memory_usage_threshold=0.99`` when
you have not set it, because an 8 GB machine sits close to Ray's default
threshold. Other configs keep Ray's memory monitor as is. Ray's object store
is limited to 512 MiB (``cluster.object_store_memory_mb``) because the replay
never passes through it; the limit applies only when RLinf starts Ray itself.

Multiple GPUs
-------------

``cluster.component_placement.supermimic`` lists GPUs, one rank per GPU.
Ranks run synchronous data parallelism: each steps its own environments,
keeps its own replay, samples its share of each batch, and all-reduces
gradients weighted by its sample count before clipping, so every rank applies
the same update. Rank 0 broadcasts the initial weights and evaluates.
``train/param_divergence`` checks that ranks hold identical weights every
``algorithm.param_check_interval`` updates.

There are two ways to use more GPUs:

* **Fixed global batch.** Keep a config and add GPUs. The global batch,
  learning rate, and update-to-data ratio stay those of one GPU, and each rank
  takes a share; this is one-GPU training spread over several GPUs.

  .. code-block:: bash

     bash examples/embodiment/run_supermimic.sh supermimic_lafan1 \
         cluster.component_placement.supermimic=0-3

* **The paper's two-GPU recipe.** Each GPU keeps the one-GPU sizes, so the
  global batch doubles, gradients are averaged over both GPUs, and the
  learning rate is :math:`2\sqrt{2} \times 10^{-4}` (Table A.5):

  .. code-block:: bash

     export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered
     bash examples/embodiment/run_supermimic.sh supermimic_bones_seed_2gpu

  The same overrides apply to LAFAN1:
  ``cluster.component_placement.supermimic=0-1 env.num_envs=8192
  algorithm.wm_window_batch=4096 algorithm.imagine_batch=8192
  algorithm.lr=2.8284271e-4``.

Collectives use NCCL on GPU with the timeout ``RLINF_TIMEOUT`` (minutes). If a
rank fails, RLinf's worker group stops the driver and the other ranks at once
instead of waiting for that timeout.

Multiple nodes with Slurm (experimental)
----------------------------------------

``examples/embodiment/slurm_supermimic.sh`` starts one Ray node per Slurm
node, exports a unique ``RLINF_NODE_RANK`` before each ``ray start``, and runs
the driver on the first node with a placement over every GPU. Partition,
account, and GPU count come from your ``sbatch`` options; ``log_path`` and the
data must be on storage every node can read and write.

.. code-block:: bash

   sbatch --nodes=2 --gpus-per-node=8 --partition=<partition> \
       examples/embodiment/slurm_supermimic.sh supermimic_lafan1 \
       env.motion_path=/shared/lafan1_g1 runner.logger.log_path=/shared/runs/lafan1

This script has been checked with stubbed ``srun`` and ``ray`` only; it has not
run on a Slurm cluster.

Checkpoints and resume
----------------------

``runner.save_interval`` writes ``log_path/<experiment_name>/global_step_<N>``
every N updates, and the last update is always written. A checkpoint holds
``supermimic.pt`` (networks, optimizers, schedules, counters, reset-bin
statistics), one ``rank_<r>.pt`` per rank (RNG streams), and
``checkpoint_meta.json`` (file sizes and SHA-256, world size, model
dimensions, goal window, format version). Files are written to a ``.tmp``
directory and renamed when every rank is done, so an interrupted save leaves
no partial checkpoint.

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1 \
       runner.logger.log_path=/data/runs/lafan1 runner.resume_dir=auto

``runner.resume_dir=auto`` picks the newest complete checkpoint of that run; a
path picks that checkpoint. A checkpoint of another format, dimension, or goal
window is refused. A different number of ranks loads the weights and
optimizers and restarts the RNG streams. The replay and simulator are not
saved, so a resumed run warms up a new replay and is a continuation, not an
exact restore.

Evaluate
--------

``evaluations/eval_supermimic.py`` loads the networks from a checkpoint and
evaluates on one GPU without Ray. Each training config has an eval config in
``evaluations/supermimic/``; a folder of checkpoints picks the newest one.

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       eval.checkpoint=/data/runs/lafan1/supermimic_lafan1
   bash evaluations/run_eval.sh supermimic_velocity_eval \
       eval.checkpoint=/data/runs/velocity/supermimic_velocity

**Motion tracking** plays every clip once from frame 0 with sensor and action
noise off and writes ``per_clip.csv``. Normalized return divides the reward
sum by 0.1 times the clip's scheduled length, so an early failure keeps the
unplayed part in the denominator.

**Velocity tracking** is chosen with ``eval.velocity``:

* ``g2`` (the velocity eval config's default) runs the Appendix G.2 step
  test: 672 commands on a 15 x 15 grid in each of the (vx, vy), (vx, wz), and
  (vy, wz) planes without the standing command, four trials each, 5 s from
  mjlab's standing pose. The value of a trial is the norm of the mean velocity
  error over the last 2 s divided by the command norm; a command reports the
  median over the trials that did not fall. A command whose trials all fell
  has no value (NaN in the CSV, ``null`` in ``summary.json``) and is counted
  in ``eval_velocity_g2/commands_all_trials_fell``. Trials differ in the sim
  context and the initial joint noise, which the paper does not specify. Results go
  to ``velocity_g2.csv`` and ``eval_velocity_g2/*``.
* ``smoke`` holds seven fixed commands from a reference start and writes
  ``velocity_smoke.csv`` and ``eval_velocity_smoke/*``. It checks that the
  velocity encoder runs; it is not a paper metric.

``runner.only_eval=true`` evaluates through the training entry instead, using
the training config's environment count. With ``runner.val_check_interval``
set, training evaluates every that many updates and after the last one.

Metrics and logs
----------------

Logging goes through RLinf's ``MetricLogger`` (TensorBoard, WandB, SwanLab
under ``log_path``). With several ranks, counts are summed, timers and memory
take the largest rank, shared counters come from rank 0, and losses are
averaged; ``runner.log_per_rank_metrics=true`` adds ``rank_<r>/`` copies.
Useful keys:

* ``wm/one_step_loss`` and ``wm/k_step_loss``: world-model error after one and
  eight steps.
* ``imagination/survival`` and ``imagination/termination_prob``: the 32-step
  survival weight and the mean termination probability.
* ``actor/grad_norm`` (before clipping), ``actor/encoder_grad_norm``, and
  ``actor/decoder_grad_norm``.
* ``critic/loss``, ``critic/value_mean``, ``critic/target_mean``.
* ``reset/bin_return_mean`` and ``reset/bin_return_min``: motion reset-bin
  returns.
* ``train/nonfinite_skips``, ``train/param_divergence``, ``time/comm_s``.
* ``memory/torch_peak_allocated``, ``memory/torch_peak_reserved``
  (PyTorch's allocator), ``memory/warp_mempool_peak`` (Warp's pool), and
  ``memory/device_used`` (the whole GPU as the driver sees it). These counters
  overlap; do not add them.

Troubleshooting
---------------

* **CUDA out of memory.** The error names the rank and its per-rank sizes.
  Add GPUs to keep the batch sizes, or use a separate smaller config;
  ``toolkits/supermimic/profile_sim.py`` measures memory for given sizes.
* **Ray kills a worker for low memory.** Free host memory rather than raising
  ``RAY_memory_usage_threshold``; the smoke launch is the only exception. On a
  host with many CPU cores and little RAM, Ray starts one idle worker per core
  (about 50 MB each) that SuperMimic does not use;
  ``RAY_prestart_worker_first_driver=0`` skips them and leaves the memory
  monitor as it is.
* **"No verified G1 reference clips".** ``SUPERMIMIC_MOTION_PATH`` or
  ``env.motion_path`` does not point at prepared clips.
* **"reference clip q0 differs from the mjlab G1 default joint pose".** The
  clips were converted with another default pose; reconvert them.
* **Checkpoint format or goal window refused.** The checkpoint was written by
  another version or config; it cannot be resumed with this one.
* **"linesearch iterations limit reached".** A MuJoCo Warp solver warning that
  untrained policies trigger; it does not stop training.

Performance and hardware
------------------------

``toolkits/supermimic/profile_sim.py`` measures collection throughput, update
time, replay size, and the memory counters above for a list of environment
counts, each in its own process:

.. code-block:: bash

   python toolkits/supermimic/profile_sim.py --motion-path /data/lafan1_g1 \
       --num-envs 256 1024 4096 --steps 30 --warmup 20 --update-iters 3 \
       --out profile.json

On an 8 GB laptop GPU under WSL2, 1024 environments collected about 3,900
environment transitions per second, one update with a quarter of the paper
batches (1024 imagined rollouts, 512 windows) took about 21 s at a 4.2 GB
PyTorch peak, and one update at the paper batches needed about 11 GB, more
than that card has. These numbers come from a launch-bound laptop setup and
do not predict datacenter GPUs; measure seconds per update on your hardware
before planning a full run.

What has been run
-----------------

On one 8 GB GPU with MuJoCo Warp and the LAFAN1 clips: short training runs of
the LAFAN1 configs (680-d, 689-d, 730-d, 99-d observation, velocity) at small
sizes, resume including after a killed driver, motion and velocity
evaluation, and the G.2 step test. The multi-GPU
arithmetic is tested with two CPU (Gloo) processes. Not yet run: paper-scale
training, NCCL on real GPUs, multiple nodes, Slurm, BONES-SEED data, and a
training run long enough to show that the policy learns to track.
