SuperMimic: G1 Motion Tracking with a Differentiable World Model
================================================================

This guide trains and evaluates a Unitree G1 whole-body motion-tracking policy
with SuperMimic in MuJoCo Warp: install the environment, prepare reference
motion, pick a config, train on one or more GPUs, resume, and evaluate. The
algorithm, its equations, and where the implementation departs from the paper
are in :doc:`../../reference/algorithms/supermimic`. On a first run, follow
`Run It`_.

Unless stated otherwise, every command on this page runs from the RLinf
repository root with the SuperMimic environment below activated.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      SuperMimic: 32-step backpropagation through a differentiable world
      model + GAE critic

   .. grid-item-card:: Models
      :text-align: center

      World model (dynamics + termination head), encoder-decoder actor, critic

   .. grid-item-card:: Environments / Data
      :text-align: center

      MuJoCo Warp G1 (mjlab); LAFAN1, BONES-SEED

   .. grid-item-card:: Training
      :text-align: center

      One process per GPU; NCCL data parallel; Slurm multi-node
      (experimental)

| **You'll do:** install a dedicated environment -> prepare the LAFAN1 clips
  -> train, resume, and evaluate with the smoke config -> train the paper
  config on one or more GPUs.
| **Prerequisites:** an NVIDIA GPU; the RLinf repository
  (:doc:`installation <../../start/installation>`); LAFAN1 G1 clips (see
  `Prepare Reference Motion`_).

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Item
     - Description
   * - Simulator
     - MuJoCo Warp through `mjlab <https://github.com/mujocolab/mjlab>`_
       (commit ``8bb2bc9``), G1 29-dof, 50 Hz control with four 5 ms physics
       steps.
   * - Task
     - Track reference motion clips (LAFAN1 or BONES-SEED retargeted to the
       G1); optionally track velocity commands with a second goal encoder.
   * - Observation
     - The last 10 frames of joint positions relative to the default pose,
       joint velocities, base angular velocity, projected gravity, and the
       previous action (93 values per frame), with uniform sensor noise.
   * - Action
     - 29 joint setpoints :math:`q_{sp} = \alpha a + q_0` for low-gain PD
       control.
   * - Goal
     - 10 reference frames of 68 values (680-d), or one of the variants in
       `Configurations`_.
   * - Entry points
     - Training uses ``examples/embodiment/run_supermimic.sh`` (not the generic
       ``run_embodiment.sh``); evaluation uses
       ``evaluations/run_eval.sh supermimic_*``, which runs
       ``evaluations/eval_supermimic.py``.

How SuperMimic Works
--------------------

One process per GPU repeats three steps. It steps every environment once in
MuJoCo Warp and writes the transitions to the replay. At the update-to-data
ratio, it fits the world model's dynamics and termination head on ``k``-step
windows from the replay. It then starts 32-step imagined rollouts from replay
states inside the world model, backpropagates the return through the
differentiable world model to update the actor, and fits the critic to GAE
targets. When an environment ends, the returns of bins along the motion
timeline decide where it restarts, so low-return segments are drawn more
often. Equations, network structure, and differences from the paper and the
community implementations are on the
:doc:`algorithm page <../../reference/algorithms/supermimic>`.

Installation
------------

SuperMimic needs an NVIDIA GPU (MuJoCo Warp). From the RLinf repository root,
create a dedicated virtual environment with RLinf's installer:

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

In every new shell, return to the repository root and run
``source .venv-supermimic/bin/activate`` first.

The installer selects torch 2.14.0 for this environment only, because mjlab
requires ``torch>=2.14.0``; RLinf's project-wide torch pin is unchanged. torch
2.14.0 requires ``setuptools>=77.0.3``, so for this environment the installer
also replaces the project-wide ``setuptools<75.9`` override with
``setuptools>=77.0.3,<82`` (setuptools 82 removes ``pkg_resources``). When
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

Prepare Reference Motion
------------------------

Motion data is not part of the repository. A clip folder holds ``clip_*.pt``
files (loaded in file-name order) and optionally a ``manifest.json``. The
format is in `Clip File Format`_.

**LAFAN1 (the paper's 40 clips).** The clips are the ``g1/`` folder of the
Hugging Face dataset
`lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_.
The script pins revision ``ce1572906efe6157840e8474d5a0d7aa87481e74``, checks
every CSV's size and SHA-256 against it, and writes ``manifest.json``. The
dataset card declares no license; the retargets derive from Ubisoft La Forge
`LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_,
released under CC BY-NC-ND 4.0 (non-commercial, no derivatives). Read those
terms before passing ``--accept-license``.

.. code-block:: bash

   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1

The 40 clips are 2.45 h of motion. With ``manifest.json`` present, the loader
refuses a clip whose SHA-256 differs or that the manifest does not list, so do
not rename, add, remove, or edit the prepared files.

**BONES-SEED (experimental).** The official release is
`bones-studio/seed <https://huggingface.co/datasets/bones-studio/seed>`_ on
Hugging Face. G1 trajectories live in ``g1/csv/{date}/{motion}.csv``. The
layout is described in
`bones-studio/seed-viewer <https://github.com/bones-studio/seed-viewer>`_.
Read ``LICENSE.md`` in the download before converting. The script does not
download the data.

The public header is the one in NVIDIA ProtoMotions'
`SEED G1 CSV preparation <https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst>`_
and in the ``Rotation.from_euler("xyz")`` call used by
GR00T-WholeBodyControl: 120 Hz; ``root_translateX/Y/Z`` in centimeters;
``root_rotateX/Y/Z`` as extrinsic XYZ Euler angles in degrees; 29 joint
columns named ``<joint>_dof`` in degrees, matching the mjlab G1 joint names.
The script reorders columns by those names and does not swap axes. In scipy,
lowercase ``xyz`` means extrinsic. This convention has not been checked
against an official CSV.

.. code-block:: bash

   python toolkits/supermimic/convert_bones_seed.py \
       --input /data/bones-seed/g1/csv \
       --output /data/bones_g1
   python toolkits/supermimic/bones_feasibility.py --clips /data/bones_g1 \
       --out /data/bones_g1_filtered
   export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered

``--input`` is one CSV or a directory, and a directory is scanned
recursively. The default resample is 120 Hz to 50 Hz. Velocities are
recomputed on that grid by the existing ``resample_states``. An existing
``clip_*.pt`` is left in place unless ``--overwrite`` is set. One broken CSV
is reported and the rest of the directory continues. A clip's ``name`` is its
path relative to ``--input``, without the suffix, so the same motion name
under two dates does not share an output file and the Appendix E name filter
still sees words such as ``stair`` and ``sit``. Feasibility stays in
``bones_feasibility.py``: it writes per-frame ``mj_geomDistance`` results
into a copy, and the loader filters on those arrays.

A few CSVs can exercise the training entry without the paper's 4096
environments:

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_bones_seed \
       env.num_envs=8 algorithm.total_updates=3 runner.save_interval=3

The BONES-SEED configs load every clip that passes the filters
(``env.num_clips: null``) and log how many each filter removed. The paper's
142,220 / 129,667 / 129,514 counts belong to its release. The converter is
tested on synthetic CSVs only. No full BONES-SEED dump has been converted,
and no official CSV has been checked.

Clip File Format
~~~~~~~~~~~~~~~~

Each ``clip_*.pt`` is a dictionary saved with ``torch.save``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Content
   * - ``fps``
     - 50, or 30. A 30 Hz clip is resampled onto the 50 Hz control grid and
       its velocities are recomputed with central differences; a 50 Hz clip
       is used as is, so **its velocity columns must be correct**. Other rates
       are refused.
   * - ``joint_names``
     - The 29 joint names, in exactly the order of
       ``rlinf.envs.sim.g1_mimic.constants.G1_JOINT_NAMES`` (mjlab's G1 joint
       order).
   * - ``q0``
     - The 29-d default joint pose (rad). It must equal mjlab's default joint
       pose, ``asset_default_state()[7:36]``.
   * - ``state``
     - A ``[T, 71]`` float tensor laid out as below, with no NaN or Inf and a
       normalized root quaternion.
   * - ``name``
     - Optional; read only by the BONES-SEED name filter.

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Columns
     - Content
     - Unit
   * - ``0:3``
     - Root (pelvis) position, world frame
     - m
   * - ``3:7``
     - Root quaternion, wxyz
     - —
   * - ``7:36``
     - Joint positions, in ``joint_names`` order
     - rad
   * - ``36:39``
     - Root linear velocity, world frame
     - m/s
   * - ``39:42``
     - Root angular velocity, root frame
     - rad/s
   * - ``42:71``
     - Joint velocities
     - rad/s

This minimal example writes a 5 s clip of standing still and runs the
loader's own checks on it:

.. code-block:: python

   from pathlib import Path

   import torch

   from rlinf.envs.sim.g1_mimic.constants import G1_JOINT_NAMES
   from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
   from rlinf.envs.sim.g1_mimic.motion import decode_motion_payload

   default = asset_default_state().float()  # mjlab standing keyframe, [71]
   T = 250                                  # 5 s at 50 Hz
   state = torch.zeros(T, 71)
   state[:, :36] = default[:36]             # root pose and joints; velocities stay 0
   payload = {
       "fps": 50,
       "joint_names": list(G1_JOINT_NAMES),
       "q0": default[7:36],
       "state": state,
   }
   decode_motion_payload(payload, "clip_000.pt")  # raises ValueError if invalid
   out = Path("/data/my_clips")
   out.mkdir(parents=True, exist_ok=True)
   torch.save(payload, out / "clip_000.pt")

Without ``manifest.json`` the loader checks the format only, not the file
contents.

Run It
------

This walks through training, saving, resuming, evaluating, and viewing curves
with ``supermimic_smoke`` (8 environments, 3 updates; fits one 8 GB GPU).
Checkpoints, curves, and evaluation results go under one ``RUN_DIR``, which is
how each step finds the previous one's output (``run_eval.sh`` still writes its
own run log to the repository's ``logs/``; see `Evaluate`_).

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export RUN_DIR=/data/runs/smoke

   # 1. Train 3 updates, saving a checkpoint each update and writing TensorBoard
   bash examples/embodiment/run_supermimic.sh supermimic_smoke env.num_clips=40 \
       runner.logger.log_path=$RUN_DIR runner.save_interval=1 \
       'runner.logger.logger_backends=[tensorboard]'

   # 2. Resume from the newest checkpoint and train to update 5
   bash examples/embodiment/run_supermimic.sh supermimic_smoke env.num_clips=40 \
       runner.logger.log_path=$RUN_DIR runner.save_interval=1 \
       'runner.logger.logger_backends=[tensorboard]' \
       runner.resume_dir=auto algorithm.total_updates=5

   # 3. Evaluate the newest checkpoint (smoke has the supermimic_lafan1
   #    networks, so it uses that eval config)
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       eval.checkpoint=$RUN_DIR/supermimic_smoke eval.output_dir=$RUN_DIR/eval

   # 4. View the training curves
   tensorboard --logdir $RUN_DIR/tensorboard

Step 2's log should show ``Resuming SuperMimic networks, ... from
.../global_step_3``. If it shows ``resume_dir=auto: no complete checkpoint
under ..., starting from scratch`` instead, its ``log_path`` differs from step
1 or step 1 wrote no checkpoint, and the run started over rather than resumed.

Afterwards ``RUN_DIR`` contains:

.. code-block:: text

   /data/runs/smoke/
   ├── supermimic_smoke/              # log_path/<experiment_name>
   │   ├── global_step_1/ ... global_step_5/
   │   │   ├── supermimic.pt          # networks, optimizers, schedules, counters
   │   │   ├── rank_0.pt              # per-rank RNG state
   │   │   └── checkpoint_meta.json   # sizes, SHA-256, world size, dimensions, goal window
   ├── tensorboard/                   # events.out.tfevents.* and config.yaml
   └── eval/                          # summary.json, per_clip.csv, config.yaml

The smoke run only shows that the pipeline works. A policy after 3 to 5
updates does not track motion, and an ``eval/fail_rate`` near 1 is expected.

Configurations
--------------

All configs are in ``examples/embodiment/config/``, with the shared
hyperparameters in ``supermimic_base.yaml``. Batch sizes are global: with
several GPUs each rank takes an equal share.

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
   * - ``supermimic_lafan1_zita``
     - zita-ch's interface and schedule: the 730-d goal together with the
       99-d observation, an analytic termination head, 500 world-model-only
       updates, 64 warmup steps, 16000 updates, and a checkpoint every 2000.
       The reward anchor is the pelvis, inputs use a running norm, the
       residual uses the body frame, and the state loss divides by the
       running residual standard deviation. The critic uses zita-ch's extra
       features. The simulator stays mjlab.
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
``algorithm.wm_only_updates`` defaults to 0. A positive value fits only the
world model for that many updates. ``algorithm.critic_features`` defaults to
``root4`` (the 4-d root error); ``zita`` selects zita-ch's extra features.

When the next four keys are omitted, the code follows zita-ch: a pelvis
reward anchor, running input normalization, a body-frame residual, and a
state loss divided by the running residual standard deviation.
``supermimic_base.yaml`` writes the paper names, so ``supermimic_lafan1``
does not silently become that recipe.

* ``algorithm.reward_anchor``: ``pelvis`` or ``torso`` (``torso_link``).
  The paper config writes ``torso``. ``termination_body`` still chooses the
  failure body.
* ``algorithm.input_norm``: ``running`` keeps a Welford mean and variance,
  after zita-ch's fixed scale on :math:`\xi`. ``none`` leaves the raw
  values. The paper config writes ``none``.
* ``algorithm.residual_frame``: ``body`` puts linear velocity in the full
  body frame and stores the world-rate angular increment back as a body
  rate. ``heading`` puts position and linear velocity in the yaw frame and
  adds the angular residual to the stored body rate. The paper config
  writes ``heading``.
* ``algorithm.wm_state_loss``: ``running`` divides the frame-local error by
  the running residual standard deviation and averages the 70 components.
  ``paper`` is an unweighted square. ``scaled`` uses fixed block scales.
  The paper config writes ``paper``.

``supermimic_lafan1_zita`` sets ``pelvis``, ``running``, ``body`` and
``running``, overriding the base config. The simulator stays mjlab. It
refuses a checkpoint saved before those four fields were recorded. One run
on a single 80 GB A800 with 4096 environments finished 16000 updates. The
first compilation took about 30 minutes, and the 16000 updates took about
4.5 hours. At update 16000, ``eval/normalized_return`` on the 40 LAFAN1
clips was 0.856, and every clip played through to its end. The same run
at update 14000 scored 0.800, with a failure rate of 0.10: 36 of the 40
clips played through to the end. zita-ch reports 0.829 at update 14000,
so this run is 0.029 lower at the same update.

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
       runner.logger.log_path=/data/runs/lafan1_zita \
       'runner.logger.logger_backends=[tensorboard]'

Evaluate with ``supermimic_lafan1_zita_eval`` on one GPU, without Ray.
Point ``eval.checkpoint`` at the ``supermimic_lafan1_zita`` directory under
that ``log_path`` to use the newest complete checkpoint, or at one
``global_step_<N>`` directory. The run above scored
``eval/normalized_return`` 0.856 at update 16000 and 0.800 at update
14000. zita-ch reports 0.829 at update 14000. To score update 14000,
point the checkpoint at that save:

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_zita_eval \
       eval.checkpoint=/data/runs/lafan1_zita/supermimic_lafan1_zita/global_step_14000 \
       eval.output_dir=/data/runs/lafan1_zita/eval_14000

Common Keys
~~~~~~~~~~~

These are the keys most often changed to set the scale and length of a run.
Defaults come from ``supermimic_base.yaml`` (``supermimic_smoke`` and other
configs override some of them). Each can be passed as a Hydra override after
``run_supermimic.sh <config>``.

.. list-table::
   :header-rows: 1
   :widths: 27 12 61

   * - Key
     - Default
     - Meaning and effect
   * - ``algorithm.k``
     - 8
     - World-model multi-step training window. Fixed at 8; validation refuses
       other values.
   * - ``algorithm.H``
     - 32
     - Imagination horizon, the number of steps backpropagated through the
       world model. Fixed at 32.
   * - ``algorithm.gamma``, ``algorithm.lam``
     - 0.99, 0.95
     - GAE discount and :math:`\lambda` for the critic and actor targets.
   * - ``algorithm.total_updates``
     - 320000
     - Number of training **updates**, not simulator steps (see below).
   * - ``algorithm.updates_per_env_step``
     - 0.4
     - Global updates owed per vectorized simulator step (update-to-data
       ratio). Larger values train more on each sample.
   * - ``algorithm.lr``, ``algorithm.lr_decay_updates``
     - 2.0e-4, 250000
     - Learning rate shared by the world model, actor, and critic; cosine
       decay to 0.03 x ``lr`` over the first ``lr_decay_updates`` updates,
       then constant.
   * - ``algorithm.wm_batches``
     - 10
     - World-model minibatches per update.
   * - ``algorithm.wm_window_batch``
     - 2048
     - ``k``-step windows per world-model minibatch, global.
   * - ``algorithm.imagine_batch``
     - 4096
     - Imagined rollouts per update, global. A main driver of GPU memory.
   * - ``algorithm.critic_epochs``, ``algorithm.critic_minibatches``
     - 5, 4
     - Passes of the critic over the imagined data per update, and
       minibatches per pass.
   * - ``algorithm.warmup_sim_steps``
     - 40
     - Vectorized simulator steps collected after each start or resume,
       before the first update, to warm the replay; they owe no updates.
   * - ``env.num_envs``
     - 4096
     - Parallel simulator environments, global. Collection throughput and
       memory grow with it.
   * - ``env.replay_length``
     - 1000
     - Ring-buffer replay capacity per environment, in control steps (20 s at
       50 Hz). Replay memory is proportional to ``num_envs x replay_length``.
   * - ``env.num_clips``
     - 40
     - Load the first N clips in file-name order; ``null`` loads all.
   * - ``runner.save_interval``
     - -1
     - Save a checkpoint every N updates. The default -1 saves only after the
       last update, so an interrupted long run has nothing to resume from.
   * - ``runner.logger.logger_backends``
     - ``[]``
     - The default writes no TensorBoard, WandB, or SwanLab, only console
       logs. Set ``[tensorboard]`` for curves.
   * - ``runner.val_check_interval``
     - -1
     - Evaluate every N updates during training; off by default.
   * - ``runner.logger.log_path``
     - see text
     - Root of checkpoints and logs. Without it, ``run_supermimic.sh`` uses
       ``logs/<time>-<config>``, a new folder on every launch.

**Updates versus simulator steps.** Each loop iteration collects one
vectorized simulator step (every environment steps once) and adds
``updates_per_env_step`` to the updates owed; each whole update owed is then
run. So ``total_updates=320000`` is about 320000 / 0.4 = 800,000 vectorized
simulator steps plus the warmup, and environment transitions are that times
``env.num_envs``. In the logs, ``train/global_update`` counts updates, and
``sim_vector_steps`` and ``env_transitions`` count simulator steps and
transitions.

**Global sizes.** ``env.num_envs``, ``algorithm.wm_window_batch``, and
``algorithm.imagine_batch`` are totals over all ranks, split evenly;
``updates_per_env_step`` and ``total_updates`` also count global updates.
Adding GPUs without changing them keeps the same training mathematics; see
`Multiple GPUs`_.

Train on One GPU
----------------

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1 \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000 \
       'runner.logger.logger_backends=[tensorboard]'

``run_supermimic.sh <config> [hydra overrides...]`` passes everything after
the config name to Hydra. The command sets three keys explicitly because
their defaults do not suit a long run:

* ``runner.logger.log_path``: without it, the run goes to
  ``logs/<time>-<config>``, a new folder on every launch, which a resume must
  then find. With a fixed folder, resume and evaluation use it directly.
* ``runner.save_interval``: the default -1 saves only at the end. Choose an
  interval from the measured seconds per update (see
  `Performance and Hardware`_).
* ``runner.logger.logger_backends``: the default ``[]`` writes no curves.
  With ``[tensorboard]``, view them with
  ``tensorboard --logdir /data/runs/lafan1/tensorboard``.

For a quick check on a small GPU:

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
``train/param_divergence`` checks that ranks hold identical weights and
buffers every ``algorithm.param_check_interval`` updates. Running
normalization merges each rank's batch count, mean, and variance before
updating the shared statistics.

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

Multiple Nodes with Slurm (Experimental)
----------------------------------------

``examples/embodiment/slurm_supermimic.sh`` starts one Ray node per Slurm
node, exports a unique ``RLINF_NODE_RANK`` before each ``ray start``, and runs
the driver on the first node with a placement over every GPU. Partition,
account, and GPU count come from your ``sbatch`` options.

Before submitting:

* Run ``sbatch`` from the RLinf repository root with the SuperMimic
  environment activated. ``sbatch`` exports the current environment to the
  job, so every node uses that environment's ``python`` and ``ray``.
* Put the code, the virtual environment, the data, and ``log_path`` on storage
  every node can read and write; each rank writes its own checkpoint shard.
* Export site network settings such as ``NCCL_SOCKET_IFNAME`` before
  ``sbatch`` when your network needs them.

.. code-block:: bash

   sbatch --nodes=2 --gpus-per-node=8 --partition=<partition> \
       examples/embodiment/slurm_supermimic.sh supermimic_lafan1 \
       env.motion_path=/shared/lafan1_g1 runner.logger.log_path=/shared/runs/lafan1 \
       runner.save_interval=1000 'runner.logger.logger_backends=[tensorboard]'

This script has been checked with stubbed ``srun`` and ``ray`` only; it has not
run on a Slurm cluster.

Checkpoints and Resume
----------------------

``runner.save_interval`` writes ``log_path/<experiment_name>/global_step_<N>``
every N updates, and the last update is always written. A checkpoint holds
``supermimic.pt`` (networks, optimizers, schedules, counters, reset-bin
statistics), one ``rank_<r>.pt`` per rank (RNG streams), and
``checkpoint_meta.json`` (file sizes and SHA-256, world size, model
dimensions, goal window, format version). Files are written to a ``.tmp``
directory and renamed when every rank is done, so an interrupted save leaves
no partial checkpoint.

To resume, launch with the **same** config name and ``log_path`` as the
original run, for example after the command in `Train on One GPU`_:

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1 \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000 \
       'runner.logger.logger_backends=[tensorboard]' \
       runner.resume_dir=auto

``runner.resume_dir=auto`` picks the newest complete checkpoint under
``log_path/<experiment_name>/`` and trains on to ``algorithm.total_updates``;
a ``global_step_<N>`` path picks that checkpoint. When ``auto`` finds no
complete checkpoint it does **not** fail: it logs ``resume_dir=auto: no
complete checkpoint under ..., starting from scratch`` and trains from the
beginning, so check the log for ``Resuming SuperMimic ... from
.../global_step_<N>``.

A checkpoint of another format, dimension, or goal window is refused. A
different number of ranks loads the weights and optimizers and restarts the
RNG streams. The replay and simulator are not saved, so a resumed run warms up
a new replay and is a continuation, not an exact restore.

Evaluate
--------

``evaluations/eval_supermimic.py`` loads the networks from a checkpoint and
evaluates on one GPU without Ray. ``evaluations/run_eval.sh`` runs it for any
config name that starts with ``supermimic_``, so
``run_eval.sh supermimic_lafan1_eval`` and
``run_eval.sh supermimic supermimic_lafan1_eval`` are the same.
``supermimic_push_recovery`` uses ``evaluations/eval_supermimic_push.py``
instead. Fields of
RLinf's generic eval configs such as ``rollout.model.model_path`` and
``runner.task_type: embodied_eval`` do not apply; SuperMimic takes the
checkpoint from ``eval.checkpoint``.

``eval.checkpoint`` is a ``global_step_<N>`` folder or a folder of
checkpoints such as ``log_path/<experiment_name>``, in which case the newest
complete one is used. Eval configs per training config:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Training config
     - Eval config (``evaluations/supermimic/``)
   * - ``supermimic_lafan1``, ``supermimic_smoke``
     - ``supermimic_lafan1_eval`` (smoke has the networks and goal of
       ``supermimic_lafan1``)
   * - ``supermimic_lafan1_anchor``
     - ``supermimic_lafan1_anchor_eval``
   * - ``supermimic_lafan1_goal730``
     - ``supermimic_lafan1_goal730_eval``
   * - ``supermimic_lafan1_obs99``
     - ``supermimic_lafan1_obs99_eval``
   * - ``supermimic_lafan1_zita``
     - ``supermimic_lafan1_zita_eval``
   * - ``supermimic_velocity``
     - ``supermimic_velocity_eval``
   * - ``supermimic_bones_seed``, ``supermimic_bones_seed_2gpu``
     - ``supermimic_bones_seed_eval`` (the two-GPU recipe changes only sizes
       and the learning rate)

A checkpoint that does not match the eval config (dimensions or goal window)
is refused.

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       eval.checkpoint=/data/runs/lafan1/supermimic_lafan1 \
       eval.output_dir=/data/runs/lafan1/eval
   bash evaluations/run_eval.sh supermimic_velocity_eval \
       eval.checkpoint=/data/runs/velocity/supermimic_velocity \
       eval.output_dir=/data/runs/velocity/eval

**Where results go.** ``run_eval.sh`` writes its run log to
``logs/<time>-<eval config>/eval_embodiment.log`` in the repository. Results go
to ``eval.output_dir``, or, when it is not set, to ``eval_global_step_<N>/``
in that log folder. The results folder holds:

* ``summary.json``: ``meta`` (checkpoint path, ``global_step``, seed, device,
  clip count, data path, the checkpoint's world size) and ``metrics`` (the
  aggregates below; a value that does not exist is ``null``).
* ``per_clip.csv``: one row per clip.
* ``velocity_g2.csv`` or ``velocity_smoke.csv``: one row per command in a
  velocity evaluation.
* ``config.yaml``: the full config the evaluation used.

``eval/*``, ``eval_velocity_g2/*``, and ``eval_velocity_smoke/*`` are metric
name prefixes in ``summary.json`` and the logs, not folders.

**Motion tracking** plays every clip once from frame 0 with sensor and action
noise off and writes ``per_clip.csv``. The normalized return of clip :math:`i`
is

.. math::

   \mathrm{NR}_i = \frac{\sum_{t} r_t}{0.1 \times T_i},

where :math:`r_t` is the reward of each control step that ran, 0.1 is the
largest reward of one step, and :math:`T_i` is the clip's frame count on the
50 Hz control grid (its scheduled control steps). The denominator uses the
scheduled length, not the steps that ran, so an early failure keeps the
unplayed part in the denominator and :math:`\mathrm{NR}_i \le 1`.
``eval/normalized_return`` is the unweighted mean over clips.

**Velocity tracking** is chosen with ``eval.velocity``:

* ``g2`` (the velocity eval config's default) runs the Appendix G.2 step
  test: 672 commands on a 15 x 15 grid in each of the (vx, vy), (vx, wz), and
  (vy, wz) planes without the standing command, four trials each, 5 s from
  mjlab's standing pose. The value of a trial is the norm of the mean velocity
  error over the last 2 s divided by the command norm; a command reports the
  median over the trials that did not fall. A command whose trials all fell
  has no value (NaN in the CSV, ``null`` in ``summary.json``) and is counted
  in ``eval_velocity_g2/commands_all_trials_fell``. Trials differ in the sim
  context and the initial joint noise, which the paper does not specify.
  Results go to ``velocity_g2.csv``, with aggregates under the
  ``eval_velocity_g2/`` prefix.
* ``smoke`` holds seven fixed commands from a reference start, writes
  ``velocity_smoke.csv``, and reports aggregates under the
  ``eval_velocity_smoke/`` prefix. It checks that the velocity encoder runs;
  it is not a paper metric.

**Stationary standing-pose push recovery.** Paper §4.3 holds a static
T-pose. This entry is a different protocol and does not claim to reproduce
that T-pose. ``evaluations/run_eval.sh supermimic_push_recovery`` runs
``evaluations/eval_supermimic_push.py`` on one GPU without Ray. It loads
an existing tracking checkpoint and does not train. The reference stays
one standing pose for the whole trial, so the goal does not switch to a
walk.

The paper specifies a static T-pose in simulation and velocity pushes of
0.5–4 m/s. The critic policy settles in 0.5–2.4 s; the critic-free policy
keeps stumbling at 0.5 m/s and falls at 4 m/s. It does not publish the
T-pose joint angles, the speed grid, the direction, the frame, the trial
count, the pre-push wait, or a numerical definition of "settles". This
implementation holds mjlab's knees-bent keyframe, ``KNEES_BENT_KEYFRAME``,
the same pose as ``q0`` and ``reset_standing``. Forward kinematics puts
each wrist about 0.10 m forward, 0.07 m outward, and 0.31 m below its
``shoulder_roll_link``. Shoulder roll is ±0.2 rad and each elbow is 0.6
rad, so both arms hang beside the torso. The asset has only
``HOME_KEYFRAME`` and ``KNEES_BENT_KEYFRAME``, and the MJCF has no
keyframe. With every joint at zero the forearm points about 0.20 m forward
and 0.19 m below the shoulder, with almost no lateral offset, so that rest
pose is not a T-pose either. Because the paper does not publish the joint
angles, this code does not invent a horizontal-arm pose and call it the
paper pose. The default speeds are 0.5, 1, 2, and 4 m/s, four trials each;
that grid is this implementation's choice. ``eval.direction=random`` draws
a horizontal direction, or pass ``eval.direction=[1.0,0.0]``.
``eval.frame`` is ``world`` or ``heading``. ``eval.push_mode=add`` adds
that horizontal velocity to the current world-frame root linear velocity;
``set`` replaces xy and keeps z. The write goes into MuJoCo Warp's root
velocity and ``forward`` refreshes derived state without integrating. The
next ``step`` integrates it. The write does not change root position,
angular velocity, or joint state.

A fall is the training ``failure_signals`` test against this static
reference: height error above 0.25 m or tilt error above 0.8. The settle
test is this implementation's. After the push the root's horizontal speed
must stay within ``eval.speed_threshold`` (default 0.15 m/s), the root tilt
within ``eval.tilt_threshold`` (default 0.35), and the torso orientation
error within ``eval.torso_ori_threshold`` (default 0.5 rad), for
``eval.stable_seconds`` (default 0.5 s) without a break.
``eval.max_xy_drift`` defaults to ``null`` and does not require a return to
the original horizontal position. ``recovery_time_s`` runs from the push to
the control step that completes the stable window;
``stable_onset_time_s`` is that window's first frame. A trial that does not
recover leaves the time blank rather than zero. A fall, a joint-limit
violation, or a non-finite state before the push is
``initialization_failed`` and is left out of the post-push fall rate.
``eval.settle_seconds`` (default 1 s) and ``eval.max_seconds`` (default
5 s) are this implementation's wait and cutoff, not the paper's 0.5–2.4 s.

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_push_recovery \
       eval.checkpoint=/data/runs/lafan1/supermimic_lafan1 \
       eval.output_dir=/data/runs/lafan1/push_recovery

``summary.json`` reports, per speed, the trial count, recovery rate, fall
rate, timeout rate, and the mean and median recovery time of the trials
that recovered. A speed with no recovery has a ``null`` recovery time.
``per_trial.csv`` has one row per trial, including the direction, the
measured velocity change, the outcome, and the reason.
``eval.save_trajectory=true`` also writes ``trajectory.csv``. An anchor,
730-d goal, or 99-d observation checkpoint needs matching ``algorithm``
overrides on the command line.

On an 8 GB RTX 4060, a checkpoint from 3 updates was run at 0.5 m/s and
1 m/s, one trial each, with 0.4 s before the push and at most 0.8 s after
it, world +x. Both trials were standing when pushed. The measured
horizontal speed changes were 0.5 m/s and 1.0 m/s. Both then fell under
the training fall test, and the recovery times were empty. That shows the
evaluator runs on the knees-bent standing pose. It does not show recovery
under the paper's T-pose protocol.

``runner.only_eval=true runner.resume_dir=<checkpoint>`` evaluates through
the training entry instead, using the training config's environment count;
the results go to the logs only. With ``runner.val_check_interval`` set,
training evaluates every that many updates and after the last one. Motion
and velocity evaluation can use that path. Push recovery uses only the
entry above.

Visualization and Results
-------------------------

Logging goes through RLinf's ``MetricLogger`` (TensorBoard, WandB, SwanLab).
The default ``runner.logger.logger_backends: []`` writes no backend. With
``[tensorboard]``, the event files and the run's ``config.yaml`` go to
``log_path/tensorboard/``:

.. code-block:: bash

   tensorboard --logdir /data/runs/lafan1/tensorboard

With several ranks, counts are summed, timers and memory take the largest
rank, shared counters come from rank 0, and losses are averaged;
``runner.log_per_rank_metrics=true`` adds ``rank_<r>/`` copies. Useful keys:

* ``train/global_update``, ``sim_vector_steps``, ``env_transitions``: updates,
  vectorized simulator steps, and environment transitions.
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

The console also prints a ``supermimic metrics {...}`` line per update and a
``supermimic eval {...}`` line per evaluation, so values are visible even with
no backend enabled.

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
* **A BONES-SEED CSV is rejected.** The header needs ``Frame``,
  ``root_translateX/Y/Z``, ``root_rotateX/Y/Z``, and 29 ``<joint>_dof``
  columns. Root position is in centimeters and angles are in degrees. The
  message names the file and the missing column.
* **A resumed run starts at update 0.** The log shows ``resume_dir=auto: no
  complete checkpoint under ...``: the ``log_path`` or config name differs
  from the original run, or the original run never wrote a checkpoint
  (``save_interval`` was -1 and it did not finish).
* **Checkpoint format or goal window refused.** The checkpoint was written by
  another version or config; it cannot be resumed or evaluated with this one.
* **No TensorBoard files.** ``runner.logger.logger_backends`` is still the
  default ``[]``.
* **"linesearch iterations limit reached".** A MuJoCo Warp solver warning that
  untrained policies trigger; it does not stop training.

Performance and Hardware
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

What Has Been Run
-----------------

On one 8 GB GPU with MuJoCo Warp and the LAFAN1 clips: short training of the
LAFAN1 configs (680-d, 689-d, 730-d, 99-d observation, velocity) at small
sizes, including resume after the driver was killed, motion and velocity
evaluation, and the G.2 step test. Multi-GPU arithmetic is also tested with
two CPU (Gloo) processes.

A later acceptance run used two 80 GB A800 GPUs, the 40 LAFAN1 clips, and
local scheduling (not inside a Slurm job). The run reported below reused an
existing virtual environment, so its own stage 01 is not applicable. An
earlier run of the same commit had already passed a from-scratch ``install.sh``
install, and the training and evaluation below used that environment. What
passed:

* Unit tests: 10 files, 201 passed, 1 skipped because the machine does not
  have 4 accelerators.
* One-GPU MuJoCo Warp smoke training: 3 updates, monotonic counters, finite
  metrics, and a complete checkpoint.
* Checkpoint resume after the driver was killed, on one GPU and on two. A
  two-GPU checkpoint continued on one GPU. After one rank was killed, the
  remaining processes exited nonzero within 20.3 s and left no orphan
  processes.
* Motion evaluation of all 40 clips. Repeating it on the same checkpoint
  stayed within tolerance (absolute :math:`5\times10^{-5}`, relative
  :math:`2\times10^{-4}`) but was not bitwise identical. ``only_eval`` and
  velocity evaluation passed. Velocity evaluation used the Appendix G.2
  command grid with 2 trials per command, not the paper's 4.
* One-GPU profiling measured throughput and memory at 128 and 512
  environments. 4096 environments was only a linear estimate from the 512
  measurement, not a measurement.
* A small two-GPU run on real NCCL: ``train/param_divergence`` stayed 0,
  the two ranks' counters matched term by term, and both rank checkpoints
  were complete. The global sizes were 64 environments and 256 imagined
  rollouts, not the paper's two-GPU recipe. A short velocity run at the
  same width also finished.
* A 12-update learning-signal run at that same two-GPU size. Metrics stayed
  finite and counters stayed monotonic. Normalized return moved from
  0.001892 to 0.001893 and the failure rate stayed at 1. One seed and a
  short budget do not show that the policy learned to track.
* One full ``supermimic_lafan1_zita`` training run: one 80 GB A800, 4096
  environments, 16000 updates. The first compilation took about 30
  minutes, and the 16000 updates took about 4.5 hours. Motion evaluation
  of the update-16000 checkpoint on all 40 clips gave
  ``eval/normalized_return`` 0.856, ``eval/fail_rate`` 0, and
  ``eval/clip_end_rate`` 1. The update-14000 checkpoint on the same 40
  clips gave ``eval/normalized_return`` 0.800, ``eval/fail_rate`` 0.10,
  and ``eval/clip_end_rate`` 0.90. zita-ch reports 0.829 at update 14000,
  so this run is 0.029 lower at the same update. The simulator was
  mjlab. This page still has no learning curve for that run.

Not yet run: the paper config ``supermimic_lafan1`` (680-d goal, 320000
updates), the paper's two-GPU recipe, multiple nodes, Slurm, real
BONES-SEED data, and paper §4.3 T-pose recovery (this entry is a standing
pose and does not stand in for that result).
