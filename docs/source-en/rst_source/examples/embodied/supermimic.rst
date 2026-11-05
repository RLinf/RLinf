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
| **Prerequisites:** an NVIDIA GPU, or the experimental AMD MI300X path
  (see `Installation`_); the RLinf repository
  (:doc:`installation <../../start/installation>`); LAFAN1 G1 clips (see
  `Prepare Reference Motion`_).

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Item
     - Description
   * - Simulator
     - MuJoCo Warp through `mjlab <https://github.com/mujocolab/mjlab>`_
       (NVIDIA commit ``8bb2bc9``; AMD uses 1.2.0), G1 29-dof,
       50 Hz control with four 5 ms physics steps.
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

The default NVIDIA path runs simulation through MuJoCo Warp. From the RLinf
repository root, create a dedicated virtual environment with RLinf's installer:

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

In every new shell, return to the repository root and run
``source .venv-supermimic/bin/activate`` first.

The NVIDIA installer selects torch 2.14.0 for this environment only, because
its pinned mjlab requires ``torch>=2.14.0``; RLinf's project-wide torch pin is unchanged. torch
2.14.0 requires ``setuptools>=77.0.3``, so for this environment the installer
also replaces the project-wide ``setuptools<75.9`` override with
``setuptools>=77.0.3,<82`` (setuptools 82 removes ``pkg_resources``). When
the installed torch is a CUDA 12 build, it also installs the CUDA 12 build of
``warp-lang`` 1.18.0 (the PyPI wheel is a CUDA 13 build). The script selects
an ``x86_64`` or ``aarch64`` wheel for the host architecture, treating
``arm64`` as ``aarch64``. The SuperMimic packages are listed in ``requirements/embodied/envs/supermimic.txt``.
Installing that file with ``pip`` into an existing RLinf environment upgrades
its torch, so keep SuperMimic in its own environment. ``docker/Dockerfile``
has an ``embodied-supermimic-image`` target that runs the same install.

The G1 model, joint order, actuator gains, and the standing keyframe come from
mjlab. Every reference clip carries the default pose ``q0`` it was converted
with, and the simulator refuses clips whose ``q0`` differs from mjlab's
default joint pose, because the goal and observation subtract ``q0`` while the
PD target adds mjlab's.

AMD MI300X (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~

The AMD path uses ROCm Warp. Prepare Linux x86_64, the ROCm 7.2 SDK
(including ``hipcc``), and a C++ build toolchain. See the
`ROCm Warp gfx942 Docker build steps <https://github.com/ROCm/warp/blob/36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71/docker/rocm/Dockerfile>`_
for an environment recipe. From the repository root, run:

.. code-block:: bash

   export ROCM_PATH=/opt/rocm  # Adjust to the installed SDK path
   bash requirements/install.sh --platform amd --rocm 7.2 --no-flash-attn \
       embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

The script installs torch 2.9.1, torchvision 0.24.0 and its Triton dependencies
from AMD's Radeon flat wheel index; use a recent uv that supports flat indexes.
It builds ROCm Warp ``1.13.0+rocm.0`` at commit
``36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71``, then installs mjlab 1.2.0,
MuJoCo 3.8.1 and mujoco-warp 3.8.1 from
``requirements/embodied/envs/supermimic_amd.txt``. This path uses separate
dependency pins to retain the ROCm Warp build.

``ROCM_PATH`` points to the ROCm SDK and defaults to ``/opt/rocm``.
Warp's source checkout defaults to ``../warp-rocm`` beside the repository;
set ``SUPERMIMIC_WARP_PATH`` to reuse another checkout. Installation checks
out the commit above and rebuilds it. ``SUPERMIMIC_HIP_ARCH`` selects the
compile target and defaults to MI300X's ``gfx942``. After installation, use
the same data preparation, training and evaluation entries below. Select
the training GPU with ``cluster.component_placement.supermimic``.

`ZJLi2013's PR #1767 feedback <https://github.com/RLinf/RLinf/pull/1767#issuecomment-6593698481>`_
reports two MI300X runs based on commit ``0a19b154`` with their two old-stack
compatibility edits. Both used ``supermimic_lafan1_zita``, seed 0 and the
40 LAFAN1 clips; the only config change was ``runner.save_interval=500``.
The reported checkpoints are:

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 22 22

   * - Run
     - Update 14000: clips without failure
     - Update 14000: normalized return
     - Update 16000: clips without failure
     - Update 16000: normalized return
   * - Run A
     - 39/40
     - 0.832
     - 38/40
     - 0.832
   * - Run B
     - 40/40
     - 0.851
     - 39/40
     - 0.844

Clips without failure counts reference motions that reached the end;
normalized return measures tracking return normalized by the full clip
length. They are different metrics. The commenter attributes the gap to
the A800 report to run-to-run variation. These two runs support similar
tracking performance, but do not establish strict cross-platform equivalence.
They evaluate motions included in training.

Run A took about 4 h 15 min for 16000 updates, with roughly 0.97 s per update
in steady state after update 600 and about 6 min for the first actor compile.
The commenter reports that ``torch.compile`` worked on ROCm and no training
scan fell back to eager. The A800 value of 0.856 cited in the comment belongs
to the PR's earlier update-16000 report; the optimized A800 run recorded
elsewhere on this page scored 0.850 in about 2 h. They are different runs.

These are third-party results for the old baseline with their compatibility
edits. The new AMD installer branch and the current derived-constant fix
have not been validated locally through a full AMD training run, and other
AMD models have not been validated. After installation, follow the data
preparation, training and evaluation steps below; these historical numbers
are not a performance guarantee for the current installer branch.

Prepare Reference Motion
------------------------

Motion data is not part of the repository. A clip folder holds ``clip_*.pt``
files (loaded in file-name order) and optionally a ``manifest.json``. The
format is in `Clip File Format`_.

Complete `Installation`_ and prepare data in the same SuperMimic virtual
environment. By default, the LAFAN1 converter reads G1's default joint pose
``q0`` through mjlab. To prepare data in a CPU RLinf environment with PyTorch
and NumPy, provide the pose with ``--q0-file`` as shown below. Copy the complete
output clip directory, including ``manifest.json``, to the training machine.

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

   source .venv-supermimic/bin/activate
   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1

For conversion on a machine without mjlab, first export the pose from the
training environment:

.. code-block:: python

   import torch
   from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state

   torch.save(asset_default_state()[7:36].clone(), "g1_q0.pt")

Copy ``g1_q0.pt`` to the data-preparation machine and pass it to the converter:

.. code-block:: bash

   python toolkits/supermimic/prepare_lafan1.py \
       --source-dir /data/lafan1_csv --out /data/lafan1_g1 \
       --q0-file ./g1_q0.pt --accept-license

The file must contain a floating-point Tensor of shape ``[29]`` with finite
joint angles in radians, in ``G1_JOINT_NAMES`` order. The converter loads it
on CPU and records its name and SHA-256 in ``manifest.json``. The simulator
still checks that the clips' pose matches the training asset's default pose.

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

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
       env.motion_format=bones_seed env.num_clips=null env.motion_path=$SUPERMIMIC_BONES_PATH \
       env.num_envs=8 env.replay_length=64 algorithm.total_updates=3 \
       algorithm.wm_only_updates=0 algorithm.wm_window_batch=64 algorithm.imagine_batch=8 runner.save_interval=3

These BONES-SEED overrides load every clip that passes the filters
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

Use the main recipe at a small scale to check training, saving, resuming and
evaluation. These overrides use 8 environments and 3 updates, keeping the
same policy interface as the full zita-aligned run. Checkpoints and results
share ``RUN_DIR``. This checks the workflow, not tracking quality.

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export RUN_DIR=/data/runs/smoke
   SMOKE_ARGS=(env.num_envs=8 env.num_clips=40 env.replay_length=64
       algorithm.total_updates=3 algorithm.wm_only_updates=0
       algorithm.wm_window_batch=64 algorithm.imagine_batch=8
       runner.logger.experiment_name=smoke runner.save_interval=1
       runner.logger.log_path=$RUN_DIR 'runner.logger.logger_backends=[tensorboard]')

   # Train, then resume from update 3 and continue to update 5.
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita "${SMOKE_ARGS[@]}"
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita "${SMOKE_ARGS[@]}" \
       runner.resume_dir=auto algorithm.total_updates=5

   # The small run keeps the main recipe's checkpoint interface.
   bash evaluations/run_eval.sh supermimic_lafan1_zita_eval env.num_envs=8 \
       eval.checkpoint=$RUN_DIR/smoke eval.output_dir=$RUN_DIR/eval
   tensorboard --logdir $RUN_DIR/tensorboard

The resume log should contain ``Resuming SuperMimic networks`` and
``global_step_3``. If it instead says ``no complete checkpoint``, check the
log path and whether the first run saved successfully. Checkpoints live in
``$RUN_DIR/smoke/global_step_<N>``, TensorBoard files in
``$RUN_DIR/tensorboard``, and evaluation tables in ``$RUN_DIR/eval``.
A policy after 3 to 5 updates is not expected to track motion.

With no arguments, ``run_supermimic.sh`` also selects the main recipe with
small batches, 8 environments, one clip and 3 updates. Passing the recipe
name explicitly uses its full configured sizes unless you override them.

Configurations
--------------

Three YAML files cover the main workflow. The training recipe contains the
complete parameters used for the reported 16,000-update LAFAN1 run; the
other two inherit it and override evaluation settings. Batch sizes are global.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Purpose
   * - ``examples/embodiment/config/supermimic_lafan1_zita.yaml``
     - Full zita-aligned training: 730-d goal, 99-d observation, 500 world-model-only updates, 64 warmup steps, 16,000 updates, saving every 2,000 updates.
   * - ``evaluations/supermimic/supermimic_lafan1_zita_eval.yaml``
     - Matching motion evaluation; velocity evaluation uses explicit overrides below.
   * - ``evaluations/supermimic/supermimic_push_recovery.yaml``
     - Standing-pose push evaluation. Explicit policy overrides preserve its original 680-d goal / 93-d observation checkpoint interface.

The main training recipe uses seed 0, the analytic termination head and
zita critic features. ``algorithm.termination_bias_init`` remains optional.
Small runs and multi-GPU launches use overrides rather than separate YAMLs.

To run the original paper-oriented settings, define the following Bash array
in the shell where you launch training or evaluation. This changes the goal,
observation, normalization and critic together; keep the same array when
evaluating its checkpoint.

.. code-block:: bash

   PAPER_ARGS=(
       algorithm.total_updates=320000 algorithm.warmup_sim_steps=40
       algorithm.wm_only_updates=0 algorithm.termination_head=learned
       algorithm.goal_encoding=paper algorithm.goal_frame_dim=68 algorithm.obs_dim=93
       algorithm.reward_anchor=torso algorithm.input_norm=none
       algorithm.residual_frame=heading algorithm.wm_state_loss=paper
       algorithm.critic_features=root4 algorithm.matmul=tf32 env.com_sampling=per_reset
       runner.save_interval=-1 runner.logger.experiment_name=paper
   )
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita "${PAPER_ARGS[@]}"

The following variants add overrides after ``"${PAPER_ARGS[@]}"``. Give
each run a distinct ``runner.logger.experiment_name``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variant
     - Additional overrides
   * - Torso pose anchor (689-d goal)
     - ``algorithm.goal_anchor=true``
   * - 730-d goal with 93-d observations
     - ``algorithm.goal_encoding=goal730 algorithm.goal_frame_dim=73``
   * - 99-d observations with a 680-d goal
     - ``algorithm.obs_dim=99``
   * - Motion and velocity training
     - ``algorithm.velocity_tracking=true``
   * - BONES-SEED
     - ``env.motion_format=bones_seed env.num_clips=null env.motion_path=$SUPERMIMIC_BONES_PATH``

For example, train the velocity encoder and evaluate the same interface:

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita "${PAPER_ARGS[@]}" \
       algorithm.velocity_tracking=true runner.logger.experiment_name=velocity \
       runner.logger.log_path=/data/runs/velocity runner.save_interval=2000
   bash evaluations/run_eval.sh supermimic_lafan1_zita_eval "${PAPER_ARGS[@]}" \
       algorithm.velocity_tracking=true eval.velocity=g2 \
       eval.checkpoint=/data/runs/velocity/velocity eval.output_dir=/data/runs/velocity/eval

When the next four keys are omitted, the code follows zita-ch: a pelvis
reward anchor, running input normalization, a body-frame residual, and a
state loss divided by the running residual standard deviation.
The paper overrides above select a separate recipe explicitly.

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

``env.com_sampling`` chooses when the torso CoM offset in :math:`\xi` is
drawn. ``per_env`` draws it once per environment, as zita-ch does;
``per_reset`` draws a new one on every reset. A new offset recomputes
MuJoCo Warp's model constants for every world, which was the largest part
of collection time. Friction and joint offsets are drawn on every reset
either way. The paper config writes ``per_reset``, ``supermimic_lafan1_zita``
writes ``per_env``, and an omitted key means ``per_env``.

The simulator stores the CoM offset and the derived constants used by
recomputation separately for each environment. When the CoM changes, each
environment uses its own constants. On versions that store the solver's
inertia scale as an array, ``stat.meaninertia`` also has one entry per
environment. These arrays are expanded before simulation graphs are recaptured.

``supermimic_lafan1_zita`` sets ``pelvis``, ``running``, ``body`` and
``running`` in the complete training recipe. The simulator stays mjlab. It
refuses a checkpoint saved before those four fields were recorded. Linear
layers run under bfloat16 autocast (``algorithm.matmul: bf16``);
integration, kinematics, the losses and the Adam weights stay float32.

In the optimized run before the review fixes, 16000 updates on one 80 GB
A800 with 4096 environments took 1 h 59 min
51 s, from the Ray cluster being ready to the metrics line of update 16000.
The same zita-ch code, measured on an A800 of the same model, took 2 h
15 min for 16000 updates. zita-ch's own report is update 14000 in 1.47 h
on an RTX PRO 6000. The compile cache from an earlier start on the same
machine was present (see `Performance and Hardware`_), and the run reached
update 100 in 110 s. The first 500 updates train only the world model, at
about 26 s per 100 updates; after that each 100 updates took 43-47 s,
44.9 s on average. That timing run used this launch:

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   mkdir -p /data/runs/lafan1_zita
   PYTHONUNBUFFERED=1 bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
       cluster.component_placement.supermimic=0 \
       runner.logger.log_path=/data/runs/lafan1_zita \
       'runner.logger.logger_backends=[tensorboard]' \
       > /data/runs/lafan1_zita/console.log 2>&1

* ``cluster.component_placement.supermimic`` picks the GPU, 0 by default.
  Keep other jobs off that GPU while timing. If a Ray cluster is already
  running on the machine, point ``RAY_ADDRESS`` at it.
* ``> console.log 2>&1``: every 100 updates the console prints one
  timestamped ``supermimic metrics`` line. Time the run from the first
  ``[INFO ... RLinf]`` line to the line with
  ``'train/global_update': 16000.0``. ``PYTHONUNBUFFERED=1`` writes each
  line to the file at once, so progress is visible during the run.
* Do not set ``SUPERMIMIC_SCAN_EAGER``, ``SUPERMIMIC_FK_EAGER`` or
  ``SUPERMIMIC_PROFILE``. The first two turn off the training compile, and
  the third syncs the GPU between pieces. Each makes training much slower.
* On a new machine, or after ``/tmp`` is cleared, the first start compiles
  for a few minutes first, and the total grows by that much.

The update-16000 checkpoint of this run scored ``eval/normalized_return``
0.850 on the 40 LAFAN1 clips, and the update-14000 checkpoint scored
0.828. zita-ch reports 0.829 at update 14000. An earlier run used the code
from before the speed work, drew the CoM offset on every reset, compiled
for about 30 minutes and trained for about 4.5 hours.

Evaluate with ``supermimic_lafan1_zita_eval`` on one GPU, without Ray.
Point ``eval.checkpoint`` at the ``supermimic_lafan1_zita`` directory under
that ``log_path`` to use the newest complete checkpoint, or at one
``global_step_<N>`` directory. The config saves every 2000 updates, so
update 14000, the one zita-ch reports, has a checkpoint. To score update
14000, point the checkpoint at that save:

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_zita_eval \
       eval.checkpoint=/data/runs/lafan1_zita/supermimic_lafan1_zita/global_step_14000 \
       eval.output_dir=/data/runs/lafan1_zita/eval_14000

Common Keys
~~~~~~~~~~~

These are the keys most often changed to set the scale and length of a run.
Defaults come from ``supermimic_lafan1_zita.yaml``; the paper overrides above change some of them. Each can be passed as a Hydra override after
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
     - 16000
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
     - 64
     - Vectorized simulator steps collected after each start or resume,
       before the first update, to warm the replay; they owe no updates.
   * - ``algorithm.matmul``
     - ``bf16``
     - ``tf32`` runs float32 matmuls on TF32 tensor cores. ``bf16`` also
       puts each network's linear layers under bfloat16 autocast;
       integration, kinematics, the losses and the Adam weights stay
       float32. ``supermimic_lafan1_zita`` writes ``bf16``. TF32 is a
       process-wide switch and is on in both modes.
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
     - 2000
     - Save a checkpoint every N updates. Set -1 to save only after the last update.
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
run. So ``total_updates=16000`` is about 16000 / 0.4 = 40,000 vectorized
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

Select the training GPU with ``cluster.component_placement.supermimic``.
The default is GPU 0 in RLinf's device enumeration. The command below uses
that default; append ``cluster.component_placement.supermimic=2`` to use
GPU 2 instead.

RLinf sets device visibility for training Workers from the placement, so it
may replace a ``CUDA_VISIBLE_DEVICES`` or ``HIP_VISIBLE_DEVICES`` value set
before launch. Put the device selection in the placement. When a Worker is
isolated to one visible GPU, that device is numbered ``cuda:0`` inside the
process: a Worker assigned node GPU 2 can still report ``cuda:0``. Check the
placement together with node-level GPU monitoring to identify the actual
device in use.

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000 \
       'runner.logger.logger_backends=[tensorboard]'

``run_supermimic.sh <config> [hydra overrides...]`` passes everything after
the config name to Hydra. The command sets three keys explicitly to select a fixed log directory, a checkpoint interval and curves:

* ``runner.logger.log_path``: without it, the run goes to
  ``logs/<time>-<config>``, a new folder on every launch, which a resume must
  then find. With a fixed folder, resume and evaluation use it directly.
* ``runner.save_interval``: the main recipe saves every 2000 updates. Choose an
  interval from the measured seconds per update (see
  `Performance and Hardware`_).
* ``runner.logger.logger_backends``: the default ``[]`` writes no curves.
  With ``[tensorboard]``, view them with
  ``tensorboard --logdir /data/runs/lafan1/tensorboard``.

For a quick check on a small GPU:

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh

Only the no-argument smoke launch sets ``RAY_memory_usage_threshold=0.99`` when
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

     bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
         cluster.component_placement.supermimic=0-3

* **The paper's two-GPU recipe.** Each GPU keeps the one-GPU sizes, so the
  global batch doubles, gradients are averaged over both GPUs, and the
  learning rate is :math:`2\sqrt{2} \times 10^{-4}` (Table A.5):

  .. code-block:: bash

     export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered
     bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita "${PAPER_ARGS[@]}" \
         env.motion_format=bones_seed env.num_clips=null env.motion_path=$SUPERMIMIC_BONES_PATH \
         cluster.component_placement.supermimic=0-1 env.num_envs=8192 \
         algorithm.wm_window_batch=4096 algorithm.imagine_batch=8192 algorithm.lr=2.8284271e-4

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
       examples/embodiment/slurm_supermimic.sh supermimic_lafan1_zita \
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

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
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
``run_eval.sh supermimic_lafan1_zita_eval`` and
``run_eval.sh supermimic supermimic_lafan1_zita_eval`` are the same.
``supermimic_push_recovery`` uses ``evaluations/eval_supermimic_push.py``
instead. Fields of
RLinf's generic eval configs such as ``rollout.model.model_path`` and
``runner.task_type: embodied_eval`` do not apply; SuperMimic takes the
checkpoint from ``eval.checkpoint``.

Motion, velocity and push-recovery evaluation stop with an error if an
active rollout's policy produces NaN/Inf actions. The error lists the active
row indices in the current batch; finished rows remain masked. For
``non-finite policy actions in active evaluation rows``, first check that
the checkpoint matches the evaluation config, then inspect policy outputs
and input data for numerical problems.

Motion-tracking and velocity evaluation also check active simulation and
reference states at initialization and after each step. Non-finite values
raise ``non-finite evaluation states in active rows`` before reward or metric
calculation; finished rows remain masked.

``eval.checkpoint`` is a ``global_step_<N>`` folder or a folder of
checkpoints such as ``log_path/<experiment_name>``, in which case the newest
complete one is used. Eval configs per training config:

Use ``supermimic_lafan1_zita_eval`` for the main training recipe and its small runs. For any variant, pass the same model-interface overrides used in training, such as ``"${PAPER_ARGS[@]}"`` and ``algorithm.velocity_tracking=true``. Dimensions, goal encoding, normalization, residual frame and critic features must match the checkpoint.

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_zita_eval \
       eval.checkpoint=/data/runs/lafan1_zita/supermimic_lafan1_zita \
       eval.output_dir=/data/runs/lafan1_zita/eval

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
       eval.checkpoint=/data/runs/lafan1/supermimic_lafan1_zita \
       eval.output_dir=/data/runs/lafan1/push_recovery

For a checkpoint from the main zita-aligned recipe, also pass all policy overrides below. The push evaluator otherwise keeps its original paper interface.

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_push_recovery \
       algorithm.goal_encoding=goal730 algorithm.goal_frame_dim=73 algorithm.obs_dim=99 \
       algorithm.reward_anchor=pelvis algorithm.input_norm=running \
       algorithm.residual_frame=body algorithm.wm_state_loss=running algorithm.critic_features=zita \
       eval.checkpoint=/data/runs/lafan1_zita/supermimic_lafan1_zita \
       eval.output_dir=/data/runs/lafan1_zita/push_recovery

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
* ``time/sim_s``, ``time/replay_sample_s``, ``time/world_model_s``,
  ``time/bptt_s``, ``time/critic_s``: host time of each piece. The GPU runs
  asynchronously, so without ``SUPERMIMIC_PROFILE=1`` these only show
  kernel launches and whatever syncs happen along the way; do not add them
  up as the time per update. Read the time per update from the console
  timestamps.
* ``memory/torch_peak_allocated``, ``memory/torch_peak_reserved``
  (PyTorch's allocator), ``memory/warp_mempool_peak`` (Warp's pool), and
  ``memory/device_used`` (the whole GPU as the driver sees it). These counters
  overlap; do not add them.

TensorBoard records every update. The console prints a
``supermimic metrics {...}`` line every 100 updates, and also the last
update when it is not a multiple of 100; update 0 goes to TensorBoard only.
Each evaluation prints a ``supermimic eval {...}`` line. Values are visible
on the console even with no backend enabled.

Troubleshooting
---------------

* **CUDA out of memory.** The error names the rank and its per-rank sizes.
  Add GPUs to keep the batch sizes, or reduce ``env.num_envs``,
  ``env.replay_length``, or the training batch sizes.
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
* **The log stops at "Compiling a SuperMimic training scan".** The first
  large batch is waiting for the compile, a few minutes with a cold cache;
  see `Performance and Hardware`_. Do not delete the inductor cache
  directory during a run.
* **"SuperMimic scan <name> stayed eager".** That piece failed to compile
  and runs eagerly from then on. Results are unchanged; training is slower.
* **Solver iteration caps.** The mjlab adapter caps the MuJoCo Warp solver
  at ``iterations=10`` and ``ls_iterations=20`` and turns off the
  one-line-per-world warning when a cap is hit. The caps themselves are
  unchanged.

Performance and Hardware
------------------------

The following measurements describe earlier training runs. Compare the console timestamps and the built-in memory metrics on your hardware when choosing a training scale.

**The zita run on an A800.** One 80 GB A800, one GPU,
``supermimic_lafan1_zita`` (4096 environments, ``algorithm.matmul: bf16``),
warm compile cache:

* The first 500 updates train only the world model, at about 26 s per 100
  updates. After that, 43-47 s per 100 updates, 44.9 s on average, or
  about 0.45 s per update. 16000 updates took 1 h 59 min 51 s. The same
  zita-ch code, measured on an A800 of the same model, took 2 h 15 min.
* ``eval/normalized_return`` is 0.850 at update 16000 and 0.828 at update
  14000. zita-ch reports 0.829 at update 14000.
* PyTorch peak allocated 17.1 GB (18.8 GB reserved), whole GPU as the
  driver sees it 20.1 GB, Warp pool peak 0.48 GB, host RSS about 4.0 GB.
* At the end: 40064 vectorized simulator steps, about 164 million
  environment transitions.

**Compilation and cache.** CUDA work on large batches (at least 256 rows)
is compiled with ``torch.compile`` into seven graphs: goal, post-processing
and bootstrap during collection, one each; world-model fitting and the
imagined rollout, each split into a step graph called from a Python loop
and a score graph run once over every step after the loop. The critic
update, sampling and the optimizers are not compiled. Each trainer keeps its
own in-process compilation cache and reuses it across updates. The compiled kernels
go to PyTorch's default inductor cache, ``/tmp/torchinductor_<user>``. A
new process on the same machine still retraces the graphs but reads the
kernels from disk; the run above reached update 100 in 110 s. On a new
machine, or after ``/tmp`` is cleared, compilation starts over and takes a
few minutes with a cold cache. ``SUPERMIMIC_SCAN_EAGER=1`` and
``SUPERMIMIC_FK_EAGER=1`` turn off the compile of the training scans and
of forward kinematics. Results are unchanged. They are meant for unit tests
on a GPU, where they cut tens of minutes to a few; do not set them for
training.

**Per-piece timing.** ``SUPERMIMIC_PROFILE=1`` syncs the GPU around each
piece, so the ``time/*`` metrics become the real time of each piece, and
training slows down. Workers take their environment from the Ray node, so
with a running Ray cluster set it before ``ray start``.

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
* A full ``supermimic_lafan1_zita`` run before the review fixes: one 80 GB
  A800, 4096 environments, 16000 updates, warm compile cache, 1 h 59 min
  51 s from the Ray cluster being ready to update 16000. The same zita-ch
  code, measured on an A800 of the same model, took 2 h 15 min for 16000
  updates. About 0.45 s per update in steady state, 17.1 GB PyTorch peak
  allocated. ``eval/normalized_return`` is 0.850 at update 16000 and 0.828
  at update 14000. zita-ch reports 0.829 at update 14000.
* An earlier full ``supermimic_lafan1_zita`` run, with the code from before
  the speed work and the CoM offset drawn on every reset: the first
  compilation took about 30 minutes, and the 16000 updates took about
  4.5 hours. The simulator was mjlab. This page still has no learning
  curve for either run.

Not yet run: the documented paper overrides (680-d goal, 320000
updates), the paper's two-GPU recipe, multiple nodes, Slurm, real
BONES-SEED data, and paper §4.3 T-pose recovery (this entry is a standing
pose and does not stand in for that result).
