SuperMimic: G1 Motion Tracking with a Differentiable World Model
================================================================

This guide trains and evaluates a Unitree G1 whole-body motion-tracking
policy with SuperMimic in MuJoCo Warp: install the environment, prepare
reference motion, choose GenCoord or MaxCoord, train on one or more GPUs,
resume, and evaluate. The algorithm and where it departs from the paper are
in :doc:`../../reference/algorithms/supermimic`. On a first run, follow
`Run It`_.

Unless stated otherwise, every command on this page runs from the RLinf
repository root with the SuperMimic environment below activated.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      SuperMimic: 32-step backpropagation through a world model, plus a GAE critic

   .. grid-item-card:: Model
      :text-align: center

      World model, motion policy, and critic; Muon on hidden matrices, fused Adam on the rest

   .. grid-item-card:: Environment
      :text-align: center

      MuJoCo Warp G1 (mjlab); LAFAN1 and BONES-SEED

   .. grid-item-card:: Training
      :text-align: center

      One env, rollout, and FSDP actor worker on each GPU

| **You'll do:** install a dedicated environment -> prepare the LAFAN1 clips
  -> run a small train, resume, and eval -> train GenCoord or MaxCoord.
| **Prerequisites:** an NVIDIA GPU, or the experimental AMD MI300X path
  (see `Installation`_); the RLinf repository
  (:doc:`installation <../../start/installation>`); LAFAN1 G1 clips
  (see `Prepare Reference Motion`_).

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Item
     - Contents
   * - Simulator
     - MuJoCo Warp through `mjlab <https://github.com/mujocolab/mjlab>`_
       (NVIDIA pinned at commit ``8bb2bc9``, AMD uses 1.2.0). G1 has 29
       degrees of freedom, the control rate is 50 Hz, and each control step
       is four 5 ms physics steps.
   * - Task
     - Track reference clips retargeted to the G1 (LAFAN1 or BONES-SEED).
   * - State
     - ``gencoord`` is the 71-d simulator state, encoded in 68 dimensions.
       ``maxcoord`` is the pose and twist of the 14 tracked bodies, encoded
       in 214 dimensions. A checkpoint loads only into the representation it
       was trained with.
   * - Action
     - 29 joint setpoints, followed by a low-gain PD controller. Training
       action noise has standard deviation 0.1.
   * - Goal
     - Ten reference frames, five control steps apart, held at the last frame
       of the clip. The actor goal is 640-d: joint positions, joint
       velocities, and a 6-d rotation of each frame's root relative to the
       current root. The world-model goal also appends the relative root
       position and is 670-d.
   * - Entry points
     - Train with ``examples/embodiment/train_embodied_agent.py``. Evaluate
       with ``evaluations/run_eval.sh supermimic_lafan1_eval``, which calls
       ``evaluations/eval_embodied_agent.py``.

How SuperMimic Works
--------------------

SuperMimic repeats three steps. It steps every environment once in MuJoCo
Warp and writes the transitions to the replay. At the update-to-data ratio
it fits the world model on ``k = 8`` step windows from the replay. It then
rolls the world model out for ``H = 32`` steps from replay states,
backpropagates the return through that model to update the actor, and fits
the critic to GAE targets. When an environment ends, returns stored in 0.2 s
bins along the motion decide where it restarts, so low-return segments are
drawn more often. Equations are on the
:doc:`algorithm page <../../reference/algorithms/supermimic>`.

In RLinf the three steps run on three worker groups, with one worker of each
group on every GPU:

* ``SuperMimicEnvWorker`` owns that GPU's environments, the reset sampler,
  and the reward.
* ``SuperMimicRolloutWorker`` holds a copy of the policy and the critic. It
  returns actions, and the critic value used to bootstrap a timed-out
  episode in the reset return.
* ``EmbodiedSuperMimicFSDPPolicy`` is the FSDP2 actor. It owns the replay,
  the world model, the actor, the critic, and their optimizers.

``SuperMimicRunner`` is the control plane. It decides how many control steps
to collect before the next update is owed, runs those steps, runs the owed
updates, and sends the new weights to the rollout workers. It also handles
logging, evaluation, and checkpoints.

Installation
------------

The default NVIDIA path runs simulation through MuJoCo Warp. From the RLinf
repository root, create a dedicated virtual environment:

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

In every new shell, return to the repository root and run
``source .venv-supermimic/bin/activate`` first.

The NVIDIA environment uses torch 2.14.0 and ``warp-lang`` 1.18.0
(``requirements/embodied/envs/supermimic.txt``). Keep it separate from other
RLinf environments because these requirements upgrade torch.
``docker/Dockerfile`` also provides an ``embodied-supermimic-image`` target.
The G1 model, joint order, actuator gains, and standing keyframe come from
mjlab.

AMD MI300X (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~

The AMD path uses ROCm Warp. Prepare Linux x86_64, the ROCm 7.2 SDK
(including ``hipcc``), and a C++ build toolchain. See the
`ROCm Warp gfx942 Docker build steps <https://github.com/ROCm/warp/blob/36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71/docker/rocm/Dockerfile>`_
for an environment recipe. From the repository root:

.. code-block:: bash

   export ROCM_PATH=/opt/rocm  # set this to the SDK install path
   bash requirements/install.sh --platform amd --rocm 7.2 --no-flash-attn \
       embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

The script installs torch 2.9.1 and torchvision 0.24.0 from AMD's Radeon
wheel index (use a recent uv that supports flat indexes), builds ROCm Warp
``1.13.0+rocm.0`` at the commit linked above, and installs mjlab 1.2.0,
MuJoCo 3.8.1, and mujoco-warp 3.8.1 from
``requirements/embodied/envs/supermimic_amd.txt``. ``SUPERMIMIC_WARP_PATH``
sets the Warp checkout (default ``../warp-rocm`` beside the repository) and
``SUPERMIMIC_HIP_ARCH`` the compile target (default ``gfx942``). Data
preparation, training, and evaluation then work as below.

The AMD installer has not been validated through a full training run. The
numbers under `Reported Tracking Results`_ are LAFAN1 checkpoint evaluations
on NVIDIA GPUs, not an AMD run.

Prepare Reference Motion
------------------------

Motion data is not part of the repository. A clip folder holds ``clip_*.pt``
files (loaded in file-name order) and optionally a ``manifest.json``. The
format is in `Clip File Format`_.

**LAFAN1 (the paper's 40 clips).** The clips are the ``g1/`` folder of the
Hugging Face dataset
`lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_.
The script pins revision ``ce1572906efe6157840e8474d5a0d7aa87481e74``, checks
every CSV's size and SHA-256, and writes ``manifest.json``. The dataset card
declares no license; the retargets derive from Ubisoft La Forge
`LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_,
released under CC BY-NC-ND 4.0 (non-commercial, no derivatives). Read those
terms before passing ``--accept-license``.

.. code-block:: bash

   source .venv-supermimic/bin/activate
   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1

The converter reads G1's default joint pose ``q0`` through mjlab. To convert
on a machine without mjlab, export the pose in the training environment and
pass it with ``--q0-file``:

.. code-block:: bash

   python -c "import torch; from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state; torch.save(asset_default_state()[7:36].clone(), 'g1_q0.pt')"
   python toolkits/supermimic/prepare_lafan1.py \
       --source-dir /data/lafan1_csv --out /data/lafan1_g1 \
       --q0-file ./g1_q0.pt --accept-license

Copy the whole output folder, including ``manifest.json``, to the training
machine. The 40 clips are 2.45 h of motion. With the manifest present, the
loader refuses a clip whose SHA-256 differs or that the manifest does not
list, so do not rename, add, remove, or edit the prepared files. The loader
resamples 30 Hz clips to 50 Hz.

**BONES-SEED (experimental).** The official release is
`bones-studio/seed <https://huggingface.co/datasets/bones-studio/seed>`_ on
Hugging Face, with G1 trajectories in ``g1/csv/{date}/{motion}.csv``. Read
``LICENSE.md`` in the download first; the script does not download the data.
It follows the public header in NVIDIA ProtoMotions'
`SEED G1 CSV preparation <https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst>`_:
120 Hz, ``root_translateX/Y/Z`` in centimeters, ``root_rotateX/Y/Z`` as
extrinsic XYZ Euler angles in degrees, and 29 ``<joint>_dof`` columns in
degrees named after the mjlab G1 joints. This convention has not been checked
against an official CSV.

.. code-block:: bash

   python toolkits/supermimic/convert_bones_seed.py \
       --input /data/bones-seed/g1/csv \
       --output /data/bones_g1
   python toolkits/supermimic/bones_feasibility.py --clips /data/bones_g1 \
       --out /data/bones_g1_filtered
   export SUPERMIMIC_MOTION_PATH=/data/bones_g1_filtered

``--input`` is one CSV or a directory scanned recursively. Clips are
resampled to 50 Hz, existing outputs are kept unless ``--overwrite`` is set,
and a broken CSV is reported without stopping the rest. To train, add the
BONES-SEED overrides from `Configurations`_. The converter is tested on
synthetic CSVs only.

Clip File Format
~~~~~~~~~~~~~~~~

Each ``clip_*.pt`` is a dictionary saved with ``torch.save``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Contents
   * - ``fps``
     - 50 or 30. A 30 Hz clip is resampled onto the 50 Hz control grid and
       its velocities are recomputed by central difference. A 50 Hz clip is
       used as stored, so **its velocity columns must be correct**. Other
       rates are refused.
   * - ``joint_names``
     - 29 joint names, in the same order as
       ``rlinf.envs.sim.g1_mimic.constants.G1_JOINT_NAMES`` (the mjlab G1
       order).
   * - ``q0``
     - The 29-d default joint pose in radians. It must equal mjlab's default,
       ``asset_default_state()[7:36]``.
   * - ``state``
     - A float tensor ``[T, 71]`` with the columns below. No NaN or Inf, and
       the root quaternion is normalized.
   * - ``name``
     - Optional. The BONES-SEED converter writes it.

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Columns
     - Contents
     - Unit
   * - ``0:3``
     - Root (pelvis) position, world frame
     - m
   * - ``3:7``
     - Root quaternion, wxyz
     - --
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

This example writes a 5 s clip of standing still and runs the loader's own
checks on it:

.. code-block:: python

   from pathlib import Path

   import torch

   from rlinf.envs.sim.g1_mimic.constants import G1_JOINT_NAMES
   from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
   from rlinf.envs.sim.g1_mimic.motion import decode_motion_payload

   default = asset_default_state().float()  # mjlab standing keyframe, [71]
   T = 250                                  # 5 s at 50 Hz
   state = torch.zeros(T, 71)
   state[:, :36] = default[:36]             # root pose and joint positions
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

Shrink the GenCoord recipe to 8 environments and 3 updates to check
training, saving, resuming, and evaluation. Checkpoints, logs, and results
share ``RUN_DIR``. This checks the workflow, not tracking quality.

.. code-block:: bash

   export PYTHONPATH=$PWD:$PYTHONPATH
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   export RUN_DIR=/data/runs/smoke
   SMOKE_ARGS=(env.train.total_num_envs=8 algorithm.replay_length=64
       algorithm.wm_window_batch=64 algorithm.imagine_batch=8
       runner.save_interval=1
       runner.logger.experiment_name=smoke runner.logger.log_path=$RUN_DIR)

   # Train 3 updates, then resume from update 3 and continue to update 5.
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord "${SMOKE_ARGS[@]}" runner.max_epochs=3
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord "${SMOKE_ARGS[@]}" runner.max_epochs=5 \
       runner.resume_dir=auto

   # Evaluate the last checkpoint. Action noise is off, as in the reported results.
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/smoke/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt \
       env.eval.action_noise_std=0 \
       runner.eval_export.output_dir=$RUN_DIR/eval
   tensorboard --logdir $RUN_DIR/tensorboard

The resume log should contain ``Resuming from`` and ``global_step_3``. If it
instead says ``resume_dir=auto found no complete checkpoint``, check
``runner.logger.log_path``, ``runner.logger.experiment_name``, and whether
the first run saved. Checkpoints live in
``$RUN_DIR/smoke/checkpoints/global_step_<N>``, TensorBoard files in
``$RUN_DIR/tensorboard``, and evaluation tables in ``$RUN_DIR/eval``. A
policy after 3 to 5 updates is not expected to track motion.

For MaxCoord, change ``--config-name`` to ``supermimic_lafan1_maxcoord``.
Evaluating a MaxCoord checkpoint also needs
``actor.model.representation=maxcoord``.

Configurations
--------------

Two training YAMLs differ only in the state representation. The two
evaluation configs inherit GenCoord and change evaluation settings.
Environment counts and batch sizes are global.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - Role
   * - ``examples/embodiment/config/supermimic_lafan1_gencoord.yaml``
     - GenCoord training. Defaults are 4096 environments, 40 clips,
       ``runner.max_epochs=400000``, and a checkpoint every 2000 updates.
       `Reported Tracking Results`_ overrides the length to 16000 updates.
   * - ``examples/embodiment/config/supermimic_lafan1_maxcoord.yaml``
     - Inherits the previous file and sets
       ``actor.model.representation`` to ``maxcoord``.
   * - ``evaluations/supermimic/supermimic_lafan1_eval.yaml``
     - Motion-tracking evaluation on the 40 clips.
   * - ``evaluations/supermimic/supermimic_push_recovery.yaml``
     - Pushes applied to the knees-bent standing pose. This is not the
       paper's Section 4.3 T-pose protocol.

The seed is 0. Network weights stay float32 and matmuls use TF32.
``actor.fsdp_config.amp_autocast.enabled`` defaults to ``False``. Setting it
to ``True`` casts matmuls inside the networks to ``precision`` (bf16 by
default); kinematics, rewards, losses, and optimizer state stay float32.
That is not the paper's TF32 setting. Hidden matrices use Muon
(``adjust_lr_fn: match_rms_adamw``); the remaining parameters use fused
Adam. The gradient norm stays on device for the clip, so an optimizer step
does not wait on the host.

``env.train.motion_on_cpu=true`` (``env.eval.motion_on_cpu`` for evaluation)
keeps a large library and its FK cache in host memory and moves only the
queried rows to the GPU. The actor worker holds a second copy for
imagination.

Append this to a BONES-SEED launch, and point ``SUPERMIMIC_MOTION_PATH`` at
the filtered directory:

.. code-block:: bash

   env.train.motion_format=bones_seed env.train.num_clips=null \
       env.eval.motion_format=bones_seed env.eval.num_clips=null

Common Keys
~~~~~~~~~~~

These are the keys most often changed to set the scale and length of a run.
Defaults come from ``supermimic_lafan1_gencoord.yaml``. Each can be passed
as a Hydra override.

.. list-table::
   :header-rows: 1
   :widths: 40 12 48

   * - Key
     - Default
     - Meaning
   * - ``runner.max_epochs``
     - 400000
     - Global updates. The reported run overrides this to 16000.
   * - ``runner.save_interval``
     - 2000
     - Write a checkpoint every this many updates, and after the last one.
   * - ``runner.val_check_interval``
     - -1
     - Do not evaluate during training. Evaluate with the commands in
       `Evaluate`_.
   * - ``env.train.total_num_envs``
     - 4096
     - Environments across all GPUs. Must divide evenly by the GPU count.
   * - ``env.train.num_clips``
     - 40
     - Clips to load. ``null`` loads every clip in the directory.
   * - ``algorithm.replay_ratio``
     - 16
     - How many times each transition is drawn by the world model on
       average. With the batches below this is 0.4 global updates per
       control step.
   * - ``algorithm.wm_batches``
     - 10
     - World-model batches in one update.
   * - ``algorithm.wm_window_batch``
     - 2048
     - Windows in one world-model batch, summed over GPUs.
   * - ``algorithm.imagine_batch``
     - 4096
     - Imagined rollouts, summed over GPUs.
   * - ``algorithm.k`` / ``algorithm.H``
     - 8 / 32
     - World-model window and imagination horizon.
   * - ``actor.optim.lr``
     - 2.0e-4
     - Learning rate shared by the three phases. The schedule is constant.
   * - ``actor.optim.clip_grad``
     - 1.0
     - Global gradient-norm cap. The actor then clamps each element to
       ``clip_grad_value`` 0.1.
   * - ``actor.optim.fused``
     - True
     - Fused Adam for parameters that are not updated by Muon. The world
       model and the critic use the same flag.
   * - ``actor.fsdp_config.amp_autocast.enabled``
     - False
     - When true, network matmuls use ``precision`` (bf16). Weights stay
       float32.
   * - ``env.train.action_noise_std``
     - 0.1
     - Training action noise. The reported evaluations set it to 0 on
       ``env.eval``.

**Updates versus simulator steps.** At the default scale each control step
owes 0.4 updates, so ``runner.max_epochs=16000`` is 40,000 control steps.
The global step counts updates and ``env/env_steps`` counts control steps.
``env.train.total_num_envs``, ``algorithm.wm_window_batch``, and
``algorithm.imagine_batch`` are totals over all GPUs and must divide evenly
by the GPU count. Adding GPUs without changing them keeps the same training
mathematics.

Train on One GPU
----------------

The recipe puts one env, one rollout, and one actor worker on every GPU
that RLinf sees (``cluster.component_placement`` maps ``actor,env,rollout``
to ``all``). On a machine with one GPU:

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.max_epochs=16000 \
       runner.logger.log_path=/data/runs/gencoord

MaxCoord:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_maxcoord \
       runner.max_epochs=16000 \
       runner.logger.log_path=/data/runs/maxcoord

Training does not render. To use one GPU of a larger machine, remove the
combined placement and point all three workers at the same device index
(index 0 below):

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.max_epochs=16000 \
       runner.logger.log_path=/data/runs/gencoord \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor:0,env:0,rollout:0}'

Multiple GPUs
-------------

Every GPU in the placement hosts one env, one rollout, and one actor
worker, and the three groups have the same size. Env rank :math:`r` exchanges
data only with rollout and actor rank :math:`r`. The actor workers are one
FSDP2 group: parameters, gradients, and optimizer state are sharded. Reset-bin
statistics are reduced across env ranks.

Leaving the config unchanged and adding GPUs keeps the global batch, the
learning rate, and the update ratio, and splits each batch across ranks.
Multi-node and Slurm launches of this recipe have not been validated.

Checkpoints and Resume
----------------------

``runner.save_interval`` writes
``<log_path>/<experiment_name>/checkpoints/global_step_<N>`` every N updates
and after the last update. A checkpoint contains:

* ``actor/``: the FSDP checkpoint. ``dcp_checkpoint/`` stores the sharded
  networks, each phase's optimizer and learning-rate schedule, and each
  rank's RNG. ``model_state_dict/full_weights.pt`` is the full weight file
  used for evaluation. ``supermimic_counters.json`` stores each phase's
  optimizer-step count.
* ``env/env_rank_<r>.pt``: reset-bin statistics and that env rank's RNG.
* ``runner_state.json``: the fractional update debt and the control-step
  count. It is written last, so a checkpoint without it is incomplete.

Resume with the same config name, ``log_path``, and ``experiment_name``:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.max_epochs=16000 \
       runner.logger.log_path=/data/runs/gencoord \
       runner.resume_dir=auto

``runner.resume_dir=auto`` picks the latest complete checkpoint. If none
exists it does not raise; it logs a line and starts from scratch, so check
that the log contains ``Resuming from .../global_step_<N>``. GenCoord and
MaxCoord weights have different shapes and cannot be loaded into each
other. The replay and the simulator state are not saved, so a resumed run
refills the replay before it trains. Resuming on a different GPU count has
not been tested.

Evaluate
--------

Evaluation uses ``evaluations/eval_embodied_agent.py``. ``evaluations/run_eval.sh``
reads ``evaluations/supermimic/<config>.yaml`` when the config name starts
with ``supermimic_``. ``runner.ckpt_path`` is the checkpoint's
``full_weights.pt``.

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=/data/runs/gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt \
       env.eval.action_noise_std=0 \
       runner.eval_export.output_dir=/data/runs/gencoord/eval_2000

Add ``actor.model.representation=maxcoord`` for a MaxCoord checkpoint and
point ``runner.ckpt_path`` at that run. ``env.eval.eval_tasks`` is
``[tracking]`` or ``[push_recovery]``. The 40 LAFAN1 clips fit in
``env.eval.total_num_envs=40``. ``max_steps_per_rollout_epoch=13656`` covers
the longest clip (8194 frames at 30 fps, resampled to 50 Hz).

**Output.** ``runner.eval_export.enabled`` is on in the evaluation configs.
Results go to ``runner.eval_export.output_dir``, or to
``<log_path>/<experiment_name>/eval_results/`` when that is unset. The
directory contains ``summary.json``, ``per_clip.csv``, and the resolved
``config.yaml``. The same metrics are logged with an ``eval/`` prefix.

**Motion tracking** plays each clip once from frame 0. Clip :math:`i`'s
normalized return is the sum of the rewards actually executed, divided by
:math:`0.1 \times T_i`, where :math:`T_i` is the clip length on the 50 Hz
grid and 0.1 is the per-step reward ceiling. Steps that were never played
because the clip failed stay in the denominator. ``normalized_return`` is
the unweighted mean over the 40 clips. ``fail_rate`` is the fraction of
clips that fell, and ``clip_end_rate`` is the fraction that reached the last
frame. ``fail_step`` and ``fail_time_s`` are averaged only over the clips
that fell.

**Standing push recovery** uses ``supermimic_push_recovery``. It loads a
tracking checkpoint, holds mjlab's knees-bent standing pose, and after one
second sets the root's horizontal velocity to 0.5, 1, 2, or 4 m/s, four
trials each. A fall uses the training failure test. The paper's Section 4.3
protocol is a T-pose; this config does not implement that pose. A MaxCoord
checkpoint still needs ``actor.model.representation=maxcoord``.

Logs and Results
----------------

Logs go through RLinf's ``MetricLogger``. The recipe writes TensorBoard
events under ``<log_path>/tensorboard/``. Training metrics are averaged
across actor ranks. The usual keys are:

* ``env/env_steps``, ``env/sim/reward_mean``, ``env/sim/terminal_rate``.
* ``train/world_model/*_loss`` and ``train/world_model/grad_norm``: per-channel
  world-model errors, and the mean pre-clip gradient norm of that phase's
  optimizer steps.
* ``train/actor/reward``, ``train/actor/grad_norm``.
* ``train/critic/loss``, ``train/critic/value_mean``, ``train/critic/target_mean``.
* ``train/imagination/survival``, ``train/imagination/termination_prob``,
  ``train/imagination/step_reward``.
* ``train/world_model_optimizer_steps``, ``train/actor_optimizer_steps``,
  ``train/critic_optimizer_steps``.
* ``time/*``: host timers for the runner and the workers. GPU work is
  asynchronous, so one phase's time can include kernels queued by the
  previous phase.

When a batch of updates finishes and the metrics are copied to the host, a
non-finite value stops training. That check is not a sync on every optimizer
step.

Reported Tracking Results
-------------------------

Both runs use the commands in `Train on One GPU`_, with
``runner.max_epochs=16000``, the 40 LAFAN1 clips, 4096 environments, and
otherwise the config defaults. The numbers below are checkpoint
evaluations, not the return printed during training: each clip starts at
frame 0 and ``env.eval.action_noise_std=0``. The training target is 16000
updates; these rows are the checkpoints at updates 2000 and 4000. One such
evaluation takes about 6.5 minutes.

.. list-table::
   :header-rows: 1
   :widths: 22 18 22 18 20

   * - Representation
     - Update
     - ``normalized_return``
     - ``fail_rate``
     - ``clip_end_rate``
   * - GenCoord
     - 2000
     - 0.704
     - 0.15
     - 0.85
   * - MaxCoord
     - 2000
     - 0.815
     - 0.025
     - 0.975
   * - GenCoord
     - 4000
     - 0.790
     - 0.125
     - 0.875
   * - MaxCoord
     - 4000
     - 0.820
     - 0.075
     - 0.925

On 40 clips, 15% is 6 falls, 2.5% is 1, 12.5% is 5, and 7.5% is 3. The
paper reports a normalized return of 0.82 on these 40 clips. MaxCoord
reaches 0.820 at update 4000. That is not a reproduction of the paper run:
training has not reached 16000 updates, reset bins are 0.2 s rather than
the paper's 1 s, and the paper's main result uses filtered BONES-SEED
rather than these 40 clips.

The other tracking errors at update 4000 (means over 40 clips;
``fail_step`` and ``fail_time_s`` are means over the clips that fell):

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Metric
     - GenCoord
     - MaxCoord
   * - ``body_pos_err``
     - 0.029
     - 0.024
   * - ``torso_pos_err``
     - 1.740
     - 1.245
   * - ``torso_ori_err``
     - 0.112
     - 0.092
   * - ``free_root_pos_err``
     - 1.740
     - 1.245
   * - ``free_root_ori_err``
     - 0.066
     - 0.054
   * - ``free_root_xy_drift``
     - 3.639
     - 2.551
   * - ``free_root_yaw_drift``
     - 1.045
     - 1.070
   * - ``episode_length_s``
     - 211.9
     - 215.2
   * - ``fail_step``
     - 6096.2
     - 6060.0
   * - ``fail_time_s``
     - 121.9
     - 121.2
   * - ``steps``
     - 1.06e4
     - 1.08e4

Scope
-----

Numerical checks live in ``tests/unit_tests/test_supermimic.py``. The CI
configs ``tests/e2e_tests/embodied/g1_mimic_supermimic.yaml``,
``g1_mimic_supermimic_maxcoord.yaml``, and
``tests/e2e_tests/evaluations/g1_mimic_supermimic_eval.yaml`` cover a
small training run, resume, and tracking evaluation. The tables above are
checkpoint evaluations of the 40 LAFAN1 clips at updates 2000 and 4000.
The end of a 16000-update run, multi-node, Slurm, training on real
BONES-SEED, and the paper's Section 4.3 T-pose recovery do not have numbers
here.
