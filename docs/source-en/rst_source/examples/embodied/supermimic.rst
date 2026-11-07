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

      Env, rollout and FSDP actor workers on every GPU; Ray multi-node
      (experimental)

| **You'll do:** install a dedicated environment -> prepare the LAFAN1 clips
  -> train, resume, and evaluate at a small scale -> train the main recipe
  or apply the paper-oriented overrides.
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
     - Ten frames of joint positions relative to the default pose, joint
       velocities, base angular velocity, projected gravity and previous
       action, with sensor noise. The main recipe has 99 values per frame,
       including 6-d reference orientation; the paper overrides use 93.
   * - Action
     - 29 joint setpoints :math:`q_{sp} = \alpha a + q_0` for low-gain PD
       control.
   * - Goal
     - The main recipe uses a 730-d goal (10 frames of 73 values). The paper
       overrides use 680-d; see `Configurations`_ for the other interfaces.
   * - Entry points
     - Training uses RLinf's ``examples/embodiment/train_embodied_agent.py``;
       evaluation uses ``evaluations/run_eval.sh supermimic_*``, which runs
       ``evaluations/eval_embodied_agent.py``.

How SuperMimic Works
--------------------

SuperMimic repeats three steps. It steps every environment once in MuJoCo
Warp and writes the transitions to the replay. At the update-to-data ratio,
it fits the world model's dynamics and termination head on ``k``-step
windows from the replay. It then starts 32-step imagined rollouts from replay
states inside the world model, backpropagates the return through the
differentiable world model to update the actor, and fits the critic to GAE
targets. When an environment ends, the returns of bins along the motion
timeline decide where it restarts, so low-return segments are drawn more
often. Equations, network structure, and differences from the paper and the
community implementations are on the
:doc:`algorithm page <../../reference/algorithms/supermimic>`.

In RLinf the three steps run on three worker groups, with one worker of each
group on every GPU:

* ``SuperMimicEnvWorker`` owns that GPU's share of the environments, the
  reset sampler and the rewards.
* ``SuperMimicRolloutWorker`` holds a copy of the policy and the motion
  critic. It returns actions, and the critic values that bootstrap the
  reset-sampler returns of episodes ending in a timeout.
* ``EmbodiedSuperMimicFSDPPolicy`` is the FSDP2 actor. It owns the replay,
  the world model, the actor, the critics and their optimizers.

``SuperMimicRunner`` is the control plane. It works out how many control
steps to collect before the next update is owed, runs the three workers for
those steps, runs the owed updates on the actor, and sends the new weights
to the rollout workers before collection resumes. It also handles logging,
evaluation and checkpoints. Within one control step, the env worker sends
the observation to its rollout worker and receives the action, sends the
transition to its actor worker, and then exchanges the bootstrap features
and values with the rollout worker before resetting the episodes that ended.

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

The NVIDIA environment uses torch 2.14.0 and ``warp-lang`` 1.18.0,
with dependencies listed in ``requirements/embodied/envs/supermimic.txt``.
Keep it separate from other RLinf environments because these requirements
upgrade torch. The installer selects the CUDA 12 Warp wheel by host
architecture when needed. ``docker/Dockerfile`` also provides an
``embodied-supermimic-image`` target using the same installation.

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
GPUs as described in `Train on One GPU`_.

The AMD installer and current simulator changes have not been validated
through a full AMD training run. Historical third-party MI300X results
are listed under `Reported Tracking Results`_; they are not a performance
guarantee for this installation path. Other AMD models remain unvalidated.

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

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 \
       env.train.motion_format=bones_seed env.train.num_clips=null \
       env.train.motion_path=$SUPERMIMIC_BONES_PATH \
       env.train.total_num_envs=8 algorithm.replay_length=64 runner.max_epochs=3 \
       algorithm.wm_only_updates=0 algorithm.wm_window_batch=64 \
       algorithm.imagine_batch=8 runner.save_interval=3 \
       runner.logger.log_path=/data/runs/bones_smoke

These BONES-SEED overrides load every clip that passes the filters
(``env.train.num_clips: null``) and log how many each filter removed. The paper's
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
evaluation. These overrides use 8 environments and 3 updates and keep the
policy interface of the full ``supermimic_lafan1_goal730`` run. Checkpoints,
logs and results share ``RUN_DIR``. This checks the workflow, not tracking
quality.

.. code-block:: bash

   export PYTHONPATH=$PWD:$PYTHONPATH
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export RUN_DIR=/data/runs/smoke
   SMOKE_ARGS=(env.train.total_num_envs=8 algorithm.replay_length=64
       algorithm.wm_only_updates=0 algorithm.wm_window_batch=64
       algorithm.imagine_batch=8 runner.save_interval=1
       runner.logger.experiment_name=smoke runner.logger.log_path=$RUN_DIR)

   # Train 3 updates, then resume from update 3 and continue to update 5.
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 "${SMOKE_ARGS[@]}" runner.max_epochs=3
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 "${SMOKE_ARGS[@]}" runner.max_epochs=5 \
       runner.resume_dir=auto

   # Evaluate the last checkpoint on the 40 training clips.
   bash evaluations/run_eval.sh supermimic_lafan1_goal730_eval \
       runner.ckpt_path=$RUN_DIR/smoke/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval
   tensorboard --logdir $RUN_DIR/tensorboard

The resume log should contain ``Resuming from`` and ``global_step_3``. If it
instead says ``resume_dir=auto found no complete checkpoint``, check
``runner.logger.log_path``, ``runner.logger.experiment_name`` and whether the
first run saved. Checkpoints live in
``$RUN_DIR/smoke/checkpoints/global_step_<N>``, TensorBoard files in
``$RUN_DIR/tensorboard``, and evaluation tables in ``$RUN_DIR/eval``. A
policy after 3 to 5 updates is not expected to track motion.

Configurations
--------------

Three YAML files cover the main workflow. The training recipe holds every
parameter of the 16,000-update LAFAN1 run reported below; the two evaluation
configs inherit it and change only evaluation settings. Environment counts
and batch sizes are global.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Purpose
   * - ``examples/embodiment/config/supermimic_lafan1_goal730.yaml``
     - Training: 730-d goal, 99-d observation, 500 world-model-only updates,
       64 warmup steps, 16,000 updates, a checkpoint every 2,000 updates.
   * - ``evaluations/supermimic/supermimic_lafan1_goal730_eval.yaml``
     - Motion-tracking evaluation of that recipe; velocity tasks are selected
       with overrides (see `Evaluate`_).
   * - ``evaluations/supermimic/supermimic_push_recovery.yaml``
     - Standing-pose push recovery of a policy with the paper interface
       (680-d goal, 93-d observation); see `Evaluate`_ for other interfaces.

The recipe composes three RLinf config groups:
``env/g1_mimic.yaml`` for ``env.train`` and ``env.eval`` (motion data,
simulation and the evaluation protocols), ``model/supermimic.yaml`` for
``actor.model`` (the policy interface, whose defaults are the paper's
choices) and ``hybrid_engines/fsdp.yaml`` for ``actor.fsdp_config``. On top
of them it selects the goal730 interface: ``actor.model.goal_encoding:
goal730``, ``obs_dim: 99``, ``critic_features: body_error``, ``input_norm:
running`` and ``wm_state_loss: running``, with ``algorithm.reward_anchor:
pelvis``, ``residual_frame: body`` and ``termination_head: analytic``. It
uses seed 0. Small runs and multi-GPU launches use overrides rather than
separate YAMLs.

To run the original paper-oriented settings, define the following Bash array
in the shell where you launch training or evaluation. This changes the goal,
observation, normalization and critic together; keep the same array when
evaluating its checkpoint.

.. code-block:: bash

   PAPER_ARGS=(
       runner.max_epochs=320000 algorithm.warmup_sim_steps=40
       algorithm.wm_only_updates=0 algorithm.termination_head=learned
       algorithm.reward_anchor=torso algorithm.residual_frame=heading
       actor.model.goal_encoding=paper actor.model.obs_dim=93
       actor.model.input_norm=none actor.model.wm_state_loss=paper
       actor.model.critic_features=root4
       actor.fsdp_config.amp_autocast.enabled=false
       env.train.com_sampling=per_reset runner.logger.experiment_name=paper
   )
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 "${PAPER_ARGS[@]}" \
       runner.logger.log_path=/data/runs/paper

The following variants add overrides after ``"${PAPER_ARGS[@]}"``. Give
each run a distinct ``runner.logger.experiment_name``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variant
     - Additional overrides
   * - Torso pose anchor (689-d goal)
     - ``actor.model.goal_anchor=true``
   * - 730-d goal with 93-d observations
     - ``actor.model.goal_encoding=goal730``
   * - 99-d observations with a 680-d goal
     - ``actor.model.obs_dim=99``
   * - Motion and velocity training
     - ``actor.model.velocity_tracking=true``
   * - BONES-SEED
     - ``env.train.motion_format=bones_seed env.train.num_clips=null
       env.train.motion_path=$SUPERMIMIC_BONES_PATH``

The anchor and velocity tracking need the paper goal and the 93-d
observation; the config check refuses other combinations. For example, train
the velocity encoder and run the Appendix G.2 step test on the same
interface:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 "${PAPER_ARGS[@]}" \
       actor.model.velocity_tracking=true runner.logger.experiment_name=velocity \
       runner.logger.log_path=/data/runs/velocity
   bash evaluations/run_eval.sh supermimic_lafan1_goal730_eval "${PAPER_ARGS[@]}" \
       actor.model.velocity_tracking=true 'env.eval.eval_tasks=[velocity_g2]' \
       env.eval.total_num_envs=2688 env.eval.max_steps_per_rollout_epoch=250 \
       runner.ckpt_path=/data/runs/velocity/velocity/checkpoints/global_step_320000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=/data/runs/velocity/eval

Four keys select how the world model and the rewards are defined. The goal730
recipe sets ``pelvis``, ``running``, ``body`` and ``running``; the paper
overrides select ``torso``, ``none``, ``heading`` and ``paper``.

* ``algorithm.reward_anchor``: ``pelvis`` or ``torso`` (``torso_link``).
  ``algorithm.termination_body`` still chooses the failure body.
* ``actor.model.input_norm``: ``running`` normalizes model inputs using
  running statistics; ``none`` leaves the raw values.
* ``algorithm.residual_frame``: ``body`` or ``heading`` selects the frame of
  the predicted state residual; see
  :doc:`the algorithm reference <../../reference/algorithms/supermimic>`
  for the component-wise definitions.
* ``actor.model.wm_state_loss``: ``running`` scales state errors by running
  residual statistics, ``scaled`` uses fixed scales, and ``paper`` uses
  unweighted squared error.

A checkpoint records these keys and the rest of the policy interface in
``supermimic_interface.json``. Resuming or evaluating it with a config that
differs in any of them is refused.

For a large LAFAN1 library, ``env.train.motion_on_cpu=true`` (and
``env.eval.motion_on_cpu=true`` for evaluation) keeps both reference states
and the full reference FK cache in host memory. Queries gather rows there
and transfer only the requested states, body poses and velocities to the
GPU. The cache still occupies host RAM. With ``motion_on_cpu=false``, the
library and its FK cache stay on the GPU. BONES-SEED libraries use CPU
storage with the same query-transfer behavior. The actor worker keeps its
own copy of the motion library for imagination, so each GPU holds two.

``env.train.com_sampling`` chooses when the torso CoM offset in :math:`\xi`
is drawn. ``per_env`` draws it once per environment, as zita-ch does;
``per_reset`` draws a new one on every reset and incurs the cost of
recomputing simulation constants. Friction and joint offsets are drawn on
every reset either way. ``env/g1_mimic.yaml`` defaults to the paper's
``per_reset``; ``supermimic_lafan1_goal730`` sets ``per_env``.

``actor.fsdp_config.amp_autocast`` (bfloat16 in the recipe) runs the network
forwards of the actor and rollout workers under autocast. Integration,
kinematics, the losses and the Adam weights stay float32. TF32 matmuls are
on in every SuperMimic process.

Historical tracking results and timing are summarized under
`Reported Tracking Results`_ and `Performance and Hardware`_.

Common Keys
~~~~~~~~~~~

These are the keys most often changed to set the scale and length of a run.
Defaults come from ``supermimic_lafan1_goal730.yaml``; the paper overrides
above change some of them. Each can be passed as a Hydra override.

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
   * - ``runner.max_epochs``
     - 16000
     - Number of training **updates**, not simulator steps (see below).
   * - ``algorithm.updates_per_env_step``
     - 0.4
     - Global updates owed per control step (update-to-data ratio). Larger
       values train more on each sample.
   * - ``actor.optim.lr``
     - 2.0e-4
     - Learning rate of the motion actor. The world-model, critic and
       velocity optimizers (``actor.world_model_optim``,
       ``actor.critic_optim``, ``actor.velocity_actor_optim``,
       ``actor.velocity_critic_optim``) use the same value.
   * - ``actor.optim.total_training_steps``, ``actor.optim.min_lr_rate``
     - 250000, 0.03
     - Cosine decay to 0.03 x ``lr`` over the first 250,000 steps of each
       optimizer's schedule, then constant. A schedule advances once per
       update in which its phase trained.
   * - ``actor.optim.clip_grad``, ``actor.optim.clip_grad_value``
     - 1.0, 0.1
     - Every phase clips its gradient to a global norm of 1.0. The actor
       phases then clamp each element to 0.1.
   * - ``algorithm.max_nonfinite_skips``
     - 20
     - An optimizer step with a non-finite gradient norm is skipped; more
       than this many in a row stops the run.
   * - ``actor.optim.fused``
     - ``True``
     - Fused Adam on CUDA for every phase. A step with a non-finite gradient
       norm is skipped inside the kernel, so the optimizer steps do not wait
       for the host; the training metrics and skip flags come back together
       after the iteration's updates, and the skips are counted then.
       Multi-GPU runs still sum row counts on the host. ``False`` uses
       PyTorch's default Adam and checks each step on the host.
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
     - Control steps collected after each start or resume, before the first
       update, to warm the replay; they owe no updates.
   * - ``actor.fsdp_config.amp_autocast.enabled``
     - ``true`` (bf16)
     - bfloat16 autocast of the network forwards; see above.
   * - ``actor.model.torch_compile``
     - ``True``
     - Compile the training scans, the env's reward and feature pieces and FK
       on large CUDA batches. ``False`` keeps them eager.
   * - ``rollout.gc_after_weight_sync``
     - ``False``
     - ``True`` runs ``gc.collect()`` and empties the device cache after each
       weight sync, as the RLinf rollout does by default. SuperMimic syncs
       after every update, where a full collection each time is a large
       share of the step.
   * - ``actor.torch_profile_update``
     - ``null``
     - A global update index. The actor's work in that update is traced with
       ``torch.profiler``; operator tables (``.txt``) and a Chrome trace
       (``.json``) go to ``<log_path>/<experiment_name>/torch_profile``.
   * - ``env.train.total_num_envs``
     - 4096
     - Parallel simulator environments, global. Collection throughput and
       memory grow with it.
   * - ``algorithm.replay_length``
     - 1000
     - Ring-buffer replay capacity per environment, in control steps (20 s at
       50 Hz); at least 10, the observation history the policy reads. Replay
       memory is proportional to ``total_num_envs x replay_length``.
   * - ``env.train.num_clips``
     - 40
     - Load the first N clips in file-name order; ``null`` loads all.
   * - ``runner.save_interval``
     - 2000
     - Save a checkpoint every N updates and after the last one; -1 saves
       nothing.
   * - ``runner.log_interval``
     - 100
     - Record the metrics to the logger backends and print RLinf's metrics
       table every N updates, after the last one and after every evaluation.
   * - ``runner.logger.logger_backends``
     - ``[tensorboard]``
     - RLinf logger backends: TensorBoard, WandB or SwanLab.
   * - ``runner.val_check_interval``
     - -1
     - Evaluate every N updates during training; off by default.
   * - ``runner.logger.log_path``
     - ``../results``
     - Root of checkpoints and logs, relative to the launch directory. Set it
       explicitly for real runs.

**Updates versus simulator steps.** Every control step (every environment
steps once) adds ``updates_per_env_step`` to the updates owed. The runner
collects until an update is due, runs every whole update owed, sends the new
weights to the rollout workers, and resumes collection. So
``runner.max_epochs=16000`` is 16000 / 0.4 = 40,000 control steps plus the
warmup, and environment transitions are that times
``env.train.total_num_envs``. The logs count updates in the global step and
control steps in ``env/env_steps``. Owed updates are capped at the remaining
budget. RLinf saves and evaluates when the global step is a multiple of
``runner.save_interval`` or ``runner.val_check_interval`` and after the
last update; when one iteration runs several updates
(``updates_per_env_step`` above 1), a multiple can be stepped over.

**Global sizes.** ``env.train.total_num_envs``, ``algorithm.wm_window_batch``
and ``algorithm.imagine_batch`` are totals over all GPUs and must divide
evenly by the number of GPUs; ``updates_per_env_step`` and
``runner.max_epochs`` also count global updates. Adding GPUs without
changing them keeps the same training mathematics; see `Multiple GPUs`_.

Train on One GPU
----------------

The recipe puts one env, one rollout and one actor worker on every GPU that
RLinf sees (``cluster.component_placement`` maps ``actor,env,rollout`` to
``all``). On a machine with one GPU that is the whole placement:

.. code-block:: bash

   export PYTHONPATH=$PWD:$PYTHONPATH
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000

RLinf's shared launcher starts the recipe with its defaults. It writes each
launch to a new ``logs/<time>-<config>`` folder in the repository and passes
no Hydra overrides, so call ``train_embodied_agent.py`` directly to change
keys or to resume. Training does not render; ``MUJOCO_GL=disabled`` replaces
the launcher's EGL default on machines without EGL:

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   MUJOCO_GL=disabled bash examples/embodiment/run_embodiment.sh supermimic_lafan1_goal730

To train on one GPU of a larger machine, replace the placement with one
entry per worker group. Hydra cannot override the recipe's combined
``actor,env,rollout`` key in place, so the first override removes it:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor: 2, env: 2, rollout: 2}' \
       runner.logger.log_path=/data/runs/lafan1

RLinf sets device visibility for its workers from the placement, so it may
replace a ``CUDA_VISIBLE_DEVICES`` or ``HIP_VISIBLE_DEVICES`` value set
before launch. When a worker is isolated to one visible GPU, that device is
numbered ``cuda:0`` inside the process: a worker assigned node GPU 2 can
still report ``cuda:0``. Check the placement together with node-level GPU
monitoring to identify the actual device in use.

The ``train_embodied_agent.py`` commands set two keys explicitly:

* ``runner.logger.log_path``: the recipe's ``../results`` is relative to the
  launch directory. With a fixed folder, resume and evaluation find the
  checkpoints directly.
* ``runner.save_interval``: the recipe saves every 2000 updates. Choose an
  interval from the measured seconds per update (see
  `Performance and Hardware`_).

TensorBoard files go to ``<log_path>/tensorboard``:
``tensorboard --logdir /data/runs/lafan1/tensorboard``.

On a host with many CPU cores and little RAM, Ray starts one idle worker per
core; ``RAY_prestart_worker_first_driver=0`` skips them.
``++cluster.object_store_memory_mb=512`` caps Ray's object store, which the
replay never passes through; the cap applies only when RLinf starts Ray
itself.

Multiple GPUs
-------------

Every GPU in the placement hosts one env, one rollout and one actor worker,
and the three groups must have the same size. The env worker of rank
:math:`r` exchanges data only with the rollout and actor workers of rank
:math:`r`. The actor workers form one FSDP2 group: parameters, gradients
and optimizer states are sharded over it, and each rank weights its loss by
its share of the global rows. The reset-bin statistics and the input
normalization statistics are merged over all env ranks. Details are in the
:doc:`algorithm reference <../../reference/algorithms/supermimic>`.

There are two ways to use more GPUs:

* **Fixed global batch.** Keep a config and add GPUs. The global batch,
  learning rate, and update-to-data ratio stay those of one GPU, and each rank
  takes a share; this is one-GPU training spread over several GPUs. With
  four visible GPUs the default ``all`` placement does this; otherwise:

  .. code-block:: bash

     python examples/embodiment/train_embodied_agent.py \
         --config-name supermimic_lafan1_goal730 \
         '~cluster.component_placement' \
         '+cluster.component_placement={actor: 0-3, env: 0-3, rollout: 0-3}' \
         runner.logger.log_path=/data/runs/lafan1_4gpu

* **The paper's two-GPU recipe.** Each GPU keeps the one-GPU sizes, so the
  global batch doubles, gradients are averaged over both GPUs, and the
  learning rate is :math:`2\sqrt{2} \times 10^{-4}` (Table A.5):

  .. code-block:: bash

     export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered
     python examples/embodiment/train_embodied_agent.py \
         --config-name supermimic_lafan1_goal730 "${PAPER_ARGS[@]}" \
         env.train.motion_format=bones_seed env.train.num_clips=null \
         env.train.motion_path=$SUPERMIMIC_BONES_PATH \
         '~cluster.component_placement' \
         '+cluster.component_placement={actor: 0-1, env: 0-1, rollout: 0-1}' \
         env.train.total_num_envs=8192 algorithm.wm_window_batch=4096 \
         algorithm.imagine_batch=8192 actor.optim.lr=2.8284271e-4 \
         runner.logger.log_path=/data/runs/bones_2gpu

  The same overrides apply to LAFAN1 without the three ``env.train.motion*``
  and ``num_clips`` keys.

Collectives use NCCL on GPU with the timeout ``RLINF_TIMEOUT`` (minutes). If a
worker fails, RLinf's worker group stops the driver and the other workers at
once instead of waiting for that timeout.

Multiple Nodes (Experimental)
-----------------------------

Start one Ray node on each allocated machine following
:doc:`/rst_source/guides/multi_node`, with a unique
``RLINF_NODE_RANK`` set before ``ray start``. Activate the same SuperMimic
environment on every node and use shared paths for code, motion data and
checkpoints.

For Slurm, obtain an allocation through your site's job script or an
interactive allocation, for example:

.. code-block:: bash

   salloc --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --partition=your_partition

Inside that allocation, start Ray on the allocated nodes using the multi-node
guide, then run the training entry point once on the head node:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 cluster.num_nodes=2 \
       env.train.motion_path=/shared/lafan1_g1 \
       runner.logger.log_path=/shared/runs/lafan1 runner.save_interval=1000

The ``all`` placement spans all sixteen GPUs. Training batch sizes are global
and are split across those ranks. Set network interfaces and release the Ray
nodes through your site's allocation lifecycle. Multi-node and Slurm execution
have not been validated for this integration.

Checkpoints and Resume
----------------------

``runner.save_interval`` writes
``<log_path>/<experiment_name>/checkpoints/global_step_<N>`` every N updates
and after the last update. A checkpoint holds:

* ``actor/``: the FSDP checkpoint of the actor workers. ``dcp_checkpoint/``
  has the sharded networks, the optimizers and learning-rate schedules of
  every phase, and each rank's RNG; ``model_state_dict/full_weights.pt`` has
  the full weights for evaluation; ``supermimic_interface.json`` records the
  policy interface; ``supermimic_counters.json`` counts optimizer steps.
* ``env/env_rank_<r>.pt``: the reset-bin statistics and each env rank's RNG.
* ``runner_state.json``: the fractional owed update and the control-step
  count. It is written last, so a checkpoint without it is incomplete.

To resume, launch with the **same** config name, ``log_path`` and
``experiment_name`` as the original run, for example after the first command
in `Train on One GPU`_:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000 runner.resume_dir=auto

``runner.resume_dir=auto`` picks the newest complete checkpoint under
``<log_path>/<experiment_name>/checkpoints/`` and trains on to
``runner.max_epochs``; a ``global_step_<N>`` path picks that checkpoint.
When ``auto`` finds no complete checkpoint it does **not** fail: it logs
``resume_dir=auto found no complete checkpoint under ...; starting from
scratch`` and trains from the beginning, so check the log for
``Resuming from .../global_step_<N>``.

A checkpoint with another policy interface is refused. The replay and
simulator are not saved, so a resumed run warms up a new replay and is a
continuation, not an exact restore. Resuming on a different number of GPUs
has not been tested. Checkpoints written by the earlier single-process
implementation (``global_step_<N>/supermimic.pt``) cannot be loaded.

Evaluate
--------

Evaluation runs through RLinf's ``evaluations/eval_embodied_agent.py``. The
env workers run the evaluation jobs, the rollout workers run the policy, and
``SuperMimicEvalRunner`` merges the rows of every env rank, logs the metrics
and writes the result files. ``evaluations/run_eval.sh`` reads
``evaluations/supermimic/<config>.yaml`` for any config name that starts
with ``supermimic_``. ``runner.ckpt_path`` is the ``full_weights.pt`` of a
checkpoint; the ``supermimic_interface.json`` beside it must match the
evaluation config.

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_goal730_eval \
       runner.ckpt_path=/data/runs/lafan1/supermimic_lafan1_goal730/checkpoints/global_step_16000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=/data/runs/lafan1/eval

For any variant, pass the same ``actor.model.*`` and ``algorithm.*``
overrides used in training, such as ``"${PAPER_ARGS[@]}"`` and
``actor.model.velocity_tracking=true``.

``env.eval.eval_tasks`` selects ``tracking`` and/or ``velocity_smoke``,
``velocity_g2`` alone, or ``push_recovery`` alone. The jobs (clips, commands
or trials) are split evenly over the env ranks, and each environment runs
one job per rollout epoch, so ``env.eval.rollout_epoch`` times a rank's
environments must cover that rank's jobs. ``env.eval.max_steps_per_rollout_epoch``
must cover the longest job; the recipe's 13656 covers the longest LAFAN1
clip, which has 8194 frames at 30 fps. An environment whose job has ended waits for the end of the epoch.

**Where results go.** With ``runner.eval_export.enabled`` (on in the
evaluation configs), the runner writes to ``runner.eval_export.output_dir``,
or to ``<log_path>/<experiment_name>/eval_results/`` when it is not set;
``run_eval.sh`` sets ``log_path`` to ``logs/<time>-<config>`` in the
repository. The results folder holds:

* ``summary.json``: ``meta`` (checkpoint path, experiment name and the
  evaluation settings) and ``metrics`` (the aggregates below; a value that
  does not exist is ``null``).
* ``per_clip.csv``: one row per clip.
* ``velocity_smoke.csv`` or ``velocity_g2.csv``: one row per command.
* ``per_trial.csv`` for push recovery, and ``trajectory.csv`` with
  ``env.eval.push_recovery.save_trajectory=true``.
* ``config.yaml``: the full config the evaluation used.

The same metrics go to the RLinf logger under the ``eval/`` prefix, for
example ``eval/normalized_return``, ``eval/velocity_g2/median_relative_error``
and ``eval/push/recovery_rate``; in ``summary.json`` they have no prefix.

During training, ``runner.val_check_interval`` runs the ``env.eval``
evaluation every that many updates and after the last one. With
``runner.eval_export.enabled=true`` its files go to
``eval_results/global_step_<N>/``.

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
``normalized_return`` is the unweighted mean over clips.

**Velocity tracking** evaluates an encoder trained with
``actor.model.velocity_tracking=true``; keep that override and select a task:

* ``'env.eval.eval_tasks=[velocity_g2]'`` runs the Appendix G.2 step test:
  672 commands on a 15 x 15 grid in each of the (vx, vy), (vx, wz), and
  (vy, wz) planes without the standing command, four trials each
  (``env.eval.velocity_g2.trials``), 5 s from mjlab's standing pose. The
  value of a trial is the norm of the mean velocity error over the last 2 s
  divided by the command norm; a command reports the median over the trials
  that did not fall. A command whose trials all fell has no value (NaN in the
  CSV, ``null`` in ``summary.json``) and is counted in
  ``velocity_g2/commands_all_trials_fell``. Trials differ in the sim context
  and the initial joint noise, which the paper does not specify. The 2688
  trials need ``env.eval.total_num_envs`` times ``env.eval.rollout_epoch`` of
  at least 2688 and ``env.eval.max_steps_per_rollout_epoch`` of at least 250.
* ``'env.eval.eval_tasks=[tracking,velocity_smoke]'`` adds seven fixed
  commands from a reference start (``env.eval.velocity_smoke``), writes
  ``velocity_smoke.csv``, and reports aggregates under ``velocity_smoke/``.
  It checks that the velocity encoder runs; it is not a paper metric. With
  the 40 LAFAN1 clips this makes 47 jobs, so also set
  ``env.eval.rollout_epoch=2``.

For repeatable G.2 evaluation, keep ``env.eval.seed``, the environment
count, hardware and the other evaluation settings fixed.

**Stationary standing-pose push recovery.** Paper §4.3 holds a static
T-pose. This task is a different protocol and does not claim to reproduce
that T-pose. ``evaluations/run_eval.sh supermimic_push_recovery`` loads an
existing tracking checkpoint and does not train. The reference stays one
standing pose for the whole trial, so the goal does not switch to a walk.

The reference uses mjlab's knees-bent standing pose, with arms beside the
torso. The settings live under ``env.eval.push_recovery``. The default
speeds are 0.5, 1, 2 and 4 m/s, four trials each, one trial per
environment. ``direction: random`` samples a horizontal direction, or set
``direction=[1.0,0.0]``. ``frame`` selects ``world`` or ``heading``.
``push_mode: add`` adds horizontal root velocity; ``set`` replaces it while
preserving vertical velocity. These are settings of this protocol.

A fall is the training ``failure_signals`` test against this static
reference: height error above 0.25 m or tilt error above 0.8. The settle
test is this implementation's. After the push the root's horizontal speed
must stay within ``speed_threshold`` (default 0.15 m/s), the root tilt
within ``tilt_threshold`` (default 0.35), and the torso orientation error
within ``torso_ori_threshold`` (default 0.5 rad), for ``stable_seconds``
(default 0.5 s) without a break. ``max_xy_drift`` defaults to ``null`` and
does not require a return to the original horizontal position.
``recovery_time_s`` runs from the push to the control step that completes
the stable window; ``stable_onset_time_s`` is that window's first frame. A
trial that does not recover leaves the time blank rather than zero. A fall,
a joint-limit violation, or a non-finite state before the push is
``initialization_failed`` and is left out of the post-push fall rate.
``settle_seconds`` (default 1 s) and ``max_seconds`` (default 5 s) are this
implementation's wait and cutoff, not the paper's 0.5–2.4 s.

``supermimic_push_recovery.yaml`` uses the paper interface. For a
``supermimic_lafan1_goal730`` checkpoint, pass that recipe's interface:

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_push_recovery \
       actor.model.goal_encoding=goal730 actor.model.obs_dim=99 \
       actor.model.critic_features=body_error actor.model.input_norm=running \
       actor.model.wm_state_loss=running \
       algorithm.reward_anchor=pelvis algorithm.residual_frame=body \
       runner.ckpt_path=/data/runs/lafan1/supermimic_lafan1_goal730/checkpoints/global_step_16000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=/data/runs/lafan1/push_recovery

``summary.json`` reports, per speed, the trial count, recovery rate, fall
rate, timeout rate, and the mean and median recovery time of the trials
that recovered. A speed with no recovery has a ``null`` recovery time.
``per_trial.csv`` has one row per trial, including the direction, the
measured velocity change, the outcome, and the reason.

Visualization and Results
-------------------------

Logging goes through RLinf's ``MetricLogger`` (TensorBoard, WandB, SwanLab).
The recipe writes TensorBoard event files and the run's ``config.yaml`` to
``<log_path>/tensorboard/``:

.. code-block:: bash

   tensorboard --logdir /data/runs/lafan1/tensorboard

Training metrics are averaged over the actor ranks;
``runner.per_worker_log=true`` also logs each rank. Useful keys:

* ``env/env_steps``: control steps collected. ``env/sim/reward_mean``,
  ``env/sim/terminal_rate`` and ``env/sim/episodes_finished`` describe the
  simulated episodes; ``env/reset/bin_return_mean`` and
  ``env/reset/bin_return_min`` the motion reset-bin returns.
* ``train/wm/one_step_loss`` and ``train/wm/k_step_loss``: world-model error
  after one and eight steps.
* ``train/imagination/survival`` and ``train/imagination/termination_prob``:
  the 32-step survival weight and the mean termination probability.
* ``train/actor/grad_norm`` (before clipping),
  ``train/actor/encoder_grad_norm``, and ``train/actor/decoder_grad_norm``.
* ``train/critic/loss``, ``train/critic/value_mean``,
  ``train/critic/target_mean``; ``train/velocity/*`` with velocity tracking.
* ``train/wm_only``, ``train/*_optimizer_steps`` and
  ``train/nonfinite_skips``.
* ``time/*``: RLinf timers of the runner (``time/generate_rollouts``,
  ``time/actor_training``, ``time/sync_weights``, ``time/eval``) and of the
  workers (``time/env/collect``, ``time/rollout/serve``,
  ``time/actor/ingest``, ``time/actor/run_updates`` and its phases). Worker
  timers measure host time; the GPU runs asynchronously, so a phase can
  absorb kernels queued by the one before it.

The metrics are recorded every ``runner.log_interval`` updates (100 by
default), after the last update and after every evaluation. A record holds the
``train/*`` and ``time/*`` values of the iteration that made it due, while the
``env/sim/*`` statistics cover every control step since the previous record.
Recording every update would slow training: TensorBoard blocks the training
loop once ten events wait to be written, which on network storage costs tens
of milliseconds per update. Each record also prints RLinf's metrics table
(progress, elapsed time, ETA and the latest values) to the console and appends
it to ``<log_path>/metrics.log``.

.. _supermimic-reported-results:

Reported Tracking Results
~~~~~~~~~~~~~~~~~~~~~~~~~

The results in this section were obtained with the earlier single-process
implementation of this integration, with one SuperMimic process per GPU and
the recipe now named ``supermimic_lafan1_goal730``. The algorithm and recipe
values are unchanged, but these runs have not yet been repeated with the
current env, rollout and actor workers.

The optimized A800 run used 40 training clips, 4096 environments and the
code before the review fixes. Results measure tracking of training motions,
not generalization or a full reproduction of the paper's 680-d recipe.

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Run
     - Update
     - Mean normalized return
   * - RLinf, optimized A800 run
     - 14000
     - 0.828
   * - RLinf, optimized A800 run
     - 16000
     - 0.850
   * - zita-ch, community report
     - 14000
     - 0.829

The community's GPU, simulator and recipe differ, so this is not a
controlled comparison. See `Performance and Hardware`_ for measured
runtime and memory.

MI300X Community Reproduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`ZJLi2013's PR #1767 feedback <https://github.com/RLinf/RLinf/pull/1767#issuecomment-6593698481>`_
reports two MI300X runs based on commit ``0a19b154`` with their two old-stack
compatibility edits. Both used the same recipe, seed 0 and the 40 LAFAN1
clips; the only config change was a checkpoint every 500 updates. The
reported checkpoints are:

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

These third-party runs used the old baseline and compatibility edits.
They do not validate the current AMD installer or establish performance
for other AMD GPUs. Installation instructions are under
`AMD MI300X (Experimental)`_.

Troubleshooting
---------------

* **Non-finite evaluation outputs.** Evaluation stops if an active policy
  produces NaN/Inf actions, or if an active simulation or reference state is
  non-finite. Check the checkpoint/config match, policy inputs and motion
  data when the error reports ``non-finite policy actions in active
  evaluation rows`` or ``non-finite evaluation states in active rows``.
* **CUDA out of memory.** Add GPUs to keep the global sizes, or reduce
  ``env.train.total_num_envs``, ``algorithm.replay_length``, or the training
  batch sizes.
* **Ray kills a worker for low memory.** Free host memory rather than raising
  ``RAY_memory_usage_threshold``. On a host with many CPU cores and little
  RAM, ``RAY_prestart_worker_first_driver=0`` stops Ray from starting idle
  workers.
* **"No verified G1 reference clips".** ``SUPERMIMIC_MOTION_PATH`` or
  ``env.train.motion_path`` does not point at prepared clips.
* **"reference clip q0 differs from the mjlab G1 default joint pose".** The
  clips were converted with another default pose; reconvert them.
* **A BONES-SEED CSV is rejected.** The header needs ``Frame``,
  ``root_translateX/Y/Z``, ``root_rotateX/Y/Z``, and 29 ``<joint>_dof``
  columns. Root position is in centimeters and angles are in degrees. The
  message names the file and the missing column.
* **A resumed run starts at update 0.** The log shows ``resume_dir=auto
  found no complete checkpoint under ...``: the ``log_path`` or
  ``experiment_name`` differs from the original run, or the original run
  never wrote a checkpoint (``save_interval`` was -1).
* **"... was trained with <key>=..., but this config has ...".** The
  checkpoint's ``supermimic_interface.json`` differs from the config; pass
  the overrides used in training. A checkpoint without that file predates the
  current structure and cannot be loaded.
* **The log stops at "Compiling a SuperMimic training scan".** The first
  large batch is waiting for the compile, a few minutes with a cold cache;
  see `Performance and Hardware`_. Do not delete the inductor cache
  directory during a run.
* **"SuperMimic scan <name> stayed eager".** That piece failed to compile
  and runs eagerly from then on. Results are unchanged; training is slower.

Performance and Hardware
------------------------

The following measurements describe training runs of the earlier
single-process implementation (see `Reported Tracking Results`_). They
have not been repeated with the current worker structure, whose collection
adds a worker-to-worker exchange per control step. Measure seconds per
update on your hardware before planning a full run.

**The A800 run.** One 80 GB A800, one GPU, the recipe with 4096
environments and bfloat16 autocast, warm compile cache:

* The first 500 updates train only the world model, at about 26 s per 100
  updates. After that, 43-47 s per 100 updates, 44.9 s on average, or
  about 0.45 s per update. 16000 updates took 1 h 59 min 51 s. The
  zita-ch code, measured on an A800 of the same model, took 2 h 15 min.
* PyTorch peak allocated 17.1 GB (18.8 GB reserved), whole GPU as the
  driver sees it 20.1 GB, Warp pool peak 0.48 GB, host RSS about 4.0 GB.
* At the end: 40064 vectorized simulator steps, about 164 million
  environment transitions.

**Compilation.** A first large run compiles training scans and FK. The
PyTorch kernel cache is under ``/tmp/torchinductor_<user>``; starting on a
new machine or after clearing that cache takes longer. Keep
``actor.model.torch_compile`` at ``True`` for training; ``False`` keeps the
scans, the env pieces and FK eager, which suits small smoke runs.

On an 8 GB laptop GPU under WSL2, 1024 environments collected about 3,900
environment transitions per second, one update with a quarter of the paper
batches (1024 imagined rollouts, 512 windows) took about 21 s at a 4.2 GB
PyTorch peak, and one update at the paper batches needed about 11 GB, more
than that card has. These numbers come from a launch-bound laptop setup and
do not predict datacenter GPUs.

Validation Scope
----------------

The training, resume and evaluation runs reported above used the earlier
single-process implementation. For the current env, rollout and actor
workers, the FSDP2 phase freezing, the actor's training phases, its
checkpoint round trip, the update schedule and the evaluation summaries have
been checked in small CPU tests. On one GPU, the fused-Adam step matched the
host-checked step on identical gradients, skipped non-finite steps without
moving the weights or the Adam state, and survived a checkpoint round trip.
The CI configs ``tests/e2e_tests/embodied/g1_mimic_supermimic*.yaml`` and
``tests/e2e_tests/evaluations/g1_mimic_supermimic_eval.yaml`` exercise
training, resume, two-GPU velocity tracking and evaluation through the RLinf
entries on a generated standing clip; ``tests/unit_tests/test_supermimic.py``
holds the numerical checks.
Full training with the current structure, the paper overrides, the paper's
two-GPU recipe, multiple nodes, Slurm, real BONES-SEED data and paper §4.3
T-pose recovery remain unvalidated.

Acknowledgements
----------------

Thanks to `CuboiLeo <https://github.com/CuboiLeo>`__ and
`zita-ch <https://github.com/zita-ch>`__ for publishing their SuperMimic
reproductions. The :doc:`algorithm page <../../reference/algorithms/supermimic>`
lists the choices this implementation takes from each.
