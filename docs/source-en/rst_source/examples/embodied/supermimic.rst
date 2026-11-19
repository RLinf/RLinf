SuperMimic: G1 Motion Tracking with a Differentiable World Model
================================================================

SuperMimic trains a whole-body motion tracking policy for the Unitree G1 by
backpropagating the tracking return through a learned world model. This page
takes you through one run in MuJoCo Warp: install the dependencies, prepare
the 40 LAFAN1 clips, train the GenCoord recipe on one GPU, evaluate the
checkpoint the run writes, and find the files it leaves behind. The
algorithm and its differences from the paper are on the
:doc:`algorithm page <../../reference/algorithms/supermimic>`.

.. toctree::
   :hidden:

   Motion data <supermimic_motion>
   Evaluation <supermimic_eval>

Overview
--------

You train one G1 tracking policy on the 40 LAFAN1 clips and evaluate it on
the same clips.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      SuperMimic: 32-step backpropagation through a world model, plus a critic

   .. grid-item-card:: Models
      :text-align: center

      World model, tracking policy, and critic

   .. grid-item-card:: Environments / Data
      :text-align: center

      MuJoCo Warp G1 (mjlab); LAFAN1 clips

   .. grid-item-card:: Training
      :text-align: center

      One GPU; several GPUs on one node

| **You'll do:** install → prepare LAFAN1 → train on one GPU → evaluate
  the checkpoint → find the outputs.
| **Prerequisites:** one NVIDIA GPU; the RLinf repository
  (:doc:`installation <../../start/installation>`); network access to
  Hugging Face for the clips.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Description
   * - Observation
     - The robot's state and the next ten reference frames of the clip, five
       control steps apart. GenCoord, used on this page, describes the state
       in the simulator's generalized coordinates; MaxCoord describes it by
       the poses and velocities of 14 tracked bodies.
   * - Action
     - Setpoints for the 29 joints at 50 Hz, tracked by a PD controller.
       Training adds noise to the action; evaluation applies it as is.
   * - Reward
     - How closely the robot follows the clip, at most 0.1 per control step.
       A fall ends the episode.

How SuperMimic Works
--------------------

SuperMimic repeats three steps. It steps every environment once in MuJoCo
Warp and stores the transitions. It fits a world model on short windows of
the stored transitions. It then rolls the world model forward 32 steps,
backpropagates the tracking return through it to improve the policy, and
fits a critic to the imagined returns. Episodes restart more often in the
parts of a clip where the policy earns little return. The equations are on
the :doc:`algorithm page <../../reference/algorithms/supermimic>`.

Installation
------------

Create a dedicated virtual environment from the RLinf repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

This virtual environment pins its own torch and ``warp-lang`` versions, so
keep it apart from other RLinf virtual environments. ``docker/Dockerfile``
also has an ``embodied-supermimic-image`` build target. For AMD GPUs, see
`Run on Different Hardware Backends`_.

Every command below runs from the repository root with this virtual
environment active.

Prepare LAFAN1
--------------

The clips are the ``g1/`` folder of the Hugging Face dataset
`lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_.
The dataset card declares no license. The retargeted clips come from Ubisoft
La Forge `LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_,
released under CC BY-NC-ND 4.0 (non-commercial, no derivatives); read those
terms before you pass ``--accept-license``.

Download, check, and convert the 40 clips:

.. code-block:: bash

   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license

The script downloads a pinned revision, checks the size and SHA-256 of every
file, and writes the converted clips with a ``manifest.json``. Its last line
reads:

.. code-block:: text

   [lafan1] wrote 40 clips, 2.45 h, manifest /data/lafan1_g1/manifest.json

Training refuses a clip that the manifest does not list or whose checksum
differs, so leave the files in ``/data/lafan1_g1`` as written.

Convert on Another Machine
~~~~~~~~~~~~~~~~~~~~~~~~~~

If the training machine cannot reach Hugging Face, convert on one that can.
That machine needs an RLinf checkout and Python with torch, numpy, and
omegaconf. The converter also needs the G1's default joint pose, which only
the training machine can export.

1. On the training machine, write the pose to ``g1_q0.pt``:

   .. code-block:: bash

      python -c "import torch; from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state; torch.save(asset_default_state()[7:36].clone(), 'g1_q0.pt')"

2. Copy ``g1_q0.pt`` to the conversion machine.
3. On the conversion machine, from its RLinf checkout, download and convert
   with that pose:

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 \
          --q0-file /path/to/g1_q0.pt --accept-license

4. Copy the whole ``/data/lafan1_g1`` folder to the training machine at the
   same path.

Run It
------

Training, resuming, and evaluation share one shell setup and one run
directory, so the evaluation reads the checkpoint that this training run
writes.

Set Up the Shell
~~~~~~~~~~~~~~~~

Run this block in every new shell before any command below:

.. code-block:: bash

   export PYTHONPATH=$PWD:$PYTHONPATH
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   export RUN_DIR=/data/runs/supermimic_gencoord
   GPU_PLACEMENT=('~cluster.component_placement'
       '+cluster.component_placement={actor:0,env:0,rollout:0}')
   TRAIN_ARGS=(--config-name supermimic_lafan1_gencoord
       runner.max_epochs=6000
       actor.cuda_graph=True
       runner.logger.log_path=$RUN_DIR
       "${GPU_PLACEMENT[@]}")

* ``SUPERMIMIC_MOTION_PATH`` is the clip folder from `Prepare LAFAN1`_.
* ``RUN_DIR`` receives the logs, checkpoints, and evaluation results.
* ``GPU_PLACEMENT`` puts every worker on GPU 0. Change ``0`` to use another
  GPU; without it, training and evaluation take every GPU on the machine.
* ``TRAIN_ARGS`` selects the GenCoord config and sets 6000 updates with CUDA
  graph replay.

.. warning::

   ``TRAIN_ARGS`` exists only in the shell that defined it. In a new shell
   without it, the training command falls back to the entry script's default
   config, which is not SuperMimic.

Train on One GPU
~~~~~~~~~~~~~~~~

This run uses the config defaults: 4096 environments, world-model batches of
2048 windows, and 4096 imagined rollouts per update, with TF32 matmuls and
float32 weights. Start training:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py "${TRAIN_ARGS[@]}"

After the first few updates the log prints
``Recorded the SuperMimic update as a CUDA graph.``. Every 100 updates it
prints a metrics table that starts with ``Global Step:``, followed by the
update count and ``/6000``. Training writes a checkpoint every 2000 updates
and after the last one.

If training stops, run the setup block in a new shell and resume from the
latest complete checkpoint:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py "${TRAIN_ARGS[@]}" \
       runner.resume_dir=auto

The log names the checkpoint after ``Resuming from``. If it reports
``resume_dir=auto found no complete checkpoint``, the run started from
scratch; check ``RUN_DIR`` and the config name. Resume on the same number of
GPUs.

Evaluate the Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~

Play each of the 40 clips once from its first frame with the weights from
update 6000:

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval_6000 \
       "${GPU_PLACEMENT[@]}"

The log ends with ``Wrote SuperMimic evaluation results to`` and the output
folder. To evaluate update 2000 or 4000, replace ``6000`` in both paths.
:doc:`supermimic_eval` defines the metrics and adds standing push recovery.

Find the Outputs
~~~~~~~~~~~~~~~~

Everything from this run is under ``RUN_DIR``:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Path under ``$RUN_DIR``
     - Contents
   * - ``metrics.log``
     - The metrics tables printed during training
   * - ``tensorboard/``
     - TensorBoard events of the training run
   * - ``supermimic_lafan1_gencoord/checkpoints/global_step_<N>/``
     - One checkpoint, which ``runner.resume_dir=auto`` resumes from
   * - ``.../global_step_<N>/actor/model_state_dict/full_weights.pt``
     - The weights that evaluation loads
   * - ``eval_6000/summary.json``
     - The evaluation metrics over the 40 clips
   * - ``eval_6000/per_clip.csv``
     - One row of metrics per clip
   * - ``eval_6000/config.yaml``
     - The resolved evaluation config

Visualization and Results
-------------------------

Watch training in TensorBoard:

.. code-block:: bash

   tensorboard --logdir $RUN_DIR/tensorboard

The x-axis is the update count. ``env/sim/reward_mean``, the simulated
reward per step, rises toward its ceiling of 0.1, and
``env/sim/terminal_rate``, the share of environments that fall each step,
drops toward zero. ``train/world_model/state_loss`` shows how well the world
model predicts the next state. :doc:`../../reference/metrics` explains the
shared metric names.

The evaluation result is ``normalized_return`` in
``$RUN_DIR/eval_6000/summary.json``: the reward each clip collected divided
by the most it could collect, averaged over the 40 clips. ``fail_rate`` is
the share of clips in which the robot fell, and ``body_pos_err`` the mean
position error of the tracked bodies in meters.

One run of this recipe on one NVIDIA A800, seed 0:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Updates
     - ``normalized_return``
     - ``fail_rate``
     - ``body_pos_err`` (m)
   * - 6000
     - 0.850
     - 0
     - 0.025

All 40 clips played to their last frame. The paper reports a normalized
return of 0.82 on these clips.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/supermimic_gencoord_curves.png
   :alt: SuperMimic GenCoord training curves
   :width: 100%

   Training curves of the same run. The imagined return per second includes
   the critic's value at the end of each imagined rollout, so it exceeds the
   5 per second that the simulated reward alone can reach.

Other Setups
------------

Train on Several GPUs
~~~~~~~~~~~~~~~~~~~~~

The environment count, the world-model batch, and the number of imagined
rollouts are totals over all GPUs, so more GPUs split the same update. In the setup block,
use a new ``RUN_DIR`` and list the GPUs in ``GPU_PLACEMENT``. For two GPUs,
change these two lines, run the whole block, and then run the training
command:

.. code-block:: bash

   export RUN_DIR=/data/runs/supermimic_gencoord_2gpu
   GPU_PLACEMENT=('~cluster.component_placement'
       '+cluster.component_placement={actor:0-1,env:0-1,rollout:0-1}')

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Per update
     - Total
     - Per GPU, 2 GPUs
     - Per GPU, 4 GPUs
   * - Environments
     - 4096
     - 2048
     - 1024
   * - World-model windows per batch (10 batches)
     - 2048
     - 1024
     - 512
   * - Imagined rollouts
     - 4096
     - 2048
     - 1024

For four GPUs, use ``0-3``. The learning rate and the number of updates stay
the same. To evaluate on one GPU, set ``GPU_PLACEMENT`` back to GPU 0 and
keep the new ``RUN_DIR``. Multi-node and Slurm launches have not been
validated with this recipe.

Train MaxCoord
~~~~~~~~~~~~~~

Use a new ``RUN_DIR`` and change ``--config-name`` in ``TRAIN_ARGS`` to
``supermimic_lafan1_maxcoord``. Its checkpoints are under
``$RUN_DIR/supermimic_lafan1_maxcoord/checkpoints``, and its evaluation
command adds ``actor.model.representation=maxcoord``.

To train on other clips, see :doc:`supermimic_motion`.

.. _supermimic-hardware:

Run on Different Hardware Backends
----------------------------------

AMD ROCm
~~~~~~~~

The AMD path runs MuJoCo Warp on ROCm Warp and targets the MI300X
(``gfx942``). You need Linux x86_64, the ROCm 7.2 SDK with ``hipcc``, and a
C++ toolchain; the
`ROCm Warp gfx942 Docker build steps <https://github.com/ROCm/warp/blob/36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71/docker/rocm/Dockerfile>`_
give a working environment. Install from the repository root:

.. code-block:: bash

   export ROCM_PATH=/opt/rocm  # set this to the SDK install path
   bash requirements/install.sh --platform amd --rocm 7.2 --no-flash-attn \
       embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

The script installs torch 2.9.1 from AMD's wheel index, which needs a recent
uv that reads flat indexes. It builds ROCm Warp ``1.13.0+rocm.0`` from the
commit above and installs mjlab 1.2.0, MuJoCo 3.8.1, and mujoco-warp 3.8.1.
``SUPERMIMIC_WARP_PATH`` sets the Warp checkout (default ``../warp-rocm``)
and ``SUPERMIMIC_HIP_ARCH`` the compile target (default ``gfx942``).

`Prepare LAFAN1`_ and `Run It`_ then apply unchanged. This path has not been
validated through a full training run.
