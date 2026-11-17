SuperMimic: G1 Motion Tracking with a Differentiable World Model
================================================================

This guide reproduces the reported SuperMimic result for Unitree G1 whole-body
motion tracking in MuJoCo Warp. You install the environment, prepare the 40
LAFAN1 clips, train the GenCoord recipe for 6000 updates on one GPU, resume
it if it stops, evaluate the final checkpoint, and compare the numbers with
`Visualization and Results`_. Every step uses the same run directory, so the
checkpoint you evaluate is the one your training wrote. The algorithm and
where it departs from the paper are in
:doc:`../../reference/algorithms/supermimic`; other clip sources are in
:doc:`supermimic_motion`, and evaluation metrics and push recovery in
:doc:`supermimic_eval`.

.. toctree::
   :hidden:

   Motion data <supermimic_motion>
   Evaluation <supermimic_eval>

Unless stated otherwise, every command on this page runs from the RLinf
repository root with the SuperMimic environment below activated.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      SuperMimic: 32-step backpropagation through a world model, plus a GAE critic

   .. grid-item-card:: Models
      :text-align: center

      World model, motion policy, and critic; Muon on hidden matrices, fused Adam on the rest

   .. grid-item-card:: Environments / Data
      :text-align: center

      MuJoCo Warp G1 (mjlab); LAFAN1, experimental BONES-SEED

   .. grid-item-card:: Training
      :text-align: center

      One env, rollout, and FSDP actor worker on each GPU

| **You'll do:** install -> prepare LAFAN1 -> train 6000 updates -> resume
  if needed -> evaluate update 6000 -> compare with the reported table.
| **Prerequisites:** one NVIDIA GPU (the reported runs use an A800), or the
  experimental AMD MI300X path; the RLinf repository
  (:doc:`installation <../../start/installation>`); network access to
  Hugging Face for the clips.

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
     - Track reference clips retargeted to the G1.
   * - State
     - ``gencoord`` is the 71-d simulator state, encoded in 68 dimensions.
       ``maxcoord`` is the pose and twist of the 14 tracked bodies, encoded
       in 214 dimensions. A checkpoint loads only into the representation it
       was trained with.
   * - Action
     - 29 joint setpoints, followed by a low-gain PD controller. Training
       adds action noise with standard deviation 0.1; evaluation applies the
       policy action as is.
   * - Goal
     - Ten reference frames, five control steps apart, held at the last frame
       of the clip.

How SuperMimic Works
--------------------

SuperMimic repeats three steps. It steps every environment once in MuJoCo
Warp and stores the transitions. When enough data has been collected, it
fits a world model on short windows of those transitions. It then rolls the
world model forward 32 steps, backpropagates the tracking return through it
to improve the policy, and fits a critic to the imagined returns. Episodes
restart more often in the parts of each clip where the policy earns little
return. In RLinf every GPU runs one simulation worker, one worker that serves
policy actions, and one FSDP training worker. The equations and the
differences from the official code are on the
:doc:`algorithm page <../../reference/algorithms/supermimic>`.

Installation
------------

The NVIDIA path runs simulation through MuJoCo Warp. From the RLinf
repository root, create a dedicated virtual environment:

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

In every new shell, return to the repository root and run
``source .venv-supermimic/bin/activate`` first.

The environment uses torch 2.14.0. With a CUDA 12 torch, the script pins
``warp-lang`` 1.18.0+cu12. Keep this environment separate from other RLinf
environments, because its requirements upgrade torch. ``docker/Dockerfile`` also
provides an ``embodied-supermimic-image`` target.

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
MuJoCo 3.8.1, and mujoco-warp 3.8.1. ``SUPERMIMIC_WARP_PATH`` sets the Warp
checkout (default ``../warp-rocm`` beside the repository) and
``SUPERMIMIC_HIP_ARCH`` the compile target (default ``gfx942``). The steps
below then apply unchanged.

This installer has not been validated through a full training run.

Prepare LAFAN1
--------------

The clips are the ``g1/`` folder of the Hugging Face dataset
`lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_.
The script downloads the 40 CSVs at pinned revision
``ce1572906efe6157840e8474d5a0d7aa87481e74``, checks every file's size and
SHA-256, converts them, and writes ``manifest.json``. The dataset card
declares no license; the retargets derive from Ubisoft La Forge
`LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_,
released under CC BY-NC-ND 4.0 (non-commercial, no derivatives). Read those
terms before passing ``--accept-license``.

On the training machine:

.. code-block:: bash

   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license

The last line reads ``wrote 40 clips, 2.45 h``. The folder holds
``clip_000.pt`` to ``clip_039.pt``, ``manifest.json``, and the downloaded
CSVs under ``csv/``. With the manifest present, the loader refuses a clip
whose SHA-256 differs or that the manifest does not list, so do not rename,
add, remove, or edit the prepared files.

Converting on Another Machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the training machine cannot reach Hugging Face, convert on a machine that
can. That machine needs an RLinf checkout and Python with torch, numpy, and
omegaconf, but not mjlab. The converter reads G1's default joint pose from
mjlab, so export it on the training machine first.

1. On the training machine, with the SuperMimic environment active, write
   the pose to ``g1_q0.pt``:

   .. code-block:: bash

      python -c "import torch; from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state; torch.save(asset_default_state()[7:36].clone(), 'g1_q0.pt')"

2. Copy ``g1_q0.pt`` to the conversion machine.
3. On the conversion machine, from its RLinf checkout, download and convert
   with that pose:

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 \
          --q0-file /path/to/g1_q0.pt --accept-license

4. Copy the whole ``/data/lafan1_g1`` folder, including ``manifest.json``,
   to the training machine at the same path.

Run It
------

Set these once in the shell that launches training and evaluation. Every
path below follows from ``RUN_DIR``:

.. code-block:: bash

   export PYTHONPATH=$PWD:$PYTHONPATH
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   export RUN_DIR=/data/runs/supermimic_gencoord

Train
~~~~~

The reported run is GenCoord on one GPU with the config defaults: 4096
environments, world-model batches of 2048 windows, and 4096 imagined
rollouts per update. The command sets 6000 updates, CUDA graph replay, and
a fixed GPU. Matmuls use TF32 and weights stay float32.

.. code-block:: bash

   TRAIN_ARGS=(--config-name supermimic_lafan1_gencoord
       runner.max_epochs=6000
       actor.cuda_graph=True
       runner.logger.log_path=$RUN_DIR
       '~cluster.component_placement'
       '+cluster.component_placement={actor:0,env:0,rollout:0}')
   python examples/embodiment/train_embodied_agent.py "${TRAIN_ARGS[@]}"

The two placement lines put the env, rollout, and actor workers on GPU 0;
change ``0`` to use another GPU. The run takes a little over an hour on one
A800. The log shows ``Recorded the SuperMimic update as a CUDA graph.``
after the first few updates, and a metrics table every 100 updates that
starts with ``Global Step:`` followed by the update count and ``/6000``.
Training writes a checkpoint every 2000 updates, under
``$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_<N>``.

Resume
~~~~~~

If training stops, rerun the same command with ``runner.resume_dir=auto``:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py "${TRAIN_ARGS[@]}" \
       runner.resume_dir=auto

``auto`` picks the latest complete checkpoint under
``$RUN_DIR/supermimic_lafan1_gencoord/checkpoints``. A checkpoint is
complete when it contains ``runner_state.json``, which is written last. The
checkpoint also holds the sharded networks, optimizers, and random state under
``actor/``, the reset statistics under ``env/``, and
``actor/model_state_dict/full_weights.pt`` for evaluation. Check that the log
contains ``Resuming from`` and the ``global_step_<N>`` you expect. If it says
``resume_dir=auto found no complete checkpoint``, the run started from
scratch: check ``RUN_DIR`` and the config name. The replay is not saved, so
a resumed run collects for a few seconds before its next update. Resume
with the same number of GPUs.

Evaluate
~~~~~~~~

Evaluate the final weights on the 40 clips. Each clip plays once from frame
0 and the policy action is applied as is:

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval_6000

The evaluation takes about 7 minutes. ``$RUN_DIR/eval_6000/summary.json``
holds ``normalized_return``, ``fail_rate``, and ``body_pos_err``, which the
table below reports; ``per_clip.csv`` has one row per clip. The metrics are
defined in :doc:`supermimic_eval`.

Train on Several GPUs
~~~~~~~~~~~~~~~~~~~~~

The environment count, the world-model batch, and the imagination batch are
totals over all GPUs, so adding GPUs splits the same update across them. Use a new ``RUN_DIR`` and list the
GPUs in the placement. With two GPUs:

.. code-block:: bash

   export RUN_DIR=/data/runs/supermimic_gencoord_2gpu
   TRAIN_ARGS=(--config-name supermimic_lafan1_gencoord
       runner.max_epochs=6000
       actor.cuda_graph=True
       runner.logger.log_path=$RUN_DIR
       '~cluster.component_placement'
       '+cluster.component_placement={actor:0-1,env:0-1,rollout:0-1}')
   python examples/embodiment/train_embodied_agent.py "${TRAIN_ARGS[@]}"

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

For four GPUs, use ``0-3`` in the placement. Each GPU hosts one env, one
rollout, and one actor worker. The learning rate and the number of updates
do not change with the GPU count. Resume and evaluation work as above with
the new ``RUN_DIR``. Multi-node and Slurm launches of this recipe have not
been validated.

MaxCoord
~~~~~~~~

To train the MaxCoord representation, use a new ``RUN_DIR`` and change
``--config-name`` to ``supermimic_lafan1_maxcoord``. Its checkpoints are under
``$RUN_DIR/supermimic_lafan1_maxcoord/checkpoints``, and evaluation needs
``actor.model.representation=maxcoord``. MaxCoord has no reported numbers
yet.

Configuration
-------------

The two training configs differ only in the state representation, and the
evaluation configs inherit GenCoord:

* ``examples/embodiment/config/supermimic_lafan1_gencoord.yaml`` and
  ``supermimic_lafan1_maxcoord.yaml`` for training;
* ``evaluations/supermimic/supermimic_lafan1_eval.yaml`` for motion tracking
  and ``supermimic_push_recovery.yaml`` for standing push recovery.

These are the keys most often changed. Each can be passed as a Hydra
override.

.. list-table::
   :header-rows: 1
   :widths: 40 12 48

   * - Key
     - Default
     - Meaning
   * - ``runner.max_epochs``
     - 400000
     - Global updates. The reported runs set 6000.
   * - ``runner.save_interval``
     - 2000
     - Updates between checkpoints. The last update also writes one.
   * - ``env.train.total_num_envs``
     - 4096
     - Environments across all GPUs. Must divide evenly by the GPU count.
   * - ``env.train.num_clips``
     - 40
     - Clips to load. ``null`` loads every clip in the folder.
   * - ``algorithm.replay_ratio``
     - 16
     - How many times each collected step is reused for the world model.
       At the defaults this is 0.4 updates per control step, so 6000
       updates collect about 15,000 control steps.
   * - ``algorithm.wm_window_batch``
     - 2048
     - World-model windows per batch, summed over GPUs.
   * - ``algorithm.imagine_batch``
     - 4096
     - Imagined rollouts per update, summed over GPUs.
   * - ``actor.cuda_graph``
     - False
     - Replay each update after the first three as one CUDA graph. The
       reported runs set ``True``. It requires a constant learning rate.
   * - ``actor.fsdp_config.amp_autocast.enabled``
     - False
     - Run network matmuls in bf16. Weights, losses, and optimizer state
       stay float32. This is not the paper's TF32 setting.

Visualization and Results
-------------------------

TensorBoard events go to ``$RUN_DIR/tensorboard``:

.. code-block:: bash

   tensorboard --logdir $RUN_DIR/tensorboard

The x-axis is the update count. ``env/sim/reward_mean`` and
``env/sim/terminal_rate`` show simulated tracking, ``train/actor/reward``
the imagined reward, and ``train/world_model/state_loss`` the world-model
fit. :doc:`../../reference/metrics` explains the shared metric names.

Reported Tracking Results
~~~~~~~~~~~~~~~~~~~~~~~~~

GenCoord on the 40 LAFAN1 clips with 4096 environments and seed 0, trained
with the commands in `Run It`_ and evaluated as in `Evaluate`_, on NVIDIA
A800 GPUs. RLinf runs use CUDA graph replay; the bf16 rows also set
``actor.fsdp_config.amp_autocast.enabled=True``. The official row is the
official SuperMimic code, which the authors of this integration ran on the
same cluster. Every run is evaluated at update 6000, about 15,000 control
steps.

.. list-table::
   :header-rows: 1
   :widths: 14 8 10 10 20 14 20

   * - Run
     - GPUs
     - Matmuls
     - Updates
     - ``normalized_return``
     - ``fail_rate``
     - ``body_pos_err`` (m)
   * - Official
     - 1
     - TF32
     - 6000
     - 0.830
     - 0.025
     - 0.027
   * - RLinf
     - 1
     - TF32
     - 6000
     - 0.846
     - 0
     - 0.024
   * - RLinf
     - 1
     - bf16
     - 6000
     - 0.857
     - 0
     - 0.023
   * - RLinf
     - 2
     - TF32
     - 6000
     - 0.843
     - 0.025
     - 0.023
   * - RLinf
     - 2
     - bf16
     - 6000
     - 0.833
     - 0.05
     - 0.023

* These runs predate the current version, which splits the three networks
  into separate FSDP units and changes how replay windows and critic
  minibatches are drawn. The current version has not yet been rerun at this
  scale, so these numbers are not yet confirmed for it.
* Each row is one seed. A ``fail_rate`` of 0.025 is one fall in 40 clips.
  Earlier checkpoints vary more between runs: across the four RLinf runs,
  ``normalized_return`` ranges from 0.672 to 0.732 at update 2000 and from
  0.765 to 0.794 at update 4000.
* The paper reports a normalized return of 0.82 on these 40 clips.

Training Speed
~~~~~~~~~~~~~~

Seconds per update for the same GenCoord recipe on A800 GPUs, with CUDA
graph replay. Each RLinf run had its GPUs to itself and ran 1200 updates;
the numbers cover the last 1000. ``time/step`` is the mean of the logged
per-update timer, which covers collection, the update, and the weight sync.
Wall clock divides the elapsed time of those 1000 updates by 1000 and also
covers logging. The runs used the same earlier version as the table above;
the current version has not been timed yet.

.. list-table::
   :header-rows: 1
   :widths: 16 10 12 22 22

   * - Run
     - GPUs
     - Matmuls
     - ``time/step`` (s)
     - Wall clock (s)
   * - Official
     - 1
     - TF32
     - --
     - 0.515
   * - RLinf
     - 1
     - TF32
     - 0.603
     - 0.637
   * - RLinf
     - 1
     - bf16
     - 0.515
     - 0.548
   * - RLinf
     - 2
     - TF32
     - 0.486
     - 0.521
   * - RLinf
     - 2
     - bf16
     - 0.449
     - 0.483
   * - RLinf
     - 4
     - TF32
     - 0.421
     - 0.456
   * - RLinf
     - 4
     - bf16
     - 0.407
     - 0.442

The authors of this integration measured the official time on the same
cluster, as the wall clock between two checkpoints of an official MaxCoord
run.
About 4% of RLinf's time is a pause of about 2 s that recurs once a minute
on the rollout side.

Scope
-----

CI runs the unit tests in ``tests/unit_tests/test_supermimic.py``, a short
GenCoord training run with a resume and a standalone evaluation, and a short
two-GPU MaxCoord run, all on a generated standing clip. The tables above
evaluate GenCoord at update 6000 on the 40 LAFAN1 clips. MaxCoord accuracy,
runs longer than 6000 updates, multi-node, Slurm, training on BONES-SEED,
and the paper's Section 4.3 T-pose recovery have no numbers here.
