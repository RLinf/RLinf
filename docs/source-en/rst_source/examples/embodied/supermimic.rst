SuperMimic
==========

Train a 29-DoF Unitree G1 whole-body motion-tracking policy with **SuperMimic** (`arXiv:2610.28480 <https://arxiv.org/abs/2610.28480>`__, `project page <https://supermimic.github.io>`__) in MuJoCo Warp. SuperMimic learns a differentiable dynamics world model from simulation rollouts, backpropagates exact tracking returns through 32-step imagined rollouts with a critic for value bootstrapping, and focuses resets on low-return clip segments. On 40 LAFAN1 clips, a 6000-update 4-GPU GenCoord run finishes in ~23 minutes on NVIDIA A800 and reaches a normalized return of ``0.856`` with ``0`` falls.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      SuperMimic (Differentiable World Model + Actor-Critic)

   .. grid-item-card:: Models
      :text-align: center

      Dynamics World Model · Motion Policy · Critic

   .. grid-item-card:: Environments / Data
      :text-align: center

      :doc:`G1 Motion Tracking <g1_mimic>` · 40 LAFAN1 clips

   .. grid-item-card:: Training
      :text-align: center

      1–4 NVIDIA GPUs (FSDP2) · :ref:`AMD ROCm (experimental) <supermimic-hardware>`

| **You'll do:** install & prepare LAFAN1 → launch training → watch TensorBoard → evaluate checkpoints.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · one or more NVIDIA GPUs (or AMD ROCm) · network access to Hugging Face for LAFAN1 clips.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 28 24 26

   * - Recipe
     - Config
     - State Representation
     - Task
   * - GenCoord (default)
     - ``supermimic_lafan1_gencoord``
     - Generalized coordinates (68-dim)
     - 40 LAFAN1 clips on ``g1_mimic``
   * - MaxCoord
     - ``supermimic_lafan1_maxcoord``
     - Tracked-body poses & twists (214-dim)
     - 40 LAFAN1 clips on ``g1_mimic``

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

For the 14 tracked bodies, PD control parameters, reward weights, and domain randomization, see :doc:`G1 Motion Tracking Environment <g1_mimic>`.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Summary
   * - Observation
     - Current G1 state encoding (GenCoord or MaxCoord) plus 10 future reference frames spaced 5 control steps apart (0.1 s stride).
   * - Action
     - 29-dim target joint angles at 50 Hz (Gaussian noise ``0.1`` in training; deterministic in evaluation).
   * - Reward
     - 6 tracking kernels plus action-rate and soft joint-limit penalties, scaled by ``0.02 s`` (ceiling ``0.1`` per step).
   * - Prompt
     - Not used.

How SuperMimic Works
--------------------

Each global update runs three phases in lockstep across GPUs:

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/supermimic_loop.svg
   :align: center
   :width: 100%
   :alt: SuperMimic three-phase training loop: simulation collection, world-model fitting, and imagined policy and critic optimization

   SuperMimic three-phase training loop in RLinf.

1. **Collect and return-based reset.** Parallel G1 environments step in MuJoCo Warp with exploration noise and append transitions to each GPU's local replay buffer. Finished episodes reset preferentially to low-return clip segments so training focuses on difficult motions.
2. **Fit the world model.** The dynamics network unrolls ``k = 8`` consecutive steps from replay states under recorded actions, predicts velocity increments, and integrates them into the next pose using Bryson-weighted per-channel MSE.
3. **Optimize policy and critic.** Starting from replay states, the policy rolls forward ``H = 32`` steps inside the frozen world model. Differentiable forward kinematics computes the tracking reward and a smooth termination probability weights survival at every step, sending exact first-order gradients back to the policy. The critic then fits GAE lambda-returns computed on the same imagined rollouts.

``actor.model.representation`` switches between ``gencoord`` (default, 68-dim generalized coordinates with differentiable forward kinematics) and ``maxcoord`` (214-dim tracked-body poses and twists with chain inverse kinematics). Checkpoints load only into the representation they were trained with. For state layouts, equations, network dimensions, optimizers, FSDP2 sharding, and differences from the paper, see :doc:`../../reference/algorithms/supermimic`.

.. _supermimic-hardware:

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. Install**

      Create a dedicated virtual environment from the repository root (pins PyTorch, MuJoCo Warp, and ``warp-lang`` separately from other RLinf environments):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
         source .venv-supermimic/bin/activate

      **2. Prepare LAFAN1**

      Download, verify, and convert the 40 LAFAN1 Unitree G1 clips (`lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_, originating from Ubisoft La Forge `LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_ under CC BY-NC-ND 4.0; review terms before passing ``--accept-license``):

      .. code-block:: bash

         python toolkits/supermimic/prepare_lafan1.py --out ./data/lafan1_g1 --accept-license

      A successful run ends with ``[lafan1] wrote 40 clips, 2.45 h, manifest data/lafan1_g1/manifest.json``. For offline conversion or custom ``clip_*.pt`` files, see :doc:`supermimic_motion`.

      **3. Launch**

      Set the clip directory, disable MuJoCo's OpenGL renderer, and launch the 6000-update GenCoord run on GPU 0 (CUDA graph replay and bf16 mixed precision are enabled by default):

      .. code-block:: bash

         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/lafan1_g1
         bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord

      The first run spends ~2–3 minutes in ``torch.compile`` warmup before printing ``Recorded the SuperMimic update as a CUDA graph.`` after update 3. Every 100 updates it prints a metrics table starting with ``Global Step: <N>/6000``. Checkpoints are written under ``logs/<timestamp>-supermimic_lafan1_gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_<N>/`` every 2000 updates and at step 6000.

   .. tab-item:: AMD ROCm (Experimental)
      :sync: rocm

      The AMD path runs MuJoCo Warp on ROCm Warp and targets MI300X (``gfx942``). Prerequisites: Linux x86_64, ROCm 7.2 SDK with ``hipcc``, a C++ toolchain (see the `ROCm Warp gfx942 Dockerfile <https://github.com/ROCm/warp/blob/36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71/docker/rocm/Dockerfile>`_), and a recent ``uv`` supporting flat wheel indexes.

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export ROCM_PATH=/opt/rocm
         bash requirements/install.sh --platform amd --rocm 7.2 --no-flash-attn \
             embodied --env supermimic --venv .venv-supermimic
         source .venv-supermimic/bin/activate

      The installer fetches PyTorch 2.9.1 from AMD's index, builds ROCm Warp ``1.13.0+rocm.0``, and installs mjlab 1.2.0, MuJoCo 3.8.1, and MuJoCo Warp 3.8.1. To customize the Warp checkout path or target GPU architecture, set ``SUPERMIMIC_WARP_PATH`` (default ``../warp-rocm``) or ``SUPERMIMIC_HIP_ARCH`` (default ``gfx942``).

      **2. Prepare LAFAN1**

      .. code-block:: bash

         python toolkits/supermimic/prepare_lafan1.py --out ./data/lafan1_g1 --accept-license

      **3. Launch**

      .. code-block:: bash

         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/lafan1_g1
         bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord

      .. note::

         Full end-to-end training convergence on AMD ROCm is untested.

.. dropdown:: Multi-GPU Training

   Environment count, world-model batches, and imagined rollouts are global totals divided evenly across GPUs, so the learning rate and total update count stay unchanged:

   .. list-table::
      :header-rows: 1
      :widths: 40 20 20 20

      * - Per Update
        - Global Total
        - Per GPU (2 GPUs)
        - Per GPU (4 GPUs)
      * - Parallel environments
        - 4096
        - 2048
        - 1024
      * - World-model windows (×10 batches)
        - 2048
        - 1024
        - 512
      * - Imagined rollouts
        - 4096
        - 2048
        - 1024

   To run on two GPUs (use ``0-3`` for four GPUs):

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord \
          '~cluster.component_placement' \
          '+cluster.component_placement={actor:0-1,env:0-1,rollout:0-1}'

.. dropdown:: Resume Training

   Pass the original run directory and ``runner.resume_dir=auto`` on the same GPU count (the log prints ``Resuming from <path>`` when a checkpoint loads):

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord \
          runner.logger.log_path=./logs/<timestamp>-supermimic_lafan1_gencoord \
          runner.resume_dir=auto

.. dropdown:: Train with MaxCoord

   Switch the config name to ``supermimic_lafan1_maxcoord``:

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh supermimic_lafan1_maxcoord

.. dropdown:: Evaluate a Checkpoint

   Run motion-tracking evaluation on all 40 LAFAN1 clips (append ``rollout.model.representation=maxcoord`` for MaxCoord checkpoints; for standing push recovery and metric definitions, see :doc:`../../evaluations/guides/supermimic`):

   .. code-block:: bash

      bash evaluations/run_eval.sh supermimic_lafan1_eval \
          runner.ckpt_path=./logs/<timestamp>-supermimic_lafan1_gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
          runner.eval_export.output_dir=./eval_results/supermimic_6000

.. dropdown:: Key Algorithm Configuration

   Default config: ``examples/embodiment/config/supermimic_lafan1_gencoord.yaml`` (``supermimic_lafan1_maxcoord.yaml`` inherits it).

   .. list-table::
      :header-rows: 1
      :widths: 36 16 48

      * - Config Key
        - Default
        - Meaning
      * - ``actor.model.representation``
        - ``gencoord``
        - State representation: ``gencoord`` or ``maxcoord``
      * - ``algorithm.k`` / ``algorithm.H``
        - ``8`` / ``32``
        - World-model unroll window and policy imagination horizon
      * - ``algorithm.replay_ratio``
        - ``16``
        - Average draws per transition by the world model (0.4 updates/step at 4096 envs)
      * - ``algorithm.wm_batches`` / ``wm_window_batch``
        - ``10`` / ``2048``
        - World-model batches per update and global windows per batch
      * - ``algorithm.imagine_batch``
        - ``4096``
        - Global number of imagined rollouts per policy update
      * - ``actor.cuda_graph``
        - ``True``
        - Capture each global update as one CUDA graph (set ``False`` to disable)
      * - ``actor.fsdp_config.mixed_precision.param_dtype``
        - ``bf16``
        - Cast network weights to bf16 once per phase while keeping master weights, optimizer state, gradient reductions, kinematics, and losses in float32 (set ``null`` for TF32)

Visualization and Results
-------------------------

Watch training progress in TensorBoard (``tensorboard --logdir ./logs``); see :doc:`../../reference/metrics` for shared metric names. Key metrics (x-axis is update count):

* ``env/sim/reward_mean``: Mean simulated per-step tracking reward (rises toward its ceiling of ``0.1``).
* ``env/sim/terminal_rate``: Fraction of environments falling per step (drops toward ``0``).
* ``train/world_model/state_loss``: Bryson-weighted world-model prediction loss.

Tracking Quality (LAFAN1, 40 Clips)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4-GPU GenCoord run on NVIDIA A800 (seed 0, default bf16; ~23 min wall time):

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Updates
     - ``normalized_return``
     - ``fail_rate``
     - ``body_pos_err`` (m)
   * - 6000
     - 0.856
     - 0.000
     - 0.023

All 40 clips complete without falling (the paper reports ``0.82`` normalized return on these clips).

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/supermimic_gencoord_curves.png
   :alt: SuperMimic GenCoord training curves
   :width: 100%

   Training curves of a 6000-update GenCoord run on NVIDIA A800. Simulated reward reaches up to 5.0/s (0.1 per 50 Hz control step); the imagined return (top right) includes the terminal critic bootstrap value and therefore exceeds 5.

Throughput Comparison
~~~~~~~~~~~~~~~~~~~~~

Seconds per global update for the GenCoord recipe on dedicated NVIDIA A800 GPUs versus the official single-GPU implementation (wall-clock mean over 400 updates after 200 warm-up updates; over 1000 updates, 4-GPU bf16 gives the same 0.227 s). RLinf bf16 rows use the default config (CUDA graphs and ``no_shard``); TF32 rows disable bf16 via ``actor.fsdp_config.mixed_precision.param_dtype=null`` and ``rollout.amp_autocast.enabled=False``:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Implementation
     - GPUs
     - Matmul
     - s/update
   * - Official
     - 1
     - TF32
     - 0.515
   * - RLinf
     - 1
     - TF32
     - 0.490
   * - RLinf
     - 1
     - bf16
     - 0.383
   * - RLinf
     - 2
     - TF32
     - 0.341
   * - RLinf
     - 2
     - bf16
     - 0.285
   * - RLinf
     - 4
     - TF32
     - 0.263
   * - RLinf
     - 4
     - bf16
     - 0.227

On 1 GPU with TF32, RLinf is ~5% faster per update than the single-process official code (0.490 s vs. 0.515 s). With default bf16 mixed precision, 1 GPU reaches 0.383 s (1.34× faster) and 4 GPUs reach 0.227 s (2.27× faster, finishing 6000 updates in ~23 minutes).
