Diffusion Model Fine-Tuning with ART
====================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   SD3 and Wan2.2 generation models trained with DiffusionART.

Fine-tune SD3 or Wan2.2 with DiffusionART to improve text rendering using OCR
rewards. ART combines rewards and clean latent endpoints within each prompt
group into a shared direction, then trains against frozen velocity targets at
states visited during generation. Complete the :doc:`shared setup <video_models>`,
choose an image or video recipe, and inspect the training results below.

Overview
--------

Both recipes use LoRA with FSDP2 and deterministic ODE generation. They split
each prompt's candidates into two disjoint update windows that share targets
frozen before the first update. Corrective masking suppresses updates that would
move an already-drifted prediction farther outward. For the derivation and
reference implementation, see the
`DiffusionART paper <https://arxiv.org/abs/2610.12572>`_,
`official code <https://github.com/vanzll/DiffusionART>`_, and
`project page <https://vanzll.github.io/diffusionart-project/>`_.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: Algorithm
      :text-align: center

      DiffusionART

   .. grid-item-card:: Output
      :text-align: center

      Image / Video

   .. grid-item-card:: Reward
      :text-align: center

      OCR / Video OCR

| **You'll do:** select a recipe → launch training → inspect OCR reward and generated samples.
| **Prerequisites:** :doc:`Shared installation, model, and dataset setup <video_models>`. The recorded Wan experiment used four NVIDIA A800 GPUs.

Tasks
~~~~~

Choose the config for the output you want to generate.

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   * - Model
     - Config
     - Task
   * - SD3.5 Medium
     - ``sd3_art_ocr``
     - Image OCR
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_art_video_ocr``
     - Text-to-video OCR

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The environment scores each completed candidate; video rewards are averaged
before ART builds guidance for the prompt group.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Value
     - Meaning
   * - Observation
     - A text prompt containing quoted target text.
   * - Action
     - The generated image or video.
   * - Reward
     - One scalar OCR similarity score in ``[0, 1]`` per candidate.

Installation and Assets
-----------------------

Follow the shared :ref:`installation <diffusion-model-setup>`,
:ref:`model download <diffusion-model-checkpoints>`, and
:ref:`OCR dataset download <diffusion-ocr-dataset>` instructions. Both ART
recipes set ``env.*.reward.use_gpu: False`` and use the CPU OCR installation.

Run Training
------------

Open ``examples/diffusion/config/sd3_art_ocr.yaml`` or
``examples/diffusion/config/wan22_ti2v_5b_art_video_ocr.yaml`` before launching.
In the selected YAML, set ``actor.model.model_path``,
``env.train.dataset.path``, and ``env.eval.dataset.path`` to your downloaded
directories. Check ``cluster.num_nodes``, ``cluster.component_placement``, and
the batch settings against your hardware and the ART constraints below.

After checking the YAML, run its config name from the repository root:

.. code:: bash

   bash examples/diffusion/run_diffusion.sh sd3_art_ocr

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr

The launcher generates and scores candidates, updates LoRA parameters, and
writes logs under ``logs/<timestamp>-<config>/``.

Configure the ART Update
------------------------

When changing rollout size or placement, each actor shard must receive
complete prompt groups.
``algorithm.art_updates_per_rollout`` must be ``1`` or ``2`` and divide
``algorithm.group_size``; ``env.train.group_size`` must match that group size.

.. warning::

   Set ``actor.global_batch_size`` to
   ``env.train.total_num_envs * env.train.rollout_epoch`` divided by
   ``algorithm.art_updates_per_rollout``. For 96 environments, 12 rollout
   epochs, and two updates, use ``actor.global_batch_size=576``. Each per-rank
   update window must be divisible by ``actor.micro_batch_size``. Environment
   counts must preserve complete groups across both env and actor ranks.

ART requires ``actor.training_backend=fsdp``, ``rollout.pipeline_stage_num=1``,
``algorithm.update_epoch=1``, and ``runner.weight_sync_interval=1``. Keep
``env.train.max_steps_per_rollout_epoch=actor.model.num_action_chunks`` so that
each rollout epoch produces one complete candidate. SD3 uses
``actor.model.sd3.train_sampler=ode`` with ``actor.model.sd3.noise_level=0.0``;
Wan records ODE trajectories.

.. list-table:: Main Update Options
   :header-rows: 1
   :widths: 40 60

   * - Option
     - Effect
   * - ``algorithm.art_timestep_fraction``
     - Fraction of denoising states used for training: ``0.5`` for SD3 and ``0.25`` for Wan in the supplied recipes.
   * - ``algorithm.art_guidance_gain``
     - Scales reward guidance; both recipes use ``1.0``.
   * - ``algorithm.art_corrective_mask_enabled``
     - Enables the corrective movement mask; both recipes use ``True``.
   * - ``algorithm.art_mask_threshold``
     - Movement threshold for the mask; both recipes use ``1e-5`` with ``art_mask_metric=sde_kl``.

.. _diffusion-art-results:

Expected Results
----------------

Watch ``env/video_ocr`` for Wan or ``env/avg`` for SD3 and inspect generated
samples. ``actor/active_group_frac`` shows whether prompt groups supply guidance;
``actor/skipped_zero_guidance`` marks rounds with no usable guidance on any rank.
Use :doc:`Training Metrics </rst_source/reference/metrics>` for logging and
TensorBoard setup. OCR reward alone does not measure visual quality or temporal
consistency, and ART and NFT losses are not directly comparable.

The runs started on November 11, 2026 used four NVIDIA A800 80 GB GPUs,
Wan2.2 TI2V 5B, LoRA (rank ``32``, alpha ``64``), FSDP2, BF16, and CPU OCR.
The first table compares **40,320 generated trajectories**: round 35 for ART
and round 70 for NFT. For each run, reward is the mean ``env/video_ocr`` over
the last 5760 trajectories at that budget. The NFT entry reports the mean and
sample standard deviation across two runs; the ART entry is one run.

.. list-table:: Wan Video OCR at 40,320 Generated Trajectories
   :header-rows: 1
   :widths: 24 8 16 12 20 20

   * - Method
     - Runs
     - Trajectories per Round
     - Learning Rate
     - Training Reward
     - Elapsed Time (h)
   * - ART
     - 1
     - 1152
     - 1e-4
     - 0.8584
     - 10.01
   * - NFT
     - 2
     - 576
     - 4e-4
     - 0.7565 ± 0.0544
     - 11.29 ± 0.52

At this sample budget, ART's reward is **13.5% higher** than the NFT mean
(an absolute increase of ``0.1019``), with **11.3% less elapsed training time**.
Elapsed time comes from the training-loop log and excludes model initialization.
ART first reached a reward of ``0.8`` over the same 5760-trajectory window at
21,888 trajectories (5.48 h); neither NFT run reached that windowed threshold
within 40,320 trajectories.

The NFT run with the same trajectories per round and learning rate as ART
recorded 29 rounds before it was stopped. Compare both at **33,408 generated
trajectories**, using the same last-5760-trajectory reward window. Each entry
below is one run.

.. list-table:: Matched Trajectories per Round and Learning Rate
   :header-rows: 1
   :widths: 18 22 18 22 20

   * - Method
     - Trajectories per Round
     - Learning Rate
     - Training Reward
     - Elapsed Time (h)
   * - ART
     - 1152
     - 1e-4
     - 0.8446
     - 8.31
   * - NFT
     - 1152
     - 1e-4
     - 0.1076
     - 8.23

With matched trajectories per round and learning rate, ART's reward is
``0.7370`` higher (**685.2% relative improvement**), with similar elapsed time.
This comparison covers only the first 29 rounds, not final convergence.
These are training rewards, not held-out evaluation scores. The 576-trajectory
NFT configuration in the first table also changes learning rate and reward
reduction; that comparison does not isolate the algorithm's effect.

To reproduce the configurations, use ``wan22_ti2v_5b_art_video_ocr`` for ART or
``wan22_ti2v_5b_nft_video_ocr`` for NFT with the model and data paths prepared
above. Set ``cluster.component_placement`` to ``{"actor,env,rollout": "0,1,2,3"}``,
``env.train.total_num_envs=96``, ``algorithm.group_size=8``, and
``runner.val_check_interval=-1``. Set ``env.train.reward.use_gpu`` and
``env.eval.reward.use_gpu`` to ``False``. Under ``actor.model.wan22_ti2v_5b``,
use ``max_generation_batch_size=2``, ``compile_transformer_forward=True``, and
``offload_auxiliary_modules=True``; retain the recipes' 512 × 512 resolution,
five frames, ten denoising steps, and guidance scale of ``1.0``.

For ART, set ``env.train.rollout_epoch=12``, ``actor.global_batch_size=576``,
``actor.micro_batch_size=2``, ``actor.optim.lr=1e-4``,
``runner.save_interval=10``, ``runner.max_epochs=70``, and
``runner.max_steps=70``. For NFT with 576 trajectories per round, use
``env.train.rollout_epoch=6``, ``actor.global_batch_size=288``,
``actor.micro_batch_size=1``, ``actor.optim.lr=4e-4``,
``runner.save_interval=-1``, ``runner.max_epochs=70``, and
``runner.max_steps=70``. Set ``env.*.reward.reward_type`` to ``video_level``
for ART or ``frame_level`` for this NFT configuration.

For NFT with 1152 trajectories per round, use the NFT recipe with the ART
overrides above, including ``video_level`` rewards, but set
``runner.max_epochs=200`` and ``runner.max_steps=200``; the recorded run was
stopped after round 29. Both NFT configurations retain
``algorithm.nft_tau=[1.0,0.01,0,70]``.

All runs used ``actor.seed=42`` and environment seeds of ``42``. These settings
do not fix every global RNG used for LoRA initialization and sampling; repeat
runs and record the revision, dependencies, and RNG seeds when comparing results.

.. _diffusion-art-libero:

LIBERO Object Task 0
--------------------

Use Pi0 to compare candidate action chunks from the same robot state on LIBERO
Object task 0. The two recipes differ only in how they choose that state:
``libero_object_0_art_openpi`` shares a fixed prefix and branches at zero-based decision index 8;
``libero_object_0_art_states_openpi`` restores snapshots sampled along collected
trajectories. Neither recipe groups later, independently reached states merely
because they have the same decision index.

Both load ``RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT``. Rollout uses
the Hugging Face backend; training uses FSDP1 NO_SHARD. The vision-language backbone stays
frozen, while the action expert and its associated trainable layers are updated
without LoRA. Each observation supplies two camera images, robot state, and the
task instruction. Pi0 generates a full action tensor; the environment executes
five 7-dimensional actions per decision.

Fixed Prefix and Snapshot Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ART needs candidates with the same conditioning state. A shared task ID alone
is insufficient: object poses, robot state, observations, and instruction must
agree before the trained chunk. The two collection paths establish this
condition in different ways:

.. list-table:: Object Task 0 Collection Modes
   :header-rows: 1
   :widths: 18 41 41

   * - Mode
     - State construction
     - Trained decision
   * - branch8
     - Start each group from the same reset state. Execute its leader's action
       chunks for decisions 0 through 7.
     - Decision 8, after 40 environment steps. Candidates and their subsequent
       continuations sample independently.
   * - ``states``
     - First collect trajectories with the frozen current policy. Save live
       group-leader snapshots at decision boundaries; sample task-matched
       temporal strata and restore each selected snapshot into all K members.
     - The first decision after restoration, not necessarily the task's initial
       decision. Subsequent continuation is independent.

branch8 sets ``algorithm.art_branch_chunk_index=8`` and
``art_state_sampling=False``. The rollout checks simulator state, robot
observations, task instructions, and both images throughout the prefix and at
the branch. If a group fails those checks or terminates before branching, its
training contribution is masked rather than compared across different states.

``states`` sets ``algorithm.art_state_sampling=True`` and
``art_branch_chunk_index=null``. Snapshots include physics, controller and
gripper state, observation caches, and simulator-process RNG state. Restoration
also preserves elapsed steps and the previous reward, so it does not grant a
fresh episode budget. The state bank is collected again after each policy
synchronization and cleared after restoration. This mode adds a collection
rollout and snapshot storage; it is not a free replacement for a fixed branch.

In both modes, K=8 candidates use different policy sampling noise to generate
different chunks. They share the state before that chunk, not the entire future
trajectory. The return sums environment rewards from the branch/restoration
through termination or the original 240-step deadline. A group with identical
returns supplies zero guidance. Only the selected chunk is trained; the
continuation supplies its Monte Carlo evaluation.

Action Guidance and Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the state fixed, compute the reward-action covariance over the group's
valid executed coordinates, then RMS-normalize it to obtain guidance H. Pi0
uses the same velocity sign as the SD3/Wan implementation: the frozen target is
``v_old - H`` at cached reverse-denoising states, not a forward-noised resample.
The model still processes its complete action tensor, but guidance, restoration,
and the trust-region mask cover valid executed coordinates only.

The recipes set ``algorithm.art_reference_source=actor``: recompute v_old once
on the actor before either update and freeze it through both. This avoids using
cross-backend replay differences as the reference displacement. The
``rollout`` option retains cached rollout velocities for comparisons. FP32
master parameters accumulate small optimizer updates before copying back to
low-precision model weights. Neither setting establishes a success-rate result;
validate the recipes with the actual checkpoint and simulator.

Compute guidance from all K candidates before splitting each group's candidates
between two disjoint update windows. No trajectory occurs in both windows.
With 128 environments, K=8, one trained chunk per environment, and two updates,
there are 16 groups and ``actor.global_batch_size=64`` per update. All four
cached denoising steps are trained with ``art_timestep_fraction=1.0``. The common
learning rate is 5e-7; both modes run 100 rounds by default.

Run Branch8 and States
~~~~~~~~~~~~~~~~~~~~~~

Install the OpenPi/LIBERO environment as described in :doc:`Pi0 <embodied/pi0>`.
Skip installation if that environment already exists:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero \
     --venv .venv-openpi-libero
   source .venv-openpi-libero/bin/activate

Run from the repository root. Set MODEL_PATH to your existing SFT checkpoint;
the commands do not download model weights. Both recipes select Object task 0
for training and evaluation:

.. code-block:: bash

   set -o pipefail
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
   export HYDRA_FULL_ERROR=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
   export MODEL_PATH=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

First run the fixed-prefix recipe on four visible GPUs. Log directories,
experiment names, and console log filenames start with the timestamp. The bash
entry point accepts the config name, robot platform, and Hydra overrides;
LOG_DIR and MEGA_LOG_FILE select the output paths:

.. code-block:: bash

   RUN_NAME="$(date +%Y%m%d_%H%M%S)_object0_branch8_T4_lr5e7_seed42"
   LOG_DIR="$PWD/logs/$RUN_NAME"
   LOG_DIR="$LOG_DIR" MEGA_LOG_FILE="$LOG_DIR/$RUN_NAME.log" \
     bash examples/embodiment/run_embodiment.sh libero_object_0_art_openpi LIBERO \
     actor.model.model_path="$MODEL_PATH" \
     rollout.model.model_path="$MODEL_PATH" \
     env.train.total_num_envs=128 env.eval.total_num_envs=48 \
     actor.global_batch_size=64 actor.micro_batch_size=2 \
     actor.seed=42 actor.optim.lr=5e-7 \
     actor.optim.use_fp32_master_params=true \
     algorithm.art_reference_source=actor \
     actor.model.num_steps=4 algorithm.art_timestep_fraction=1.0 \
     runner.max_steps=100 runner.max_epochs=100 \
     runner.eval_at_start=true runner.val_check_interval=10 \
     runner.save_interval=-1 \
     runner.logger.experiment_name="$RUN_NAME"

For snapshot sampling, start a separate run from the same SFT checkpoint and
use the states recipe. Execute the two commands sequentially on the same GPUs:

.. code-block:: bash

   RUN_NAME="$(date +%Y%m%d_%H%M%S)_object0_states_T4_lr5e7_seed42"
   LOG_DIR="$PWD/logs/$RUN_NAME"
   LOG_DIR="$LOG_DIR" MEGA_LOG_FILE="$LOG_DIR/$RUN_NAME.log" \
     bash examples/embodiment/run_embodiment.sh libero_object_0_art_states_openpi LIBERO \
     actor.model.model_path="$MODEL_PATH" \
     rollout.model.model_path="$MODEL_PATH" \
     env.train.total_num_envs=128 env.eval.total_num_envs=48 \
     actor.global_batch_size=64 actor.micro_batch_size=2 \
     actor.seed=42 actor.optim.lr=5e-7 \
     actor.optim.use_fp32_master_params=true \
     algorithm.art_reference_source=actor \
     actor.model.num_steps=4 algorithm.art_timestep_fraction=1.0 \
     runner.max_steps=100 runner.max_epochs=100 \
     runner.eval_at_start=true runner.val_check_interval=10 \
     runner.save_interval=-1 \
     runner.logger.experiment_name="$RUN_NAME"

Both runs evaluate the original full task before training and every 10 rounds.
Evaluation does not branch or restore training snapshots. Check
``eval/num_trajectories`` for the actual evaluation budget; 48 parallel slots do
not imply 48 unique reset states. Checkpoint saving is disabled in these
commands; set a positive ``runner.save_interval`` if you need saved models.

Read Results and Validate
~~~~~~~~~~~~~~~~~~~~~~~~~

Use full-task evaluation success to compare the modes. Training continuation
success is conditional on the sampled branch state and is not the same metric.
Monitor ``actor/active_group_frac``, ``actor/branch_valid_group_frac``,
``actor/branch_state_error_max``, and ``actor/branch_image_error_max`` to check
signal availability and same-state grouping. ``actor/mask_frac_global`` and
``actor/outward_frac_global`` pool active-item counts before forming ratios;
the ``actor/u1/`` and ``actor/u2/`` versions separate the two updates.
``actor/mask_frac`` and ``actor/outward_frac`` also weight micro-batches and
actor ranks by active-item count, rather than averaging local ratios.

The optional ``algorithm.art_replay_diagnostics=true`` adds replay forwards.
``actor/replay_mse`` compares the pre-update actor with rollout velocities;
``actor/update_u1_exec_rms`` and ``actor/update_u2_exec_rms`` compare against the
frozen actor reference. These are distinct from success metrics. Diagnostics
are disabled in the default recipes.

Run the CPU regressions for candidate grouping, return slicing, frozen targets,
and snapshot restoration with:

.. code-block:: bash

   python -m pytest tests/unit_tests/test_diffusion_art.py \
     tests/unit_tests/test_libero_snapshot.py

These tests do not replace GPU training or real MuJoCo/checkpoint validation.
