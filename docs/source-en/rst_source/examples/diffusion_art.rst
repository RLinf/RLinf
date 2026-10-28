Diffusion Model Fine-Tuning with ART
====================================

This page describes DiffusionART fine-tuning for SD3.5 Medium image text
rendering and Wan2.2 TI2V 5B video text rendering. It covers the algorithm
and its equations, OCR data, training configuration, recorded ART/NFT
reward and runtime results, and training metrics. An experimental Pi0 recipe
extends ART to multi-step LIBERO action trajectories.

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

      SD3.5 Medium / Wan2.2 TI2V 5B / Pi0

   .. grid-item-card:: Algorithm
      :text-align: center

      Image OCR / video OCR / LIBERO

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
     - OCR prompts; ``visual.video_ocr`` with a video-level mean score.
   * - Robot manipulation
     - ``libero_object_0_art_openpi`` (``examples/embodiment/config/``)
     - LIBERO Object task 0; environment episode returns.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Each image/video episode generates one image or video. ART uses one scalar reward for each
candidate, then compares candidates generated from the same prompt.
The :ref:`LIBERO action recipe <diffusion-art-libero>` instead groups complete
robot trajectories and aligns their action chunks by decision index.

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

.. _diffusion-art-libero:

LIBERO Action Fine-Tuning
------------------------------

The ``libero_object_0_art_openpi`` recipe fine-tunes Pi0 on LIBERO Object task 0
from the same SFT checkpoint as
``examples/embodiment/config/libero_object_0_nft_openpi.yaml``. It uses the
existing environment/rollout pipeline, complete episode returns, cached rollout
velocities, and two ART updates per rollout.

.. warning::

   This is an experimental trajectory-level adaptation. A group shares a task
   and initial simulator state; its observations diverge after different
   actions. Aggregation at later decisions therefore differs from the paper's
   same-condition image/video setting. Success rates, guidance strength, and
   drift thresholds require LIBERO experiments. ``art_first_chunk_only=True``
   restricts training to the initial decision as a same-initial-observation
   control.

Episode Returns and Action Guidance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One robot episode contains multiple decisions, each generating an action chunk
through its own denoising process. The recipe sums rewards up to and including
termination, then uses that episode return at each valid decision. This carries
terminal success feedback to earlier actions.

For group :math:`g`, decision :math:`h`, and executed action coordinate :math:`j`,
the valid trajectories form :math:`\mathcal V_{g,h,j}`. The covariance is computed
only from those trajectories:

.. math::

   C_{g,h,j}=\frac{1}{n_{g,h,j}-1}
   \sum_{i\in\mathcal V_{g,h,j}}
   (R_i-\bar R_{g,h,j})(a_{i,h,j}-\bar a_{g,h,j}).

Here :math:`R_i` is the complete episode return, :math:`a_{i,h,j}` is a normalized
model-space action coordinate, and :math:`n_{g,h,j}=|\mathcal V_{g,h,j}|`.
Coordinates with fewer than two valid trajectories receive zero guidance.
The covariance is normalized over supported action coordinates before assigning
guidance to each trajectory. Constant-return groups have zero guidance.

The transformer still receives its full action horizon and model dimensions.
The covariance, target loss, and drift proxy use executed actions and real
environment dimensions. An action after termination is excluded even when it
shares a chunk with earlier valid actions. Loss normalization accounts for
valid chunks across actor ranks.

Both updates retain the velocities cached during rollout. Each reset group
contributes half its trajectories to each window, and every chunk from one
trajectory stays in the same window. Group guidance is computed before this
split. Pi0 uses the same velocity sign as the SD3/Wan implementation, so the
target is :math:`v_{\mathrm{old}}-H_{g,h}`. The existing corrective mask applies
to the valid action coordinates.

Configuration and Launch
~~~~~~~~~~~~~~~~~~~~~~~~

The OpenPi environment and checkpoint setup follows :doc:`Pi0 <embodied/pi0>`.
The LIBERO installer selects the simulator assets used by this recipe:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero \
      --venv .venv-openpi-libero
   source .venv-openpi-libero/bin/activate

This creates a separate OpenPi environment. With the Pi0 SFT checkpoint already
available locally, the following command starts the experiment from the
repository root:

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1

   python examples/embodiment/train_embodied_agent.py \
      --config-path "$EMBODIED_PATH/config" \
      --config-name libero_object_0_art_openpi \
      actor.model.model_path=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
      rollout.model.model_path=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

The defaults collect 64 episodes, with 240 environment steps and 5 actions per
decision. Each rollout contains :math:`64\times(240/5)=3072` action chunks before
termination masking. Two disjoint updates therefore require
``actor.global_batch_size=1536``. In general, the required batch size is
``total_num_envs * rollout_epoch * decisions / art_updates_per_rollout``.
With ``algorithm.art_first_chunk_only=True``, the decision count is one and
the matching override is ``actor.global_batch_size=32`` for this recipe.

The model uses ``openpi.rl_mode=art``, ``is_nft=False``, and
``noise_method=flow_ode`` with zero transition noise. Training requires
``auto_reset=False``, ``ignore_terminations=False``, and fixed grouped reset
states. The four-step sampler and ``art_timestep_fraction=0.25`` select one
rollout denoising state per action chunk. The full recipe is in
``examples/embodiment/config/libero_object_0_art_openpi.yaml``.

Fixed Branch Sampling
~~~~~~~~~~~~~~~~~~~~~

A shared action prefix extends the same-condition comparison to a later robot
decision. ``algorithm.art_branch_chunk_index`` selects one zero-based decision
for every group. Before that decision, each group executes its first member's
sampled action chunk. At the branch and throughout the continuation, each member
samples independently with the frozen rollout policy. The actor trains only the
branch chunk, using its observations, denoising states, and cached old velocities.
The return includes rewards from the branch through termination:

.. math::

   R_i^{(b)}=\sum_{t=t_b}^{T_i}r_{i,t}.

The branch mode uses the existing launch command with these overrides:

.. code-block:: bash

   algorithm.art_first_chunk_only=False \
   algorithm.art_branch_chunk_index=8 \
   actor.global_batch_size=32

With five actions per chunk, index 8 branches after 40 environment steps. Index 0
provides an initial-decision control. Each episode contributes one training chunk,
so 64 episodes and two disjoint updates require a global batch size of 32.
``art_first_chunk_only`` and a non-null branch index are mutually exclusive.
The branch mode requires synchronous batches containing complete groups on every
environment, rollout, and actor rank. Evaluation retains the standard full-task
rollout.

During the prefix and at the branch, the rollout checks task identity, robot
observations, exposed simulator state, and both camera images. The default maximum
state difference is ``art_branch_state_atol=1e-5``; the maximum per-camera mean
absolute pixel difference is ``art_branch_image_atol=1.0``. A mismatch at any
prefix decision, or any group member terminating before the branch, masks the
entire group for training while preserving batch shapes. These checks compare
the exposed state and observations; they do not restore controller snapshots.

Validation and Metrics
~~~~~~~~~~~~~~~~~~~~~~

The tensor/config tests exercise termination masks, group alignment, frozen
targets, and cached-state replay. The model test uses small Gemma/SigLIP
backbones and generated observations, so it runs without a checkpoint:

.. code-block:: bash

   python -m pytest tests/unit_tests/test_diffusion_art.py

A startup experiment with ``runner.max_steps=2``, ``runner.max_epochs=2``,
``runner.val_check_interval=1``, and ``runner.save_interval=1`` exercises
training, evaluation, and checkpoint saving before a full run. GPU LIBERO
training and checkpoint resume require the actual model and simulator assets.

``actor/art_reward_mean`` and ``actor/art_reward_std`` describe episode returns.
``actor/valid_chunk_frac`` and ``actor/valid_action_frac`` report the fractions
remaining after termination and optional reward filtering. The existing
``active_group_frac``, ``mask_frac``, and ``movement_score`` diagnose guidance
and drift. ART/NFT comparisons use the same SFT initialization, reset/evaluation
states, environment interactions, optimizer-update counts, and elapsed time;
their loss magnitudes are different objectives.

Branch experiments also report ``actor/branch_chunk_index``,
``actor/branch_valid_group_frac``, ``actor/branch_state_error_max``, and
``actor/branch_image_error_max``. The errors summarize prefix checks, and the
valid-group fraction excludes groups that terminated before the branch. A short
index-0 run followed by an index-8 run checks initial-decision behavior and later
branching before increasing the training budget. These experiments require GPU
LIBERO validation; the unit tests cover sampling, cache alignment, return slicing,
and invalid-group masking.

Full-output velocity diagnostics use ``algorithm.art_replay_diagnostics=true``.
``actor/replay_{region}_rms`` measures pre-update replay error;
``actor/drift_u1_{region}_rms`` and ``actor/drift_u2_{region}_rms`` measure drift
from the same frozen rollout velocities after each optimizer update. The
``region`` suffix is ``exec`` for valid executed coordinates, ``future`` for
later action tokens in environment dimensions, ``padding`` for remaining model
dimensions, ``unexec`` for all unexecuted coordinates, or ``full`` for the whole
output. Entirely invalid rows are excluded; squared errors and coordinate counts
are summed across microbatches and ranks before computing RMS. These probes use
fixed cached states and the selected denoising timesteps, not regenerated action
chains. The training loss and trust-region mask still cover executed coordinates.
Full velocity caching increases rollout storage and transport; enabling these
diagnostics adds eval-mode forward passes before and after the updates.
The same option also reports ``actor/art_output_restore_ratio`` and
``actor/art_output_restore_cosine``, which compare the restoration and guidance
parts of the loss gradient with respect to the executed velocity outputs, after
loss scaling, accumulation, and rank averaging.

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

In these records, ART reaches 0.8 with 30.6% less elapsed time and 33.3% fewer
trajectories than NFT-576. The two NFT runs change learning rate and sampling
amount together, so this comparison cannot isolate either factor. Different
record lengths also prevent comparing their final observed rewards as equal
budget endpoints. These are training rewards from single runs, not held-out
success rates or a reproduction of the paper's video benchmarks.

Runtime and Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~

Execution costs cover the common interval
``5760 < cumulative trajectories <= 40320``. Each run contributes 34560
trajectories: ART and NFT-1152 rounds 6--35, and NFT-576 rounds 11--70.

.. list-table:: Mean Runtime in the Common Interval
   :header-rows: 1
   :widths: 34 22 22 22

   * - Metric
     - ART
     - NFT, 1152, 1e-4
     - NFT, 576, 4e-4
   * - Round time (s)
     - 1026.24
     - 971.35
     - 495.00
   * - Actor training per round (s)
     - 105.23
     - 61.56
     - 63.37
   * - Seconds per 1000 trajectories
     - 890.83
     - 843.18
     - 859.37
   * - Trajectories per minute
     - 67.35
     - 71.16
     - 69.82

At 1152 trajectories per round, ART spends about 44 s more in actor training
and 55 s more per round in this interval. The Wan ART recipe selects two
denoising states per trajectory, precomputes their old velocities, and
backpropagates through both; the NFT recipe samples one training state per
trajectory. These additional forwards and backwards are a likely contributor
to the actor-time difference; the existing timers do not isolate their cost.
Generation remains the largest phase in all three records.

``algorithm.art_timestep_fraction=0.1`` defines a timestep-cost ablation. With
the configured ten rollout steps, it selects one training state instead of two.
A comparison with unchanged remaining settings and initialization excluded from
timing measures the effect of this reduction. Reward versus elapsed time and
throughput capture different effects because fewer states also change the
learning budget.

The metric definitions below relate update diagnostics to reward and runtime.
ART and NFT losses have different definitions, and logged gradient scales alone
do not establish which algorithm is more stable. The :ref:`ART Objective and Implementation <diffusion-art-algorithm>` distinguishes the implementation from the paper's
image, video, and pairwise experiments.

.. _diffusion-training-metrics:

Diffusion Training Metrics
--------------------------

These metrics describe image and video generator training with ART or NFT.
Task rewards measure text rendering, update diagnostics describe model changes,
and timers record execution cost. The terminal table shortens namespaces:
``actor/art_loss`` in Training/Actor is logged as ``train/actor/art_loss``;
``video_ocr`` in Environment is ``env/video_ocr``.

Task Reward and Rollout Signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a prompt requesting the text "OPEN", OCR reads the generated letters and
compares them with that target. The scorer removes spaces and ignores case,
then applies a normalized edit-distance score:

.. math::

   R_{\mathrm{OCR}}=1-\frac{\min(d(\hat y,y),|y|)}{|y|}.

Here :math:`y` is the target text, :math:`\hat y` is the recognized text, and
:math:`d` counts insertions, deletions, and substitutions. A substring match also
receives one. Video OCR scores every frame; ``reward_type=video_level`` uses the
mean frame score, while ``frame_level`` preserves frame scores for downstream
grouping. A reward of 0.8 expresses text similarity, not an 80% rate of completely
correct videos.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Metric in the Terminal Table
     - Meaning
   * - ``avg`` / ``video_ocr`` / ``ocr``
     - Reward-backend scores. ``reward.key`` selects the training reward. Consistent backend fields support comparison across runs; logged aggregates can differ with frame and chunk reductions.
   * - ``return``
     - Episode reward sum. In the one-step, scalar-reward ART recipe it coincides with the episode score.
   * - ``episode_len``
     - Environment steps per episode. One generated video can be one episode with several internal denoising steps.
   * - ``num_trajectories``
     - Completed candidates in the current collection round, rather than a cumulative counter.
   * - ``rewards``
     - Rollout-side aggregate of the reward tensor used for training.
   * - ``advantages_mean`` / ``advantages_min`` / ``advantages_max``
     - Relative advantage statistics for NFT's grouped comparison. A centered mean is expected after normalization. ART computes guidance directly from rewards.
   * - ``loss_mask_fraction``
     - Fraction of rollout entries marked valid for training. This is distinct from ART's corrective mask and NFT's clipping decisions.

ART Guidance and Update Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ART estimates a direction from a complete prompt group, then fits a frozen
velocity target at selected denoising states. The
:ref:`ART Objective and Implementation <diffusion-art-algorithm>` defines the equations and velocity convention.
Its loss is :math:`\mathcal L=w\operatorname{MSE}(v_\theta,v_{\mathrm{old}}-H_c)`.

.. list-table:: Terminal Keys Under Training/Actor
   :header-rows: 1
   :widths: 36 64

   * - Metric
     - Meaning
   * - ``actor/art_reward_mean``
     - Mean scalar candidate reward after ``art_reward_reduction``.
   * - ``actor/art_reward_std``
     - Population standard deviation of those rewards on each actor shard, averaged across shards. It is not the within-prompt reward spread.
   * - ``actor/active_group_frac``
     - Fraction of prompt groups with nonzero guidance. Constant rewards or a zero aggregate direction make a group inactive.
   * - ``actor/covariance_rms``
     - Mean RMS of the raw reward--endpoint covariance over active groups. Its scale depends on rewards and latent coordinates.
   * - ``actor/guidance_rms``
     - Mean per-candidate RMS of direction times magnitude, including inactive candidates. Under unit normalization it often tracks the active-group fraction.
   * - ``actor/skipped_zero_guidance``
     - One when every actor shard has zero guidance and optimization is skipped; zero when at least one shard has a signal.
   * - ``actor/art_loss``
     - Weighted, corrective-masked loss averaged over training items, including zero-loss entries.
   * - ``actor/unmasked_loss``
     - Weighted loss before corrective masking, averaged over active items. Its denominator differs from ``actor/art_loss``.
   * - ``actor/old_deviate``
     - Mean velocity MSE :math:`D^{-1}\|v_\theta-v_{\mathrm{old}}\|_2^2` over training items.
   * - ``actor/movement_score``
     - Mean movement proxy. ``sde_kl`` uses reference SDE noise and step size; ``fixed_mse`` uses velocity MSE.
   * - ``actor/outward_frac``
     - Fraction of active items whose descent direction has positive inner product with displacement from the old velocity.
   * - ``actor/mask_frac``
     - Fraction of active items blocked by both excessive movement and the outward-direction condition.
   * - ``actor/total_loss``
     - Equals ``actor/art_loss`` in the ART worker.

Ratios and losses aggregate micro-batch/timestep statistics and are then averaged
across workers. ``outward_frac`` counts outward directions at any movement size;
``mask_frac`` also requires crossing the threshold. A smaller masked loss can
result from blocking more items. Reward and active/masked fractions provide
the context for interpreting it.

NFT Energies and Clipping
~~~~~~~~~~~~~~~~~~~~~~~~~

NFT compares two velocity branches built from the current change
:math:`\Delta v=v_\theta-v_{\mathrm{old}}`:

.. math::

   v_\pm=v_{\mathrm{old}}\pm\beta\Delta v,\qquad
   E_\pm=\operatorname{reduce}\left[
      w_\pm\bigl(T(v_\pm)-x_{\mathrm{target}}\bigr)^2\right].

The configured transform :math:`T` converts velocity to the prediction space,
such as ``nft_target_space=x0``. The energy measures prediction error, with
``nft_weight_mode`` and ``nft_energy_reduction`` controlling its scale. Positive
and negative identify branches; the energy means normally include all candidates.
For ``nft_loss_form=mse``, the transformed advantage is a weight
:math:`r\in[0,1]`, and the unclipped per-item loss is

.. math::

   \ell_{\mathrm{NFT}}=\frac{A_{\max}}{\beta}
   \left[rE_+ +(1-r)E_-\right],
   \qquad A_{\max}=\text{advantage clipping bound}.

The bound is ``adv_clip_max``. The clipped implementation selects the maximum
of the clipped and unclipped per-item losses. Its objective and scale differ
from ART's target MSE.

.. list-table:: Terminal Keys Under Training/Actor
   :header-rows: 1
   :widths: 36 64

   * - Metric
     - Meaning
   * - ``actor/E_pos_mean`` / ``actor/E_neg_mean``
     - Mean positive/negative branch energies over all candidates.
   * - ``actor/E_pos_mean_pos_only``
     - Positive-branch energy for valid items with transformed advantage :math:`r>0.5`.
   * - ``actor/E_neg_mean_neg_only``
     - Negative-branch energy for valid items with :math:`r<0.5`.
   * - ``actor/delta_E_mean``
     - Mean :math:`E_+-E_-`. Opposing contributions can cancel in this aggregate.
   * - ``actor/delta_v_norm``
     - Norm of the velocity change over the configured gather/reduction axes. It is not ART's mean squared deviation.
   * - ``actor/clip_frac``
     - Fraction of reduced velocity entries with a clipping coefficient below one. The counting unit depends on ``nft_gather_type``.
   * - ``actor/clip_loss_frac``
     - Fraction of valid loss entries where the clipped loss exceeds the unclipped loss and is selected.
   * - ``actor/nft_tau``
     - Current actor mixing coefficient in the rollout/reference EMA after schedule interpolation. A smaller value retains more historical parameters.
   * - ``actor/nft_loss``
     - NFT objective before gradient-accumulation scaling, including configured clipping.
   * - ``actor/total_loss``
     - Loss divided by the gradient-accumulation factor in the NFT worker. With SFT co-training, it also includes the SFT contribution.
   * - ``actor/grad_norm`` / ``actor/lr``
     - Optimizer-reported gradient norm and current learning rate; also emitted by ART.

Energy means, losses, and gradients help diagnose a run, but their scales depend
on weighting, target space, and gradient accumulation. Task rewards under
matched time or sample budgets provide a basis for comparing algorithm performance.

Runtime and Progress
~~~~~~~~~~~~~~~~~~~~

The runner's round timer and disjoint phase timers describe end-to-end
performance. Nested timers expose costs within each phase; adding overlapping
spans would count some execution time more than once.

.. list-table:: Terminal Keys Under Time
   :header-rows: 1
   :widths: 43 57

   * - Metric
     - Meaning
   * - ``step``
     - Runner time for the current round.
   * - ``generate_rollouts``
     - Collection phase, including generation, reward feedback, and trajectory delivery.
   * - ``actor_training`` / ``actor/run_training``
     - Runner and worker views of actor training, including ART/NFT input preparation.
   * - ``sync_weights``
     - Runner weight synchronization phase.
   * - ``actor/actor/recv_traj``
     - Actor wait/receive span for trajectories, overlapping rollout collection.
   * - ``actor/actor/compute_adv`` / ``cal_adv_and_returns``
     - Worker and runner spans for advantage/return processing. ART uses this stage to report rollout statistics.
   * - ``rollout/predict``
     - Model generation calls, including generation and decoding in the wrapper.
   * - ``env/env_interact_step`` / ``env/env/bootstrap_step``
     - Environment interaction, including reward scoring, and bootstrap work.
   * - ``env/interact`` / ``env/run_interact_once``
     - Environment interaction spans, including communication and waits for outputs.
   * - ``rollout/generate_one_epoch`` / ``rollout/rollout/generate``
     - Nested rollout-loop spans within ``generate_rollouts``.
   * - ``actor/actor/sync_model_to_rollout`` / ``rollout/sync_model_from_actor``
     - Actor-side and rollout-side views of weight transfer; these can overlap.

Global Step in the header counts collection/training rounds; one ART round can
contain two optimizer updates. Elapsed is cumulative wall time, ETA is an
estimate, and header Step Time is elapsed time divided by completed rounds.
Time ``step`` records
individual round costs. Seconds per 1000 trajectories and trajectories per
minute account for differences in sampling amounts. The
:ref:`Training Results <diffusion-art-results>` reports a comparison over the same
cumulative sampling interval.

Diagnose LIBERO Updates and State Sampling
------------------------------------------

Use ``libero_10_0_art_diagnostics_openpi`` for Long task 0 or
``libero_object_0_art_diagnostics_openpi`` for Object task 0. Both select 128
training environments, K=8, all four denoising steps, actor-side reference
velocities, FP32 master AdamW, and five rounds. The existing recipes retain
rollout-side reference velocities and leader-only state collection.

First separate rollout replay error from parameter updates. Set
``algorithm.art_reference_source=actor`` to recompute the frozen reference once
before both ART updates. With ``algorithm.art_replay_diagnostics=True``, compare
``actor/replay_mse`` (actor versus rollout before updating),
``actor/reference_pre_mse`` (actor versus the selected reference), and
``actor/update_u1_exec_rms`` / ``actor/update_u2_exec_rms`` (actor-reference
update displacement). ``actor/drift_u1_*`` and ``actor/drift_u2_*`` continue to
use the rollout reference. The ``update_u*`` metrics require actor references.

Enable ``algorithm.art_parameter_diagnostics=True`` for a short FSDP1 NO_SHARD
check. ``actor/param_u1_bf16_changed_frac`` and
``actor/param_u2_bf16_update_rms`` measure actual model-weight changes;
``actor/param_u2_master_update_rms`` measures FP32 master accumulation. These
probes keep CPU copies of trainable parameters and measure cumulative change
from before the first update. Set ``algorithm.art_replay_batch_sizes=[1,2,8]``
to compare a fixed subset under different actor inference batch sizes.

Read ``actor/mask_frac_global`` and ``actor/outward_frac_global`` for
active-item-count-weighted ratios. The ``actor/u1/`` and ``actor/u2/`` versions
separate the two updates; ``movement_score_active`` uses the same active-item
denominator. The existing ``mask_frac`` and ``outward_frac`` remain available
with their microbatch-average definition for old-log comparisons.

Next freeze the SFT policy with ``algorithm.art_probe_only=True``. This skips
backward, optimizer, scheduler, and post-initial weight synchronization. Select
one of these state pools with ``algorithm.art_state_sampler``:

.. list-table:: State Sampling Ablations
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Collection and selection
   * - ``legacy``
     - Keep all live leader states and sample task-matched temporal strata.
   * - ``uniform``
     - Keep a bounded reservoir from every trajectory and sample temporal strata.
   * - ``success_mixture``
     - Use the bounded pool; request approximately ``art_success_state_fraction``
       successful-source states (default 0.7), falling back to the available
       outcome pool when necessary.
   * - ``early_failure``
     - Use the bounded pool; retain all successful-source states and only failed
       source states before ``art_failure_max_step_fraction`` of the horizon
       (default 0.5).

``algorithm.art_states_per_trajectory`` defaults to 4 for bounded pools.
Reservoir admission happens before snapshot capture. Source outcomes are labeled
when collection ends; restoration preserves the original remaining budget.
Probe JSONL files under ``<log_path>/art_probe/`` record each group's returns,
source outcome, source trajectory/reset identity, sampled time, and remaining
steps. These probes compare sampling distributions without changing rewards.

Run the experiment matrix from an activated OpenPi/LIBERO environment:

.. code-block:: bash

   python examples/embodiment/run_art_ablation.py diagnose \
     --checkpoint /path/to/model/RLinf-Pi0-LIBERO-Long-SFT --gpus 0,1,2,3
   python examples/embodiment/run_art_ablation.py probe \
     --checkpoint /path/to/model/RLinf-Pi0-LIBERO-Long-SFT --gpus 0,1,2,3
   python examples/embodiment/run_art_ablation.py summary

``diagnose`` runs four five-round arms: rollout/actor references crossed with
FP32 master disabled/enabled. ``probe`` runs each of the four samplers for ten
frozen-policy rounds. ``summary`` reports signal yield separately for screening
and held-out initial reset IDs. Inspect source composition and all-fail/all-success
rates as well as active fractions; the summary does not select a winner.

After selecting a sampler, run ``train --sampler uniform`` (replace ``uniform``
with your selection) for two 30-round arms: the corrected numerical baseline
with ``legacy`` sampling and the same baseline with the selected sampler.
``confirm --sampler uniform`` repeats both arms for 50 rounds with seeds 42
and 43. ``object --checkpoint /path/to/model/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT``
runs a 25-round Object task 0 baseline. Every phase writes timestamp-first run
names, console logs, and an exact command manifest. Add ``--dry-run`` to inspect
commands before execution; pass ``--override key=value`` for a Hydra override.

Training phases evaluate at step 0 and every 10 rounds (every 5 for ``object``).
``runner.eval_repeats`` repeats the full available reset set, while
``runner.eval_seed`` pairs policy-noise seeds across arms without advancing the
training RNG. Three repeats give 150 episodes when the task has 50 reset states;
these are repeated initial states, not 150 distinct scenes. Confirmation uses
five repeats. ``eval`` performs four repeats at step 0 with zero training rounds;
read ``num_trajectories`` to verify the completed evaluation budget.
