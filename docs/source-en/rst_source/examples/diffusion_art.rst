Diffusion Model Fine-Tuning with ART
====================================

This page describes DiffusionART fine-tuning for SD3.5 Medium image text
rendering and Wan2.2 TI2V 5B video text rendering. It covers the algorithm
and its equations, OCR data, training configuration, recorded ART/NFT
reward and runtime results, and training metrics.

Overview
--------

The image recipe targets text rendering in single images; the video recipe
scores text across frames. Optical character recognition (OCR) reads
the generated text and compares it with the text requested in the prompt.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      DiffusionART with scalar rewards

   .. grid-item-card:: Models
      :text-align: center

      SD3.5 Medium / Wan2.2 TI2V 5B

   .. grid-item-card:: Environments / Data
      :text-align: center

      Image OCR / video OCR prompts

   .. grid-item-card:: Training
      :text-align: center

      LoRA + FSDP2; Wan comparison on 4 NVIDIA A800 GPUs

Tasks
~~~~~

The configs in ``examples/diffusion/config/`` associate each task with a
Diffusers checkpoint and an OCR reward backend.

.. list-table::
   :header-rows: 1
   :widths: 20 39 41

   * - Task
     - Config
     - Input and Reward
   * - Text to image
     - ``sd3_art_ocr``
     - OCR prompts; ``visual.ocr`` image text similarity.
   * - Text to video
     - ``wan22_ti2v_5b_art_video_ocr``
     - OCR prompts; ``visual.video_ocr`` with a video-level mean score.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Each episode generates one image or video. ART uses one scalar reward for each
candidate, then compares candidates generated from the same prompt.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - Observation
     - ``task_descriptions`` sampled from the dataset.
   * - Action
     - A generated image or video tensor.
   * - Reward
     - OCR text similarity in ``[0, 1]``; video training averages frame scores.
   * - Prompt
     - A description containing quoted target text, such as a sign reading "OPEN".

ART Training Pipeline
---------------------

Each prompt produces a group of scored candidates. ART combines the
clean latent endpoints and their rewards into a shared direction: candidates
above the group mean pull the direction toward their endpoints, while candidates
below the mean push it away. It then trains the denoising model toward a frozen
velocity target at states collected during generation.

The supplied recipes split each prompt's candidates between two disjoint update
windows. Both updates use targets from before the first update. The second
update therefore balances reward guidance with a correction for drift introduced
by the first. A mask suppresses updates that exceed the movement threshold and
would move farther outward. The :ref:`ART Objective and Implementation <diffusion-art-algorithm>` specifies the equations and configuration
constraints behind this update structure.

.. _diffusion-art-algorithm:

ART Objective and Implementation
--------------------------------

The ART objective combines grouped reward guidance, frozen velocity targets,
and a corrective mask. The equations below relate the paper to the SD3 and
Wan velocity convention and the configuration constraints in RLinf.
The method is presented in `DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models <https://vanzll.github.io/diffusionart-project/>`_.

Aggregation-First Reward Guidance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An image or video receives its reward after generation, while the model makes
many intermediate denoising predictions. ART turns these endpoint scores into
a direction shared by states from the same prompt. For example, if several
signs spell "OPEN" correctly and others miss letters, ART uses their latent
endpoints and scores to estimate which direction is associated with better
text rendering. The reward function can be a black box.

DiffusionNFT also learns from endpoint rewards. Its forward-process objective
regresses predictions at states formed by adding noise to completed samples.
The ART paper identifies a forward--reverse consistency force in these targets:
the per-sample forward velocity and the model's reverse velocity can differ.
Positive and negative branches only partly cancel this force after aggregation
across states and model gradients. The remaining force can interfere with reward
guidance. This analysis concerns the objective; it does not imply that every
NFT configuration collapses.

The paper also studies exponential moving averages (EMA). In RLinf's NFT
implementation, the rollout/reference parameters follow

.. math::

   \bar\theta_{k+1}=(1-\tau_k)\bar\theta_k+\tau_k\theta_{k+1}.

Here ``nft_tau`` is the current-parameter mixing coefficient. A smaller value
retains more of the previous reference; ``1`` uses the current actor. The paper
attributes EMA's stabilization to restoration toward a lagged reference, with
task-dependent tuning. ART instead constructs guidance directly from a prompt
group and obtains restoration from a second update against a frozen target.

From Endpoint Scores to a Direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current rollout policy defines a sample distribution conditioned on a
prompt :math:`c`. Increasing the weight of better samples gives
the reward-tilted distribution

.. math::

   p_\lambda(x_0\mid c)=
   \frac{p_{\mathrm{old}}(x_0\mid c)\exp(\lambda R(x_0,c))}{Z_c(\lambda)},
   \qquad
   \left.\frac{\partial}{\partial\lambda}
   \mathbb E_{p_\lambda}[x_0\mid c]\right|_{\lambda=0}
   =\operatorname{Cov}_{p_{\mathrm{old}}}(R,x_0\mid c).

The normalization :math:`Z_c` makes the probabilities sum to one. The derivative
says that a small preference for higher rewards moves the mean sample along
the reward--endpoint covariance. Here :math:`\lambda` denotes the paper's
tilting parameter :math:`\tau`; it is separate from NFT's EMA coefficient.

For :math:`K` candidates of one prompt, paper Equation (5) estimates this vector
and normalizes its root mean square (RMS):

.. math::

   C_c=\frac1K\sum_{i=1}^K
   (R_i-\bar R_c)(x_{0,i}-\bar x_{0,c}),
   \qquad
   G_c=\frac{C_c}{\operatorname{RMS}(C_c)},
   \qquad
   \operatorname{RMS}(z)=\sqrt{\frac1D\sum_{j=1}^D z_j^2}.

Each :math:`x_{0,i}` is a clean latent endpoint with :math:`D` coordinates, not a
decoded RGB image. A score above the group mean contributes positively; a score
below it contributes negatively. Normalization fixes the direction's scale
across prompts with different reward ranges. Constant rewards give no relative
signal, so RLinf assigns that group zero guidance.

``_art_scalar_covariance`` computes this statistic over complete prompt groups
before micro-batching. The recipes use ``art_covariance_reduction=unbiased``,
which divides by :math:`K-1`; ``population`` uses the paper's :math:`K`.
A positive scalar factor cancels under ideal RMS normalization. The historical
direction path first divides by reward mean absolute deviation, normalizes by
RMS with an epsilon floor, and the worker normalizes that result again. For
signals above those floors, the default unit-magnitude guidance is
:math:`H_c=G_c`. Epsilon floors can attenuate very small nonzero signals.
``art_zero_signal_mode=exact`` explicitly clears constant-reward roundoff.

Frozen Velocity Targets
~~~~~~~~~~~~~~~~~~~~~~~

The same group guidance applies to states actually visited by the ODE rollout.
In the paper's clean-directed velocity convention, the target is
:math:`v^{\mathrm{ART}}_{\mathrm{tgt}}=v^{(0)}+G_c`. RLinf's Diffusers models
use the opposite sign: a descending-noise Euler step is

.. math::

   x_{\sigma'}=x_\sigma+(\sigma'-\sigma)v_\theta(x_\sigma,c),
   \qquad \sigma'<\sigma.

With this convention, the implemented target and loss are

.. math::

   v_{\mathrm{tgt}}=\operatorname{stopgrad}(v_{\mathrm{old}}-H_c),
   \qquad
   \mathcal L_\theta(s)=w(s)\frac1D
   \|v_\theta(s)-v_{\mathrm{tgt}}(s)\|_2^2.

Reducing velocity by :math:`H_c` moves the next latent toward :math:`H_c`
because the timestep difference is negative. This explains the minus sign in
``_art_loss`` relative to the paper. The stop-gradient operation keeps the old
prediction and reward direction fixed while only the current model receives
gradients.

``_precompute_old_velocities`` recomputes and caches the old predictions at all
selected states before either optimizer update. Both disjoint windows reuse
this cache. The rollout tensors hold states, current and next noise levels, and
endpoints; SD3 and Wan use the same ART actor through their velocity-forward
interfaces.

Two-Update Restoration
~~~~~~~~~~~~~~~~~~~~~~

After one parameter update, predictions can move even at states reserved for
the second window because they share model parameters. Keeping the original
target makes the next update respond to that movement. The displacement is
:math:`d(s)=v_\theta(s)-v_{\mathrm{old}}(s)` in RLinf's convention. Holding the
mask fixed during differentiation gives

.. math::

   -\nabla_{v_\theta}\mathcal L
   =\frac{2w}{D}(-H_c-d(s)).

Before the first update, :math:`d=0` and the output-space descent direction is
proportional to :math:`-H_c`. In the second update, :math:`-d` pulls predictions
back toward the old field as drift grows. This is Equation (6) with the velocity
sign converted. A walk toward a fixed destination follows the same principle:
after a large first step, the remaining distance determines the next step.

``art_updates_per_rollout=2`` creates two equally sized, disjoint windows, each
with half the candidates from every prompt. Every candidate belongs to one
window; both windows use the covariance estimated from the full group.
Gradient accumulation spans the selected timesteps as well as micro-batches.
Setting the option to ``1`` is a restoration ablation and changes the training
dynamics.

Corrective Movement Mask
~~~~~~~~~~~~~~~~~~~~~~~~

The second update can still move predictions too far. ART measures movement
using reference Gaussian transitions whose means depend on the old and current
velocities. Paper Equation (7) defines the per-coordinate KL proxy

.. math::

   m(s)=\frac1D D_{\mathrm{KL}}(q_\theta(\cdot\mid s)\|q_{\mathrm{old}}(\cdot\mid s))
   =\frac{\|\mu_\theta(s)-\mu_{\mathrm{old}}(s)\|_2^2}
   {2D\sigma_{\mathrm{ref}}^2(s)}.

These Gaussian transitions define a movement scale while generation follows
the ODE trajectory. In code, for :math:`h=\sigma-\sigma'` and reference noise
level :math:`\eta`, the proxy is

.. math::

   m(s)=\frac{a^2\operatorname{MSE}(v_\theta,v_{\mathrm{old}})}{2b^2},
   \qquad
   (a,b^2)=
   \begin{cases}
      (h,\eta^2),&\sigma\simeq1,\\
      (h(1+\eta^2/2),\eta^2\sigma h/(1-\sigma)),&\text{otherwise}.
   \end{cases}

The implementation adds numerical floors at the schedule boundaries. With
``art_mask_metric=sde_kl``, Equation (8) becomes, in RLinf's velocity convention,

.. math::

   w(s)=\sigma\left[1-
      \mathbf1\{m(s)\ge\delta\}
      \mathbf1\{\langle d(s),-H_c-d(s)\rangle>0\}\right].

The mask suppresses a state only when it has crossed the movement threshold
and its descent direction would increase the displacement. A correction toward
the old field remains eligible even beyond the threshold. The factor
:math:`\sigma` emphasizes high-noise states; zero-guidance states receive zero
loss. Mask decisions are detached from autograd.

Update Configuration
~~~~~~~~~~~~~~~~~~~~

The configs ``sd3_art_ocr.yaml`` and ``wan22_ti2v_5b_art_video_ocr.yaml``
set ``algorithm.loss_type=embodied_art``, ``adv_type=raw``, and
``actor.model.<model_type>.rl_mode=art``. ART builds its direction directly from
scalar rewards rather than the GRPO advantage tensor.

.. list-table:: ART Options Under ``algorithm``
   :header-rows: 1
   :widths: 35 22 43

   * - Option
     - Recipe Value
     - Effect
   * - ``art_updates_per_rollout``
     - ``2``
     - Disjoint optimizer windows; ``1`` is supported for ablation.
   * - ``art_covariance_reduction``
     - ``unbiased``
     - Covariance denominator; ``population`` uses the group size.
   * - ``art_direction_mode``
     - ``historical``
     - MAD and epsilon-floored RMS path; alternative: ``normalized_aggregate``.
   * - ``art_scale_mode``
     - ``unit_normalized_covariance``
     - Active-group magnitude; alternatives: ``reward_mad``, ``raw_covariance``, ``reward_std_normalized_covariance``.
   * - ``art_guidance_gain``
     - ``1.0``
     - Multiplier on the chosen magnitude.
   * - ``art_zero_signal_mode`` / ``art_zero_signal_atol``
     - ``exact`` / ``0.0``
     - Explicitly clears exactly constant rewards.
   * - ``art_loss_space``
     - ``velocity_linear_sigma``
     - Noise-level weighting; ``velocity_uniform`` uses unit weight.
   * - ``art_mask_metric``
     - ``sde_kl``
     - Movement proxy; ``fixed_mse`` uses velocity MSE directly.
   * - ``art_mask_threshold``
     - ``1e-5``
     - Positive movement threshold :math:`\delta`.
   * - ``art_reference_noise_level``
     - ``0.8``
     - Reference SDE scale :math:`\eta` for the movement proxy.
   * - ``art_corrective_mask_enabled``
     - ``True``
     - Enables outward movement masking.
   * - ``art_exclude_final_step``
     - ``False``
     - Optionally removes the last recorded transition from the timestep pool.
   * - ``art_timestep_fraction``
     - SD3 ``0.5``; Wan ``0.25``
     - Selects ``max(1, int(pool_size * fraction))`` states per candidate without replacement.
   * - ``art_reward_reduction``
     - ``mean``
     - Reduces each candidate's reward entries to a scalar; alternative: ``sum``.
   * - ``art_eps``
     - ``1e-5``
     - Numerical floor for covariance-direction normalization.

The model's ``timestep_fraction`` is a separate field.
``algorithm.art_timestep_fraction`` controls ART's training-state count.
With ten recorded transitions, the recipes train on five states for SD3 and
two for Wan per candidate. The NFT ``resample`` path selects one random state
per candidate per preparation pass. These counts affect actor runtime and
the training compute consumed per trajectory.

Batch and Placement Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each actor shard must contain complete comparison groups.
Here :math:`E` denotes ``env.train.total_num_envs``, :math:`L` denotes
``env.train.rollout_epoch``, :math:`U` denotes ``art_updates_per_rollout``, and
:math:`G` denotes ``algorithm.group_size``. The configuration validator requires

.. math::

   N=EL,\qquad B_{\mathrm{global}}=N/U,\qquad
   E\bmod\bigl(\operatorname{lcm}(W_{\mathrm{env}},W_{\mathrm{actor}})G\bigr)=0.

For example, 96 environments and 12 rollout epochs generate 1152 candidates;
two update windows require ``actor.global_batch_size=576``. The per-rank window
must also divide evenly into ``actor.micro_batch_size``.

The implementation requires FSDP, ``rollout.pipeline_stage_num=1``,
``algorithm.update_epoch=1``, and ``runner.weight_sync_interval=1``.
The update count :math:`U\in\{1,2\}` must divide :math:`G`, and
``env.train.group_size`` must equal ``algorithm.group_size``.
Each rollout epoch must produce one complete candidate:
``env.train.max_steps_per_rollout_epoch=actor.model.num_action_chunks``.
SD3 requires ``train_sampler=ode`` and ``noise_level=0.0``. The Wan ART path
records deterministic ODE trajectories.

Implementation and Paper Scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current recipes cover scalar-reward SD3.5 image OCR and Wan2.2 text-to-video
OCR experiments. The paper evaluates a broader set of models and rewards:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Scenario
     - Paper Experiments
     - Current ART Recipes
   * - Text to image
     - SD3.5 Medium and FLUX.1-dev; GenEval, OCR, PickScore, and multiple rewards.
     - SD3.5 Medium with scalar OCR reward and LoRA.
   * - Text to video
     - HunyuanVideo-13B; text alignment, visual quality, and motion quality.
     - Wan2.2 TI2V 5B in ``t2v`` mode with scalar video OCR reward and LoRA.
   * - Image to video
     - Distilled Wan2.1-I2V-14B with visual and motion quality rewards.
     - Requires an ART task/data recipe and validation for that setup.
   * - Pairwise feedback
     - Winner/loser or tie labels substituted for scalar rewards.
     - Requires a pairwise reward producer and a corresponding recipe.

The worker follows the scalar aggregation, frozen-target, two-window, and
corrective-mask construction. RLinf supplies distributed placement, trajectory
collection, gradient accumulation, logging, and checkpointing. The implementation
entry points are ``EmbodiedARTFSDPPolicy`` in
``rlinf/workers/actor/fsdp_art_policy_worker.py``, the SD3 and Wan wrappers under
``rlinf/models/diffusion/``, and the ART constraints in ``rlinf/config.py``.
Training diagnostics are defined in :ref:`Diffusion Training Metrics
<diffusion-training-metrics>`.

Installation
------------

The diffusion environment runs from an RLinf checkout on a Linux NVIDIA GPU
host. Its installation command selects CPU OCR scoring by default:

.. code-block:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

This creates a Python 3.10 environment with Diffusers, PEFT, and PaddleOCR.
Both ART configs use ``env.*.reward.use_gpu=False``. GPU OCR scoring uses an
installation with ``--paddle-gpu`` and train/eval reward fields set to ``True``.
The :doc:`Diffusion Models <video_models>` page covers shared dependencies
and download requirements.

Model Checkpoints
-----------------

Each task uses its corresponding Diffusers checkpoint. SD3.5 weight downloads
require acceptance of the model access terms on Hugging Face.

.. code-block:: bash

   huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
      --local-dir /path/to/stable-diffusion-3.5-medium

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
      --local-dir /path/to/Wan2.2-TI2V-5B-Diffusers

These commands save Diffusers-format checkpoints. ``actor.model.model_path``
identifies the matching local directory at launch.

OCR Dataset
-----------

The recipes use OCR prompts from `DiffusionNFT
<https://github.com/NVlabs/DiffusionNFT/tree/main/dataset/ocr>`_. Each line is one
prompt; ``train.txt`` and ``test.txt`` define the two splits.

.. code-block:: bash

   mkdir -p /path/to/dataset/ocr
   curl -fL https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt \
      -o /path/to/dataset/ocr/train.txt
   curl -fL https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt \
      -o /path/to/dataset/ocr/test.txt

These commands create the directory expected by ``visual.ocr_dataset``.
``env.train.dataset.path`` and ``env.eval.dataset.path`` identify that directory.

Training Commands
-----------------

The following commands run from the repository root and use the checked-in
batch settings. Model and dataset arguments identify local asset directories.

.. code-block:: bash

   bash examples/diffusion/run_diffusion.sh sd3_art_ocr \
      actor.model.model_path=/path/to/stable-diffusion-3.5-medium \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr \
      actor.model.model_path=/path/to/Wan2.2-TI2V-5B-Diffusers \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

The script starts the synchronous runner and records logs under
``logs/<timestamp>-<config>/``. The actor trains LoRA parameters with FSDP2,
the rollout worker generates candidates, and the environment scores them.
The two ART YAML files contain the remaining options. ``runner.max_steps=5``
limits a run to a short startup check; compilation and initialization can
dominate the timing of such a run.

.. warning::

   Each actor shard must contain complete prompt groups, and
   ``actor.global_batch_size`` must equal the rollout candidate count divided by
   ``algorithm.art_updates_per_rollout``. The
   :ref:`ART Objective and Implementation <diffusion-art-algorithm>` gives the exact constraints.

Comparison Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

The comparison uses Wan2.2 on four NVIDIA A800 GPUs. The following commands
express three separate runs on the same GPU set.

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0,1,2,3
   COMMON_ARGS=(
      actor.model.model_path=/path/to/Wan2.2-TI2V-5B-Diffusers
      env.train.dataset.path=/path/to/dataset/ocr
      env.eval.dataset.path=/path/to/dataset/ocr
      env.train.total_num_envs=96
      env.eval.total_num_envs=32
      actor.micro_batch_size=2
      actor.model.wan22_ti2v_5b.max_generation_batch_size=2
      actor.model.wan22_ti2v_5b.offload_auxiliary_modules=true
      actor.model.wan22_ti2v_5b.compile_transformer_forward=true
      env.train.reward.use_gpu=false
      env.eval.reward.use_gpu=false
      rollout.pipeline_stage_num=1
      runner.max_steps=200
      runner.max_epochs=200
      runner.save_interval=10
   )

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr \
      "${COMMON_ARGS[@]}" env.train.rollout_epoch=12 \
      actor.global_batch_size=576 actor.optim.lr=1e-4 \
      runner.logger.experiment_name=wan22-art-n1152-lr1e-4

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_nft_video_ocr \
      "${COMMON_ARGS[@]}" env.train.rollout_epoch=12 \
      actor.global_batch_size=576 actor.optim.lr=1e-4 \
      runner.logger.experiment_name=wan22-nft-n1152-lr1e-4

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_nft_video_ocr \
      "${COMMON_ARGS[@]}" env.train.rollout_epoch=6 \
      actor.global_batch_size=288 actor.optim.lr=4e-4 \
      runner.logger.experiment_name=wan22-nft-n576-lr4e-4

The first two runs generate ``96 * 12 = 1152`` candidates per round; the third
generates ``96 * 6 = 576``. All use two optimizer updates per round with these
batch sizes. Both NFT runs use the same implementation, with different learning
rates and sampling amounts. These commands express the configuration choices;
exact curves also depend on the code revision, dependencies, prompts, and seed.

.. _diffusion-art-results:

Training Results
----------------

``env/video_ocr`` records task reward. Generated samples provide additional
evidence of text rendering quality. :ref:`Diffusion Training Metrics
<diffusion-training-metrics>` defines ART guidance, masks, NFT clipping, and
timers. OCR reward measures text similarity; visual quality and temporal
consistency are separate evaluation criteria.

The tables summarize one recorded run per configuration: 148 ART rounds, 111
NFT rounds at 1152 trajectories, and 70 NFT rounds at 576 trajectories.
The reward field is ``video_ocr`` throughout. Smoothed rewards average the last
5760 trajectories: five rounds for ART and NFT-1152, ten for NFT-576.

.. list-table:: Recorded Runs
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - Run
     - Trajectories per Round
     - Learning Rate
     - Recorded Rounds
     - Recorded Elapsed Time
   * - ART
     - 1152
     - 1e-4
     - 148
     - 41:26:57
   * - NFT-1152
     - 1152
     - 1e-4
     - 111
     - 29:29:42
   * - NFT-576
     - 576
     - 4e-4
     - 70
     - 09:35:58

Reward Improvement
~~~~~~~~~~~~~~~~~~

Time and sample costs correspond to the first recorded crossing of each
smoothed reward threshold. Each entry reports elapsed hours and cumulative generated
trajectories; a crossing requires a complete 5760-trajectory averaging window.

.. list-table:: Time and Trajectories to Target Reward
   :header-rows: 1
   :widths: 16 28 28 28

   * - Target
     - ART, 1152, 1e-4
     - NFT, 1152, 1e-4
     - NFT, 576, 4e-4
   * - 0.2
     - 2.30 h / 9216
     - 10.18 h / 43776
     - 2.44 h / 10368
   * - 0.5
     - 3.18 h / 12672
     - 13.67 h / 58752
     - 3.58 h / 14976
   * - 0.8
     - 5.43 h / 21888
     - Unreached within 29.50 h
     - 7.83 h / 32832

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
