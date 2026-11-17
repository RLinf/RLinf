SuperMimic
==========

SuperMimic trains a humanoid whole-body tracking policy by backpropagating
an imagined return through a differentiable world model. RLinf implements the
method of *SuperMimic: Scaling Humanoid Whole-Body Control with One
Differentiable World Model* (Liao et al., arXiv:2610.28480, `project page
<https://supermimic.github.io>`_) for the Unitree G1 in MuJoCo Warp. The
state layouts and the optimizers follow `HybridRobotics/supermimic
<https://github.com/HybridRobotics/supermimic>`_ (Apache-2.0). This page
describes the algorithm and where it differs from the paper. Installation, data, and
commands are in :doc:`../../examples/embodied/supermimic`.

Training Loop
-------------

One update has three steps:

1. **Collect.** The policy acts in simulation. Training noise is
   :math:`\mathcal{N}(0, 0.1^2)`, and the replay stores the noisy action that
   was executed. Episodes start at a reference frame drawn from the return
   sampler.
2. **Fit the world model** on ``k = 8`` step windows from the replay. One
   update has 10 batches of 2048 windows, summed over all GPUs.
3. **Improve the policy.** Imagine ``H = 32`` steps in the world model and
   backpropagate, then fit the critic to GAE targets on those rollouts for
   5 epochs of 4 minibatches.

``algorithm.replay_ratio = 16`` is how many times each transition is drawn
by the world model on average. With the batches above and 4096 environments,
the run performs 0.4 global updates per control step, so the 6000-update run
in the user guide collects about 15,000 control steps.

The replay does not change during an update. Each update therefore draws all
10 world-model batches and the imagination start states before it trains,
and the table of legal ``k``-step window starts is built once and shared by
the 10 batches. A window never crosses an episode end before its last step.
Each env is drawn uniformly among those with a legal window, then a start
uniformly within that env.

World Model
-----------

``actor.model.representation`` selects the layout. A checkpoint loads only
into the layout it was trained with.

* ``gencoord`` stores the 71-d simulator state. The encoding is the 68-d
  vector ``[z, gravity_b, v_b, w_b, q, qd]``. The network predicts a 35-d
  velocity increment ``[dv_b, dw_b, dqd]``. Body poses are computed from the
  predicted state by the G1 kinematic tree.
* ``maxcoord`` stores the world pose and twist of the 14 tracked bodies
  (182-d). The encoding is each body's position, 6-d rotation, and twist in
  the pelvis frame, plus the pelvis height and gravity (214-d). The network
  predicts ``[dv, dw]`` per body (84-d). Joint angles come back through
  inverse kinematics.

Positions integrate :math:`(D+1)/(2D)` of the increment, where :math:`D = 4`
is the number of physics substeps in one control step. Velocities integrate
the whole increment. A window starts from the recorded state and rolls
forward under the recorded actions. The state loss is grouped by channel and
weighted with Bryson's rule: each group's mean squared error is divided by
that group's nominal error squared, then combined by channel count. The
nominal errors are ``NOMINAL_ERRORS`` in
``rlinf/algorithms/supermimic/objectives.py``. This is not the paper's
unweighted :math:`\|\hat s - s\|^2`.

The reward is the tracking reward of the predicted state after differentiable
forward kinematics. The anchor is the pelvis: the root terms compare the
pelvis, and the body-pose terms re-anchor the reference at the pelvis xy and
yaw. The six tracking kernels are root position, root orientation, body
position, body orientation, body linear velocity, and body angular velocity.
Their weights sum to 5 and are multiplied by :math:`\Delta t = 0.02`, so the
per-step reward ceiling is 0.1. An action-rate penalty and a joint-limit
penalty are added.

The failure test uses the pelvis. The episode fails when the absolute height error
exceeds 0.25 m or the absolute tilt error exceeds 0.8. Tilt is the
difference of the gravity direction's z component in the robot and reference
pelvis frames. The 10 s training time limit and the end of the reference
clip also end the episode, and neither counts as a failure.

Actor and Critic
----------------

The policy reads the current state encoding and the 640-d actor goal, not a
stack of proprioceptive history. The goal is ten reference frames, five
control steps apart, held at the clip's last frame. Its content is each
frame's joint positions and joint velocities, plus a 6-d rotation of each
frame's root relative to the current root. The world-model goal also appends
the relative root position and is 670-d. The encoding and the goal are
normalized by running moments.

The imagined return uses a soft survival weight :math:`c_0 = 1`,
:math:`c_{t+1} = c_t (1 - \hat d_{t+1})`. The gradient passes through the
reward, the termination probability, and the bootstrap value. The world
model and the critic are frozen for this step. Each network is a stack of
``Linear -> LayerNorm -> GELU`` blocks and a final linear layer. The policy
encoder and decoder use hidden widths 2048, 1024, 512, 256. The dynamics
network uses 2048, 2048, 2048. The critic uses 256, 128. The latent size is
32.

The critic fits GAE returns computed backwards on the same imagined
rollout, with :math:`\gamma = 0.99` and :math:`\lambda = 0.95`. Targets are
held fixed while the critic fits, and values are clamped to the largest
discounted return the tracking reward can produce.

Optimizers
----------

Each phase has two optimizers. Hidden matrices use Muon with
``adjust_lr_fn`` ``match_rms_adamw``, weight decay 0.1, and momentum 0.95.
The remaining parameters (first and last layers, norms, and biases) use
fused Adam. The actor's Adam uses :math:`\beta = (0.9, 0.95)`; the world
model and the critic use :math:`(0.9, 0.999)`. The learning rate is
:math:`2\times 10^{-4}` with a constant schedule.

Gradients are clipped to a global norm of 1.0. The actor then clamps each
element to 0.1. The norm stays on the GPU, so this step does not read it on
the host. A non-finite gradient is not skipped. When a batch of updates
finishes and the metrics are copied to the host, a non-finite value stops
training. ``actor.cuda_graph`` is off by default. When it is on, the first
three updates run eagerly, the fourth is recorded as one CUDA graph, and
every later update replays it. The official code captures the world-model
update and the actor-and-critic update as two graphs; RLinf captures the
three phases in one. Replay changes speed, not the update written above.
``actor.fsdp_config.amp_autocast.enabled`` is also off by default; turning it
on runs network matmuls in bf16 while weights, losses, and optimizer state
stay float32.

Parallel Training
-----------------

Every GPU runs one env, one rollout, and one actor worker, and env rank
:math:`r` exchanges data only with rollout and actor rank :math:`r`. The
environment count and both batches are global and split evenly across ranks.
Each actor rank keeps a replay of its own envs and draws its share of every
batch from it.

The actor workers form one FSDP2 group, and the dynamics network, the
policy, and the critic are three FSDP units
(``module_classes_to_wrap: [Dynamics, MotionPolicy, Critic]``). Before each
phase every unit is resharded and the phase's parameters are made trainable.
Each forward of a phase all-gathers the networks it reads once, so an 8-step
world-model window or a 32-step imagination runs on gathered weights rather
than gathering at every step:

* the world-model phase gathers the dynamics network;
* the actor phase gathers all three, with the dynamics network and the critic
  frozen;
* the critic phase gathers only the critic.

Gradients are reduce-scattered by FSDP. The pre-clip norm is RLinf's
``get_grad_norm`` over the phase's sharded parameters. Reset-bin statistics
and the input-normalization moments are reduced across env ranks.

Return-Based Reset
------------------

The motion timeline is cut into 0.2 s bins (10 frames at 50 Hz). Each bin
keeps an estimate of the discounted return earned from that bin. A reset
mixes three terms: 50% uniform over bins, 40%
:math:`\mathrm{softmax}(-25\, G_b / G_{max})`, and 10% on the distribution
accumulated from the square of that softmax. :math:`G_{max} = 0.1 / (1 -
\gamma) = 10`. The paper uses 1 s bins. The distribution is refreshed every
20 control steps; an episode that ends on a refresh step draws its start
from the distribution before that refresh, as in the official code. Returns
are binned over all frames of the timeline, but a start is drawn within the
chosen bin's share of the first ``frames - 2`` frames, so a draw can land in
the neighboring bin. The official sampler maps frames the same way.

Differences from the Paper and the Official Code
-------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Item
     - Paper or official code
     - This implementation
   * - State
     - The official code has GenCoord and MaxCoord
     - The same. The two YAMLs differ only in this choice
   * - World-model loss
     - The paper uses an unweighted square
     - Bryson weighting by a nominal error
   * - Goal
     - The paper says 10 future frames
     - 10 frames, five control steps apart
   * - Reset bins
     - The paper uses 1 s
     - 0.2 s, with shares 50% / 40% / 10%
   * - Precision
     - Figure 5 of the paper is TF32
     - Float32 weights and TF32 matmuls. bf16 is used for network matmuls
       only when ``amp_autocast`` is enabled
   * - Optimizer
     - The official code uses Muon plus fused Adam, and captures the
       world-model update and the actor-and-critic update as two CUDA graphs
     - Muon plus fused Adam, with the norm left on the GPU. Off by default;
       with ``actor.cuda_graph=True`` each update replays one CUDA graph
   * - Simulator scoring
     - The official environment scores reward and failure on body poses
       from before the last physics substep
     - Body poses are refreshed after the last substep, before reward and
       failure are scored
   * - Critic minibatches
     - The official code trims the imagined rows to a multiple of 4 before
       shuffling, so the last rows are never fitted
     - All rows are shuffled first, then trimmed. Only a row count that is
       not a multiple of ``critic_minibatches`` differs
   * - Parallelism
     - The official training entry is one process on one GPU
     - One env, rollout, and FSDP2 actor worker per GPU, with the three
       networks as separate FSDP units. Batches are global, so the GPU count
       does not change the update
   * - Reported data
     - The paper's main result is filtered BONES-SEED. The official default
       downloads one Weights & Biases clip
     - The local 40 LAFAN1 clips. Evaluation numbers are in the user guide

Entry Points
------------

Training starts at ``examples/embodiment/train_embodied_agent.py`` and
evaluation at ``evaluations/eval_embodied_agent.py``. Both select the
SuperMimic runner when ``algorithm.loss_type`` is ``supermimic``.

* Runners: ``rlinf/runners/supermimic_runner.py`` and
  ``rlinf/runners/supermimic_eval_runner.py``.
* Env worker: ``rlinf/workers/env/supermimic_env_worker.py``.
* Rollout worker: ``rlinf/workers/rollout/hf/supermimic_rollout_worker.py``.
* Actor worker: ``rlinf/workers/actor/fsdp_supermimic_policy_worker.py``.
* Networks: ``rlinf/models/embodiment/supermimic/``.
* Loss, replay, and the reset sampler: ``rlinf/algorithms/supermimic/``.
* Simulation, reference motion, reward, and evaluation:
  ``rlinf/envs/sim/g1_mimic/``.

Limitations
-----------

Velocity tracking is not implemented. Standing push recovery holds mjlab's
knees-bent keyframe, not the T-pose in Section 4.3 of the paper. The
BONES-SEED converter has not been checked against an official CSV.
Multi-node training and resuming on a different GPU count are unvalidated.
The tracking table in the user guide evaluates GenCoord at update 6000. Its
runs predate the three FSDP units, the shared window table, and the
critic shuffle described above; those changes have not yet been rerun at
that scale.

Acknowledgments
---------------

Thanks to `CuboiLeo <https://github.com/CuboiLeo>`__
(`SuperMimic-Reproduced <https://github.com/CuboiLeo/SuperMimic-Reproduced>`__)
and `zita-ch <https://github.com/zita-ch>`__
(`supermimic_vibed <https://github.com/zita-ch/supermimic_vibed>`__) for
publishing their community reproductions, which earlier versions of this
integration consulted. The current settings follow the official code and the
paper as described above.
