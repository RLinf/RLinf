SuperMimic
==========

SuperMimic trains a humanoid whole-body tracking policy by backpropagating
an imagined return through a differentiable world model. RLinf implements the
method of *SuperMimic: Scaling Humanoid Whole-Body Control with One
Differentiable World Model* (Liao et al., arXiv:2610.28480, `project page
<https://supermimic.github.io>`_) for the Unitree G1 in MuJoCo Warp. The
state layouts and the optimizers follow `HybridRobotics/supermimic
<https://github.com/HybridRobotics/supermimic>`_ (Apache-2.0). This page is
the algorithm and where it differs from the paper. Installation, data, and
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
each control step owes 0.4 global updates. ``runner.max_epochs=16000`` is
therefore 40,000 control steps.

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

Failure reads the pelvis. The episode fails when the absolute height error
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
and dynamics hidden widths step down from 2048 to 256, the critic is
``(256, 128)``, and the latent size is 32.

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
training. The official implementation captures the whole update in a CUDA
graph. FSDP2 collectives cannot be captured that way, so this code does not.
The update written above is unchanged.

Return-Based Reset
------------------

The motion timeline is cut into 0.2 s bins (10 frames at 50 Hz). Each bin
keeps an estimate of the discounted return earned from that bin. A reset
mixes three terms: 50% uniform over bins, 40%
:math:`\mathrm{softmax}(-25\, G_b / G_{max})`, and 10% on the distribution
accumulated from the square of that softmax. :math:`G_{max} = 0.1 / (1 -
\gamma) = 10`. The paper uses 1 s bins.

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
     - The official code uses Muon plus fused Adam, and captures the whole
       update in a CUDA graph
     - Muon plus fused Adam, with the norm left on the GPU. No CUDA graph
   * - Parallelism
     - The official training entry is one process
     - One env, rollout, and FSDP2 actor worker per GPU
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
The tracking tables in the user guide are the checkpoints at updates 2000
and 4000; the training target of that run is 16000 updates.
