SuperMimic
==========

SuperMimic trains a humanoid whole-body tracking policy by backpropagating an
imagined return through a learned, differentiable world model, instead of
weighting a likelihood ratio as PPO does. RLinf implements the method of
*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model* (Liao et al., arXiv:2610.28480, `project page <https://supermimic.github.io>`_)
for the Unitree G1 on MuJoCo Warp. This page describes the algorithm and where
the implementation departs from the paper; installation, data, training, and
evaluation commands are in :doc:`../../examples/embodied/supermimic`.

The code is in ``rlinf/algorithms/supermimic/`` (objectives, networks, replay,
return-based reset sampling), ``rlinf/envs/sim/g1_mimic/`` (reference motion,
kinematics, reward, observation, MuJoCo Warp adapter), and
``rlinf/workers/supermimic/`` (training loop, evaluation, checkpoints).

Training loop
-------------

Each iteration of Algorithm A.1 has three steps, run in one process per GPU:

1. **Collect.** The policy acts in the simulator with Gaussian action noise
   :math:`\epsilon_t \sim \mathcal{N}(0, 0.1^2)`; the replay buffer stores the
   executed noisy action. Episodes start from return-based reset points.
2. **Fit the world model** on ``k = 8``-step windows sampled from the replay.
3. **Improve the policy** through the frozen world model over ``H = 32``
   imagined steps, then fit the critic to the imagined GAE targets.

``algorithm.updates_per_env_step = 0.4`` sets the update-to-data ratio: one
update (one world-model fit with 10 batches, one actor step, and 5 critic
epochs of 4 minibatches) per 2.5 vector steps.

World model
-----------

The dynamics network predicts a 70-d residual from heading-invariant features
of the 71-d state :math:`s` (root height, root rotation without yaw as Rot6D,
heading-frame linear velocity, body-frame angular velocity, joint positions
and velocities), the executed action :math:`a`, and the per-episode sim
context :math:`\xi` (foot friction, joint calibration offsets, torso
centre-of-mass offset):

.. math::

   \hat s_{t+1} = \hat s_t \oplus f_\omega(\phi(\hat s_t), a_t, \xi)

where :math:`\oplus` adds position and velocity residuals in the heading frame
and composes the rotation residual as a rotation vector. Its last layer is
zero, so the initial model predicts no change. It is fit on autoregressive
windows seeded at a logged state and rolled forward on logged actions
(Eq. 1):

.. math::

   \mathcal{L}_{WM}(\omega) = \mathbb{E}_{\mathcal{B}} \sum_{j=1}^{k}
   \Big[ \| \hat s_j - s_j \|^2_{\Sigma} + \mathrm{BCE}(\hat d_j, d_j) \Big]

The state error uses per-block scales and the relative rotation
:math:`\log(R_{pred} R_{target}^\top)`. Reward and observation heads are
analytic: the reward is computed from the predicted state through
differentiable forward kinematics of the G1 tree, and the observation from the
state, the previous action, and :math:`\xi`. The termination head is a learned
MLP on the height and tilt errors against the reference (and their
magnitudes); the BCE label is 1 only for a tracking failure.

Actor objective
---------------

The policy :math:`a_t = \pi_\theta(o_{t-9..t} + \eta_t, g_t)` acts on the
noisy observation history and the goal. With a soft survival weight
:math:`c_0 = 1`, :math:`c_{t+1} = c_t (1 - \hat d_{t+1})`, the imagined
objective is (Eq. 2)

.. math::

   \hat J(\theta) = \mathbb{E}\Big[ \sum_{t=0}^{H-1} \gamma^t \hat c_t \hat r_{t+1}
   + \gamma^H \hat c_H V_\psi(\hat s_H, g_H, \xi) \Big]

and its gradient is taken by backpropagation through time: through the reward,
through :math:`\hat c_t` (the termination head is differentiable), and through
the bootstrap value. The world model and critic are frozen during this step.
Gradients are clipped to a global norm of 1.0 and then element-wise to 0.1.

Critic and GAE
--------------

The critic :math:`V_\psi(s, g, \xi)` (plus a 4-d root error, critic only) is
fit to GAE returns computed backward over the same imagined trajectory
(Eq. 3), starting from :math:`\hat R_H = V_H`:

.. math::

   \hat R_t = \hat r_{t+1} + \gamma (1 - \hat d_{t+1})
   \big[ (1 - \lambda) V_{t+1} + \lambda \hat R_{t+1} \big],
   \qquad \mathcal{L}_V = \mathbb{E} \sum_{t=0}^{H-1} (V_t - \hat R_t)^2

with :math:`\gamma = 0.99` and :math:`\lambda = 0.95`. The targets are fixed
while the critic is fit.

Termination, timeouts, and clip ends
------------------------------------

A motion episode fails when the pelvis height error exceeds 0.25 m or the
tilt error (difference of the gravity direction's z component in the robot
and reference pelvis frames) exceeds 0.8. Only a failure is a world-model
positive and an evaluation failure. The 10 s training cap and the end of the
reference clip end an episode but are not failures; imagination ignores both
and holds the last reference frame past the end of a clip. The return used
for reset sampling bootstraps a timeout with the motion critic and does not
bootstrap a failure or a clip end.

Time alignment
--------------

Everything conditioned at robot time :math:`t` reads reference frames from
:math:`m_t`, and the transition :math:`s_t \to s_{t+1}` is scored against
:math:`m_{t+1}`:

* the 680-d goal :math:`g_t` holds frames :math:`m_t .. m_{t+9}`;
* the 9-d anchor, the critic root error, and the 99-d observation's relative
  orientation compare :math:`s_t` with :math:`m_t`;
* the reward :math:`r_{t+1}` and the termination :math:`d_{t+1}` compare
  :math:`s_{t+1}` with :math:`m_{t+1}`;
* in imagination, step :math:`t` uses the goal at :math:`\tau_0 + t` and scores
  :math:`\hat s_{t+1}` against :math:`m_{\tau_0 + t + 1}`.

The paper describes the goal as ten *future* frames, and CuboiLeo's
implementation uses :math:`m_{t+1} .. m_{t+10}` with its anchor and critic
error against :math:`m_{t+1}`. Starting the 680-d window at the current frame
is this implementation's choice; both conventions are self-consistent, but
the policy sees one frame less of the future. The 730-d goal keeps zita-ch's
:math:`m_{t+1} .. m_{t+10}`. Checkpoints record the window in
``checkpoint_meta.json`` and refuse to load into a config with another one.

Reward
------

The motion reward is BeyondMimic's tracking reward without its self-collision
penalty, with the joint-limit weight lowered to -2, all multiplied by
:math:`\Delta t = 0.02` s (Table A.2): body position, orientation, linear and
angular velocity over 14 tracked bodies, root position and orientation, action
smoothness, and joint limits. As in BeyondMimic, the root terms compare
``torso_link`` in the world frame, and the body pose terms first move the
reference onto the robot torso's xy position and yaw (``update_relative_body_poses``),
so global drift stays in the root terms. Body velocities are compared in the
world frame.

Return-based reset sampling
---------------------------

Appendix D.1: the motion timeline is cut into 1 s bins, each bin keeps an EMA
:math:`G_b` of the discounted return from it, and resets mix 50% uniform over
bins, 40% :math:`\mathrm{softmax}_b(-25\, G_b / G_{max})`, and 10% a slow EMA of
that softmax. :math:`G_{max} = 0.1 / (1 - \gamma) = 10` is the largest motion
return. Temperature 25, the EMA rates 0.05 and 0.01 are community values, not
printed in the paper. Only motion-tracking episodes update :math:`G_b`; a
velocity episode's return depends on its command rather than on the
reference point, so velocity environments reset uniformly over bins.

Goal interfaces and velocity tracking
-------------------------------------

Appendix B trains several goal interfaces through one world model with a
shared decoder :math:`a = \mathrm{dec}_\theta(\mathrm{enc}_{\theta,i}(g), o)`.
The velocity interface (Appendix G, RL encoder) maps the command
:math:`(v_x, v_y, \omega_z)` and the observation history to the 32-d latent.
Half of the environments and half of every imagined batch use it, with the
Table A.6 reward, a soft fall termination, and its own critic. The decoder's
weights are frozen on the velocity update, and a 0.1-weighted MSE pulls the
velocity latent toward the stop-gradient motion latent on rollouts with
reference commands. The distillation encoder of Appendix G is not
implemented.

Paper, community, and implementation choices
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 18 20

   * - Item
     - Paper
     - CuboiLeo
     - zita-ch
     - This implementation
   * - State and residual
     - not specified
     - 71-d state, scaled residual, explicit integration
     - 71-d, other frames
     - 71-d state, unscaled heading-frame residual
   * - World-model state loss
     - :math:`\|\hat s - s\|^2`
     - per-block scales
     - --
     - CuboiLeo's scales, SO(3) rotation error
   * - 680-d goal window
     - ten future frames
     - :math:`m_{t+1}..m_{t+10}`
     - --
     - :math:`m_t..m_{t+9}`
   * - 730-d goal
     - --
     - --
     - :math:`m_{t+1}..m_{t+10}`
     - as zita-ch
   * - 9-d anchor body
     - --
     - torso
     - --
     - torso
   * - Body re-anchoring
     - BeyondMimic
     - torso
     - pelvis
     - torso
   * - Termination head
     - learned or analytic
     - learned
     - analytic sigmoid
     - learned; ``termination_head: analytic`` is experimental
   * - Input normalization
     - not specified
     - fixed scales
     - running normalization
     - none
   * - Reference velocity
     - not specified
     - central difference
     - central difference
     - central difference
   * - Reset sampler constants
     - bins, 50/40/10
     - temperature 25, EMA 0.05/0.01
     - --
     - as CuboiLeo
   * - Velocity step test
     - G.2, 672 commands x 4 trials
     - --
     - --
     - G.2; trials differ by sim context and joint noise

CuboiLeo (`CuboiLeo/SuperMimic-Reproduced <https://github.com/CuboiLeo/SuperMimic-Reproduced>`_)
and zita-ch (`zita-ch/supermimic_vibed <https://github.com/zita-ch/supermimic_vibed>`_)
are community reproductions used as design references; no code was copied
from them. The table describes their ``main`` branches as consulted during
development; no commit was recorded, so later changes in those repositories
take precedence over this table.

Known limitations
-----------------

* **Termination-head cold start.** With PyTorch's default initialization the
  head outputs about 0.5, so the 32-step survival starts near
  :math:`10^{-9}` and the actor receives almost no gradient beyond the first
  few imagined steps until the head has seen a few hundred world-model steps.
  Failures are about 0.3% of the logged transitions; in short runs the head
  learned the base rate first and gave lower probabilities to large errors
  than to small ones. ``algorithm.termination_bias_init`` (for example -4.6,
  a 1% prior) and ``algorithm.termination_head: analytic`` remove the cold
  start and, in short local runs, gave a correctly ordered head. Both are
  experimental; the defaults are the paper setting.
* **No network input normalization.** Joint velocities reach about 20 in the
  observation and the calibration offsets in :math:`\xi` are about 0.01; the
  paper does not specify normalization and none is applied.
* **BPTT gradient spikes.** The pre-clip actor gradient norm can spike by two
  orders of magnitude; the paper's clipping bounds each step.
* **Not implemented:** the Appendix G distillation encoder and the
  push-recovery experiments. A converter for the official BONES-SEED G1 CSV
  exists, and it has not been checked on a real release.
* **Not demonstrated:** a tracking policy at paper scale, the paper's
  two-GPU recipe, and multi-node runs. A short two-GPU run on 80 GB A800s
  has finished on real NCCL with zero parameter divergence. See the user
  guide for what has and has not been run.
