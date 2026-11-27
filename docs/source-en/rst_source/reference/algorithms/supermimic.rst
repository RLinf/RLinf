SuperMimic
==========

SuperMimic trains a humanoid whole-body motion-tracking policy by backpropagating an imagined return through a differentiable world model. RLinf implements *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model* (Liao et al., `arXiv:2610.28480 <https://arxiv.org/abs/2610.28480>`__, `project page <https://supermimic.github.io>`__) for the 29-DoF Unitree G1 in MuJoCo Warp, following the state layouts and optimizers of `HybridRobotics/supermimic <https://github.com/HybridRobotics/supermimic>`__ (Apache-2.0).

For installation and training steps, see :doc:`../../examples/embodied/supermimic`; for the simulation environment, see :doc:`../../examples/embodied/g1_mimic`; for standalone evaluation, see :doc:`../../evaluations/guides/supermimic`.

Training Loop
-------------

One global update consists of three phases executed in lockstep across all GPU ranks:

1. **Collect.** The policy acts in MuJoCo Warp with exploration noise :math:`\epsilon \sim \mathcal{N}(0, 0.1^2)` and stores the executed noisy action in each rank's replay buffer. Reset episodes sample start frames from the return-based reset sampler.
2. **Fit the world model.** Unroll ``k = 8`` consecutive steps from replay states under recorded actions over 10 batches of 2048 global windows.
3. **Optimize policy and critic.** Imagine ``H = 32`` steps from replay start states inside the frozen world model, backpropagate the imagined return to the policy, and fit the critic to GAE targets on those rollouts for 5 epochs of 4 minibatches.

With ``algorithm.replay_ratio = 16``, 4096 global environments, and ``10 × 2048`` windows per update, each control step drives 0.4 global updates (a 6000-update run collects ~15000 control steps). Because the replay buffer is static during an update, all 10 world-model batches and imagination start states are sampled up front from one shared table of legal ``k``-step window starts (a window never crosses an episode boundary before its final step).

State Representations and Dynamics Integration
----------------------------------------------

``actor.model.representation`` selects the state layout shared by replay storage, the world model, and the policy. Checkpoints load only into the layout they were trained with.

.. list-table::
   :header-rows: 1
   :widths: 15 24 25 20 16

   * - Layout
     - Stored State
     - Encoded State
     - Predicted Delta
     - Body / Joint Recovery
   * - ``gencoord``
     - 71-dim simulator state
     - 68-dim ``[z, gravity_b, v_b, w_b, q, qd]``
     - 35-dim ``[dv_b, dw_b, dqd]``
     - Differentiable G1 forward kinematics
   * - ``maxcoord``
     - 182-dim world pose & twist of 14 tracked bodies
     - 214-dim per-body pelvis-frame pos, 6-dim rot, twist + pelvis height & gravity
     - 84-dim ``[dv, dw]`` per body
     - Chain inverse kinematics

Given predicted velocity increments :math:`\Delta v` over one control step :math:`\Delta t = 0.02\text{ s}` containing :math:`D = 4` physics substeps, velocities integrate the full increment and positions/orientations integrate the semi-implicit substep-averaged fraction:

.. math::

   v_{t+1} = v_t + \Delta v, \qquad
   p_{t+1} = p_t + \Delta t \left(v_t + \frac{D + 1}{2D} \Delta v\right).

Objectives
----------

World-Model Loss (Bryson Weighting)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Over an autoregressive ``k``-step window, the loss compares predicted state channels :math:`\hat{x}_c` against recorded states :math:`x_c` by channel group :math:`c` with dimension :math:`d_c` and nominal error :math:`\sigma_c`:

.. math::

   \mathcal{L}_{\text{wm}} = \frac{\sum_c d_c \, \sigma_c^{-2} \, \text{MSE}(\hat{x}_c, x_c)}{\sum_c d_c}.

.. list-table:: Nominal Errors
   :header-rows: 1
   :widths: 22 14 22 14 28

   * - Channel Group
     - :math:`\sigma_c`
     - Channel Group
     - :math:`\sigma_c`
     - Unit
   * - ``root_pos``
     - ``0.01``
     - ``body_pos``
     - ``0.05``
     - m
   * - ``root_ori``
     - ``0.02``
     - ``body_rot``
     - ``0.1``
     - rad
   * - ``joint_pos``
     - ``0.1``
     - ``body_vel``
     - ``1.0``
     - rad / (m/s)
   * - ``joint_vel``
     - ``3.14``
     - ``body_avel``
     - ``3.14``
     - rad/s

Tracking Reward and Soft Termination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the simulator and imagined rollouts evaluate the same per-step tracking reward over the pelvis root and 14 tracked bodies (with body position and orientation references re-anchored to the robot pelvis's horizontal xy and yaw):

.. math::

   r_t = \Delta t \left( \sum_{i=1}^{6} w_i \exp\!\left(-\frac{e_i}{\sigma_i^2}\right) - 0.1 \|a_t - a_{t-1}\|_2^2 - 2.0 \sum_{j=1}^{29} \delta_{\text{lim}, j} \right),

where :math:`\Delta t = 0.02`, :math:`\delta_{\text{lim}, j}` is the joint-angle violation beyond soft limits, and :math:`\sum w_i = 5` gives a per-step reward ceiling of :math:`r_{\max} = 0.1`:

.. list-table:: Tracking Kernel Parameters
   :header-rows: 1
   :widths: 44 28 28

   * - Kernel
     - Weight :math:`w_i`
     - Std :math:`\sigma_i`
   * - Root position / orientation
     - 0.5 / 0.5
     - 0.3 / 0.4
   * - Body position / orientation
     - 1.0 / 1.0
     - 0.3 / 0.4
   * - Body linear / angular velocity
     - 1.0 / 1.0
     - 1.0 / 3.14

Hard failure in simulation triggers when :math:`|z_{\text{pelvis}} - z_{\text{ref}}| > 0.25\text{ m}` or :math:`|g_z - g_{z,\text{ref}}| > 0.8`. During imagination, failure is replaced by a differentiable soft termination probability with softness :math:`\tau = 0.05`:

.. math::

   \hat{d}_t = \sigma\!\left(\frac{\max\!\left(\frac{|z_{\text{pelvis}} - z_{\text{ref}}|}{0.25},\; \frac{|g_z - g_{z,\text{ref}}|}{0.8}\right) - 1}{\tau}\right).

Imagined Return and Critic Targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the world model and critic frozen, the policy maximizes the soft-survival discounted imagined return over horizon :math:`H = 32` (:math:`\gamma = 0.99`):

.. math::

   c_0 = 1, \quad c_{t+1} = c_t (1 - \hat{d}_t), \qquad
   J(\theta) = \mathbb{E}\!\left[ \sum_{t=0}^{H-1} \gamma^t c_t r_t + \gamma^H c_H V_\psi(s_H) \right].

Gradients flow through :math:`r_t`, :math:`\hat{d}_t`, and :math:`V_\psi(s_H)`. The critic then fits fixed GAE targets (:math:`\gamma = 0.99`, :math:`\lambda = 0.95`) computed backward over the same imagined rollout, with predicted values clamped to :math:`r_{\max} / (1 - \gamma) = 10.0`:

.. math::

   \delta_t = r_t + \gamma (1 - \hat{d}_t) V(s_{t+1}) - V(s_t), \qquad
   A_t = \delta_t + \gamma \lambda (1 - \hat{d}_t) A_{t+1}, \qquad
   \hat{G}_t = A_t + V(s_t).

Network Architectures and Inputs
--------------------------------

Each network is a stack of ``Linear -> LayerNorm -> GELU`` blocks followed by a final ``Linear`` layer, with running empirical normalization on state and goal inputs.

* **Goal inputs:** 10 reference frames starting at the current frame, spaced 5 control steps apart (clamped at the clip end). Each frame holds joint positions, joint velocities, and root 6-dim rotation relative to the current root (640-dim actor goal); the world-model and critic goal appends relative root positions (670-dim).
* **Simulator context (33-dim):** Normalized foot friction (1), torso CoM offset (3), and joint encoder biases (29).
* **Dynamics (world model):** Input ``encoding_dim`` + 29 (PD joint target) + 33 (context); hidden widths ``[2048, 2048, 2048]``.
* **MotionPolicy (actor):** Goal encoder ``640 -> [2048, 1024, 512, 256] -> 32`` (with ``tanh`` latent); decoder ``32 + encoding_dim -> [2048, 1024, 512, 256] -> 29``.
* **Critic:** Input ``encoding_dim`` + 670 + 33; hidden widths ``[256, 128] -> 1``.

Optimizers and Execution
------------------------

* **Dual optimizers:** Hidden linear matrices in every stack use Muon (``adjust_lr_fn: match_rms_adamw``, weight decay ``0.1``, momentum ``0.95``, Nesterov ``True``). Remaining parameters (first/last layers, norms, biases) use fused Adam with :math:`\beta = (0.9, 0.95)` for the actor and :math:`\beta = (0.9, 0.999)` for the world model and critic. All optimizers use a constant learning rate of :math:`2\times 10^{-4}`.
* **Gradient clipping and health check:** Training clips the global gradient norm to ``1.0`` on GPU without host synchronization and clamps actor gradients element-wise to ``0.1``. Non-finite gradients stop the run when metrics copy to the host.
* **CUDA graph:** ``actor.cuda_graph`` (default ``True``) runs 3 eager warm-up updates, captures update 4 (all three phases together) as one CUDA graph, and replays it thereafter.
* **Mixed precision (bf16):** FSDP2 ``mixed_precision.param_dtype: bf16`` (with fp32 reductions and uncast forward inputs) casts network weights to bf16 once per phase at unshard while master weights, optimizer state, gradient reductions, kinematics, rewards, and losses stay float32; rollout inference uses ``rollout.amp_autocast`` in bf16.

Parallel Training (FSDP2)
-------------------------

Every GPU runs one env, one rollout, and one actor worker; env rank :math:`r` communicates only with rollout and actor rank :math:`r`. Global env counts and batch sizes split evenly across ranks, and each actor rank samples its share from its local replay buffer.

Actor workers form one FSDP2 group with three separate FSDP units (``module_classes_to_wrap: [Dynamics, MotionPolicy, Critic]``). Because the full model is ~64 MB in float32, the default config sets ``sharding_strategy: no_shard`` and ``reshard_after_forward: False``, so each rank keeps full weights and synchronizes gradients with all-reduce. With a sharded strategy such as ``full_shard``, unshard all-gathers weights and backward passes reduce-scatter gradients. Before each phase, only the active phase's parameters become trainable and the phase unshards only the networks it reads:

* **World-model phase:** unshards ``Dynamics``.
* **Actor phase:** unshards all three networks with ``Dynamics`` and ``Critic`` frozen.
* **Critic phase:** unshards ``Critic`` only.

Each update computes global gradient norms across FSDP parameters, and reset-bin statistics plus input-normalization moments reduce across env ranks.

Return-Based Reset Sampler
--------------------------

The motion library timeline is partitioned into 0.2 s bins (10 frames at 50 Hz). Each bin :math:`b` tracks an estimate :math:`G_b` of the discounted return earned when starting from that bin (capped by :math:`G_{\max} = 0.1 / (1 - \gamma) = 10`). Every 20 control steps, the reset distribution refreshes as a three-way mixture:

.. math::

   p(b) = 0.5 \, \mathcal{U}(b) + 0.4 \, \text{softmax}\!\left(-25 \frac{G_b}{G_{\max}}\right)_b + 0.1 \, p_{\text{keep}}(b),

where :math:`p_{\text{keep}}` accumulates the squared softmax distribution over time so previously difficult segments retain sampling mass. Episodes ending on a refresh step sample from the distribution prior to that refresh, matching the official sampler. Return tracking covers all frames in a clip, while episode starts sample only from the first ``frames - 2`` frames (weighted by each bin's share).

Configuration Reference
-----------------------

Core algorithm and actor sections from ``examples/embodiment/config/supermimic_lafan1_gencoord.yaml``:

.. code-block:: yaml

   algorithm:
     adv_type: null
     loss_type: supermimic
     k: 8                       # world-model training window
     H: 32                      # policy imagination horizon
     gamma: 0.99
     lam: 0.95
     replay_ratio: 16           # average world-model draws per transition
     wm_batches: 10
     wm_window_batch: 2048      # global windows per batch
     imagine_batch: 4096        # global imagined rollouts per update
     critic_epochs: 5
     critic_minibatches: 4
     replay_length: 1000

   actor:
     cuda_graph: True
     model:
       representation: gencoord  # gencoord or maxcoord
     fsdp_config:
       strategy: fsdp2
       sharding_strategy: no_shard
       reshard_after_forward: False
       wrap_policy:
         module_classes_to_wrap: [Dynamics, MotionPolicy, Critic]
       mixed_precision:
         param_dtype: bf16
         reduce_dtype: fp32
         cast_forward_inputs: False
       amp_autocast:
         enabled: False
         precision: bf16

Differences from the Paper and Official Code
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Item
     - Paper or Official Code
     - RLinf Implementation
   * - State layout
     - Official code provides GenCoord and MaxCoord
     - Same; ``supermimic_lafan1_gencoord`` and ``supermimic_lafan1_maxcoord`` differ only here
   * - World-model loss
     - Paper writes unweighted squared error
     - Bryson channel-group weighting (see Nominal Errors above)
   * - Goal window
     - Paper states 10 future frames
     - 10 frames spaced 5 control steps apart
   * - Reset bins
     - Paper states 1 s bins
     - 0.2 s bins with 50% / 40% / 10% mixture
   * - Precision
     - Paper Figure 5 uses TF32
     - Float32 master weights, optimizer state, kinematics, and losses with bf16 network matmuls by default via FSDP2 ``mixed_precision.param_dtype`` (and ``rollout.amp_autocast`` for rollout); set ``param_dtype: null`` and ``rollout.amp_autocast.enabled: False`` for full TF32
   * - CUDA graphs
     - Official code captures 2 graphs (world model; actor + critic)
     - Enabled by default (``actor.cuda_graph: True``); captures all 3 phases in 1 graph
   * - Simulator scoring
     - Official env scores body poses from before the last physics substep
     - Body poses refresh after the last substep before reward and failure scoring
   * - Critic minibatches
     - Official code trims rows to a multiple of 4 before shuffling
     - Shuffles all rows first, then trims
   * - Parallelism
     - Official entry is single-process, single-GPU
     - Multi-GPU FSDP2 with one env, rollout, and actor worker per GPU and global batching
   * - Default dataset
     - Paper main result uses filtered BONES-SEED; official default pulls one W&B clip
     - Local 40 LAFAN1 clips

Limitations
-----------

* Velocity tracking is not implemented.
* Standing push recovery uses mjlab's knees-bent keyframe rather than the Section 4.3 T-pose.
* Multi-node training and resuming across different GPU counts are unvalidated.

Acknowledgments
---------------

Thanks to `CuboiLeo <https://github.com/CuboiLeo>`__ (`SuperMimic-Reproduced <https://github.com/CuboiLeo/SuperMimic-Reproduced>`__) and `zita-ch <https://github.com/zita-ch>`__ (`supermimic_vibed <https://github.com/zita-ch/supermimic_vibed>`__) for publishing their community reproductions, which this integration consulted.
