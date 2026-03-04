Async Proximal Policy Optimization (Async PPO)
==============================================

1. Introduction
---------------

Async PPO in RLinf is an asynchronous training implementation for embodied tasks. It keeps the core optimization logic of PPO, but it does not follow the strictly synchronized pattern of "collect one rollout batch, then train one update batch". Instead, it decouples environment interaction, rollout inference, and actor training into long-running concurrent pipelines.

The goal of this implementation is not to redefine PPO mathematically. Its purpose is to remove system bottlenecks that are common in embodied training, especially when:

- environment stepping is slow and the actor spends too much time waiting for data,
- policy inference is expensive because of heavy visual encoders or large models,
- rollout horizons are long and synchronized barriers reduce end-to-end throughput.

Async PPO is most useful when system throughput is the main bottleneck. If your workload is small or standard PPO already trains efficiently and stably, the synchronous setup is usually simpler.


2. Scope
--------

In the current RLinf implementation, Async PPO is only available for embodied tasks. The relevant entrypoints and core components are:

- Training entry: ``examples/embodiment/train_async.py``
- Runner: ``rlinf/runners/async_ppo_embodied_runner.py``
- Actor worker: ``rlinf/workers/actor/async_ppo_fsdp_worker.py``
- Rollout worker: ``rlinf/workers/rollout/hf/async_huggingface_worker.py``
- Environment worker: ``rlinf/workers/env/async_env_worker.py``

The current implementation also has several explicit constraints:

- It only supports embodied tasks, not reasoning or agent tasks.
- The Async PPO path requires ``algorithm.loss_type: decoupled_actor_critic``.
- The actor model must include a value head, so ``actor.model.add_value_head: True`` is required.
- ``AsyncEnvWorker`` and ``AsyncMultiStepRolloutWorker`` do not support offload.
- When ``rollout.recompute_logprobs=True``, actor weight offload is not supported.
- ``AsyncPPOEmbodiedRunner`` does not currently implement validation; if ``runner.val_check_interval > 0`` is set, RLinf warns and skips validation.


3. Why Async PPO
----------------

The standard synchronous PPO execution pattern is:

1. Collect one batch of trajectories.
2. Stop sampling.
3. Compute advantages and run policy updates.
4. Synchronize new weights.
5. Start the next rollout round.

This design is simple, but it introduces strong global barriers. If one stage is slow, the others wait.

For embodied training, this often becomes a real system problem:

- the simulator may be slow because of physics, rendering, or reset logic,
- rollout inference may be expensive because of perception-heavy policies,
- actor training may require large batches and significant GPU time.

Async PPO removes this barrier-heavy execution pattern:

- the environment continuously produces observations,
- the rollout worker continuously consumes observations and produces actions while accumulating trajectories,
- the actor continuously consumes completed rollout batches and updates the policy.

This improves overall utilization, but it comes with a tradeoff: samples are no longer guaranteed to come from the newest policy. As a result, stale-sample control becomes a first-class design concern.


4. System Architecture
----------------------

The high-level data flow of Async PPO in RLinf is:

.. code-block:: text

   AsyncEnvWorker
       |  env_channel
       v
   AsyncMultiStepRolloutWorker
       |  actor_channel
       v
   AsyncPPOEmbodiedFSDPActor
       |  weight sync
       v
   AsyncMultiStepRolloutWorker

There is also a reverse action path:

- rollout infers actions from environment observations,
- rollout sends actions back to env through ``rollout_channel``,
- env steps forward and pushes the next observations back into ``env_channel``.

Component responsibilities are:

- ``AsyncEnvWorker`` runs the long-lived environment interaction loop,
- ``AsyncMultiStepRolloutWorker`` performs policy inference, collects trajectories, and records policy versions,
- ``AsyncPPOEmbodiedRunner`` controls training cadence, weight synchronization, and metrics aggregation,
- ``AsyncPPOEmbodiedFSDPActor`` computes advantages, computes proximal logprobs, and performs PPO updates.

The important idea is not just "async function calls". The important idea is that env and rollout are long-running services rather than per-step jobs.


5. Execution Flow
-----------------

One Async PPO training update proceeds as follows:

1. The runner initializes workers and synchronizes actor weights to rollout.
2. The env worker calls ``bootstrap_step()`` to produce initial observations.
3. The rollout worker repeatedly executes ``generate_one_epoch()``:
   - read environment output from ``env_channel``,
   - infer actions using the current rollout weights,
   - record ``prev_logprobs``, ``prev_values``, and ``forward_inputs``,
   - attach the current policy ``versions`` to samples,
   - send actions back to env through ``rollout_channel``.
4. When one rollout epoch finishes, rollout splits trajectories and sends them to ``actor_channel``.
5. The actor receives trajectories and:
   - optionally recomputes ``proximal_logprobs``,
   - computes advantages and returns,
   - flattens and shuffles the ``[T, B, ...]`` batch,
   - trains using ``global_batch_size`` and ``micro_batch_size``.
6. After one actor update, the runner increments ``global_step`` and synchronizes the new weights to rollout.
7. Rollout continues sampling with the new version. If rollout gets too far ahead, stale-sample control will throttle it.

Async PPO overlaps sampling and training, but it is not unconstrained concurrency. Weight synchronization, version tracking, and sample admission are still explicitly controlled.


6. Versioning and Sample Staleness
----------------------------------

The hardest problem in asynchronous training is sample staleness. RLinf addresses it at two levels.

6.1 Sample version tagging
^^^^^^^^^^^^^^^^^^^^^^^^^^

During rollout generation, RLinf writes the current rollout policy version into ``versions`` for each step. This tells the actor exactly which policy produced the behavior data.

During training, the actor also tracks its own current version. This means each sample is associated with:

- a behavior policy version, which actually generated the sample,
- a proximal policy version, which acts as the PPO anchor for the current update,
- a current training version, which is the policy being optimized now.

This makes the age of the data explicit rather than implicit.

6.2 Rollout-side staleness throttling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rollout worker uses ``staleness_threshold`` to bound how far rollout is allowed to get ahead of actor.

If rollout has already produced too many episodes under older policies while actor has not caught up yet, rollout pauses inside ``wait_if_stale()`` until the version gap becomes acceptable again.

This mechanism serves two purposes:

- it prevents rollout from running arbitrarily far ahead and generating large volumes of stale samples,
- it exposes a direct tradeoff between system throughput and sample freshness.

In practice:

- smaller ``staleness_threshold`` usually improves stability,
- larger ``staleness_threshold`` usually improves throughput, but increases stale-sample risk.


7. Algorithm Mechanics
----------------------

7.1 Three policy views
^^^^^^^^^^^^^^^^^^^^^^

To understand Async PPO in RLinf, it is helpful to distinguish three policy views:

- behavior policy :math:`\pi_b`, which generated the sample,
- proximal policy :math:`\pi_p`, which acts as the PPO anchor for the current update,
- current policy :math:`\pi_\theta`, which is optimized by backpropagation.

In synchronous PPO, these are often close enough that the algorithm can be explained using only "old policy" and "new policy". In the asynchronous setting, the behavior policy may lag behind significantly, so RLinf models this difference explicitly.

7.2 Decoupled actor-critic loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RLinf uses ``decoupled_actor_critic`` for Async PPO. Instead of applying standard PPO clipping directly against the behavior policy, it decouples the proximal constraint from the behavior correction.

Define

.. math::

   r_{\mathrm{prox}} =
   \frac{\pi_{\theta}(a_t \mid s_t)}
        {\pi_p(a_t \mid s_t)},
   \qquad
   w_{\mathrm{behav}} =
   \frac{\pi_p(a_t \mid s_t)}
        {\pi_b(a_t \mid s_t)}.

The actor update can then be understood as PPO clipping around the proximal policy :math:`\pi_p`, while using :math:`w_{\mathrm{behav}}` to correct for distribution shift introduced by stale behavior data.

Operationally:

- PPO clipping controls how far the current policy moves from the proximal anchor,
- behavior weighting controls how much stale data can influence the update.

7.3 Dual clip and behavior filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RLinf adds two additional stabilizers:

- ``clip_ratio_c`` applies dual clipping to cap extreme advantage-driven gradients,
- ``behave_weight_threshold`` filters out samples with excessively large behavior weights.

If training becomes unstable, these controls and the staleness metrics should be checked before changing high-level learning settings.

7.4 GAE and the value head
^^^^^^^^^^^^^^^^^^^^^^^^^^

Advantages are still computed with standard GAE:

.. math::

   \hat{A}_t^{\mathrm{GAE}}
   =
   \sum_{l=0}^{\infty} (\gamma \lambda)^l
   \left(r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})\right).

This means Async PPO still requires value estimation. In practice, the configuration must include:

- ``algorithm.adv_type: gae``
- ``algorithm.loss_type: decoupled_actor_critic``
- ``actor.model.add_value_head: True``


8. Two Ways to Obtain Proximal Logprobs
---------------------------------------

``proximal_logprobs`` is a key quantity in Async PPO. RLinf supports two ways to obtain it:

1. Explicit recomputation

   When ``rollout.recompute_logprobs=True``, the actor explicitly recomputes proximal logprobs using the current actor weights before training.

2. Approximate interpolation

   When ``rollout.recompute_logprobs=False``, RLinf uses ``versions`` to interpolate between behavior logprobs and current logprobs, constructing an approximate proximal anchor.

The tradeoff is straightforward:

- explicit recomputation is more stable and should be the default choice,
- approximate interpolation is faster, but more sensitive to stale data.

If you are debugging instability, keep explicit recomputation enabled.


9. Configuration
----------------

The following is a minimal Async PPO configuration skeleton aligned with ``examples/embodiment/config/maniskill_async_ppo_openvla.yaml``:

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-3
       env: 0-1
       rollout: 2-3

   runner:
     task_type: embodied
     max_epochs: 1000
     val_check_interval: -1
     save_interval: 40

   algorithm:
     adv_type: gae
     loss_type: decoupled_actor_critic
     normalize_advantages: True
     staleness_threshold: 1
     behave_weight_threshold: 2.0
     clip_ratio_high: 0.3
     clip_ratio_low: 0.3
     clip_ratio_c: 3.0
     value_clip: 0.2
     gamma: 0.99
     gae_lambda: 0.95
     entropy_bonus: 0.0
     rollout_epoch: 1

   env:
     train:
       total_num_envs: 16
       max_episode_steps: 80
       max_steps_per_rollout_epoch: 80

   rollout:
     backend: huggingface
     recompute_logprobs: True
     pipeline_stage_num: 1

   actor:
     training_backend: fsdp
     micro_batch_size: 40
     global_batch_size: 320
     model:
       add_value_head: True

The most important Async PPO-specific parameters are:

- ``staleness_threshold``
  Bounds how far rollout is allowed to run ahead of actor.

- ``behave_weight_threshold``
  Caps how much stale samples can contribute through behavior weighting.

- ``rollout.recompute_logprobs``
  Controls whether proximal logprobs are recomputed explicitly.

- ``actor.micro_batch_size`` and ``actor.global_batch_size``
  Determine training memory pressure and throughput.

The actor training path also enforces strict batch constraints:

- ``global_batch_size`` must be divisible by ``micro_batch_size * actor_world_size``,
- the flattened rollout size must be divisible by the per-rank training batch size.

If these conditions are violated, training will fail with assertions.


10. Launching Async PPO
-----------------------

The recommended launch path is:

.. code-block:: bash

   bash examples/embodiment/run_async.sh maniskill_async_ppo_openvla LIBERO

The equivalent Python entrypoint is:

.. code-block:: bash

   python examples/embodiment/train_async.py \
     --config-path examples/embodiment/config \
     --config-name maniskill_async_ppo_openvla

Before running:

- in multi-node mode, Ray must already be started and the training script should run only on the head node,
- ``run_async.sh`` sets ``MUJOCO_GL=egl`` and ``PYOPENGL_PLATFORM=egl``,
- ``ROBOT_PLATFORM`` must match the actual robot platform, otherwise action dimensions and normalization may not align.


11. Monitoring
--------------

When debugging Async PPO, reward alone is not enough. At minimum, monitor the following metrics.

System-side metrics:

- ``time/env/*`` to detect environment bottlenecks,
- ``time/rollout/*`` to detect inference bottlenecks,
- ``time/actor_training`` to detect training bottlenecks.

Policy-update metrics:

- ``train/actor/proximal_approx_kl``,
- ``train/actor/clip_fraction``,
- ``train/actor/dual_clip_fraction``.

Staleness-related metrics:

- ``train/actor/behav_approx_kl``,
- ``train/actor/behav_clip_fraction``,
- ``train/actor/average_version``,
- ``train/actor/current_version``.

If throughput is high but ``behav_approx_kl`` and ``behav_clip_fraction`` remain elevated, rollout is likely getting too far ahead and stale samples are starting to dominate the update.


12. Tuning Guidelines
---------------------

12.1 Optimize for stability first
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recommended starting point:

- ``staleness_threshold: 1``
- ``behave_weight_threshold: 2.0``
- ``rollout.recompute_logprobs: True``

Only after reward, KL, and stale-sample metrics look healthy should you push throughput harder.

12.2 If throughput is too low
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recommended order of changes:

1. improve ``cluster.component_placement`` to reduce contention between env, rollout, and actor,
2. increase ``env.train.total_num_envs``,
3. increase ``staleness_threshold`` modestly, for example from ``1`` to ``2``,
4. only then consider disabling explicit proximal logprob recomputation.

12.3 If training is unstable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recommended order of changes:

1. reduce ``staleness_threshold``,
2. reduce ``behave_weight_threshold``,
3. lower the actor learning rate,
4. reduce ``clip_ratio_high`` and ``clip_ratio_low``,
5. reduce ``update_epoch`` or training batch size.

12.4 If you hit OOM
^^^^^^^^^^^^^^^^^^^

Recommended order of changes:

1. reduce ``actor.micro_batch_size``,
2. keep or enable ``gradient_checkpointing``,
3. reduce ``env.train.total_num_envs``,
4. then consider reducing model or input scale.


13. Async PPO vs. Synchronous PPO
---------------------------------

The main difference can be summarized simply:

- synchronous PPO prioritizes sample freshness,
- Async PPO prioritizes system throughput while explicitly controlling stale-sample impact.

So the main change is not the high-level PPO objective itself. The main change is the execution model of the training system.

If your bottleneck is system-side, Async PPO is often worth the added complexity. If your main difficulty is algorithmic convergence, synchronous PPO is usually easier to tune first.
