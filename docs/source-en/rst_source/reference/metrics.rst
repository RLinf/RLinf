Training Metrics
================

RLinf logs scalar metrics through :doc:`MetricLogger <../guides/logger>` under five core namespaces: ``train/``, ``rollout/``, ``env/``, ``eval/``, and ``time/``. This page defines each namespace once and lists what to watch when debugging a run.

.. tip::

   For embodied tasks, track **``env/success_once``** (training) and **``eval/success_once``** (validation). Under sparse rewards, per-step ``env/reward`` is normalized by episode length and is less direct than episodic success rate.

``train/``
----------

Policy and value optimization statistics logged every actor update (on-policy PPO / GRPO).

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Metric
     - Meaning
     - What to watch
   * - ``train/actor/approx_kl``
     - Approximate KL divergence between old and updated policy.
     - Spikes above ``0.05``–``0.1`` indicate overly aggressive updates; lower learning rate or clip bounds.
   * - ``train/actor/clip_fraction``
     - Fraction of tokens or actions whose importance ratio was clipped.
     - Sustained values above ``0.3`` mean the policy is moving too fast per epoch.
   * - ``train/actor/clipped_ratio``
     - Mean clipped probability ratio.
     - Should remain close to ``1.0``.
   * - ``train/actor/grad_norm``
     - Actor gradient norm before clipping.
     - Repeated spikes or ``NaN`` point to numerical instability or bad reward scaling.
   * - ``train/actor/lr``
     - Current actor learning rate.
     - Confirm warmup and decay schedule match the config.
   * - ``train/actor/policy_loss``
     - PPO / GRPO policy gradient objective.
     - Can oscillate; judge progress by reward and ``env/success_once`` rather than loss alone.
   * - ``train/critic/value_loss``
     - Huber value-function loss.
     - Should decrease steadily early in training; flat or exploding loss hurts advantage quality.
   * - ``train/critic/value_clip_ratio``
     - Fraction of value predictions clipped against previous estimates.
     - Very high values suggest ``value_clip`` is too tight or critic learning rate is too high.
   * - ``train/critic/explained_variance``
     - Fraction of return variance explained by the critic (closer to ``1.0`` is better).
     - Values near or below ``0`` mean the critic is not fitting returns yet.
   * - ``train/entropy_loss``
     - Policy entropy (when entropy computation is enabled).
     - Rapid drop toward ``0`` signals entropy collapse and premature convergence.
   * - ``train/loss``
     - Total combined loss (actor + critic + entropy regularization).
     - Watch for ``NaN`` / ``Inf``; interpret individual components above.

``rollout/``
------------

Batch-level reward and advantage statistics collected from rollout before optimization.

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Metric
     - Meaning
     - What to watch
   * - ``rollout/rewards``
     - Mean reward across rollout chunks or completions.
     - Should trend upward over steps; flat zero means the policy never reaches reward signal.
   * - ``rollout/advantages_mean``
     - Mean advantage in the batch.
     - Close to ``0.0`` when advantage normalization is enabled.
   * - ``rollout/advantages_max``
     - Maximum advantage in the batch.
     - Extreme outliers can destabilize updates if normalization is disabled.
   * - ``rollout/advantages_min``
     - Minimum advantage in the batch.
     - Check symmetry with ``advantages_max`` to verify reward distribution.
   * - ``rollout/loss_mask_fraction``
     - Fraction of active (unmasked) tokens or steps in the batch.
     - Near ``0`` indicates truncated or degenerate rollouts.

``env/``
--------

Task-level signals reported by training environment workers.

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Metric
     - Meaning
     - What to watch
   * - ``env/success_once``
     - **Primary embodied signal.** Unnormalized episodic success rate.
     - Should trend steadily upward; compare against ``eval/success_once`` to check generalization.
   * - ``env/episode_len``
     - Mean number of environment steps per episode.
     - Often decreases as the policy solves tasks faster.
   * - ``env/return``
     - Sum of rewards over an episode.
     - Near zero under sparse rewards until episodes succeed.
   * - ``env/reward``
     - Step-level reward averaged over episode length.
     - Diluted by episode length; use ``env/success_once`` for sparse-reward tasks.

``eval/``
---------

Periodic validation metrics logged whenever ``runner.val_check_interval`` triggers evaluation (or during standalone evaluation).

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Metric
     - Meaning
     - What to watch
   * - ``eval/success_once``
     - Episodic success rate on the evaluation environment or task split.
     - Primary model-selection metric; gap vs. ``env/success_once`` shows overfitting or exploration noise.
   * - ``eval/return``
     - Mean episodic return during evaluation.
     - Tracks dense-reward task progress cleanly without exploration noise.
   * - ``eval/episode_len``
     - Mean evaluation episode length.
     - Shorter length alongside higher success indicates cleaner execution.
   * - ``eval/num_trajectories``
     - Total trajectories aggregated across evaluation workers.
     - Confirm it matches expected ``eval`` worker and episode counts.

``time/``
---------

Wall-clock durations (in seconds) collected by runner and worker timers to diagnose system bottlenecks.

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Metric
     - Meaning
     - What to watch
   * - ``time/step``
     - Total wall-clock duration of one global training step.
     - End-to-end throughput indicator; compare sum of stages against ``time/step`` to verify pipeline overlap.
   * - ``time/sync_weights`` / ``time/update_rollout_weights``
     - Weight synchronization from actor to rollout workers.
     - High cost suggests suboptimal placement or overly frequent syncs.
   * - ``time/generate_rollouts``
     - Combined rollout generation and environment interaction phase.
     - Dominant share means env stepping or rollout inference is the bottleneck.
   * - ``time/cal_adv_and_returns``
     - Advantage and return computation duration.
     - Usually small; spikes indicate host-device sync overhead.
   * - ``time/actor_training``
     - Actor forward, backward, and optimizer update phase.
     - High share suggests increasing actor GPUs or tuning ``micro_batch_size``.
   * - ``time/eval``
     - Validation round duration when evaluation runs.
     - Adjust ``val_check_interval`` if evaluation dominates wall time.
   * - ``time/env/*``, ``time/rollout/*``, ``time/actor/*``, ``time/reward/*``
     - Per-component worker breakdown aggregated across ranks (and per rank when ``runner.per_worker_log: True``).
     - Pinpoint straggler ranks or imbalanced GPU placement.

Off-Policy and SAC
------------------

Logged by ``EmbodiedSACFSDPPolicy`` and ``AsyncEmbodiedSACFSDPPolicy`` (including CrossQ and RLPD):

.. list-table::
   :header-rows: 1
   :widths: 30 36 34

   * - Metric
     - Meaning
     - What to watch
   * - ``train/sac/critic_loss``
     - Bellman regression loss over Q-networks.
     - Diverging critic loss indicates Q-value overestimation; check ``tau``, ``gamma``, or LayerNorm/BRN.
   * - ``train/sac/actor_loss``
     - Maximum-entropy policy loss.
     - Track alongside ``train/sac/critic_loss`` and ``env/success_once``.
   * - ``train/sac/alpha`` / ``train/sac/alpha_loss``
     - Learned entropy temperature and its optimization loss.
     - If ``alpha`` collapses to ``0`` or explodes, tune ``target_entropy`` or ``initial_alpha``.
   * - ``train/actor/entropy``
     - Policy action entropy.
     - Should settle near ``-target_entropy`` under automatic tuning.
   * - ``train/critic/lr`` / ``train/critic/grad_norm``
     - Critic optimizer learning rate and gradient norm.
     - Watch for gradient spikes during high-UTD updates.
   * - ``train/replay_channel_qsize``
     - Queued trajectories waiting on the async actor channel.
     - Growing queue means actor updates lag behind rollout collection.

Async PPO Staleness
-------------------

Logged by ``compute_decoupled_ppo_actor_loss`` during :doc:`Async PPO <algorithms/async_ppo>` runs:

.. list-table::
   :header-rows: 1
   :widths: 30 36 34

   * - Metric
     - Meaning
     - What to watch
   * - ``train/actor/proximal_approx_kl``
     - Approximate KL between current policy and proximal anchor.
     - Primary optimization-step size check (analogous to ``approx_kl``).
   * - ``train/actor/behav_approx_kl``
     - Approximate KL between proximal anchor and behavior policy that collected the batch.
     - Sustained high values mean rollout data is too stale; lower ``staleness_threshold``.
   * - ``train/actor/behav_clip_fraction``
     - Fraction of samples filtered by ``behave_weight_threshold``.
     - High drop rates mean stale samples are being discarded instead of trained on.
   * - ``train/actor/dual_clip_fraction``
     - Fraction of entries hitting the dual-clip bound ``clip_ratio_c``.
     - Persistent high fraction signals overly large policy shifts.
   * - ``train/actor/average_version`` / ``train/actor/current_version``
     - Mean behavior-policy version of the training batch vs. current actor version.
     - Gap ``current_version - average_version`` directly measures policy staleness in steps.

Reasoning and Agentic
---------------------

Logged by reasoning and agentic runners (``reasoning_runner.py``, ``agent_runner.py``, ``async_agent_runner.py``):

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Metric
     - Meaning
     - What to watch
   * - ``actor/training/*`` / ``critic/training/*``
     - Per-minibatch actor and critic training metrics (losses, KL, clip fraction, grad norm).
     - Same stability checks as ``train/actor/*`` and ``train/critic/*`` above.
   * - ``rollout/equal_reward_group_ratio`` / ``rollout/zero_reward_group_ratio``
     - Fraction of prompt groups with identical or all-zero rewards.
     - High ratios mean GRPO gets little learning signal; check sampling temperature or prompt difficulty.
   * - ``rollout/data_staleness/mean`` / ``rollout/data_staleness/max``
     - Version lag of completed groups in asynchronous agentic rollouts.
     - Keep bounded to avoid off-policy drift in async agent training.
   * - ``rollout/buffer/size`` / ``rollout/inflight_groups``
     - Pending groups in the async rollout buffer and active in-flight prompt groups.
     - Diagnose rollout-to-actor backpressure or under-utilized rollout workers.
   * - ``time/rollout``, ``time/reward``, ``time/actor/training``, ``time/actor/inference``, ``time/critic/training``
     - Per-stage wall-clock times for LLM generation, reward scoring, reference/old logprob inference, and training.
     - Guide GPU split between rollout, inference, and training workers.

See :doc:`Logger <../guides/logger>` to configure logging backends (TensorBoard, Weights & Biases, SwanLab) and ``runner.logger``.
