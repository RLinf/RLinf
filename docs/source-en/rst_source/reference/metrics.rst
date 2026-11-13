Training Metrics
================

Metrics are grouped by training, rollout and completed episodes. Standard
deviations use the population definition. Configure logging backends with
:doc:`Logger <../guides/logger>`.

Training Metrics
----------------

The ``train/`` namespace reports policy and value optimization statistics.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``train/actor/approx_kl``
     - Approximate KL divergence between the old and new policies.
   * - ``train/actor/clip_fraction``
     - Fraction of valid elements affected by PPO clipping.
   * - ``train/actor/dual_clip_fraction``
     - Fraction of valid elements affected by Dual-Clip; zero when disabled.
   * - ``train/actor/clipped_ratio``
     - Mean clipped probability ratio.
   * - ``train/actor/grad_norm``
     - Actor gradient norm.
   * - ``train/actor/lr``
     - Learning rate.
   * - ``train/actor/policy_loss``
     - Policy loss.
   * - ``train/critic/value_loss``
     - Value-function loss.
   * - ``train/critic/value_clip_ratio``
     - Fraction of value updates affected by clipping.
   * - ``train/critic/explained_variance``
     - Explained variance of value predictions.
   * - ``train/entropy_loss``
     - Policy entropy.
   * - ``train/loss``
     - Total training loss.

Rollout Metrics
---------------

The ``rollout/`` namespace summarizes valid training samples. GAE returns are
value-function training targets; episode returns are listed under ``env/`` below.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``rollout/advantages_max``
     - Maximum training advantage.
   * - ``rollout/advantages_mean``
     - Mean training advantage.
   * - ``rollout/advantages_min``
     - Minimum training advantage.
   * - ``rollout/advantages_std``
     - Standard deviation of training advantages.
   * - ``rollout/rewards``
     - Mean reward per action.
   * - ``rollout/rewards_mean``, ``rollout/rewards_std``
     - Reward mean and standard deviation at the configured reward granularity.
   * - ``rollout/returns_mean``, ``rollout/returns_std``
     - Mean and standard deviation of value-function training targets.
   * - ``rollout/returns_min``, ``rollout/returns_max``
     - Minimum and maximum value-function training targets.
   * - ``rollout/values_mean``, ``rollout/values_std``
     - Mean and standard deviation of value predictions for sampled states.
   * - ``rollout/advantages_raw_mean``, ``rollout/advantages_raw_std``
     - Mean and standard deviation of unnormalized GAE advantages, available when return normalization is disabled.
   * - ``rollout/loss_mask_fraction``
     - Fraction of rollout elements included in training.

Episode Metrics
---------------

Environments report completed-episode statistics under ``env/``. Independent
evaluation reports the same keys under ``eval/``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``env/success_once``
     - Fraction of episodes in which success occurred.
   * - ``env/episode_len``
     - Mean episode length in environment steps.
   * - ``env/return``, ``env/return_mean``
     - Mean episode return.
   * - ``env/return_std``
     - Standard deviation of episode returns.
   * - ``env/reward``
     - Mean environment reward per step.

Dual-Clip Configuration
-----------------------

These options control Dual-Clip in embodied PPO.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Type
     - Default and Meaning
   * - ``algorithm.use_dual_clip``
     - Boolean
     - Uses the actor's default when omitted. ``false`` disables Dual-Clip.
   * - ``algorithm.clip_ratio_c``
     - Float or null
     - Defaults to 3.0 when enabled. Accepts a finite value greater than 1; null disables the bound.
