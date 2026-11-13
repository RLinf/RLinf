Training Metrics
================

RLinf reports metrics through the :doc:`MetricLogger <../guides/logger>` under a few namespaces —
``train/``, ``rollout/``, ``env/``, and ``time/``. This page defines them once; example
pages link here instead of repeating the definitions.

.. tip::

   For embodied tasks, compare ``env/success_once`` with ``env/return_mean``.
   If each episode returns 1 on success and 0 on failure, both metrics equal
   the episode success rate.
   ``rollout/returns_mean`` is a value-training target, not episode return.

Training metrics — ``train/``
-----------------------------

Policy- and value-optimization statistics, logged every actor update.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``train/actor/approx_kl``
     - Approximate KL divergence between the old and new policies.
   * - ``train/actor/clip_fraction``
     - Fraction of updates where the probability ratio was clipped.
   * - ``train/actor/dual_clip_fraction``
     - Fraction of valid policy-loss elements bounded by Dual-Clip. Zero when Dual-Clip is disabled; padding is excluded.
   * - ``train/actor/clipped_ratio``
     - Mean of the clipped probability ratios.
   * - ``train/actor/grad_norm``
     - Gradient norm of the actor.
   * - ``train/actor/lr``
     - Current learning rate.
   * - ``train/actor/policy_loss``
     - PPO / GRPO policy loss.
   * - ``train/critic/value_loss``
     - Value-function loss.
   * - ``train/critic/value_clip_ratio``
     - Fraction of value targets whose update was clipped.
   * - ``train/critic/explained_variance``
     - Explained variance of the value predictions (closer to 1 is better).
   * - ``train/entropy_loss``
     - Policy entropy.
   * - ``train/loss``
     - Total training loss (actor + critic + entropy regularization).

Rollout metrics — ``rollout/``
------------------------------

Statistics over valid rollout elements, excluding padding through the loss mask.
Means and population standard deviations weight every valid element equally
across ranks, including when ranks have different valid counts.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``rollout/advantages_max``
     - Maximum advantage in the batch.
   * - ``rollout/advantages_mean``
     - Mean advantage in the batch.
   * - ``rollout/advantages_std``
     - Population standard deviation of the training advantages, after normalization if enabled.
   * - ``rollout/advantages_min``
     - Minimum advantage in the batch.
   * - ``rollout/rewards``
     - Legacy mean of the stored reward elements, before chunk-level summation.
   * - ``rollout/rewards_mean``, ``rollout/rewards_std``
     - Reward mean and population standard deviation. With chunk-level rewards, action rewards are summed within each chunk first.
   * - ``rollout/returns_mean``, ``rollout/returns_std``
     - Mean and population standard deviation of value-training targets. For GAE, these include discounted rewards and bootstrap values.
   * - ``rollout/returns_min``, ``rollout/returns_max``
     - Minimum and maximum value-training target.
   * - ``rollout/values_mean``, ``rollout/values_std``
     - Mean and population standard deviation of GAE value predictions at sampled transitions. The final bootstrap value is excluded.
   * - ``rollout/advantages_raw_mean``, ``rollout/advantages_raw_std``
     - Mean and population standard deviation of GAE targets minus value predictions, before advantage normalization. Logged only when return normalization is disabled.
   * - ``rollout/loss_mask_fraction``
     - Fraction of rollout elements retained by the loss mask.

Environment metrics — ``env/``
------------------------------

Task-level signals from the simulator.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``env/success_once``
     - **Recommended.** Unnormalized episodic success rate — the truest measure of task performance.
   * - ``env/episode_len``
     - Number of environment steps elapsed in the episode.
   * - ``env/return``
     - Mean completed-episode return, summed from actual environment rewards without value bootstrap.
   * - ``env/return_mean``, ``env/return_std``
     - Explicit episode-return mean and population standard deviation. The mean aliases ``env/return``; completed episodes are weighted equally across ranks.
   * - ``env/reward``
     - Step-level reward (``0`` on intermediate steps, ``1`` on success). The logged value is normalized by episode length, which makes it hard to read as real performance.

Independent evaluation uses the same episode metrics under ``eval/`` instead
of ``env/``. For binary episode returns with success rate :math:`p`, the mean
is :math:`p` and the population standard deviation is :math:`\sqrt{p(1-p)}`.

Dual-Clip configuration
-----------------------

These options apply to embodied PPO actor losses.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Type
     - Default and meaning
   * - ``algorithm.use_dual_clip``
     - Boolean
     - If omitted, uses the actor's default. Set to ``false`` to disable Dual-Clip, as in the Task22 recipe.
   * - ``algorithm.clip_ratio_c``
     - Float or null
     - Defaults to 3.0 when enabled. A finite value greater than 1 bounds the policy loss; null disables the bound.

See also the :doc:`Logger <../guides/logger>` tutorial for choosing backends (TensorBoard,
Weights & Biases, SwanLab) and configuring ``runner.logger``.
