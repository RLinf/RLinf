Training Metrics
================

RLinf reports metrics through the :doc:`MetricLogger <../guides/logger>` under a few namespaces —
``train/``, ``rollout/``, ``env/``, and ``time/``. This page defines them once; example
pages link here instead of repeating the definitions.

.. tip::

   For embodied tasks the single most useful signal is **``env/success_once``** — the
   unnormalized episodic success rate. Most other ``env/*`` values are hard to read
   directly under sparse rewards (see below).

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

Statistics of the advantages and rewards collected during rollout.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``rollout/advantages_max``
     - Maximum advantage in the batch.
   * - ``rollout/advantages_mean``
     - Mean advantage in the batch.
   * - ``rollout/advantages_min``
     - Minimum advantage in the batch.
   * - ``rollout/rewards``
     - Reward of a rollout chunk.

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
     - Episode return. Under sparse rewards this is near-zero until the terminal success step, so it is not very informative during training.
   * - ``env/reward``
     - Step-level reward (``0`` on intermediate steps, ``1`` on success). The logged value is normalized by episode length, which makes it hard to read as real performance.

See also the :doc:`Logger <../guides/logger>` tutorial for choosing backends (TensorBoard,
Weights & Biases, SwanLab) and configuring ``runner.logger``.
