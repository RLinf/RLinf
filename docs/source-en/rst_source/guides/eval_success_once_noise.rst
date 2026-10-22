Understanding Fluctuations in ``success_once``
=============================================

This guide explains why embodied evaluation metrics such as
``eval/env/success_once`` can oscillate between validation steps (for
example, swinging between 0.5 and 0.6 during PPO fine-tuning of pi05 on
RoboCasa) and how to tell statistical noise apart from real training
instability.

How ``success_once`` Is Computed
--------------------------------

Per-episode ``success_once`` is tracked by the environment wrapper
(see ``rlinf/envs/sim/robocasa/robocasa_env.py``): it records whether an
episode succeeded **at least once** before termination.

At validation time, ``rlinf.utils.metric_utils.compute_evaluate_metrics``
aggregates every episode from all evaluation rollout epochs and rollout
stages into a **single plain mean** of the binary outcome. That mean is also
reported alongside ``eval/num_trajectories``, the total number of episodes it
was computed over.

Why the Metric Oscillates
-------------------------

A mean of a binary outcome is a *binomial proportion*. Even with a perfectly
fixed policy and fixed environment seeds, re-evaluating a stochastic policy
(autoregressive action sampling) produces different episodes, and the
estimate carries sampling noise:

.. math::

   \mathrm{std}(\hat{p}) = \sqrt{\frac{p\,(1-p)}{n}}

where :math:`p` is the true success rate and :math:`n` is the number of
evaluation episodes. With :math:`p \approx 0.55`:

=====================================  =========================  ==========================
Episodes per eval (``n``)              Std of the estimate        Typical 95% noise band
=====================================  =========================  ==========================
16 (e.g. 4 envs × 4 rollout epochs)    ±0.124                     roughly 0.30 – 0.80
100                                    ±0.050                     roughly 0.45 – 0.65
1000                                   ±0.016                     roughly 0.52 – 0.58
=====================================  =========================  ==========================

So an ``env/success_once`` curve that oscillates between 0.5 and 0.6 with a
small evaluation batch is **entirely within the expected sampling noise** —
it does not by itself indicate an unstable training run. A flat-or-noisy
curve over ~20 PPO steps more likely means the evaluation is under-powered,
or that the policy has not yet improved beyond the noise floor.

Environment seeds for RoboCasa are already fixed deterministically
(``seed = cfg.seed + global_env_id``, see
``rlinf/envs/sim/robocasa/robocasa_env.py``), so the scene/reset layout does
not add extra variance; the remaining variance comes from the small episode
count and the stochasticity of the policy itself.

Diagnosing With the Built-in Tool
---------------------------------

RLinf ships ``tools/diagnose_success_once_noise.py``:

.. code-block:: bash

   # Noise band of a single evaluation point:
   # what values are within noise of 0.55 over 16 episodes?
   python tools/diagnose_success_once_noise.py single --rate 0.55 --episodes 16

   # Are two consecutive eval points (0.50 then 0.60) meaningfully different?
   python tools/diagnose_success_once_noise.py compare \
       --rate-a 0.50 --episodes-a 16 --rate-b 0.60 --episodes-b 16

   # How many episodes are needed for a ±0.03 confidence interval?
   python tools/diagnose_success_once_noise.py plan --target-width 0.06 --rate 0.55

The ``compare`` subcommand runs a two-proportion z-test. Only when it
reports a statistically significant difference should you treat a drop in
``success_once`` as evidence of training instability.

Check ``eval/num_trajectories`` in your logs: it tells you the ``n`` that
applies to your runs.

Recommended Config Changes
--------------------------

1. **Increase the evaluation batch.** Raise ``env.eval.total_num_envs``
   and/or ``env.eval.rollout_epoch`` so their product reaches the episode
   count reported by the ``plan`` subcommand (for a ±0.03 interval around
   p = 0.55, roughly 1000 episodes; ~100 episodes already narrows the noise
   band to about ±0.05).
2. **Keep seeds fixed.** Leave ``env.eval.seed`` fixed across validation
   steps so that differences between steps are attributable to the policy,
   not to different scenes.
3. **Smooth with judgment, not with hiding.** If you plot a moving average
   over validation steps, report the window size and remember that a moving
   average does not reduce the noise of a single point.
4. **Compare with the z-test, not by eye.** Before reacting to a dip, run
   the ``compare`` subcommand on the two neighboring eval points.
