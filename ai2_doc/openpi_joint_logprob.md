# OpenPI `joint_logprob`

This note explains what `actor.model.openpi.joint_logprob` means in RLinf's
OpenPI policy, with the math and the code path that implements it.

Relevant code:

- [`openpi_action_model.py`](/weka/oe-training-default/shiruic/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py)
- [`yam_ppo_openpi.yaml`](/weka/oe-training-default/shiruic/RLinf/examples/embodiment/config/yam_ppo_openpi.yaml)

## Intuition

OpenPI here is a denoising policy. It does not predict the final action in one
shot. Instead, it starts from noise and repeatedly denoises:

```text
x_T -> x_{T-1} -> ... -> x_1 -> x_0
```

At each denoising step `t`, the policy predicts a Gaussian transition from
`x_t` to `x_{t-1}` conditioned on the observation `o`:

```math
x_{t-1} \sim \pi_\theta(\cdot \mid x_t, o, t)
= \mathcal{N}(\mu_\theta(x_t, o, t), \sigma_\theta(x_t, o, t)^2)
```

Here:

- `o` is the policy observation prefix: images, language, and optional state
- `\mu_\theta` is the predicted denoised mean
- `\sigma_\theta` is the predicted standard deviation

In RLinf, this log-probability is computed in `get_logprob_norm()`:

```math
\log \pi_\theta(x_{t-1} \mid x_t, o, t)
= -\log \sigma_\theta
- \frac{1}{2}\log(2\pi)
- \frac{1}{2}\left(\frac{x_{t-1} - \mu_\theta}{\sigma_\theta}\right)^2
```

## `joint_logprob: false`

When `joint_logprob` is `false`, RLinf samples one denoising step `k` and
trains PPO using only that one transition.

The policy score is effectively:

```math
\ell_{\text{single}}(\theta)
= \log \pi_\theta(x_{k-1} \mid x_k, o, k)
```

and PPO uses that single-step log-probability in the usual surrogate loss:

```math
r(\theta) = \exp(\ell_{\text{single}}(\theta) - \ell_{\text{single}}(\theta_{\text{old}}))
```

```math
\mathcal{L}_{\text{PPO}}
= -\min\left(r(\theta) A,\ \mathrm{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A\right)
```

Operationally:

- one denoising step is chosen at random
- only one transition contributes policy gradient
- the backward graph is smaller and usually more stable

This behavior comes from:

- `sample_actions()`: one random denoise index is chosen
- `get_log_prob_value()`: `num_steps = 1` when `joint_logprob` is false

## `joint_logprob: true`

When `joint_logprob` is `true`, RLinf includes all denoising transitions in the
policy score, plus the initial noise term.

Conceptually, the trajectory log-probability is:

```math
\log \pi_\theta(x_{0:T} \mid o)
\approx \log p(x_T) + \sum_{t=1}^{T} \log \pi_\theta(x_{t-1} \mid x_t, o, t)
```

In the current RLinf implementation, the per-step log-probabilities are
collected and then averaged before PPO uses them:

```math
\bar{\ell}_\theta
= \frac{1}{T+1}
\left(
\log p(x_T) + \sum_{t=1}^{T} \log \pi_\theta(x_{t-1} \mid x_t, o, t)
\right)
```

Then PPO treats `\bar{\ell}_\theta` as the policy log-probability:

```math
r(\theta) = \exp(\bar{\ell}_\theta - \bar{\ell}_{\theta_{\text{old}}})
```

This means every denoising step contributes gradient, not just one sampled step.

## What Changes in Practice

`joint_logprob: true` gives denser supervision:

- every denoising step contributes to the update
- the model gets signal from the full denoising trajectory
- credit assignment can be better across the whole diffusion chain

But it also makes training more aggressive:

- gradients flow through more denoising steps
- the same PPO update touches more transition terms
- if `\sigma_\theta` becomes small, the term
  `((x_{t-1} - \mu_\theta) / \sigma_\theta)^2` can become very large

That is especially important for `flow_noise`, where `\sigma_\theta` is learned
by a network instead of being fixed by a simple schedule.

## Why It Can Cause NaN Gradients

Even if the forward values are finite, the backward pass can still explode.

The main reason is the Gaussian log-prob formula:

```math
\log \pi_\theta
= -\log \sigma_\theta
- \frac{1}{2}\left(\frac{x_{t-1} - \mu_\theta}{\sigma_\theta}\right)^2
+ \text{const}
```

If `\sigma_\theta` gets too small, both of these terms become dangerous:

- `-\log \sigma_\theta`
- `((x_{t-1} - \mu_\theta) / \sigma_\theta)^2`

With `joint_logprob: true`, this happens across all denoising steps instead of
just one sampled step, so the total backward signal is usually much larger.

That is why `joint_logprob: true` is often less stable than `false`, especially
with:

- `noise_method: flow_noise`
- `train_expert_only: true`
- large denoising depth

## Code Mapping

The relevant implementation points are:

- `sample_actions()`
  - if `joint_logprob` is `true`, it uses all denoise steps
  - otherwise it samples one denoise step
- `get_log_prob_value()`
  - if `joint_logprob` is `true`, it loops over all denoise steps
  - otherwise it loops once
- `get_logprob_norm()`
  - computes the Gaussian transition log-probability

In your YAM config, this flag lives here:

```yaml
actor:
  model:
    openpi:
      joint_logprob: true
```

## Practical Rule of Thumb

Use `joint_logprob: true` when you want stronger full-trajectory denoising
supervision and the training is numerically stable.

Use `joint_logprob: false` when:

- training is unstable
- gradients are exploding
- you are debugging NaN issues
- you are combining OpenPI with `flow_noise`

For NaN debugging, `joint_logprob: false` is a very reasonable first switch to
test.
