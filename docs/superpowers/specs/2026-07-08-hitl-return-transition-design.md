# HITL Return Transition Design

Date: 2026-07-08

## Goal

Add a generic opt-in HITL-aware return transition option for RECAP return
computation.

The current HITL-aware return logic hard-splits successful human-in-the-loop
episodes at the first teleoperation frame. Frames before the intervention are
assigned failed-prefix returns, and frames from the intervention onward are
assigned successful-suffix returns. This creates an abrupt value target jump at
the intervention boundary. For ALOHA sandwich episode 025, the normalized target
changes from about `-0.083` at frame 1757 to about `-0.199` at frame 1758.

The desired behavior is to keep the semantic split, but smooth the final
pre-intervention value targets over a configurable transition window.

## Current Context

The relevant pipeline is:

- `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`
  writes the returns sidecar.
- `rlinf/data/datasets/recap/value_dataset.py` reads that sidecar and uses the
  normalized `return` values as value SFT targets.
- `examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py`
  reads the same returns sidecar as `true_return` when computing continuous
  advantages.

The existing `compute_hitl_aware_returns_for_episode()` behavior is intentionally
hard-split:

- failed episodes are unchanged failed episodes, even with teleop frames;
- successful episodes without teleop are unchanged successful episodes;
- successful HITL episodes split at the first teleop frame;
- the autonomous prefix receives failed returns;
- the teleop suffix receives successful returns.

Existing unit tests encode this hard-split behavior. The transition feature
must preserve those tests when the new configuration is unset or zero.

## Non-Goals

- Do not change return computation for non-HITL episodes.
- Do not change failed HITL episodes.
- Do not change `reward` or `done` semantics.
- Do not smooth value predictions after training or only in plots.
- Do not make this ALOHA-specific.
- Do not change advantage boolean teleop override behavior.

## Behavior

For successful HITL episodes with `hitl_aware_returns: true` and a positive
transition window, the autonomous prefix still represents failure and the
teleop suffix still represents success. The only changed values are the final
pre-teleop return targets.

If `split` is the first teleop frame and the resolved transition size is `W`,
then frames `[max(0, split - W), split)` receive smoothed return targets. Frames
before that window keep the original failed-prefix returns. Frames from `split`
onward keep the original successful-suffix returns.

For ALOHA sandwich, setting `hitl_transition_chunks: 2` with
`action_horizon: 10` produces a 20-frame transition window before the first
teleop frame.

## Configuration

Add generic optional fields under `data` for return computation:

```yaml
data:
  hitl_aware_returns: true
  hitl_transition_chunks: 2
  hitl_transition_steps: null
```

Resolution rules:

- Default behavior remains unchanged. If both transition fields are unset or
  zero, the current hard split is preserved.
- `hitl_transition_steps` takes precedence when set.
- Otherwise, resolved steps are
  `hitl_transition_chunks * action_horizon`.
- If `hitl_transition_chunks > 0` and neither `hitl_transition_steps` nor
  `action_horizon` is available, configuration validation should fail with a
  clear message.
- The resolved window length is clamped to the available pre-teleop prefix.
- ALOHA sandwich should set `hitl_transition_chunks: 2`.

## Data Flow

The change belongs in return sidecar construction, not in value SFT or plotting.
That keeps value SFT and advantage computation aligned on the same target
definition.

1. `compute_returns.py` reads `is_success`, optional `teleop_mask`, `gamma`,
   `failure_reward`, `hitl_aware_returns`, and the transition config.
2. Normal episodes, failed episodes, no-teleop successes, and `split == 0`
   remain unchanged.
3. Successful HITL episodes with a positive transition window compute the
   existing failed prefix and successful suffix returns.
4. Only the final pre-teleop prefix return targets are replaced by a linear
   ramp.
5. The returns sidecar stores the smoothed `return` values.
6. Value SFT trains on those returns.
7. Advantage computation uses those same returns for `true_return`.

`reward` and `done` remain unchanged from current HITL-aware split behavior.
`done=True` remains at `split - 1`, and the suffix still starts at `split`.

## Ramp Formula

For a transition window of `W` frames before `split`:

- `start = max(0, split - W)`
- `W_eff = split - start`
- `source = prefix_returns[start:split]`
- `target_end = suffix_returns[0]`
- `alpha` increases linearly from `1 / W_eff` to `1.0`

For each transition frame:

```text
smoothed[t] = (1 - alpha[t]) * source[t] + alpha[t] * target_end
```

This makes the last frame before teleop closest to the first teleop frame's
successful-suffix target. Earlier transition frames remain closer to the
failed-prefix target.

## Edge Cases

- `W <= 0`: preserve current hard split.
- `split == 0`: preserve standard successful returns.
- `W > split`: clamp the transition window to `split`.
- Failed HITL episodes: preserve standard failed returns.
- Successful episodes without teleop: preserve standard successful returns.
- Non-ALOHA datasets may opt in if they provide `teleop_mask`.

## Testing

Unit tests should cover:

- Current hard-split behavior remains unchanged when transition config is unset
  or zero.
- Successful HITL episode with `hitl_transition_steps=2` ramps only the two
  frames immediately before first teleop.
- `hitl_transition_chunks=2` with `action_horizon=10` resolves to 20 steps.
- The transition window clamps when the prefix is shorter than the requested
  window.
- `reward` and `done` arrays are unchanged by smoothing.
- Failed HITL episodes and non-HITL successful episodes remain unchanged.
- ALOHA sandwich return config loads with `hitl_transition_chunks: 2`.

Verification should include:

- Focused unit tests for `compute_returns.py`.
- A config smoke test for
  `examples/offline_rl/config/aloha_sandwich_recap_compute_returns.yaml`.
- After recomputing ALOHA sandwich returns, a data-level sanity check that
  episode 025 no longer jumps directly from normalized `-0.083` at frame 1757
  to `-0.199` at frame 1758, and that the preceding 20 frames form a smooth
  downward ramp.

## Rollout Notes

The option is opt-in, so existing datasets preserve their current behavior
unless they explicitly configure a transition window. For ALOHA sandwich, the
pipeline should be rerun from return computation onward:

1. recompute returns with `hitl_transition_chunks: 2`;
2. retrain value SFT on the new returns sidecar;
3. recompute advantages with the new value checkpoint and same returns tag;
4. regenerate value/advantage diagnostics.

Only retraining or recomputing downstream artifacts from the smoothed sidecar
will change the plotted `value_current` shape.
