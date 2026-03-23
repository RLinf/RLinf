# Sync vs Async Embodied Training

This note explains the difference between RLinf's synchronous and asynchronous
embodied training pipelines, using the staged YAM setup as the concrete
example.

Related entrypoints:

- Sync: `examples/embodiment/train_embodied_agent.py`
- Sync staged YAM: `examples/embodiment/train_embodied_agent_staged.py`
- Async: `examples/embodiment/train_async.py`

## High-Level Idea

The difference is mostly about **where the idle time goes**.

- In the **sync pipeline**, rollout/env interaction and actor training are
  separated by a barrier. The system first collects a rollout batch, then
  stops sampling, trains, syncs weights, and starts the next batch.
- In the **async pipeline**, env interaction and rollout generation keep
  running while the actor trains on already-completed batches. Training,
  sampling, and weight sync overlap in time.

## Timeline Sketch

### Sync pipeline

```text
time -->

rollout/env: [ generate ][ generate ][ generate ] .... [ send data ]            idle
actor train:                                                   [   train   ]    idle
weight sync:                                                                     [sync]

next step  :                                                                               [ generate ] ...
```

Equivalent intuition:

```text
1. rollout/env produce one full batch
2. send trajectories to actor
3. actor trains
4. actor syncs new weights to rollout
5. rollout/env start again
```

This is simpler and gives fresher samples, but it wastes time whenever the
actor is training and rollout/env are waiting, or vice versa.

### Async pipeline

```text
time -->

rollout/env: [generate][generate][generate][generate][generate][generate]...
send data  :          [ send ]        [ send ]        [ send ]
actor train:                [  train  ][  train  ][  train  ]...
weight sync:                      [sync]     [sync]     [sync]
```

Equivalent intuition:

```text
1. env + rollout continuously produce batches
2. completed trajectories are pushed to the actor side
3. actor trains whenever enough data is ready
4. weights are synced periodically without stopping the whole pipeline
5. rollout may briefly wait only if samples get too stale
```

This improves throughput when env stepping or rollout inference is expensive,
because actor training no longer forces the whole system to pause.

## What "Send Data" and "Sync Weight" Mean

`send data`:

- Env and rollout workers finish a trajectory or rollout chunk.
- That data is sent through RLinf channels to the actor.
- The actor computes advantages / returns and then trains.

`sync weight`:

- After training, the actor publishes newer model weights.
- Rollout workers load the new weights so future actions use a newer policy.

In async PPO, rollout can continue for a while with slightly older weights.
That is expected. RLinf handles this with policy version tracking and
`staleness_threshold`.

## What the Real RLinf Code Does

Sync embodied runner:

- `EmbodiedRunner`
- used by `train_embodied_agent.py`
- used by `train_embodied_agent_staged.py`

Async embodied runner:

- `AsyncPPOEmbodiedRunner` for async PPO
- used by `train_async.py`

The async path is not just "the same code with `async def`". It is a different
pipeline design:

- env workers run as long-lived services
- rollout workers run as long-lived services
- actor keeps consuming completed trajectories and training continuously
- weight sync happens in the background / periodically instead of at a global
  barrier

## Why This Matters for Staged YAM

The staged YAM path adds one more component:

- `VLMPlannerWorker`

That worker is used for:

- **TOPReward**: dense reward from `compute_top_reward(...)`
- **subtask planning**: new instruction text from `get_next_subtask(...)`

In the current sync staged pipeline, both of those planner calls are
effectively on the critical path of env interaction:

```text
env step -> planner RPC -> wait for result -> continue stepping
```

So although the runner launches env and rollout concurrently, the env worker
still blocks on planner results during a step.

## Why Async Staged YAM Is Harder

Making **TOPReward** async is mostly a systems problem:

- env stepping should not wait for the VLM
- but trajectories still need complete reward tensors before actor training

Making **subtask planning** async is also a semantics problem:

- the planner decides a new `task_description`
- the policy is conditioned on that text
- if the planner reply arrives late, the policy may act for a few more chunks
  using the old instruction

That means "fully async planner service" introduces a new kind of staleness:

- not just **old model weights**
- but also **old task instruction**

## Async Staged YAM Design Choice

The implemented async staged design avoids the worst reward inconsistency by
keeping TOPReward anchored to the **original high-level task description**,
not the mutable subtask text.

That means:

- the planner-generated subtask is still useful as policy-side guidance
- delayed subtask arrival can still change policy conditioning later than ideal
- but the dense reward definition does **not** jump when the subtask text changes

In other words:

```text
reward anchor     = original task description
policy guidance   = latest available subtask text
```

This turns async subtask lag from a reward-definition bug into a control-lag
issue.

## Practical Risk Assessment

For YAM configs:

- `yam_ppo_openpi`: low risk from async subtask lag, because
  `subtask_interval: 0`
- `yam_ppo_openpi_topreward`: higher risk, because delayed planner replies can
  shift when the instruction actually changes

Possible failure modes:

- policy keeps following the old task text for too long
- the planner decision is based on earlier frames but gets applied later

What no longer happens with the current async design:

- TOPReward does not switch to scoring against a late-arriving subtask
- reward deltas do not mix old and new subtask instructions

## Recommended Mental Model

Use this rule of thumb:

- **Sync**: easier to reason about, lower staleness, lower hardware utilization
- **Async**: higher throughput, more overlap, but more stale state to manage

For staged YAM specifically:

- async TOPReward is a reasonable direction
- async subtask planning is still somewhat risky, but mostly because of policy
  conditioning lag rather than reward inconsistency
- if subtask timing matters a lot, you may still want to block at subtask
  refresh points even in an otherwise async pipeline

## Summary

The picture to keep in mind is:

- **sync** = generate, stop, train, sync, repeat
- **async** = generate and train at the same time, with periodic weight sync

For plain embodied async PPO, this is mostly a throughput improvement.
For staged YAM, it also changes when planner-derived rewards and instructions
become visible, so correctness and semantics matter in addition to throughput.
