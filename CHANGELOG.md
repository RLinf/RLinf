# Changelog

## Unreleased

### Behavior Subtask Reward And Replay

This change keeps the existing BEHAVIOR environment architecture while adding
subtask reward and replay/TRO state support.

The BEHAVIOR stack remains organized as:

- `BehaviorProcess`: simulator layer for OmniGibson `VectorEnvironment`, env
  wrappers, env step/reset, robot state access, and simulator-dependent replay
  metadata.
- `BehaviorProcessPool`: Ray actor pool and sharding layer for shared pool
  acquire/release, env slicing, shard dispatch, result merging, and cleanup.
- `BehaviorEnv`: Gym/RL interface layer for reset, chunk step, observation
  wrapping, reward calculation, done handling, metrics, auto reset, and fixed
  reset state ids.

Added support for replay/TRO state initialization and dumping so BEHAVIOR
training and evaluation can restore deterministic task states and collect
replayable simulator states.

Added fixed reset state id handling for deterministic evaluation over selected
activity instances.

Added BEHAVIOR stage reward modes for subtask training:

- `stage_sparse`
- `stage_weighted`
- `stage_cumulative`

Added subtask prompt handling so replay or reward metadata can update
`obs["task_descriptions"]` with the current stage prompt while keeping task
description assembly inside `BehaviorEnv`.

Added action control helpers for action masking and first-chunk action override.
These keep simulator-facing action fixes in `BehaviorProcess` while keeping the
reusable parsing and tensor logic outside the class.

Reduced unnecessary class-local helper wrappers in `behavior_env.py` without
introducing a new helper module. Pure file-local helpers remain module-private
inside `behavior_env.py` to stay close to the main branch single-file style.

Validation run:

- `python -m py_compile rlinf/envs/behavior/behavior_env.py`
- `ruff check rlinf/envs/behavior/behavior_env.py`
- `pytest tests/unit_tests/test_behavior_runtime_helpers.py tests/unit_tests/test_activity_instance_loader.py tests/unit_tests/test_behavior_process_pool.py tests/unit_tests/test_behavior_replay_pipeline.py tests/unit_tests/test_behavior_env_metrics.py`
