# PR3 BEHAVIOR tro_state Replay Evaluation Description

This note maps the PR3 diff to the behavior it enables. It is written as a
review aid for the BEHAVIOR `turning_on_radio` tro_state replay evaluation path,
especially the press-only evaluation that starts from cached press-stage states.

## Summary

PR3 adds support for evaluating BEHAVIOR policies from cached `tro_state`
states, plus the supporting machinery to produce those states from demonstration
replay. The main target is press-only evaluation for `turning_on_radio`, where
the environment starts after the move/pickup phases and success is measured by
reaching the press stage.

The change is intentionally split across these responsibilities:

- Evaluation configs define the full-task and press-only entry points.
- `instance_loader.py` restores cached `tro_state` files into an existing
  OmniGibson environment.
- `replay_initializer.py` builds replay plans from BEHAVIOR demonstration
  parquet / annotation files.
- `replay_tro_state_dumper.py` executes replay prefixes and writes mid-rollout
  `tro_state` files.
- `behavior_env.py` wires reset payloads, prompts, trunk proprio randomization,
  diagnostics, and metrics into the RLinf env surface.
- `action_controls.py`, `stage_rewards.py`, and `replay_runtime.py` keep
  action masking, stage reward/success, and replay metadata helpers out of the
  main env class.
- `utils.py` contains robot-state helpers needed after direct state restoration.

## Motivation

The standard BEHAVIOR full-task reset starts from the beginning of the task and
measures end-to-end completion. For the current work, we need a narrower
press-stage evaluation:

- Start from cached states where the robot is already near the press subtask.
- Avoid measuring unrelated move-to-radio / pickup noise.
- Keep the heldout instance IDs fixed so checkpoints are comparable.
- Report both "ever reached press success" and "successful at rollout end".

The press-stage states are not checked into this repository. The evaluation
config expects the dataset path through `PRESS_TRO_STATE_DIR`.

## Diff Map

| Area | Files | What changed |
| --- | --- | --- |
| Evaluation entry points | `evaluations/behavior/behavior_replay_tro_state_eval.yaml`, `evaluations/behavior/behavior_replay_tro_state_press_eval.yaml` | Adds full-task and press-only configs for tro_state replay evaluation. |
| Cached instance parsing / loading | `rlinf/envs/behavior/instance_loader.py` | Adds `tro_state` file discovery, range parsing for instance IDs, robot/object restore, replay metadata loading, and deterministic offline sampling. |
| State dumping | `rlinf/envs/behavior/instance_generator.py`, `rlinf/envs/behavior/replay_tro_state_dumper.py` | Adds `dump_tro_state()` and a replay-based dumper that writes mid-rollout states. |
| Replay plan construction | `rlinf/envs/behavior/replay_initializer.py` | Reads BEHAVIOR demo actions and annotations, selects replay prefix length by step or stage, and optionally adds action noise. |
| Env runtime wiring | `rlinf/envs/behavior/behavior_env.py` | Adds replay reset payloads, prompt control, trunk proprio randomization, diagnostics, and metrics. |
| Env runtime helpers | `rlinf/envs/behavior/action_controls.py`, `rlinf/envs/behavior/stage_rewards.py`, `rlinf/envs/behavior/replay_runtime.py` | Factors action masking / first-chunk overrides, stage reward / success extraction, and replay metadata injection out of `behavior_env.py`. |
| Worker RPC entry | `rlinf/workers/env/env_worker.py` | Adds `dump_behavior_replay_tro_states()` to call the env-side replay dumper across worker/stage slots. |
| Visual wrapper | `rlinf/envs/behavior/rgb_wrapper.py` | Sets the R1Pro base mass and camera resolution in the RGB wrapper. |
| Tests | `tests/unit_tests/test_activity_instance_loader.py`, `tests/unit_tests/test_behavior_env_metrics.py`, `tests/unit_tests/test_behavior_process_pool.py`, `tests/unit_tests/test_behavior_replay_pipeline.py` | Adds pure-Python coverage for parsing, replay planning, reset sharding, metrics/done extraction, and replay data structures. |

## Evaluation Configs

### Press-only config

File: `evaluations/behavior/behavior_replay_tro_state_press_eval.yaml`

Important fields:

- `runner.task_type: embodied_eval`
- `runner.only_eval: True`
- `runner.max_steps: 1`
- `env.eval.total_num_envs: 32`
- `env.eval.rollout_epoch: 4`
- `env.eval.use_fixed_reset_state_ids: True`
- `env.eval.prompt_override: "press radio"`
- `env.eval.success_stage_idx: 3`
- `env.eval.omni_config.task.activity_instance_dir: ${oc.env:PRESS_TRO_STATE_DIR}`
- `env.eval.omni_config.task.activity_instance_id: "0-99"`
- `env.eval.omni_config.task.instance_file_format: tro_state`
- `env.eval.omni_config.task.instance_resample_mode: offline`

Semantics:

- Each eval reset loads one cached `tro_state` from `PRESS_TRO_STATE_DIR`.
- Instance IDs are iterated in a fixed order when `use_fixed_reset_state_ids`
  is enabled, giving deterministic checkpoint-to-checkpoint comparisons.
- The policy prompt is overridden to `press radio`, so the model is evaluated on
  the subtask rather than the full task description.
- Success is the configured stage success, not the original full-task success.

### Full-task config

File: `evaluations/behavior/behavior_replay_tro_state_eval.yaml`

This is the same tro_state restore path without the press-only prompt and
stage-success specialization. It exists to validate that cached tro_state
instances can also drive a full-task style eval.

## tro_state Loading and Restore

File: `rlinf/envs/behavior/instance_loader.py`

Key symbols:

- `parse_activity_instance_ids()`
- `discover_activity_instance_files()`
- `load_activity_instance_tro_state()`
- `ActivityInstanceLoader.prepare_reset()`

What it does:

1. Parses `activity_instance_id` as an int, list, comma-separated string, or
   inclusive range such as `"0-99"`.
2. Discovers cached files with either `template` or `tro_state` format.
3. For `tro_state`, bootstraps OmniGibson from instance `0` first, then mutates
   the existing env in place with cached task-relevant state.
4. Restores object state, robot base pose, robot joint positions, and optional
   assisted-grasp state.
5. Writes `rlinf_replay` metadata into the scene so `behavior_env.py` can expose
   replay/stage information through `info`.

Important restore details:

- Object root poses are rebased between scene-relative and world coordinates
  when `env.scene.idx != 0`.
- Robot articulation is restored without driving stale controller targets.
- A mid-rollout restore performs settle steps before rollout continues. The
  default is `DEFAULT_MIDROLLOUT_RESTORE_SETTLE_STEPS = 120`, configurable with
  `RLINF_BEHAVIOR_TRO_RESTORE_SETTLE_STEPS`.
- `RLINF_BEHAVIOR_TRO_LOAD_AGENT_STATE=0` can disable agent state loading for
  debugging restore behavior.

## Replay Plan Construction

File: `rlinf/envs/behavior/replay_initializer.py`

Key symbols:

- `ReplayEpisode`
- `ReplayPlan`
- `BehaviorReplayInitializer`
- `maybe_make_replay_initializer()`
- `replay_plans_to_infos()`

What it does:

1. Reads BEHAVIOR demonstration episodes from
   `{dataset_root}/data/task-{task_id:04d}/episode_*.parquet`.
2. Optionally reads stage annotations from
   `{dataset_root}/annotations/task-{task_id:04d}/episode_*.json`.
3. Optionally reads subtask prompts from orchestrator files under
   `{dataset_root}/orchestrators/task-{task_id:04d}/.../task_annotated.json`.
4. Converts episode IDs to BEHAVIOR activity instance IDs.
5. Samples replay plans deterministically with `seed + seed_offset`.
6. Chooses replay prefix length by:
   - explicit `target_step`, or
   - `stage_index` plus `stage_boundary` (`start` or `end`), or
   - the full demo length.
7. Applies optional action noise for robustness experiments, globally or over
   selected stage intervals.

This module is the core "replay helper" reviewers may be referring to. It keeps
demo-file parsing and replay prefix selection out of the env runtime.

## Replay tro_state Dumping

File: `rlinf/envs/behavior/replay_tro_state_dumper.py`

Key symbols:

- `make_replay_initializer()`
- `replay_plans()`
- `dump_replay_tro_states()`
- `grasp_rejection_reason()`

What it does:

1. Resets env slots to source activity instances.
2. Uses `BehaviorReplayInitializer` to sample the demo prefix for each source
   instance.
3. Steps the vectorized OmniGibson env through those prefix actions.
4. Optionally rejects candidates where the target object or any object remains
   grasped.
5. Writes the current simulator state through `dump_tro_state()`.
6. Embeds replay metadata such as source/output instance ID, episode index,
   replay step count, requested stage index, actual stage index, and subtask
   prompts.

Worker entry point:

- `EnvWorker.dump_behavior_replay_tro_states()` builds jobs across workers,
  rollout stages, and local env slots.
- `BehaviorProcess.dump_replay_tro_states()` delegates to the dumper module
  inside the Ray actor that owns OmniGibson.

## Runtime Reset and Metadata Flow

Files: `rlinf/envs/behavior/behavior_env.py`,
`rlinf/envs/behavior/replay_runtime.py`

Key symbols:

- `BehaviorProcess._parse_reset_payload()`
- `BehaviorProcess._reset_env_indices()`
- `replay_runtime.apply_replay_tro_metadata()`
- `BehaviorEnv.env_reset_partial()`
- `BehaviorEnv._init_ordered_reset_instance_ids()`
- `BehaviorEnv._ordered_reset_ids_for_indices()`

Flow:

1. `BehaviorEnv.reset()` calls the pool reset path.
2. For fixed eval IDs, `BehaviorEnv` prepares per-env reset payloads with the
   requested `activity_instance_id`.
3. `BehaviorProcess.reset()` recognizes payload entries containing
   `instance_id`.
4. `ActivityInstanceLoader.prepare_reset()` loads the requested cached
   `tro_state`.
5. `replay_runtime.apply_replay_tro_metadata()` reads scene metadata and
   injects it into `info["replay_init"]` and `info["reward"]["task_specific"]`.
6. If a stage index is present, the task reward's active stage is set before
   downstream reward and success logic runs.

This path is why reset infos must be handled carefully: reset infos may not
include `info["done"]`. `BehaviorEnv._extract_info_done()` now treats missing or
malformed done fields as not done instead of raising.

## Stage Success, Reward, and Metrics

File: `rlinf/envs/behavior/stage_rewards.py`

Key symbols:

- `success_stage_idx`
- `is_target_stage_success()`
- `stage_sparse_reward_tensor()`
- `stage_weighted_reward_tensor()`
- `stage_cumulative_reward_tensor()`
- `extract_episode_success()`
- `extract_episode_done()`
- `_record_metrics()`

Success semantics:

- If `env.success_stage_idx` is set, episode success is computed from stage
  metadata:
  - `current_stage_idx == success_stage_idx`
  - `completion_bonus != 0.0`
- If `success_stage_idx` is not set, the env falls back to the normal
  BEHAVIOR `done.success` / `info.success` path.

Reward modes:

- `stage_sparse`: reward only when the configured target stage succeeds.
- `stage_weighted`: stage completion bonus multiplied by configured
  `stage_reward_weights`.
- `stage_cumulative`: reward is the completed stage count.
- Other modes preserve existing per-step / relative reward behavior.

Metrics added or propagated:

- `success_once`
- `success_at_end`
- `target_stage_success`
- `current_stage_idx`
- `success_stage_idx`
- `completed_stage_count`
- `total_stage_count`
- `activity_instance_id`
- `held_in_hand_at_end`
- numeric replay metadata under `infos["replay_init"]`

For press-only eval, `success_once` answers "did the policy ever press during
the rollout", while `success_at_end` answers "was the target stage successful at
the final done event".

## Prompt and Observation Handling

File: `rlinf/envs/behavior/action_controls.py`

Key symbols:

- `prompt_override`
- `use_subtask_prompt`
- `_update_stage_prompts_from_info()`
- `_compose_task_description()`
- `_task_descriptions_from_infos()`
- `_wrap_obs()`

Behavior:

- `prompt_override` replaces all task descriptions. The press-only config uses
  this for `"press radio"`.
- `use_subtask_prompt` can append stage-specific prompts discovered by
  `BehaviorReplayInitializer` from orchestrator metadata.
- `_wrap_obs()` still returns the normal OpenPI observation layout:
  `main_images`, `wrist_images`, `task_descriptions`, and `states`.

## Robot and Mechanical Control Details

These changes are important because direct `tro_state` restore bypasses the
normal reset path. Without explicit cleanup, the robot can carry stale
controller goals, stale joint velocities, stale assisted-grasp bookkeeping, or
unstable restored articulation state into the first policy step.

### Robot state helpers

File: `rlinf/envs/behavior/utils.py`

Key symbols:

- `sync_robot_after_pose_override()`
- `reset_robot_joint_state_to_reset_pose()`
- `clear_robot_grasp_state()`

Behavior:

- `sync_robot_after_pose_override()` calls `keep_still()`, resets joint
  velocities to zero, and resynchronizes the robot after direct base-pose
  overrides.
- `reset_robot_joint_state_to_reset_pose()` restores manipulation joints to the
  robot's configured reset posture while preserving the sampled base pose.
- `clear_robot_grasp_state()` best-effort releases stale assisted-grasp state on
  all robot arms.

### Mid-rollout restore helpers

File: `rlinf/envs/behavior/instance_loader.py`

Key symbols:

- `_restore_robot_joint_positions()`
- `_sync_robot_controller_no_op_goals()`
- `_settle_restored_midrollout_state()`
- `_restore_ag_state()`

Behavior:

- Restores robot joint positions with zero joint velocity.
- Updates controller no-op goals to match the restored state.
- Steps physics while holding robot/object state still so the restored state is
  stable before the policy acts.
- Restores assisted-grasp constraints when the cached state was saved from an
  assisted-grasping rollout.

### Action controls

File: `rlinf/envs/behavior/behavior_env.py`

Key symbols:

- `parse_action_mask()`
- `r1pro_noop_action()`
- `apply_action_mask()`
- `parse_first_chunk_action_override()`
- `apply_first_chunk_action_override()`

Behavior:

- `action_mask` can freeze selected action dimensions. Frozen dimensions are not
  set to zero; they are replaced with an R1Pro no-op action derived from the
  current joint state.
- The default generated mask can freeze:
  - base action dimensions `0:3`
  - trunk action dimensions `3:7`
- `r1pro_noop_action()` maps the current R1Pro joint positions into the 23-D
  action layout:
  - dimensions `3:7`: trunk joints from robot joint positions `6:10`
  - dimensions `7:14`: left arm joints from even arm joint indices
  - dimensions `14:21`: right arm joints from odd arm joint indices
  - dimensions `21` and `22`: gripper sums
- `first_chunk_action_override` can force selected action dimensions to a fixed
  value only on the first action chunk after reset. This is a targeted control
  hook for stabilizing the immediate post-restore action.

### Diagnostics

File: `rlinf/envs/behavior/behavior_env.py`

Key symbols:

- `trace_robot_joints`
- `RLINF_BEHAVIOR_TRACE_JOINTS`
- `_log_robot_joint_trace()`

Behavior:

- When enabled through config or `RLINF_BEHAVIOR_TRACE_JOINTS=1`, the env prints
  JSON records with robot joint positions, velocities, world pose, chunk step,
  and optional action values.
- This is diagnostic only; it should not change rollout behavior.

### RGB wrapper mass change

File: `rlinf/envs/behavior/rgb_wrapper.py`

The wrapper sets `robot.base_footprint_link.mass = 250.0` while the simulator is
stopped, then applies camera resolutions. This is a mechanical stability change
for the R1Pro RGB observation path and should be called out separately because it
changes physical simulation parameters.

## Tests and Verification

Added tests:

- `tests/unit_tests/test_activity_instance_loader.py`
- `tests/unit_tests/test_behavior_env_metrics.py`
- `tests/unit_tests/test_behavior_process_pool.py`
- `tests/unit_tests/test_behavior_replay_pipeline.py`
- `tests/unit_tests/test_behavior_runtime_helpers.py`

The tests are mostly pure-Python and avoid requiring OmniGibson startup. They
cover:

- activity instance ID parsing
- replay dataclasses and replay config validation
- replay info conversion
- process-pool sharding / merge behavior
- reset-path done extraction without `KeyError`
- metrics and success extraction shapes
- direct helper coverage for action controls, stage rewards, and replay runtime
  metadata injection

Observed clean verify run:

- Log directory: `logs/20260818-clean-verify-press/`
- Completed `Global Step: 1/1`
- `num_trajectories = 128`
- `success_once = 0.4140625`
- `success_at_end = 0.015625`
- `target_stage_success = 0.015625`
- `held_in_hand_at_end = 0.734375`

Observed post-refactor smoke runs with the same behavior config:

- Log directory: `logs/refactor-verify-press-20260819-085159/`
  - `num_trajectories = 128`
  - `success_once = 0.328125`
  - `success_at_end = 0.015625`
  - `target_stage_success = 0.015625`
- Log directory: `logs/refactor-verify-press-rerun-20260819-091536/`
  - `num_trajectories = 128`
  - `success_once = 0.359375`
  - `success_at_end = 0.0078125`
  - `target_stage_success = 0.0078125`

The baseline and post-refactor Hydra configs differ only in log/video output
paths and experiment name. Behavior-critical fields such as
`success_stage_idx: 3`, `prompt_override: "press radio"`,
`activity_instance_id: "0-99"`, `instance_resample_mode: offline`,
`rollout_epoch: 4`, and the OpenPI checkpoint path are unchanged.

The eval requires external assets:

- `PRESS_TRO_STATE_DIR`: directory containing cached press-stage `tro_state`
  files.
- `rollout.model.model_path`: OpenPI checkpoint path.
- BEHAVIOR / OmniGibson runtime dependencies.

## Reviewer Notes and Current Structure

The functional change is larger than a config-only eval addition, so the runtime
helpers are split by concern:

- `behavior_env.py`: env orchestration, prompt composition, trunk proprio
  randomization, diagnostics, and metric recording wrappers.
- `action_controls.py`: action mask parsing/application, first-chunk overrides,
  and R1Pro no-op action construction.
- `stage_rewards.py`: target-stage success, stage reward tensors, and episode
  success/done extraction.
- `replay_runtime.py`: replay metadata injection and stage-index helpers.
- `utils.py`: robot restore helpers used after direct state restoration.

The current PR also has two replay modules:

- `replay_initializer.py`: demo dataset parsing and replay plan selection.
- `replay_tro_state_dumper.py`: env-side replay execution and state dumping.

Those modules should stay separate. A remaining follow-up is deciding whether
the robot restore helpers in `utils.py` should become a dedicated robot-state
module.
