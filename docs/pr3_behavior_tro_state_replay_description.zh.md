# PR3 BEHAVIOR tro_state Replay Evaluation 说明

这份说明把 PR3 的 diff 映射到它实际启用的行为。它主要用于 review
BEHAVIOR `turning_on_radio` 的 `tro_state` replay evaluation 路径，尤其是
从缓存 press-stage 状态开始的 press-only evaluation。

## 总览

PR3 增加了从缓存 `tro_state` 状态评估 BEHAVIOR policy 的能力，并补上了
从 demonstration replay 生成这些状态的配套机制。主要目标是
`turning_on_radio` 的 press-only evaluation：环境从 move / pickup 阶段之后
开始，成功判定看是否到达 press 阶段。

这次改动按职责拆成了几部分：

- Evaluation configs 定义 full-task 和 press-only 的入口。
- `instance_loader.py` 把缓存的 `tro_state` 文件恢复到已有 OmniGibson 环境里。
- `replay_initializer.py` 从 BEHAVIOR demonstration 的 parquet / annotation
  文件构造 replay plan。
- `replay_tro_state_dumper.py` 执行 demo prefix，并写出 rollout 中间状态的
  `tro_state` 文件。
- `behavior_env.py` 把 reset payload、prompt、trunk proprio randomization、
  diagnostics 和 metrics 接到 RLinf env 表面。
- `action_controls.py`、`stage_rewards.py` 和 `replay_runtime.py` 把
  action masking、stage reward / success、replay metadata helper 从主 env
  class 里拆出来。
- `utils.py` 包含直接恢复状态之后所需的 robot-state helper。

## 背景动机

标准 BEHAVIOR full-task reset 会从任务开头开始，并评估端到端完成情况。
当前工作需要一个更窄的 press-stage evaluation：

- 从缓存状态开始，此时机器人已经接近 press 子任务。
- 避免把 move-to-radio / pickup 这些无关噪声混进 press 技能评估。
- 固定 heldout instance IDs，让不同 checkpoint 可以稳定比较。
- 同时报告“rollout 中是否曾经 press 成功”和“rollout 结束时是否仍然成功”。

press-stage 状态不会提交到仓库里。evaluation config 通过
`PRESS_TRO_STATE_DIR` 读取数据路径。

## Diff Map

| 区域 | 文件 | 改动内容 |
| --- | --- | --- |
| Evaluation 入口 | `evaluations/behavior/behavior_replay_tro_state_eval.yaml`, `evaluations/behavior/behavior_replay_tro_state_press_eval.yaml` | 增加 tro_state replay evaluation 的 full-task 和 press-only 配置。 |
| 缓存 instance 解析 / 加载 | `rlinf/envs/behavior/instance_loader.py` | 增加 `tro_state` 文件发现、instance ID range 解析、robot/object restore、replay metadata 加载，以及 deterministic offline sampling。 |
| 状态 dump | `rlinf/envs/behavior/instance_generator.py`, `rlinf/envs/behavior/replay_tro_state_dumper.py` | 增加 `dump_tro_state()`，以及基于 replay 的 dumper，用来写出 rollout 中间状态。 |
| Replay plan 构造 | `rlinf/envs/behavior/replay_initializer.py` | 读取 BEHAVIOR demo actions 和 annotations，按 step 或 stage 选择 replay prefix length，并可选加入 action noise。 |
| Env runtime 接线 | `rlinf/envs/behavior/behavior_env.py` | 增加 replay reset payload、prompt control、trunk proprio randomization、diagnostics 和 metrics。 |
| Env runtime helpers | `rlinf/envs/behavior/action_controls.py`, `rlinf/envs/behavior/stage_rewards.py`, `rlinf/envs/behavior/replay_runtime.py` | 将 action masking / first-chunk override、stage reward / success extraction 和 replay metadata injection 从 `behavior_env.py` 拆出。 |
| Worker RPC 入口 | `rlinf/workers/env/env_worker.py` | 增加 `dump_behavior_replay_tro_states()`，用于跨 worker / stage / env slot 调用 env 侧 replay dumper。 |
| Visual wrapper | `rlinf/envs/behavior/rgb_wrapper.py` | 在 RGB wrapper 里设置 R1Pro base mass 和 camera resolution。 |
| Tests | `tests/unit_tests/test_activity_instance_loader.py`, `tests/unit_tests/test_behavior_env_metrics.py`, `tests/unit_tests/test_behavior_process_pool.py`, `tests/unit_tests/test_behavior_replay_pipeline.py` | 增加 pure-Python 覆盖：解析、replay planning、reset sharding、metrics / done extraction 和 replay data structures。 |

## Evaluation Configs

### Press-only config

文件：`evaluations/behavior/behavior_replay_tro_state_press_eval.yaml`

重要字段：

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

语义：

- 每次 eval reset 会从 `PRESS_TRO_STATE_DIR` 加载一个缓存 `tro_state`。
- 启用 `use_fixed_reset_state_ids` 后，instance IDs 会按固定顺序遍历，
  因此 checkpoint 之间的比较是 deterministic 的。
- policy prompt 被覆盖成 `press radio`，所以模型被评估的是子任务，
  而不是完整任务描述。
- success 使用配置的 stage success，不使用原始 full-task success。

### Full-task config

文件：`evaluations/behavior/behavior_replay_tro_state_eval.yaml`

这个配置使用同一套 `tro_state` restore 路径，但没有 press-only prompt 和
stage-success 专门化。它用于验证缓存 `tro_state` instance 也能驱动
full-task 风格的 eval。

## tro_state 加载和恢复

文件：`rlinf/envs/behavior/instance_loader.py`

关键符号：

- `parse_activity_instance_ids()`
- `discover_activity_instance_files()`
- `load_activity_instance_tro_state()`
- `ActivityInstanceLoader.prepare_reset()`

它做的事情：

1. 将 `activity_instance_id` 解析成 int、list、逗号分隔字符串，或 `"0-99"`
   这样的闭区间 range。
2. 发现 `template` 或 `tro_state` 格式的缓存文件。
3. 对 `tro_state`，先用 instance `0` bootstrap OmniGibson，然后在已有 env
   上原地写入缓存的 task-relevant state。
4. 恢复 object state、robot base pose、robot joint positions，以及可选的
   assisted-grasp state。
5. 把 `rlinf_replay` metadata 写入 scene，让 `behavior_env.py` 可以通过
   `info` 暴露 replay / stage 信息。

重要 restore 细节：

- 当 `env.scene.idx != 0` 时，object root poses 会在 scene-relative 和
  world coordinates 之间重新基准化。
- Robot articulation 会被恢复，但不会沿用过期 controller targets。
- mid-rollout restore 会先执行 settle steps，再继续 rollout。默认值是
  `DEFAULT_MIDROLLOUT_RESTORE_SETTLE_STEPS = 120`，可通过
  `RLINF_BEHAVIOR_TRO_RESTORE_SETTLE_STEPS` 配置。
- `RLINF_BEHAVIOR_TRO_LOAD_AGENT_STATE=0` 可以关闭 agent state loading，
  用于调试 restore 行为。

## Replay Plan 构造

文件：`rlinf/envs/behavior/replay_initializer.py`

关键符号：

- `ReplayEpisode`
- `ReplayPlan`
- `BehaviorReplayInitializer`
- `maybe_make_replay_initializer()`
- `replay_plans_to_infos()`

它做的事情：

1. 从 `{dataset_root}/data/task-{task_id:04d}/episode_*.parquet` 读取
   BEHAVIOR demonstration episodes。
2. 可选从 `{dataset_root}/annotations/task-{task_id:04d}/episode_*.json`
   读取 stage annotations。
3. 可选从 `{dataset_root}/orchestrators/task-{task_id:04d}/.../task_annotated.json`
   读取 subtask prompts。
4. 将 episode IDs 转成 BEHAVIOR activity instance IDs。
5. 使用 `seed + seed_offset` deterministic 地采样 replay plans。
6. 按下面规则选择 replay prefix length：
   - 显式 `target_step`，或
   - `stage_index` 加 `stage_boundary`（`start` 或 `end`），或
   - 完整 demo 长度。
7. 为 robustness 实验可选添加 action noise，可以全局添加，也可以只在选定
   stage interval 上添加。

这个模块就是 reviewer 可能提到的核心 “replay helper”。它把 demo-file
解析和 replay prefix 选择从 env runtime 里拆了出来。

## Replay tro_state Dumping

文件：`rlinf/envs/behavior/replay_tro_state_dumper.py`

关键符号：

- `make_replay_initializer()`
- `replay_plans()`
- `dump_replay_tro_states()`
- `grasp_rejection_reason()`

它做的事情：

1. 将 env slots reset 到 source activity instances。
2. 使用 `BehaviorReplayInitializer` 为每个 source instance 采样 demo prefix。
3. 用这些 prefix actions 驱动 vectorized OmniGibson env。
4. 可选拒绝 target object 或任何 object 仍被 grasped 的候选状态。
5. 通过 `dump_tro_state()` 写出当前 simulator state。
6. 嵌入 replay metadata，例如 source / output instance ID、episode index、
   replay step count、requested stage index、actual stage index 和 subtask
   prompts。

Worker 入口：

- `EnvWorker.dump_behavior_replay_tro_states()` 在 workers、rollout stages 和
  local env slots 之间构造 jobs。
- `BehaviorProcess.dump_replay_tro_states()` 在拥有 OmniGibson 的 Ray actor
  内部委托 dumper module 执行。

## Runtime Reset 和 Metadata Flow

文件：`rlinf/envs/behavior/behavior_env.py`，
`rlinf/envs/behavior/replay_runtime.py`

关键符号：

- `BehaviorProcess._parse_reset_payload()`
- `BehaviorProcess._reset_env_indices()`
- `replay_runtime.apply_replay_tro_metadata()`
- `BehaviorEnv.env_reset_partial()`
- `BehaviorEnv._init_ordered_reset_instance_ids()`
- `BehaviorEnv._ordered_reset_ids_for_indices()`

流程：

1. `BehaviorEnv.reset()` 调用 pool reset 路径。
2. 对 fixed eval IDs，`BehaviorEnv` 准备带有目标 `activity_instance_id` 的
   per-env reset payloads。
3. `BehaviorProcess.reset()` 识别包含 `instance_id` 的 payload entries。
4. `ActivityInstanceLoader.prepare_reset()` 加载指定的缓存 `tro_state`。
5. `replay_runtime.apply_replay_tro_metadata()` 读取 scene metadata，并注入
   到 `info["replay_init"]` 和 `info["reward"]["task_specific"]`。
6. 如果存在 stage index，会先设置 task reward 的 active stage，再进入下游
   reward 和 success 逻辑。

这条路径说明了为什么 reset infos 必须谨慎处理：reset infos 可能没有
`info["done"]`。`BehaviorEnv._extract_info_done()` 现在会把缺失或格式错误的
done fields 当作 not done，而不是抛出异常。

## Stage Success、Reward 和 Metrics

文件：`rlinf/envs/behavior/stage_rewards.py`

关键符号：

- `success_stage_idx`
- `is_target_stage_success()`
- `stage_sparse_reward_tensor()`
- `stage_weighted_reward_tensor()`
- `stage_cumulative_reward_tensor()`
- `extract_episode_success()`
- `extract_episode_done()`
- `_record_metrics()`

Success 语义：

- 如果设置了 `env.success_stage_idx`，episode success 从 stage metadata 计算：
  - `current_stage_idx == success_stage_idx`
  - `completion_bonus != 0.0`
- 如果没有设置 `success_stage_idx`，env 回退到普通 BEHAVIOR
  `done.success` / `info.success` 路径。

Reward modes：

- `stage_sparse`：只有配置的 target stage 成功时才给 reward。
- `stage_weighted`：stage completion bonus 乘以配置的
  `stage_reward_weights`。
- `stage_cumulative`：reward 是已完成 stage 数量。
- 其他模式保持已有 per-step / relative reward 行为。

新增或透传的 metrics：

- `success_once`
- `success_at_end`
- `target_stage_success`
- `current_stage_idx`
- `success_stage_idx`
- `completed_stage_count`
- `total_stage_count`
- `activity_instance_id`
- `held_in_hand_at_end`
- `infos["replay_init"]` 下的数值型 replay metadata

对 press-only eval，`success_once` 表示“policy 在 rollout 中是否曾经 press
成功”，而 `success_at_end` 表示“最终 done event 时 target stage 是否成功”。

## Prompt 和 Observation Handling

文件：`rlinf/envs/behavior/action_controls.py`

关键符号：

- `prompt_override`
- `use_subtask_prompt`
- `_update_stage_prompts_from_info()`
- `_compose_task_description()`
- `_task_descriptions_from_infos()`
- `_wrap_obs()`

行为：

- `prompt_override` 会替换所有 task descriptions。press-only config 用它设置
  `"press radio"`。
- `use_subtask_prompt` 可以追加由 `BehaviorReplayInitializer` 从 orchestrator
  metadata 里发现的 stage-specific prompts。
- `_wrap_obs()` 仍返回普通 OpenPI observation layout：
  `main_images`、`wrist_images`、`task_descriptions` 和 `states`。

## Robot 和机械控制细节

这些改动很重要，因为直接 `tro_state` restore 会绕过普通 reset 路径。
如果不显式清理，机器人可能把过期 controller goals、过期 joint velocities、
过期 assisted-grasp bookkeeping，或者不稳定的 restored articulation state
带到 policy 第一步。

### Robot state helpers

文件：`rlinf/envs/behavior/utils.py`

关键符号：

- `sync_robot_after_pose_override()`
- `reset_robot_joint_state_to_reset_pose()`
- `clear_robot_grasp_state()`

行为：

- `sync_robot_after_pose_override()` 调用 `keep_still()`，把 joint velocities
  清零，并在直接覆盖 base pose 后重新同步 robot。
- `reset_robot_joint_state_to_reset_pose()` 将 manipulation joints 恢复到
  robot 配置的 reset posture，同时保留采样得到的 base pose。
- `clear_robot_grasp_state()` best-effort 释放所有 robot arms 上过期的
  assisted-grasp state。

### Mid-rollout restore helpers

文件：`rlinf/envs/behavior/instance_loader.py`

关键符号：

- `_restore_robot_joint_positions()`
- `_sync_robot_controller_no_op_goals()`
- `_settle_restored_midrollout_state()`
- `_restore_ag_state()`

行为：

- 恢复 robot joint positions，并将 joint velocity 置零。
- 更新 controller no-op goals，使其匹配恢复后的状态。
- 在保持 robot / object state 静止的同时推进 physics，让恢复状态在 policy
  action 前稳定下来。
- 如果缓存状态来自 assisted-grasping rollout，则恢复 assisted-grasp
  constraints。

### Action controls

文件：`rlinf/envs/behavior/behavior_env.py`

关键符号：

- `parse_action_mask()`
- `r1pro_noop_action()`
- `apply_action_mask()`
- `parse_first_chunk_action_override()`
- `apply_first_chunk_action_override()`

行为：

- `action_mask` 可以冻结选定 action dimensions。被冻结的维度不是置零，而是
  替换成从当前 joint state 推导出来的 R1Pro no-op action。
- 默认生成的 mask 可以冻结：
  - base action dimensions `0:3`
  - trunk action dimensions `3:7`
- `r1pro_noop_action()` 将当前 R1Pro joint positions 映射到 23-D action
  layout：
  - dimensions `3:7`：来自 robot joint positions `6:10` 的 trunk joints
  - dimensions `7:14`：来自偶数 arm joint indices 的 left arm joints
  - dimensions `14:21`：来自奇数 arm joint indices 的 right arm joints
  - dimensions `21` 和 `22`：gripper sums
- `first_chunk_action_override` 可以只在 reset 后第一个 action chunk 中，强制
  选定 action dimensions 为固定值。这是稳定 restore 后第一步 action 的
  定向控制 hook。

### Diagnostics

文件：`rlinf/envs/behavior/behavior_env.py`

关键符号：

- `trace_robot_joints`
- `RLINF_BEHAVIOR_TRACE_JOINTS`
- `_log_robot_joint_trace()`

行为：

- 通过 config 或 `RLINF_BEHAVIOR_TRACE_JOINTS=1` 启用时，env 会打印 JSON
  records，包含 robot joint positions、velocities、world pose、chunk step
  和可选 action values。
- 这只用于诊断，不应该改变 rollout 行为。

### RGB wrapper mass change

文件：`rlinf/envs/behavior/rgb_wrapper.py`

wrapper 会在 simulator stopped 状态下设置
`robot.base_footprint_link.mass = 250.0`，然后应用 camera resolutions。
这是针对 R1Pro RGB observation path 的机械稳定性改动。由于它改变了物理仿真
参数，应该在 review 中单独说明。

## Tests 和 Verification

新增 tests：

- `tests/unit_tests/test_activity_instance_loader.py`
- `tests/unit_tests/test_behavior_env_metrics.py`
- `tests/unit_tests/test_behavior_process_pool.py`
- `tests/unit_tests/test_behavior_replay_pipeline.py`
- `tests/unit_tests/test_behavior_runtime_helpers.py`

这些测试大多是 pure-Python，不需要启动 OmniGibson。覆盖内容包括：

- activity instance ID parsing
- replay dataclasses 和 replay config validation
- replay info conversion
- process-pool sharding / merge behavior
- reset-path done extraction，避免 `KeyError`
- metrics 和 success extraction shapes
- action controls、stage rewards 和 replay runtime metadata injection 的直接
  helper 覆盖

已观察到的 clean verify run：

- Log directory：`logs/20260818-clean-verify-press/`
- 完成 `Global Step: 1/1`
- `num_trajectories = 128`
- `success_once = 0.4140625`
- `success_at_end = 0.015625`
- `target_stage_success = 0.015625`
- `held_in_hand_at_end = 0.734375`

结构调整之后，用相同行为配置观察到的 smoke runs：

- Log directory：`logs/refactor-verify-press-20260819-085159/`
  - `num_trajectories = 128`
  - `success_once = 0.328125`
  - `success_at_end = 0.015625`
  - `target_stage_success = 0.015625`
- Log directory：`logs/refactor-verify-press-rerun-20260819-091536/`
  - `num_trajectories = 128`
  - `success_once = 0.359375`
  - `success_at_end = 0.0078125`
  - `target_stage_success = 0.0078125`

baseline 和结构调整后的 Hydra config 只在 log / video 输出路径和
experiment name 上不同。关键行为字段没有变化，包括
`success_stage_idx: 3`、`prompt_override: "press radio"`、
`activity_instance_id: "0-99"`、`instance_resample_mode: offline`、
`rollout_epoch: 4` 和 OpenPI checkpoint path。

eval 依赖外部 assets：

- `PRESS_TRO_STATE_DIR`：包含缓存 press-stage `tro_state` 文件的目录。
- `rollout.model.model_path`：OpenPI checkpoint 路径。
- BEHAVIOR / OmniGibson runtime dependencies。

## Reviewer Notes 和当前结构

这次功能改动比 config-only eval addition 大，所以 runtime helpers 已按职责
拆分：

- `behavior_env.py`：env orchestration、prompt composition、trunk proprio
  randomization、diagnostics 和 metric recording wrappers。
- `action_controls.py`：action mask parsing / application、first-chunk
  overrides 和 R1Pro no-op action 构造。
- `stage_rewards.py`：target-stage success、stage reward tensors 和 episode
  success / done extraction。
- `replay_runtime.py`：replay metadata injection 和 stage-index helpers。
- `utils.py`：直接 state restore 后使用的 robot restore helpers。

当前 PR 也有两个 replay modules：

- `replay_initializer.py`：demo dataset parsing 和 replay plan selection。
- `replay_tro_state_dumper.py`：env-side replay execution 和 state dumping。

这两个模块应该继续保持独立；后续可以再判断是否要把 `utils.py` 里的
robot restore helpers 单独拆成 robot-state module。
