# ManiSkill RLT Joint Control 设计说明

这份文档说明 RLinf 中 ManiSkill joint-control RLT 的当前设计。范围包括 Stage1 RL token 训练、Stage2 在线 TD3 训练、ManiSkill 环境包装、rollout/replay 数据流、expert intervention、eval 口径和关键调参项。

本文档以当前代码和配置为准，主要对应：

| 模块 | 文件 |
| --- | --- |
| Stage2 配置 | `examples/embodiment/config/rlt_stage2_maniskill_joint.yaml` |
| Stage1 配置 | `examples/sft/config/rlt_stage1_maniskill_joint.yaml` |
| ManiSkill env | `rlinf/envs/maniskill/maniskill_env.py` |
| rollout worker | `rlinf/workers/rollout/hf/huggingface_worker.py` |
| env worker | `rlinf/workers/env/env_worker.py` |
| actor worker | `rlinf/workers/actor/fsdp_rlt_stage2_policy_worker.py` |
| Stage2 policy | `rlinf/models/embodiment/rlt_stage2/rlt_stage2_policy.py` |
| actor/critic/loss | `rlinf/models/embodiment/rlt_stage2/components.py` |
| replay buffer | `rlinf/models/embodiment/rlt_stage2/replay_buffer.py` |
| OpenPI wrapper | `rlinf/models/embodiment/rlt_stage2/vla_wrapper.py` |
| RL token | `rlinf/models/embodiment/rlt_stage2/rl_token.py` |
| OpenPI data config | `rlinf/models/embodiment/openpi/dataconfig/rlt_maniskill_joint_dataconfig.py` |

## 1. 目标和边界

ManiSkill RLT 的目标是在已经 SFT 过的 OpenPI/VLA 策略上做在线强化学习微调。Stage2 不再训练大 VLA，而是冻结 VLA 和 Stage1 训练出的 RL token，只训练一个轻量 actor-critic。

核心结构是：

```text
obs, prompt
  -> frozen OpenPI VLA
      -> VLA embedding z
      -> reference action chunk a_tilde
  -> frozen RL token encoder
      -> z_rl
  -> x = [z_rl, proprio]
  -> direct Gaussian actor pi_theta(x, a_tilde) -> action chunk a
  -> twin-Q critic Q_phi(x, a)
```

当前 ManiSkill 版本聚焦 joint control：

| 项 | 当前语义 |
| --- | --- |
| task | `PegInsertionSideWideClearance-v1` |
| prompt | `insert the peg in the hole` |
| obs mode | ManiSkill `rgb` |
| camera | third-view + wrist camera |
| proprio | Panda qpos 前 9 维 |
| action | 10-step chunk，每步 8 维 joint action |
| control mode | `pd_joint_delta_pos` |
| reward | `only_success`，即 success sparse reward |
| sim/control freq | 100Hz physics, 10Hz control |
| Stage2 trainable params | direct actor + twin-Q critic |
| Stage2 frozen params | OpenPI VLA + RL token |

非目标：

| 非目标 | 原因 |
| --- | --- |
| 不在 Stage2 更新 VLA | 当前实现用 actor-only worker 训练小模型，rollout 缓存特征，避免在线更新大模型的显存和同步成本 |
| 不把 reward 改成 dense shaping | `reward_mode: only_success` 是实验/论文要求，调参要在 sparse reward 下完成 |
| 不把 intervention 当作 DAgger 全监督训练 | RLT 中 intervention 是 off-policy replay 的一部分，critic 学实际执行动作，actor 用 intervention 替换后的 reference 做 BC regularization |
| 不默认使用 reset seed 0 | 当前 train/eval 显式配置 seed，避免默认简单初始状态造成虚高 |

## 2. 任务、观测和动作语义

### 2.1 ManiSkill 环境包装

`ManiskillEnv` 在 `wrap_obs_mode: rlt_openpi_joint` 下会走 RLT/OpenPI joint schema。

关键行为：

| 行为 | 说明 |
| --- | --- |
| 注册 peg insertion 变体 | 对 PegInsertionSide 系列注册 RLinf 的 joint observer variant |
| robot uid | 默认使用 `PANDA_WIDE_WRISTCAM_UID`，保证 wrist camera 可用 |
| image size | 默认 384x384 |
| main image key | `3rd_view_camera` |
| wrist image key | `wide_hand_camera` |
| prompt normalization | 将旧 prompt `insert the peg into the hole` 统一到 `insert the peg in the hole` |
| proprio extraction | 从 `raw_obs["agent"]["qpos"]` 取前 9 维 |

env 输出给 OpenPI/RLT 的 observation schema：

```python
{
    "main_images": Tensor[B, H, W, C],
    "wrist_images": Tensor[B, H, W, C],
    "extra_view_images": None,
    "states": Tensor[B, 9],
    "task_descriptions": list[str],
}
```

### 2.2 OpenPI 数据语义

Stage1 和 Stage2 都依赖 `pi05_rlt_maniskill_joint` 这套 OpenPI data config。`LeRobotRLTManiSkillJointDataConfig` 将本地数据字段映射到 OpenPI 训练/推理需要的字段：

| OpenPI 字段 | 本地字段 |
| --- | --- |
| `observation/image` | `image` |
| `observation/wrist_image` | `wrist_image` |
| `observation/state` | `state` |
| `actions` | `actions` |
| prompt | `prompt_key` 或 `default_prompt` |

输出 action 经过 `ManiSkillOutputs(output_action_dim=8)` 截到 8 维，和 Stage2 YAML 的 `action_dim: 8` 对齐。

### 2.3 Action chunk 单位

当前配置：

```yaml
actor:
  model:
    num_action_chunks: 10
    action_dim: 8
```

因此一个 action chunk 的 flat 形状是：

```text
action_chunk_dim = 10 * 8 = 80
```

系统里常见的单位如下：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| control step | `[8]` | ManiSkill 10Hz 下执行的一步 joint action |
| action chunk | `[10, 8]` 或 flat `[80]` | actor/VLA 一次输出的 10 个 control step |
| chunk transition | `x, a[80], rewards[10], next_x` | critic 学习的基本 replay 单位 |
| runner global step | 一次完整 collect/train/log/eval 调度周期 | 不是 env step，也不是 TD update |
| critic update | actor worker 对 critic 做一次 optimizer step | `update_step` 计数按 critic update 增加 |
| actor update | 每 `critic_actor_ratio` 次 critic update 做一次 | 当前通常是 2 critic : 1 actor |

## 3. 两阶段训练流程

RLT 分为 Stage1 和 Stage2。Stage1 训练 RL token，Stage2 在线训练 actor-critic。

### 3.1 Stage1：训练 RL token

配置文件：

```text
examples/sft/config/rlt_stage1_maniskill_joint.yaml
```

启动脚本：

```bash
bash examples/sft/train_rlt_stage1.sh
```

Stage1 输入是 ManiSkill joint 数据集，当前配置里是：

```yaml
data:
  train_data_paths:
    - dataset_path: "/mnt/public2/xiekaizhi/rlt-openpi-sim/rlt_maniskill_joint"

actor:
  model:
    model_path: ".../rlt_maniskill_joint_pi05_sft/checkpoints/global_step_2000/actor"
    num_action_chunks: 10
    action_dim: 8
    rlt_stage1:
      config_name: "pi05_rlt_maniskill_joint"
      norm_stats_path: ${actor.openpi_data.norm_stats_path}
      embedding_dim: 2048
      vla_finetune_alpha: 0.0
```

Stage1 policy 的核心是：

```text
VLA embedding z
  -> RLTokenEncoder(z, pad_mask) -> z_rl
  -> RLTokenDecoder(z_rl, z, pad_mask) -> z_hat
  -> masked MSE(z_hat, z)
```

当前 `vla_finetune_alpha: 0.0`，所以 VLA 冻结，只训练 RL token encoder/decoder。训练完成后导出：

```text
rl_token/rl_token_model.pt
```

Stage2 会通过 `actor.model.rlt_stage2.rl_token_path` 加载这个 checkpoint。

### 3.2 Stage2：在线 TD3 微调

配置文件：

```text
examples/embodiment/config/rlt_stage2_maniskill_joint.yaml
```

常用启动方式：

```bash
bash examples/embodiment/run_embodiment.sh rlt_stage2_maniskill_joint
```

Stage2 关键配置：

```yaml
algorithm:
  loss_type: rlt_td3
  adv_type: rlt_td3
  actor_only_train_model: True
  update_epoch: 5
  train_every_transitions: 50
  max_updates_per_train_step: 1600
  critic_actor_ratio: 2
  replay_buffer:
    min_buffer_size: 1000
    capacity: 50000
  td3_bc:
    warmup_updates: 30000
    actor_loss_warmup_updates: 20000
    actor_loss_ramp_updates: 50000
    warmup_bc_weight: 7.0
    warmup_q_weight: 0.05
    online_bc_weight: 2.8
    online_q_weight: 0.16

actor:
  model:
    model_type: rlt_stage2
    model_path: ".../rlt_maniskill_joint_pi05_sft/checkpoints/global_step_1000/actor"
    rlt_stage2:
      rl_token_path: ".../rl_token_model.pt"
      actor_noise_sigma: 0.002
      ref_action_dropout: 0.5
      replay_subsample_stride: 2
```

Stage2 worker 分工：

| worker | 负责内容 |
| --- | --- |
| EnvWorker | reset/env step/chunk step/reward/done/intervention policy_info/step obs trace |
| RolloutWorker | 用 frozen VLA + RL token 编码 `x` 和 `a_tilde`，用 actor 预测动作，必要时调用 expert |
| ActorWorker | 接收 trajectory，构造 replay，训练 direct actor 和 twin-Q critic，同步 actor 权重到 rollout |

## 4. Stage2 模型设计

### 4.1 Frozen VLA wrapper

`Stage2VLAWrapper` 加载 OpenPI 模型，用于两件事：

| 方法 | 作用 |
| --- | --- |
| `extract_embeddings()` | 调 OpenPI prefix cache，拿 VLA 内部 embedding `z` 和 pad mask |
| `get_rl_chunk_reference()` | 调 OpenPI `sample_actions(..., mode="eval")`，拿 VLA reference action chunk `a_tilde` |
| `extract_proprio()` | 从 processed OpenPI observation 的 state 里取前 `proprio_dim` 维 |
| `prepare_obs()` | 接受原始 prompt 或 pre-tokenized prompt，构造 OpenPI observation |

Stage2 始终把 VLA 当 frozen feature provider，不通过 actor worker 更新 VLA。

### 4.2 RL token

RL token 模块和 Stage1 共享结构：

```text
RLTokenEncoder:
  append learnable e_rl token
  transformer encoder
  return final token as z_rl

RLTokenDecoder:
  transformer decoder
  reconstruct z_hat from z_rl
```

Stage2 只使用 encoder path：

```python
z_rl = rl_token_model.encode(embeddings, pad_mask)
```

最终给 actor/critic 的 state 是：

```text
x = concat([z_rl, proprio])
```

当前默认：

```text
z_rl dim = 2048
proprio dim = 9
x dim = 2057
```

### 4.3 Direct Gaussian Actor

当前实现使用 `DirectGaussianActor`，不是旧 residual actor。

输入：

```text
[x, a_tilde]
```

输出：

```text
action chunk mean, shape [B, 80]
```

训练/rollout 时可以加固定标准差 Gaussian noise：

```text
a = mean + Normal(0, actor_noise_sigma)
```

eval 时使用 deterministic mean：

```text
a = mean
```

当前配置：

```yaml
actor_noise_sigma: 0.002
ref_action_dropout: 0.5
```

`ref_action_dropout` 只在 actor loss 训练时使用：随机把一部分 `a_tilde` 输入置零，避免 actor 单纯复制 VLA reference，而不利用 `z_rl/proprio`。

### 4.4 Twin-Q Critic 和 target networks

critic 是 twin-Q：

```text
Q1(x, a), Q2(x, a)
```

并维护 target actor 和 target critic：

```text
target_actor
q1_target
q2_target
```

每次 update 后软更新：

```text
target = (1 - tau) * target + tau * online
```

当前：

```yaml
tau: 0.005
gamma: 0.99
critic_actor_ratio: 2
```

## 5. Rollout 和 Env 数据流

### 5.1 Train rollout 主流程

训练时每个 chunk step 的数据流：

```text
EnvWorker sends obs + policy_info
  -> RolloutWorker.predict()
      -> RLTStage2Policy.predict_action_batch()
          -> prepare obs
          -> VLA embedding
          -> RL token z_rl
          -> VLA reference a_tilde
          -> actor action
          -> forward_inputs = {x, a_tilde, action, tokenized prompt}
      -> maybe expert intervention
      -> send action chunk to EnvWorker
  -> EnvWorker.chunk_step(action chunk)
      -> obs_list, rewards[10], dones[10], infos_list
      -> update local intervention state
      -> send next obs + optional step_obs to RolloutWorker
  -> EnvWorker packs ChunkStepResult
  -> end of rollout sends Trajectory to ActorWorker
```

`forward_inputs` 是 Stage2 replay 的关键缓存。它至少包含：

| key | 含义 |
| --- | --- |
| `x` | 当前 chunk boundary 的 `[z_rl, proprio]` |
| `a_tilde` | 当前 replay 使用的 reference chunk |
| `base_a_tilde` | intervention 替换前的 VLA reference |
| `action` | 当前实际执行 action chunk，flat `[80]` |
| `intervention_flags` | 每个 control step 是否 expert/human 接管 |
| `student_control` | 当前是否由 student actor 控制 |
| `ready_for_online` | actor 是否已经过 warmup gate |
| `online_gate_step` | rollout 看到的 actor update version |
| `intervention_requested` | env 是否请求 expert takeover |

### 5.2 Warmup gate

rollout worker 根据 actor 同步过来的 `update_step` 决定是否让 student actor 上线：

```text
ready_for_online = update_step >= td3_bc.warmup_updates
```

在 warmup 未完成前：

| 项 | 行为 |
| --- | --- |
| 动作执行 | 用 base VLA reference action |
| replay 收集 | 仍然收集 |
| actor/critic 训练 | actor worker 后台训练 |
| expert intervention | 强制不执行 expert takeover |
| eval | 也会根据同步版本决定 policy 行为 |

这点很重要：`runner.global_step` 不是 warmup gate；gate 用的是 actor worker 完成的 TD3 update 数。这样避免“runner step 到了但 actor 实际还没训练”的错配。

## 6. Replay 构造

Stage2 使用自定义 numpy replay buffer，只保存已经编码好的小特征和 chunk transition，不保存图像。

每条 replay transition：

```python
{
    "x": [state_dim],
    "a": [chunk_len * action_dim],
    "a_tilde": [chunk_len * action_dim],
    "rewards": [chunk_len],
    "next_x": [state_dim],
    "next_a_tilde": [chunk_len * action_dim],
    "dones": [1],
    "intervention": [chunk_len * action_dim],
}
```

### 6.1 旧路径：chunk-boundary replay

当：

```yaml
replay_subsample_stride: 0
```

actor worker 使用 `_chunk_trajectory_to_transitions()`。

构造方式：

```text
每个 action chunk 生成 1 条 replay transition
start = chunk boundary
end = next chunk boundary
x = forward_inputs[t].x
a_tilde = forward_inputs[t].a_tilde
a = trajectory.actions[t]
rewards = trajectory.rewards[t]
next_x = forward_inputs[t + 1].x
next_a_tilde = forward_inputs[t + 1].a_tilde
```

优点：

| 优点 | 说明 |
| --- | --- |
| 快 | 不需要额外编码中间 obs |
| 简单 | replay 和 rollout chunk boundary 一一对应 |
| 稳定 | 不引入额外 sparse anchor cache 复杂度 |

缺点：

| 缺点 | 说明 |
| --- | --- |
| 数据效率低 | 每 10 个 control step 只有 1 个 replay window |
| 和论文 stride/subsample 语义不完全一致 | 无法从中间 control step 起滑窗构造 transition |

### 6.2 论文版路径：stride replay

当：

```yaml
replay_subsample_stride: 2
```

actor worker 使用 `_step_trace_to_transitions()`。

目标语义：

```text
从真实执行的 control-step trace 中，每隔 stride 个 step 取一个窗口起点
窗口长度仍为 chunk_len = 10
action/reward/done/intervention 来自真实执行序列
x/a_tilde 和 next_x/next_a_tilde 来自对应窗口起点的 VLA/RL-token 编码
```

为了避免每个 control step 都重跑 VLA，当前实现做了工程优化：

| anchor 类型 | 处理方式 |
| --- | --- |
| chunk boundary | 正常 policy forward 已经缓存 `x/a_tilde`，直接复用 |
| 非 boundary stride anchor | EnvWorker 只发送稀疏 offset 的 obs，RolloutWorker 批量编码 |
| 非 stride offset | 不编码 |

以 `chunk_len=10, stride=2` 为例，非 boundary anchor 是：

```text
2, 4, 6, 8
```

boundary offset `0/10/20/...` 由正常 `forward_inputs` 复用。

相关字段：

| 字段 | 位置 | 含义 |
| --- | --- | --- |
| `EnvOutput.step_obs` | env -> rollout | 稀疏 anchor 的 obs |
| `RolloutResult.rlt_step_trace` | rollout -> env/actor | 稀疏 anchor 的 `x/a_tilde/anchor_offsets` |
| `Trajectory.rlt_step_trace` | actor replay conversion | 多个 chunk step stack 后的 sparse trace |

### 6.3 Terminal partial chunk

stride replay 只允许 terminal partial：

```yaml
replay_allow_terminal_partial: True
```

规则：

| 场景 | 是否允许 padding |
| --- | --- |
| episode terminal 导致窗口不足 10 步 | 允许，缺失 action/reward/intervention 补 0，`done=1`，next feature 用 terminal fallback |
| rollout 边界但 episode 没结束 | 不允许，直接跳过该 partial window |

这个规则是为了避免在非 terminal rollout 边界“凭空造未来动作/奖励”，否则 critic 会学到假的 transition。

### 6.4 Intervention reference replacement

RLT intervention 的当前语义是：expert/human 接管时，实际执行动作进入 replay，同时对应 control step 的 `a_tilde` 也替换成 expert/human action。

在 rollout 端，如果整条 env 样本当前 chunk 被 expert takeover：

```python
forward_inputs["a_tilde"] = where(expert_takeover, expert_action, base_a_tilde)
forward_inputs["base_a_tilde"] = base_a_tilde
forward_inputs["action"] = executed_action
forward_inputs["intervention_flags"] = True on takeover steps
```

在 actor replay 构造端，如果 chunk 内是 step-level mixed intervention：

```python
a_tilde_chunk[intervention_mask] = executed_action[intervention_mask]
```

因此 actor BC regularizer 的 target 是：

| step 来源 | BC target |
| --- | --- |
| student/base VLA | `a_tilde`，即 base VLA reference |
| expert/human intervention | executed expert/human action |
| mixed chunk | 非 intervention step 用 base reference，intervention step 用 executed expert action |

critic 始终学实际执行动作 `a`，不学未执行的 imagined action。

## 7. TD3 训练目标

### 7.1 Critic target

对每条 chunk transition，critic target 是：

```text
R_chunk = sum_{i=0}^{C-1} gamma^i * reward_i
next_a = target_actor(next_x, next_a_tilde) + fixed Gaussian noise
next_q = min(q1_target(next_x, next_a), q2_target(next_x, next_a))
target = R_chunk + gamma^C * (1 - done) * next_q
```

其中：

```text
C = num_action_chunks = 10
```

critic loss：

```text
L_Q = MSE(Q1(x, a), target) + MSE(Q2(x, a), target)
```

### 7.2 Actor loss

actor loss：

```text
L_actor = bc_weight * MSE(a, a_tilde) - q_weight * Q_min(x, a) + delta_weight * delta_loss
```

当前 `delta_weight: 0.0`，所以主要是 BC regularizer 和 Q objective。

actor update 时：

| 行为 | 当前实现 |
| --- | --- |
| actor action | `actor_forward(... deterministic=False, apply_action_noise=True)` |
| reference dropout | 如果 `ref_action_dropout > 0`，训练 actor loss 时启用 |
| critic gradient | actor 更新时冻结 online critic 参数，只让梯度回到 actor |
| actor update ratio | 每 `critic_actor_ratio` 次 critic update 更新一次 actor |

### 7.3 Actor loss schedule

当前权重 schedule：

```yaml
td3_bc:
  actor_loss_warmup_updates: 20000
  actor_loss_ramp_updates: 50000
  warmup_bc_weight: 7.0
  warmup_q_weight: 0.05
  online_bc_weight: 2.8
  online_q_weight: 0.16
```

语义：

| update 区间 | 行为 |
| --- | --- |
| `< 20000` | 高 BC、低 Q，actor 主要贴近 reference |
| `20000` 到 `20000 + 50000` | 线性从 warmup 权重 ramp 到 online 权重 |
| ramp 结束后 | 使用 online 权重 |

注意：`td3_bc.warmup_updates` 控制 actor 何时上线；`td3_bc.actor_loss_warmup_updates` 控制 loss 权重。两者单位都是 critic update，但作用不同。

## 8. 训练调度

当前 actor worker 的训练预算不是“每个 global step 固定训练 N 次”，而是根据 replay 新增 transition 数累积 `pending_update_budget`。

关键配置：

```yaml
update_epoch: 5
train_every_transitions: 50
max_updates_per_train_step: 1600
replay_buffer:
  min_buffer_size: 1000
td3_bc:
  warmup_updates: 30000
```

调度逻辑：

```text
buffer_ready = global_min_replay_size >= replay_buffer.min_buffer_size

首次 buffer_ready 时记录：
  warmup_ready_total_transitions
  warmup_ready_total_episodes

online_transitions_added =
  global_total_transitions_added - warmup_ready_total_transitions

transition_cycles =
  online_transitions_added // train_every_transitions

desired_total_updates =
  td3_bc.warmup_updates + transition_cycles * update_epoch

pending_update_budget =
  max(desired_total_updates - current_update_step, 0)
```

每个 runner step 最多实际执行：

```text
min(pending_update_budget, max_updates_per_train_step)
```

如果 `max_updates_per_train_step: 0`，表示不限制，actor worker 会尽量一次追完所有 pending budget。

### 8.1 pending_update_budget 的含义

`train/rlt_stage2/pending_update_budget` 是“理论上应该补但还没做完的 critic update 数”。它单调上升通常说明：

| 现象 | 可能原因 |
| --- | --- |
| rollout 产生 transition 太快 | env 数大、stride replay 产生 replay window 多 |
| learner 太慢 | actor worker 单卡、batch 大、VLA 不在 actor 但 critic/actor update 仍多 |
| `train_every_transitions` 太小 | 每新增少量 transition 就产生 update budget |
| `max_updates_per_train_step` 太小 | 每轮最多追一点，长期跟不上 |

如果 pending 长期增加，eval 看到的 policy 可能明显滞后于 train env 当前采样分布。此时 train success 虚高或 gap 变大时，要优先看：

```text
pending_update_budget
updates_scheduled
critic_updates_run
actor_updates_run
ready_for_online
global_total_transitions_added
```

## 9. Expert intervention 设计

当前 ManiSkill 中的 expert intervention 是用强 expert model 模拟 human takeover。它不是每一步都介入，而是在 peg insertion 的关键局部区域检测到偏离后介入。

### 9.1 Local correction 区域

EnvWorker 维护 per-env local policy state：

```python
{
    "intervention_region": bool,
    "expert_takeover": bool,
    "deviation": bool,
    "deviation_count": int,
    "takeover_left": int,
    "takeover_used": int,
    "prev_yz_error": float,
    "prev_hole_x": float,
}
```

进入 intervention region 的主要条件：

```text
grasp
and near_hole_x
and near_hole_yz
and not success
```

其中 `near_hole_yz` 会结合 peg head/body 到 goal 的 yz 距离、hole radius、以及是否已经 prealign/partial_insert。

### 9.2 Deviation 判定

进入局部纠偏区域后，以下情况会累计 deviation：

| 条件 | 含义 |
| --- | --- |
| `yz_worse` | yz error 比上一 chunk 增大超过 `yz_error_eps` |
| `no_x_progress` | hole x 方向推进小于 `progress_eps` |
| `~safe_yz` | y/z 超出安全 margin |
| `lost_grasp` | 已在 intervention region 里但丢 grasp |
| `moved_away_from_hole` | x 方向明显远离孔 |

连续 `deviation_patience` 个 chunk 满足 deviation 后，触发 expert takeover。

当前配置：

```yaml
deviation_patience: 2
takeover_chunks: 3
takeover_max_chunks: 6
safe_yz_margin: 1.5
progress_eps: 0.01
yz_error_eps: 0.005
near_hole_x_min: -0.06
near_hole_yz_margin: 2.0
exit_hole_x_min: -0.12
fallback_hole_radius: 0.035
```

### 9.3 Takeover 执行

RolloutWorker 只有在以下条件同时满足时才真正使用 expert action：

```text
mode == train
allow_expert
rollout.expert_model configured
runner.expert_ckpt_path configured
ready_for_online
policy_info["expert_takeover"] == True
```

如果 `ready_for_online == False`，env 可以计算 requested takeover，但 rollout 不会真的让 expert 接管。这是为了让 warmup 阶段只收 base VLA 数据，不让 expert intervention 提前改变 warmup 分布。

当前 expert model 配置：

```yaml
runner:
  expert_ckpt_path: ".../global_step_8000/actor"

rollout:
  expert_model:
    model_path: ${actor.model.model_path}
    act_as_vla_reference: True
```

`act_as_vla_reference: True` 表示 expert 只作为强 VLA reference，不加载 Stage2 actor；expert 权重路径沿用 DAgger 风格，放在 `runner.expert_ckpt_path`。

### 9.4 Intervention 指标

| 指标 | 含义 |
| --- | --- |
| `env/expert_intervention_requested_rate` | env 请求 expert takeover 的比例 |
| `env/expert_intervention_actual_rate` | rollout 实际执行 expert action 的比例 |
| `env/deviation_rate` | 当前 chunk local deviation 的比例 |
| `train/replay_buffer/intervention_rate` | replay buffer 中 intervention mask 的均值，分母是所有 replay action dim |

`replay_buffer/intervention_rate` 的分母不是 episode，也不是 chunk 数，而是 replay 里所有 action element。当前 `chunk_len=10, action_dim=8`，一条 transition 有 80 个 action element。按 step mask 展开后，某个 control step 介入会贡献 8 个 true element。

## 10. Eval 设计

当前 eval 配置：

```yaml
env:
  eval:
    total_num_envs: 256
    seed: 2026
    auto_reset: False
    use_fixed_reset_state_ids: True
    max_episode_steps: 500
    max_steps_per_rollout_epoch: 500
    action_exec_chunks: 10
```

eval 行为：

| 项 | 行为 |
| --- | --- |
| action | actor deterministic mean |
| expert | `allow_expert=False`，eval 不使用 expert |
| reset ids | 固定一批由 seed 生成的 `episode_id` |
| auto reset | false，episode done 后不会在同一 eval epoch 自动重开新 episode |
| action exec | 每次执行完整 10-step chunk |
| metric | 主要看 `eval/success_once` |

train 和 eval 指标不可直接等价：

| 差异 | 影响 |
| --- | --- |
| train 有 ongoing online data collection | train success 容易受当前采样分布、done 时序、expert/sample mix 影响 |
| eval 固定 reset ids | eval 更适合比较 checkpoint，但可能受这批 state 难度影响 |
| train 可能有 expert intervention | train env success 可能包含 expert 纠偏后的成功 |
| eval 不用 expert | eval 是 student actor 的真实表现 |
| train 统计是异步 rollout 过程 | 首批/早结束样本可能先贡献成功，难样本后贡献失败 |

因此判断 Stage2 是否真的 work，应优先看：

```text
eval/success_once
actor/action_ref_abs_mean
actor/bc_loss
critic/q1_mean, critic/q2_mean
train/rlt_stage2/pending_update_budget
env/student_control_rate
env/expert_intervention_actual_rate
```

## 11. 关键指标解释

| 指标 | 解释 | 典型用法 |
| --- | --- | --- |
| `env/success_once` | train env 中每个 episode 是否曾经 success | 观察采样分布和在线控制表现，但可能虚高 |
| `eval/success_once` | eval env 中每个 episode 是否曾经 success | 主要性能指标 |
| `env/student_control_rate` | 当前 rollout 中 student actor 控制比例 | 判断是否已经过 warmup gate |
| `env/rlt_ready_for_online` | rollout 是否认为 actor 已 ready | 对齐 warmup/update_step |
| `env/expert_intervention_requested_rate` | env 请求 intervention 的比例 | 判断触发器是否敏感 |
| `env/expert_intervention_actual_rate` | 实际 expert 接管比例 | 判断 intervention 是否真的执行 |
| `env/deviation_rate` | local correction 区域内偏离比例 | 判断是否进入可纠偏阶段 |
| `train/replay_buffer/size` | 当前 replay buffer 大小 | 判断是否超过 replay_buffer.min_buffer_size |
| `train/replay_buffer/fill_ratio` | buffer 填充比例 | 判断是否开始循环覆盖 |
| `train/replay_buffer/intervention_rate` | replay action element 级 intervention 比例 | 判断 expert 样本占比 |
| `train/rlt_stage2/critic_loss` | critic TD loss | 观察 critic 是否爆炸 |
| `train/critic/q1_mean`, `train/critic/q2_mean` | twin-Q 均值 | 看 Q 是否分叉或长期不上升 |
| `train/rlt_stage2/actor_loss` | actor 总 loss | 受 BC/Q 权重共同影响 |
| `train/actor/bc_loss` | actor action 与 BC target 的 MSE | 过快下降可能只是贴近 target，不代表 Q 学好 |
| `train/actor/action_ref_abs_mean` | actor 输出和 reference 的平均绝对偏差 | 判断 actor 是否偏离 base |
| `train/actor/q_mean` | actor action 的 critic value | 判断 actor 是否被 Q 推动 |
| `train/rlt_stage2/pending_update_budget` | 未追上的 critic update 预算 | 判断 learner 是否跟上数据 |
| `train/rlt_stage2/global_total_transitions_added` | 全局累计加入 replay 的 transition 数 | 判断 replay 生成速度 |

## 12. 配置要点和推荐检查

### 12.1 Checkpoint 对齐

必须同时确认：

| 项 | 要求 |
| --- | --- |
| Stage2 `actor.model.model_path` | base/student VLA checkpoint |
| Stage1 `actor.model.model_path` | 训练 RL token 时使用的 VLA checkpoint |
| Stage2 `rl_token_path` | 应来自同语义 VLA/data config 的 Stage1 |
| `norm_stats_path` | 和 OpenPI 数据、VLA checkpoint 对齐 |
| `policy_setup` | `panda-qpos` |
| `num_action_chunks/action_dim` | Stage1、Stage2、OpenPI config、env action 都一致 |

如果 RL token 来自不同 step 的 VLA，`z_rl` 表示和当前 base VLA 可能错配。表现上可能出现 early eval 虚高、warmup 不稳定、actor BC loss 异常或 eval 不提升。

### 12.2 Seed 和 reset ids

当前 train：

```yaml
seed: 20260604
use_fixed_reset_state_ids: False
```

当前 eval：

```yaml
seed: 2026
use_fixed_reset_state_ids: True
```

含义：

| 模式 | reset 行为 |
| --- | --- |
| train | seed 决定随机序列，但每轮不固定同一批 episode_id |
| eval | seed 先生成固定 episode_id 列表，不同 checkpoint 用同一批初始状态 |

如果 eval 只看一批固定 ids，适合 checkpoint 间比较；如果要估计泛化性能，应另开多 seed eval 或扩大 eval env 数。

### 12.3 Replay stride 取舍

| 配置 | 适用场景 |
| --- | --- |
| `replay_subsample_stride: 0` | 快速验证、速度优先、先确认模型和调度是否正常 |
| `replay_subsample_stride: 2` | 更接近论文滑窗数据构造，数据效率更高，但 rollout 更重 |

如果 `Generating Rollout Epochs` 明显变慢，优先看：

```text
replay_subsample_stride
replay_feature_batch_size
total_num_envs
num_action_chunks
rollout worker GPU memory
```

### 12.4 Online update ratio

当前预算近似：

```text
每新增 train_every_transitions 条 replay transition
  -> 增加 update_epoch 次 critic update 预算
```

当前配置：

```yaml
train_every_transitions: 50
update_epoch: 5
```

也就是每 50 条 replay transition 增加 5 次 critic update。stride replay 会显著增加 transition 数，因此同样的 env rollout 下，update budget 也会变多。

如果 pending 爆炸：

| 改法 | 影响 |
| --- | --- |
| 增大 `train_every_transitions` | 降低 update/data ratio，learner 更容易跟上 |
| 减小 `update_epoch` | 降低每个周期 update 数 |
| 增大 `max_updates_per_train_step` | 不改总预算，只让每轮多追一些 |
| 关掉 stride replay | transition 生成数回到 chunk boundary，速度更接近旧版 |
| 减少 train env 数 | 降低数据产生速度 |

## 13. 常见问题

### 13.1 为什么 train success 很高但 eval 不高？

优先排查：

| 原因 | 排查指标 |
| --- | --- |
| train 包含 expert intervention | `expert_intervention_actual_rate` |
| train 初始状态/统计时序虚高 | 多 seed eval，扩大 eval env |
| learner 跟不上 replay | `pending_update_budget` 是否长期上升 |
| actor 只是贴近 BC target，没有学到更高 Q | `bc_loss` 下降但 `q1/q2/actor_q` 不升 |
| eval 固定 seed 较难 | 更换 eval seed 或做多 seed 平均 |
| base VLA / RL token / norm stats 不对齐 | 检查 checkpoint 和 `norm_stats_path` |

### 13.2 为什么 BC loss 一开始很大，后面变小？

direct actor 是从随机初始化 MLP 直接输出 action chunk mean，不是 zero residual。因此训练初期 actor mean 不一定接近 `a_tilde`，BC loss 可以很大。随着 warmup 高 BC 权重训练，actor 会快速贴近 `a_tilde` 或 intervention 替换后的 target，BC loss 下降。

### 13.3 为什么 intervention rate 很低？

当前 intervention 只在 peg 已经 grasp、接近孔、局部偏离时触发，不是全程 expert。`replay_buffer/intervention_rate` 又是 action element 级均值，所以数值会被 `[chunk_len * action_dim]` 展开稀释。低 intervention rate 不一定异常，但如果 `requested_rate` 和 `actual_rate` 都接近 0，说明触发条件可能太严或 agent 根本没进入可纠偏区域。

### 13.4 为什么 ready_for_online 第一步可能掉性能？

warmup 阶段执行 base VLA；ready 后执行 actor mean/sample。即使 actor BC loss 已下降，direct actor 也可能在某些状态下和 base VLA 有小偏差，而 peg insertion 对插入阶段很敏感。需要同时看：

```text
actor/action_ref_abs_mean
actor/bc_loss
q1/q2 stability
student_control_rate
eval/success_once at step 0 and after online
```

### 13.5 是否应该提高 online_q_weight？

不一定。Q 信号弱时，提高 `online_q_weight` 会放大 critic 的错误方向，可能更快拉坏 actor。更稳的顺序通常是：

```text
先确认 critic target 和 replay 正常
再看 Q 是否能区分成功/失败
再逐步增加 q_weight 或降低 bc_weight
```

如果 Q 均值不上升、q1/q2 只是在噪声波动，提高 Q 权重通常不是第一选择。

## 14. 与 openpi-RLT/论文语义的对齐和差异

| 设计点 | 当前 RLinf 实现 | 对齐程度 | 可能影响 |
| --- | --- | --- | --- |
| frozen VLA | Stage2 wrapper 冻结 OpenPI | 对齐 | 降低在线训练成本 |
| frozen RL token | Stage2 只用 encoder encode | 对齐 | 依赖 Stage1 checkpoint 对齐 |
| actor 形式 | direct Gaussian actor | 对齐当前目标语义 | 初始不是 base residual，早期 BC loss 可较高 |
| train sample / eval mean | train 用 noise，eval deterministic | 对齐 Gaussian actor 语义 | eval 更稳定，train 有探索 |
| target actor | `target_actor` soft update | 对齐 TD3 | target 更稳定 |
| twin-Q critic | `Q1/Q2` + target copies | 对齐 TD3 | 降低 Q 过估计 |
| intervention replaces reference | intervention step 把 `a_tilde` 替换成 executed expert action | 对齐 Algorithm 1 语义 | actor 可向 expert correction 学，而不是继续贴 base |
| chunk reward target | chunk 内 discounted reward + bootstrap | 对齐 chunk-level critic | sparse reward 下 credit assignment 仍难 |
| stride replay | 可选 `replay_subsample_stride > 0` | 更接近论文 | 更重，可能拖慢 rollout |
| sparse anchor cache | 只编码非 boundary stride anchor | 工程优化 | 比全量 step obs 编码快，但实现更复杂 |
| actor-only train model | actor worker 不加载 VLA/RL-token | RLinf 工程差异 | 依赖 rollout 缓存完整、正确 |
| eval fixed reset ids | 固定 seed 生成 eval episode ids | 工程评估选择 | checkpoint 可比，但需多 seed 看泛化 |

## 15. 文件地图

| 文件 | 作用 |
| --- | --- |
| `examples/embodiment/config/rlt_stage2_maniskill_joint.yaml` | Stage2 主配置 |
| `examples/sft/config/rlt_stage1_maniskill_joint.yaml` | Stage1 RL token 配置 |
| `examples/sft/train_rlt_stage1.sh` | Stage1 启动脚本 |
| `examples/embodiment/run_embodiment.sh` | Stage2/embodied 任务启动脚本 |
| `examples/embodiment/train_embodied_agent.py` | 根据 `loss_type` 创建 actor/rollout/env worker |
| `rlinf/envs/maniskill/maniskill_env.py` | ManiSkill obs/reward/reset/intervention info |
| `rlinf/workers/env/env_worker.py` | chunk_step、policy_info、step_obs trace、trajectory packing |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | RLT predict、expert takeover、sparse anchor encoding、eval |
| `rlinf/workers/actor/fsdp_rlt_stage2_policy_worker.py` | replay conversion、training schedule、TD3 update、weight sync |
| `rlinf/data/embodied_io_struct.py` | EnvOutput/RolloutResult/Trajectory 数据结构 |
| `rlinf/models/embodiment/rlt_stage1/rlt_stage1_policy.py` | Stage1 RL token reconstruction training |
| `rlinf/models/embodiment/rlt_stage2/rlt_stage2_policy.py` | Stage2 policy 接口 |
| `rlinf/models/embodiment/rlt_stage2/components.py` | direct actor、twin-Q critic、loss、TD target |
| `rlinf/models/embodiment/rlt_stage2/replay_buffer.py` | numpy circular replay buffer |
| `rlinf/models/embodiment/rlt_stage2/vla_wrapper.py` | OpenPI feature/reference wrapper |
| `rlinf/models/embodiment/rlt_stage2/rl_token.py` | RL token encoder/decoder |
| `rlinf/models/embodiment/openpi/dataconfig/rlt_maniskill_joint_dataconfig.py` | ManiSkill joint LeRobot/OpenPI schema |

## 16. 最小复现实验检查表

开始一组新实验前建议逐项确认：

| 检查项 | 期望 |
| --- | --- |
| Stage2 base VLA checkpoint | 和实验设定一致，例如 step1000 或 step2000 |
| RL token checkpoint | 来自同语义 base VLA/data config |
| `norm_stats_path` | 指向 joint ManiSkill 数据集对应 stats |
| `num_action_chunks/action_dim` | 10/8，全链路一致 |
| train seed | 显式设置，且 `use_fixed_reset_state_ids: False` |
| eval seed | 显式设置，且 `use_fixed_reset_state_ids: True` |
| `eval_before_train` | 如果要记录 base baseline，设为 true |
| `td3_bc.warmup_updates` | actor 上线前足够训练 |
| `pending_update_budget` | 不应长期无限上升 |
| `expert_intervention_actual_rate` | ready 后非零才说明 intervention 真执行 |
| `replay_buffer/intervention_rate` | 低值正常，但应和 actual_rate 同趋势 |
| `q1/q2` | 不应明显分叉或爆炸 |
| `actor/action_ref_abs_mean` | 不应突然大幅飙升 |
| `eval/success_once` | 最终以 eval 为准，不以 train env success 单独下结论 |
