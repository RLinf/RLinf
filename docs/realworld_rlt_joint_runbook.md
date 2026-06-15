# Realworld RLT Joint Runbook

这是一份临时操作文档，用来说明当前 RLinf 里的 Franka 真机 joint-control RLT 流程。它参考了现有文档和当前代码：

- `docs/source-zh/rst_source/tutorials/embodied/realworld_robot.rst`
- `docs/source-zh/rst_source/examples/embodied/franka.rst`
- `docs/source-zh/rst_source/examples/embodied/franka_pi0_sft_deploy.rst`
- `docs/source-zh/rst_source/examples/embodied/maniskill_rlt_joint.rst`
- `examples/embodiment/config/rlt_stage2_realworld_joint.yaml`
- `examples/embodiment/config/env/realworld_rlt_joint_peg_insertion.yaml`

本文档面向这条链路：

```text
realworld joint LeRobot data
  -> OpenPI pi0.5 joint SFT
  -> SFT realworld eval
  -> Stage1 RL-token training
  -> Stage2 realworld online RLT
```

## 0. 这套东西在干什么

RLT realworld joint 分成几步：

```text
SFT:
  用真机采集的 joint 数据，把 pi0.5 VLA 训成会输出 8D joint action 的策略。

SFT eval:
  把 SFT 出来的 actor/ checkpoint 直接拿到真机上跑，确认方向和动作尺度没错。

Stage1:
  离线训练 RL-token encoder/decoder。产物是 rl_token_model.pt。

Stage2:
  真机在线 RL。冻结 SFT VLA 和 Stage1 RL-token，只训练一个小 actor-critic。
  rollout 侧和真机交互，actor 侧从 replay buffer 里做 TD3 更新。
```

当前 realworld Stage2 配置没有用离线 demo buffer。它会先在线 warmup 收集 replay，等 replay 到 `algorithm.warmup_min_size` 后再开始训练。

## 1. 真机硬件怎么接

典型拓扑是两台机器：

```text
GPU 训练节点，通常是 Ray head，RLINF_NODE_RANK=0
  - 跑 actor
  - 跑 rollout
  - 提交训练入口脚本

机器人控制节点，通常是 NUC，RLINF_NODE_RANK=1
  - 和 Franka 在同一局域网
  - 跑 env worker
  - 跑 FrankaController
  - 接 RealSense、GELLO、键盘
```

物理连接按这个顺序检查：

1. Franka 控制柜和控制节点在同一局域网。控制节点浏览器能打开 `http://<robot_ip>/desk`。
2. GPU 节点和控制节点互通。GPU head 能 SSH 到控制节点，控制节点能连到 `<head_ip>:6379`。
3. RealSense 相机接控制节点的 USB 3 口，不要接到 GPU head，除非你专门改了相机/控制分机配置。
4. GELLO 接控制节点，串口通常是 `/dev/ttyUSB0` 或 `/dev/serial/by-id/...`。
5. 键盘奖励用的物理键盘也要接控制节点，因为 env worker 在控制节点上读 `/dev/input/eventX`。
6. Franka 急停、Desk 页面、控制节点终端都要在操作者伸手够得到或旁边有人能立刻操作的位置。

当前 joint-target RLT 配置最适合 GELLO。SpaceMouse 默认输出末端位姿 delta，不适合直接接到这个 8D absolute joint action 环境，除非另外改 wrapper/action 逻辑。

## 2. 上真机前的现场检查


1. 清空机械臂工作空间，确保 peg、hole、相机、线缆不会被手臂扫到。
2. 在 Franka Desk 里确认机器人无 error，并切到可编程控制模式。
3. 手动拖动机械臂到安全位置，确认没有明显机械干涉。
4. 确认 `target_ee_pose` 是当前孔位附近的目标位姿，`reset_joint_qpos` 是安全复位关节位姿。
5. 确认 `joint_command_topic` 是你的控制器实际监听的 topic，默认是 `/joint_states_gripper`。
6. 第一次 smoke 时把单步关节变化调小，例如 `max_joint_delta=0.03`。
7. stage2 现场要同时盯机器人和日志/键盘奖励。别让真机无人值守在线探索。

获取目标位姿可以参考现有 Franka 文档里的工具。在控制节点上：

```bash
export FRANKA_ROBOT_IP=<Franka IP>
python -m toolkits.realworld_check.test_franka_controller
```

脚本提示后输入 `getpos_euler`，把当前末端位姿写进配置或作为 Hydra override：

```bash
env.train.override_cfg.target_ee_pose='[0.5,0.0,0.1,-3.14,0.0,0.0]'
env.eval.override_cfg.target_ee_pose='[0.5,0.0,0.1,-3.14,0.0,0.0]'
```

检查相机：

```bash
python -m toolkits.realworld_check.test_franka_camera
```

检查 GELLO 串口：

```bash
ls /dev/serial/by-id/
python rlinf/envs/realworld/common/gello/gello_expert.py --port /dev/serial/by-id/<your-gello-port>
```

如果这个 GELLO 测试读不到数据，不要开 `env.train.use_gello=True`。

## 3. Ray 集群启动

Ray 会在 `ray start` 时捕获 Python、ROS、环境变量。所有和机器人、键盘、网络相关的变量都要在 `ray start` 前设置。

GPU head 节点：

```bash
cd /path/to/RLinf
source ray_utils/realworld/setup_before_ray.sh
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=<head_ip>
```

机器人控制节点：

```bash
cd /path/to/RLinf
source ray_utils/realworld/setup_before_ray.sh
source <your_catkin_ws>/devel/setup.bash
export RLINF_NODE_RANK=1
export RLT_REALWORLD_ROBOT_IP=<Franka IP>
export RLT_REALWORLD_GELLO_PORT=/dev/serial/by-id/<your-gello-port>
ray start --address=<head_ip>:6379
```

stage2要用键盘人工奖励，先在控制节点找键盘 event：

```bash
ls -l /dev/input/by-id/*-event-kbd
sudo chmod 666 /dev/input/event20
export RLINF_KEYBOARD_DEVICE=/dev/input/event20
```

这里的 `/dev/input/event20` 要替换成实际键盘。`RLINF_KEYBOARD_DEVICE` 也必须在控制节点 `ray start` 前设置；否则 env worker 读不到按键。

如果要跑 `env.train.task_mode=full_task`，同一个键盘还会用于进入 critical phase。默认进入 critical phase 的按键是 `v`，不是 `c`，避免和成功奖励键冲突。

确认集群：

```bash
ray status
```

`rlt_stage2_realworld_joint.yaml` 默认是 `cluster.num_nodes: 2`，GPU 是 rank 0，realworld 控制节点是 rank 1。

如果两个节点不共享同一个 RLinf 代码路径，在 head 节点提交训练前加：

```bash
export RLINF_CODE_WORKING_DIR=auto
```

## 4. 准备路径变量

下面几个变量贯穿 SFT、Stage1、Stage2。按实际路径改：

```bash
export DATASET_ROOT=/mnt/public2/xiekaizhi/rlt-openpi-sim/data/collected_data_id3_id4/rank_0/id_4
export NORM_STATS_PATH=${DATASET_ROOT}/norm_stats.json
export BASE_PI05_PATH=/mnt/public2/xiekaizhi/rlt-openpi-sim/pi05_base

export SFT_ACTOR_PATH=/mnt/public2/xiekaizhi/rlt-openpi-sim/clean-RLinf/RLinf-dev/logs/20260610-10:20:02-rlt_realworld_joint_pi05_sft/rlt_realworld_joint_pi05_sft/checkpoints/global_step_1000/actor
export STAGE1_RL_TOKEN_PATH=/path/to/rlt_stage1_realworld_joint/checkpoints/global_step_5000/actor/rl_token/rl_token_model.pt

export RLT_REALWORLD_ROBOT_IP=<Franka IP>
export RLT_REALWORLD_GELLO_PORT=/dev/serial/by-id/<your-gello-port>
```

`DATASET_ROOT` 必须是 LeRobot 数据目录，里面应该有 `data/`、`meta/`，并且 `norm_stats.json` 要和这份数据匹配。

SFT/Stage1/Stage2 用的是不同东西：

```text
BASE_PI05_PATH:
  pi0.5 base model，只用于 SFT 起点。

SFT_ACTOR_PATH:
  SFT 训练后的 checkpoints/global_step_xxx/actor 目录。
  SFT eval、Stage1、Stage2 都用这个目录。

STAGE1_RL_TOKEN_PATH:
  Stage1 产出的 actor/rl_token/rl_token_model.pt。
  只给 Stage2 的 rlt_stage2.rl_token_path 用。
```

不要把 `full_weights.pt` 单文件传给 Stage2。这里应传 `actor/` 目录。

## 5. SFT 训练

SFT 使用配置：

```text
examples/sft/config/rlt_realworld_joint_pi05_sft.yaml
```

如果配置文件里路径已经改好，可以直接跑：

```bash
bash examples/sft/run_vla_sft.sh rlt_realworld_joint_pi05_sft
```


SFT 关键产物是：

```text
logs/<time>/rlt_realworld_joint_pi05_sft/checkpoints/global_step_xxx/actor
```

后续 eval、Stage1、Stage2 都应使用这个 `actor/` 目录。

## 6. SFT 真机 eval 怎么跑

先评测 SFT。SFT eval 用配置：

```text
examples/embodiment/config/rlt_realworld_joint_pi05_sft_eval.yaml
```

第一次 eval 建议把动作幅度调小，并且只跑少量 episode。`run_realworld_eval.sh` 可以透传 Hydra overrides：

```bash
bash examples/embodiment/run_realworld_eval.sh rlt_realworld_joint_pi05_sft_eval \
  env.eval.override_cfg.max_joint_delta=0.03 \
  env.eval.max_episode_steps=30 \
  env.eval.max_steps_per_rollout_epoch=30
```

现场操作：

1. 操作者站在急停/Desk 旁边，看日志。
2. 启动命令后，机器人会先 reset 到 `reset_joint_qpos`。
3. 看第一段动作方向。只要出现反向、突然大幅摆动、接近撞击，立刻停。
4. 如果 SFT 能把 peg 大致带到孔附近，再考虑 Stage2。
5. 如果 SFT eval 都明显不对，别上 Stage2。maniskill里1000步sft成功率约0.35，8000步约0.90.

SFT eval 配置默认关闭 `use_gello` 和 `use_spacemouse`，这一步是纯 policy eval，不需要人在环。

如果 SFT eval 方向明显不对，优先检查：

- `actor.model.model_path` 是否指向 SFT 的 `actor/` 目录。
- `norm_stats_path` 是否是同一份数据的统计。
- `actor.model.openpi.config_name` 是否是 `pi05_rlt_realworld_joint`。
- `action_dim` 是否为 `8`。
- `env.eval.action_exec_chunks` 是否为 `10`。
- `env.eval.override_cfg.max_joint_delta` 是否太大。

## 7. Stage1 RL-token

Stage1 使用配置：

```text
examples/sft/config/rlt_stage1_realworld_joint.yaml
```

入口脚本：

```text
examples/sft/train_rlt_stage1.sh
```

```bash
bash examples/sft/train_rlt_stage1.sh rlt_stage1_realworld_joint
```

Stage1 产物是：

```text
logs/<time>/rlt_stage1_realworld_joint/checkpoints/global_step_xxx/actor/rl_token/rl_token_model.pt
```

把这个路径写到：

```bash
export STAGE1_RL_TOKEN_PATH=/path/to/rl_token_model.pt
```

Stage1 是离线训练，不需要连真机。

## 8. Stage2 在线真机训练怎么跑

Stage2 使用配置：

```text
examples/embodiment/config/rlt_stage2_realworld_joint.yaml
```

入口是

```bash
bash examples/embodiment/run_realworld.sh rlt_stage2_realworld_joint
```

但 `examples/embodiment/run_realworld.sh` 不透传额外 Hydra 参数。需要覆盖 `use_gello`、`keyboard_reward_wrapper`、`warmup_min_size`、`max_joint_delta` 时，用 Python 入口。

默认配置是 `env.train.task_mode=critical_phase`。意思是每个 episode reset 后就认为已经在关键阶段，Stage2 actor 可以在 warmup 结束后接管，replay 也会直接记录。这适合先把机器人 reset 到孔口附近，只训练对孔/插入这一段。

第一次建议跑短 smoke。开 GELLO 人在环、键盘奖励、降低 `max_joint_delta`：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name rlt_stage2_realworld_joint \
  env.train.use_gello=True \
  env.train.gello_action_mode=joint_target \
  env.train.task_mode=critical_phase \
  env.train.keyboard_reward_wrapper=single_stage \
  env.train.override_cfg.max_joint_delta=0.03 \
  env.eval.use_gello=False \
  env.eval.keyboard_reward_wrapper=single_stage \
  env.eval.override_cfg.max_joint_delta=0.03 \
  algorithm.warmup_min_size=10 \
  algorithm.warmup_post_collect_updates=10 \
  runner.max_epochs=3 \
  runner.logger.log_path=logs/rlt_stage2_realworld_smoke
```

现场操作顺序：

1. 启动命令前，把 peg/hole 放回初始状态。
2. 操作者手放在 GELLO 附近，但不要乱动；另一个人负责按键奖励。
3. 每个 episode 开始时，机器人会 reset 到 `reset_joint_qpos`，然后执行 policy/GELLO 覆盖后的 joint target。
4. 如果 policy 走得还行，就别接管，让它自己产生在线数据。
5. 如果机器人明显偏、快撞、卡住、peg 到孔口但对不准，用 GELLO 接管几步。
6. 成功插入后按 `c`。失败、危险或已经无意义继续时按 `a`。不想结束但要给中性反馈时按 `b`。
7. episode 结束后把实物恢复到下一个 episode 的初始状态，再让 reset/下一条继续。
8. smoke 结束后看视频、reward、replay buffer、intervention 相关指标，再决定是否正式跑。

确认硬件、奖励、接管、视频和 replay 都正常后，再跑正式版：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name rlt_stage2_realworld_joint \
  env.train.use_gello=True \
  env.train.gello_action_mode=joint_target \
  env.train.task_mode=critical_phase \
  env.train.keyboard_reward_wrapper=single_stage \
  env.eval.use_gello=False \
  env.eval.keyboard_reward_wrapper=single_stage \
  runner.logger.log_path=logs/rlt_stage2_realworld
```

等自动成功判定稳定、SFT 和 Stage2 都不会乱动后，再考虑关闭键盘奖励或把 `max_joint_delta` 恢复到默认 `0.08`。

### Stage2 的 `critical_phase` 和 `full_task`

当前 RLinf realworld Stage2 支持两个模式：

```yaml
env:
  train:
    task_mode: critical_phase  # 或 full_task
    critical_phase_key: v
    record_prefix_before_critical_phase: false
```

`critical_phase`：

- reset 后立刻 `in_critical_phase=True`。
- warmup 阶段仍然跑 SFT/base/reference 动作，warmup 结束后 Stage2 actor 可以控制。
- replay 从 episode 一开始就记录。
- 适合先把 `reset_joint_qpos` 设到孔口附近，只训练对孔/插入阶段。

`full_task`：

- reset 后先是非关键 prefix，`in_critical_phase=False`。
- 非关键 prefix 强制执行 SFT/base/reference 动作，不让 Stage2 actor 控制。
- 默认 `record_prefix_before_critical_phase=false`，所以 prefix 不写入 Stage2 replay。
- 操作者看到 peg 到孔口附近、需要 RL actor 学插入时，按 `critical_phase_key`，默认是 `v`。
- 按下 `v` 后，从下一次 policy chunk 开始 `in_critical_phase=True`，warmup 结束后 actor 才会接管，replay 开始记录。

这和 openpi-RLT 的 full-task 语义一致：full-task 不是“整段都当成 critical phase”。它是先让 base model 完成非关键前缀，到 critical phase 后再切到 RL policy，并且默认不把前缀写入 replay。

要跑 full-task smoke：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name rlt_stage2_realworld_joint \
  env.train.use_gello=True \
  env.train.gello_action_mode=joint_target \
  env.train.task_mode=full_task \
  env.train.critical_phase_key=v \
  env.train.keyboard_reward_wrapper=single_stage \
  env.train.override_cfg.max_joint_delta=0.03 \
  env.eval.task_mode=full_task \
  env.eval.use_gello=False \
  env.eval.keyboard_reward_wrapper=single_stage \
  env.eval.override_cfg.max_joint_delta=0.03 \
  algorithm.warmup_min_size=10 \
  algorithm.warmup_post_collect_updates=10 \
  runner.max_epochs=3 \
  runner.logger.log_path=logs/rlt_stage2_realworld_full_task_smoke
```

full-task 现场流程：

1. episode reset 到任务起始位姿，SFT/base 先走非关键 prefix。
2. 操作者不要按 `v`，直到 peg 已经接近孔口或到你定义的 critical phase 边界。
3. 到边界后按住/点按 `v` 一下，日志会打印 critical phase entered。
4. 下一段 action chunk 开始，rollout 会允许 actor 控制，replay 也开始记录。
5. 如果插入成功，按 `c` 给成功奖励并结束 episode；如果失败或危险，按 `a`。

注意：full-task 的切换以 action chunk 为粒度，不是单个控制 step 立刻切。默认 `num_action_chunks=10`，所以按 `v` 后最多晚一个 chunk 才切到 actor。这是为了安全地保持控制流稳定。

如果要让 full-task 的非关键 prefix 也进入 replay，可以设：

```bash
env.train.record_prefix_before_critical_phase=true
```

但这通常不建议，因为 prefix 会稀释 critical phase 的学习信号。

## 9. 人在环具体怎么操作

**连续几次卡住**的时候，需要人在环纠偏。人在环是通过 wrapper 实现的。以 GELLO joint-target 为例：

```text
policy 给出 8D action
  -> GELLO wrapper 读取 GELLO 当前 7D joint target + gripper
  -> 如果 GELLO action 和 policy action 不一致，用 GELLO action 覆盖 policy action
  -> env.step 执行覆盖后的 action
  -> info["intervene_action"] 记录人的动作
  -> Stage2 replay 保存 intervention/source
```

对操作者来说，要记住三件事：

1. `use_gello=True` 是在线指导模式，不是无人值守模式。GELLO 当前姿态会参与 action 覆盖，操作者必须盯着。
2. 当前 joint-target wrapper 没有“按住按钮才接管”的显式开关。只要 GELLO 读数和 policy action 差异超过阈值，就会覆盖，覆盖状态会持续约 0.5 秒。
3. 真正要教的是 critical phase：peg 接近孔、对孔、开始插入、卡住需要微调这几步。远离目标的普通移动如果 policy 能做，就尽量让 policy 自己做。

什么情况该接管（连续几次卡住）：

- 机器人要碰到夹具、桌面、相机、线缆。
- peg 明显偏离孔，继续走只会失败。
- peg 到孔口附近但角度/位置不准。
- policy 来回抖动或卡住。

什么情况不要接管：

- policy 正在稳定接近目标。
- 只是慢，但方向对。
- 不确定 GELLO 当前关节姿态是否和机器人安全姿态一致。

代码里现在有 `env.train.task_mode` 和 `env.train.critical_phase_key`。这里的 critical phase 既是操作概念，也是训练开关：它决定 Stage2 actor 什么时候能接管，以及哪些 transition 会写入 replay。

## 10. 成功奖励怎么给

当前 `FrankaJointPegInsertionEnv` 默认是稀疏奖励：

```text
自动成功: reward = 1
未成功: reward = 0
```

自动成功判定看这些字段：

```yaml
env:
  train:
    override_cfg:
      target_ee_pose: [...]
      reward_threshold: [0.015, 0.015, 0.03, 0.2, 0.2, 0.2]
      check_orientation_success: false
      success_hold_steps: 1
```

当前 joint peg insertion 默认只看 xyz 是否进入 `reward_threshold`，因为 `check_orientation_success: false`。

刚开始跑真机时，建议打开人工键盘奖励：

```bash
env.train.keyboard_reward_wrapper=single_stage
```

按键含义：

```text
c = 成功，reward=1，结束当前 episode
a = 失败，reward=-1，结束当前 episode
b = reward=0，不结束
```

注意：键盘奖励读的是控制节点上的 Linux input 设备，不是 GPU head 终端的标准输入。要在控制节点 `ray start` 前设置：

```bash
export RLINF_KEYBOARD_DEVICE=/dev/input/event20
```

什么时候按：

1. peg 明确插进孔，按 `c`。
2. 机器人已经偏到不可能成功、碰到夹具、或你打算中断本 episode，按 `a`。
3. 你想继续这个 episode，只是当前还没成功，通常不用按；需要明确给 0 时按 `b`。

如果 `target_ee_pose` 还没有完全校准，人工奖励会比自动奖励更稳。等自动 success threshold 校准好了，再考虑关闭键盘奖励。

## 11. 每次运行前后的现场 SOP

运行前：

1. Franka Desk 无 error，机器人可编程模式已打开。
2. 控制节点已 source ROS workspace，并在 `ray start` 前设置 `RLINF_NODE_RANK=1`。
3. `ray status` 看到两个节点。
4. `RLT_REALWORLD_STAGE2_BASE_PATH` 指向 SFT `actor/` 目录。
5. `RLT_REALWORLD_STAGE1_RL_TOKEN_PATH` 指向 Stage1 `rl_token_model.pt`。
6. GELLO 串口能读数。
7. 键盘 event 可读，`RLINF_KEYBOARD_DEVICE` 已设置。
8. 如果跑 full-task，确认 `critical_phase_key` 默认 `v` 没有和奖励键 `a/b/c` 冲突。
9. `max_joint_delta` 第一次 smoke 用 `0.03`。
10. peg/hole 和桌面上所有物体在相机视野内，且没有线缆进入机械臂路径。

运行中：

1. 一个 episode 一个 episode 地看，不要离开。
2. 看机器人实际动作，不要只看日志。
3. 成功及时按 `c`，明显失败及时按 `a`。
4. 需要接管时接管 critical phase，接管后尽快让 policy 自己走。
5. 出现异常速度、异常方向、控制器报错、相机掉帧，停止当前 run，先排硬件。

运行后：

1. 看 `logs/.../video/train` 或 `logs/.../video/eval`。
2. 看 `reward` 是否符合现场按键和实际成功。
3. 看 replay 是否增长。
4. 看 intervention 是否被记录。接管时应有 `intervene_action`。
5. 如果 smoke 有问题，不要扩大训练步数。

## 12. 常见坑


`RLT_REALWORLD_STAGE2_BASE_PATH` 应该指向 SFT 的 `actor/` 目录，不是 `rl_token_model.pt`。

`RLT_REALWORLD_STAGE1_RL_TOKEN_PATH` 才是 Stage1 的 `rl_token_model.pt`。

`RLINF_NODE_RANK`、ROS workspace、`RLINF_KEYBOARD_DEVICE`、GELLO 串口等环境变量要在 `ray start` 前设置。

`use_gello=True` 时必须设置 `gello_port`，例如：

```bash
export RLT_REALWORLD_GELLO_PORT=/dev/serial/by-id/<your-gello-port>
```

当前 joint-target 配置优先用 GELLO。不要直接把 SpaceMouse 当 joint-target 人在环来用。

如果 reward 一直是 0，优先检查 `target_ee_pose`、`reward_threshold`、`check_orientation_success`，或者临时打开 `keyboard_reward_wrapper=single_stage`。

如果按键没反应，优先检查键盘是不是接在控制节点、`RLINF_KEYBOARD_DEVICE` 是否在控制节点 `ray start` 前设置、event 设备是否有读权限。

如果 full-task 一直没有 actor 接管，先看日志/指标里的 `env/rlt_in_critical_phase` 和 `env/rlt_record_transition`。它们一直是 0，说明还没按到 `critical_phase_key`，或者键盘 event 没被 env worker 读到。

如果 worker 找不到 Franka/ROS 包，多半是控制节点在 `ray start` 前没有 source ROS workspace。

如果机器人 reset 都不安全，先修 `reset_joint_qpos`，不要继续训练。

## 13. 后续 Stage2 resume

Stage2 checkpoint 会在日志目录下保存：

```text
logs/<time>/checkpoints/global_step_xxx/actor
```

如果中断后继续训练，用：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name rlt_stage2_realworld_joint \
  runner.resume_dir=/path/to/checkpoints/global_step_xxx \
  runner.logger.log_path=logs/rlt_stage2_realworld_resume
```

当前 Stage2 worker 会恢复 actor checkpoint，也会恢复 replay buffer 状态。
