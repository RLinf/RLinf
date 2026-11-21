# Tianji HIL 四大工作流操作手册

本手册基于 `RLinf@tianji_hil` 代码（[tianji_env.py](../../rlinf/envs/realworld/tianji/tianji_env.py)、[tianji_hil.py](../../rlinf/envs/realworld/tianji/tianji_hil.py)、[tianji_episode_archive.py](../../rlinf/envs/realworld/tianji/tianji_episode_archive.py)）与版本化启动器（[deploy/tianji/](../../deploy/tianji/README.md)）整理四个核心工作流的启动逻辑、状态机与归档规则。

> [!NOTE]
> 运行实机前，请先复制 [deploy/tianji/site.env.example](../../deploy/tianji/site.env.example) 为未跟踪的 `site.env` 并填入本机路径与 ID；`tianji1` 上既有的 `~/Stage2-30HZ-*` 桌面目录在操作员主动切换前保持原样不变。

---

## 0. 通用前置条件：启动 LoopInfra 采集器

在运行任何 RLinf 工作流前，先通过 `deploy/tianji/` 完成采集器与环境自检：
1. 执行离线预检：`bash deploy/tianji/check_tianji_local.sh --mode offline`。
2. 启动/复用 LoopInfra 采集器（内部使用 `env -u PYTHONPATH` 隔离，默认监听 `http://127.0.0.1:8780`）：
   ```bash
   bash deploy/tianji/run_tianji_wuji_collector.sh
   ```
3. 在采集器 UI 确认双臂已连接并处于 `Position/EEF` 就绪状态，三路相机（`head`, `wrist_A`, `wrist_B`）同步流正常，保存的复位位姿（`loopinfra.reset_pose.v1`）已加载。
4. 确认契约环境变量（`LOOPINFRA_URL`, `LOOPINFRA_ROOT`, `TIANJI_CONTROL_HZ`, `TIANJI_ACTION_HORIZON`, `RLT_REF_OFFSET`, `RLINF_REALWORLD_PLATFORM=tianji`, `RLINF_SKIP_REALWORLD_SETUP=1`）已由 `site.env` / `tianji_local.env` 统一导出。

---

## 1. 模型推理（Model Inference）

用于纯策略评估/实机演示，**不启动训练更新、不计算 RL 奖励、不支持丢弃样本**。

### 路径 A：Stage2 Joint VLA / Actor 推理（推荐框架）
- **入口**：[evaluations/eval_embodied_agent.py](../../evaluations/eval_embodied_agent.py)（或 [run_tianji_stage1_local.sh](../../deploy/tianji/run_tianji_stage1_local.sh) / [run_tianji_stage2_infer_local.sh](../../deploy/tianji/run_tianji_stage2_infer_local.sh) / [stage2_30hz_1118v1.sh](../../deploy/tianji/stage2_30hz_1118v1.sh) `--run`）
- **配置**：
  - 仅基座 VLA（无 Residual Actor）：[tianji2_joint_stage1.yaml](../../examples/embodiment/config/tianji2_joint_stage1.yaml)（`rollout.joint_vla_only: true`）
  - VLA + 已训练 Residual Actor：[tianji2_rlt_stage2_infer.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_infer.yaml)（`rollout.joint_actor_eval: true`）
- **关键参数**：
  - `runner.only_eval=true`，`runner.deploy_actor_updates=false`
  - `env.eval.override_cfg.hil_mode=inference`
  - `TIANJI_ACTION_SPACE=joint`，`TIANJI_RT_SCHEDULE=1`（通过 `/api/policy_schedule` 下发 16D 关节块）
  - 小车 15/6 子阶段部署可通过 `TIANJI_CAR_NUM_SUBTASKS`（`15` 或 `6`）与 `TIANJI_CAR_BANNER` 参数化切换。

### 路径 B：Legacy Stage1 独立推理进程
- **入口**：`python -m rlinf.envs.realworld.tianji.stage1`（[stage1/__main__.py](../../rlinf/envs/realworld/tianji/stage1/__main__.py)）或 `TIANJI_ACTION_SPACE=eef bash deploy/tianji/run_tianji_stage1_local.sh`
- **说明**：使用 `hil_mode="stage1"` 与独立 TCP 服务（默认端口 `6110`），面板标题为 `HIL · Stage1 推理`，支持无机械臂 `check` 模式与物理实机 `execute` 模式。

---

## 2. HIL 数据收集（HIL Data Collection）

由人类操作员通过 PICO 遥操（在 LoopInfra 侧完成）或人工接管直接采集高质量示范/修正轨迹，**RLinf 不接入独立 Pico wrapper**（由 LoopInfra 透传）。

- **采集方式 1（纯人工 VR 采集）**：直接在 `LoopInfra` 采集器 UI（`http://127.0.0.1:8780/collector`）进行，PICO A 开始、B 结束并复位。
- **采集方式 2（RLinf 托管归档 `TianjiEpisodeArchive`）**：
  - 环境配置：[realworld_tianji.yaml](../../examples/embodiment/config/env/realworld_tianji.yaml) 配合 `override_cfg`：
    - `backend=loopinfra`
    - `hil_control=true`
    - **`hil_mode=training` 或 `hil_mode=dagger`**（`tianji_env.py` 已解耦为在 `training` 与 `dagger` 两种模式下均启用完整 `TianjiEpisodeArchive`）
    - `episode_archive=true`，`episode_archive_image_hz=15`，`tactile_raw_archive=true`
    - `recorder_dir=${runner.logger.log_path}/hil`
  - 若需同时触发 LoopInfra 侧原始录制，设置 `TIANJI_HIL_RAW_RECORDING=1`（调用 `POST /api/hil-recording`）。
- **人工接管检测**：[tianji_env.py](../../rlinf/envs/realworld/tianji/tianji_env.py) 在每步检测 LoopInfra `/api/status` 的 `human_active`、面板 `H/P` 锁存态（`human_latched`），以及相邻观测的关节变化量，自动记录 `exec_action`、`intervene_action` 与 `iv_kind`。

---

## 3. DAgger 数据收集（DAgger Data Collection）

DAgger 的目标是在实机运行冻结的 Stage1 VLA（无 Actor 在线更新、无探索噪声），当策略出现偏差时由人类通过 `H`/`P` 介入接管、调整子阶段并保存完整轨迹（含策略前缀、人工接管动作与阶段标签），同时支持丢弃失败/污染轨迹。

### 标准配置与安全启动入口
- **专用配置**：[examples/embodiment/config/tianji_dagger_collect.yaml](../../examples/embodiment/config/tianji_dagger_collect.yaml)
  - `env.train.override_cfg.hil_mode: dagger`（启用 `episode_archive: true`、`tactile_raw_archive: true`、`discard_episodes: true` 与 `H/P/Discard`，同时关闭仅用于 RL 训练的里程碑奖励、`-1` 步失败惩罚与 `tianji_shadow`）
  - `runner.deploy_actor_updates: false`（rollout 始终执行冻结的 Stage1 VLA 参考）
  - `actor.model.rollout_noise: false`（采集阶段关闭残差探索噪声）
  - `algorithm.rlt_schedule.max_updates_per_train_step: 0`、`warmup_min_size: 999999999`、`algorithm.replay_buffer.auto_save: false`（禁用 RL 优化器更新与 replay 磁盘持久化）
- **通用启动脚本**：
  ```bash
  bash deploy/tianji/run_tianji_dagger_local.sh
  ```
- **小车部署 DAgger 模式（6/15 子阶段）**：
  ```bash
  TIANJI_CAR_NUM_SUBTASKS=15 bash deploy/tianji/stage2_30hz_1118v1.sh --dagger
  # 或 6 子阶段版本：
  TIANJI_CAR_NUM_SUBTASKS=6 bash deploy/tianji/stage2_30hz_1118v1.sh --dagger
  ```
- 采集完成后可使用离线校验脚本确认轨迹完整性：
  ```bash
  python3 deploy/tianji/check_tianji_episode_archive.py "$TIANJI_RUNS_DIR/<run_id>"
  ```

---

## 4. Online RL（异步双 GPU 在线强化学习）

在 GPU0 推理与真机交互的同时，GPU1 实时进行 Residual Actor-Critic（`rlt_ac`）更新，并将新权重按步同步回 GPU0。

- **启动脚本**：
  - 新训练：[deploy/tianji/start_tianji_stage2_async.sh](../../deploy/tianji/start_tianji_stage2_async.sh)
  - 断点续训：[deploy/tianji/resume_tianji_stage2_async.sh](../../deploy/tianji/resume_tianji_stage2_async.sh) `<CHECKPOINT_DIR>`
  - 核心 Python 入口：[examples/embodiment/train_async.py](../../examples/embodiment/train_async.py)（`AsyncEmbodiedRunner`）
- **核心配置**：
  - 无触觉 / 默认 30Hz 关节版本：[tianji2_rlt_stage2_async.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_async.yaml)
  - 含 Xense 触觉版：[tianji2_rlt_stage2_tactile.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_tactile.yaml)
  - 早期 20D TCP 参考版：[tianji_rlt_stage2_async.yaml](../../examples/embodiment/config/tianji_rlt_stage2_async.yaml)
- **关键架构特性**：
  - **GPU 分配**：`cluster.component_placement` 将 `env: 0, rollout: 0`（冻结 4.1B VLA + 异步块推理，或通过 `TIANJI_PI05_SERVICE=1` 连常驻 `pi05_service`）与 `actor: 1`（FSDP Residual MLP Actor + Multi-head Q/V/Penalty Critic）隔离在两张 GPU 上。
  - **预加载门控**：`runner.init_before_start=true` 确保 Actor 初始化、断点恢复与首次权重同步完成后，HIL 面板才允许开始第一条轨迹。
  - **训练信号**：`takeover_penalty: -1.0`（`takeover_kinds: [2, 3]`）、`F` 步失败惩罚、子阶段完成里程碑奖励，以及 `direct_human_bc` 监督。
  - **流式块调度**：`stream_chunks=true` + `rtc_guidance=true` 实现跨块平滑过渡与首块/恢复块关节斜坡混合。

---

## 5. HIL Web 控制面板（`:8879`）与快捷键

`TianjiHILControl` 在 `http://127.0.0.1:8879` 提供实时状态机控制：

- **主要阶段（`phase`）**：`waiting` → `running` ↔ `paused` → `ended` → `returning` → `waiting`。
- **控制模式（`control_mode`）**：`waiting` / `resetting` / `reset_failed` / `fault` / `policy` / `human` / `human_latched` / `paused` / `hold`。
- **模式差异**：
  - `training`：完整启用 `H`/`P`、子阶段推进与里程碑奖励、`F`（步失败惩罚 `-1`）、`Discard` 丢弃本条与可选 `tianji_shadow`。
  - `dagger`：启用 `H`/`P`、子阶段切换、`Discard` 丢弃本条与完整 `TianjiEpisodeArchive`；关闭仅限 RL 的 `F` 惩罚、里程碑奖励与 `tianji_shadow`。
  - `inference` / `stage1`：仅执行评估与 `H`/`P`/子阶段切换，不计算 RL 奖励或丢弃样本。
- **快捷键（面板聚焦非输入框时）**：
  - `H`：人类接管（`human_start`，取消当前动作调度并锁存 `human_latched=true`，需按住 PICO 握持键移动）
  - `P`：交回 Policy（`human_stop`，以当前位置重置控制参考并恢复策略预取）
  - `F`：仅 `hil_mode=training` 可用，标记当前步失败（奖励 `-1`，机械臂继续执行）
  - `N`：推进到下一子阶段（`next_subtask`；在等待开始阶段若启用 Monitor 开关则 `N` 表示关闭下一条 Monitor）
  - `1`..`9`：切换至指定子阶段（`training` 模式下触发里程碑与回退重入记录）
  - `Y`：等待开始阶段开启下一条后台 Monitor（仅在 `training` 模式配置 `tianji_shadow` 时生效）

---

## 6. Episode Archive 目录结构（`episodes/episode_NNNNNN/`）

当 `episode_archive=true` 且 `hil_mode` 为 `"training"` 或 `"dagger"`（或启用 `TIANJI_CAR_1118=1` 归档）时，每条轨迹在后台线程异步写入 `${recorder_dir}/episodes/.episode_NNNNNN.partial`，结束后原子重命名为 `episode_NNNNNN/`（若点击“丢弃本条”或 `outcome=discard` 则直接删除，除非设置 `TIANJI_KEEP_DISCARDED=1`）：

```text
episode_NNNNNN/
├── manifest.json          # 结果（outcome）、步数、块数、动作编码、图像覆盖率、文件哈希、issues
├── steps.jsonl            # 逐控制步执行日志（含时间戳、iv_kind、takeover 边界、奖励）
├── chunks.jsonl           # 逐动作块元数据与 policy_input 索引
├── policy/<chunk>.npz     # Rollout 返回的原始预测（ref_chunk, action, actions_by_stage, z_rl, proprio）
├── labels/<chunk>.npz     # 环境按完成规则补全后的训练标签（executed actions, masks, rewards, iv_kind）
├── images/
│   ├── stream.bin + stream.jsonl   # 独立子进程按同步率全量采集的三相机 JPEG 流
│   └── inputs.bin + inputs.jsonl   # 实际送入策略推理的三相机 JPEG 组
└── xense/                 # （可选）触觉视频 <role>.mp4 + 帧时间索引 <role>.jsonl
```
