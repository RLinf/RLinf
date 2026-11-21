# Tianji HIL 四大工作流操作手册

本手册基于 `RLinf-tianji-hil` 实际代码（[tianji_env.py](../../rlinf/envs/realworld/tianji/tianji_env.py)、[tianji_hil.py](../../rlinf/envs/realworld/tianji/tianji_hil.py)、[tianji_episode_archive.py](../../rlinf/envs/realworld/tianji/tianji_episode_archive.py)）整理四个核心工作流的启动逻辑、状态机与归档规则。

> [!WARNING]
> **执行前提**：当前 `RLinf-tianji-hil` 的 `tianji_hil` 分支存在部分缺失文件与 `sparse-checkout`（详见 [repo_management.md](repo_management.md)）。若在 `tianji1` 上执行实机部署，请先完成代码修复或确认使用完整环境。

---

## 0. 通用前置条件：启动 LoopInfra 采集器

在运行任何 RLinf 工作流前，先在 LoopInfra 仓库完成以下就绪检查：
1. 以 `env -u PYTHONPATH` 启动 LoopInfra 采集器（默认监听 `http://127.0.0.1:8780`），控制配置为 `wuji_compat`，同步频率与 `TIANJI_CONTROL_HZ` 一致（`10` / `15` / `30` Hz）。
2. 在采集器 UI 确认双臂已连接并处于 `Position/EEF` 就绪状态，三路相机（`head`, `wrist_A`, `wrist_B`）同步流正常，保存的复位位姿（`loopinfra.reset_pose.v1`）已加载。
3. 导出一致的契约环境变量（`LOOPINFRA_URL`, `LOOPINFRA_ROOT`, `TIANJI_CONTROL_HZ`, `TIANJI_ACTION_HORIZON`, `RLT_REF_OFFSET`, `RLINF_REALWORLD_PLATFORM=tianji`, `RLINF_SKIP_REALWORLD_SETUP=1`）。

---

## 1. 模型推理（Model Inference）

用于纯策略评估/实机演示，**不启动训练更新、不计算 RL 奖励、不支持丢弃样本**。

### 路径 A：Stage2 Joint VLA / Actor 推理（推荐框架）
- **入口**：[evaluations/eval_embodied_agent.py](../../evaluations/eval_embodied_agent.py)
- **配置**：
  - 仅基座 VLA（无 Residual Actor）：[tianji2_joint_stage1.yaml](../../examples/embodiment/config/tianji2_joint_stage1.yaml)（设置 `rollout.joint_vla_only: true`，加载 `TianjiVLAEvalWorker` + `TianjiVLAEnvWorker`）
  - VLA + 已训练 Residual Actor：[tianji2_rlt_stage2_infer.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_infer.yaml)（设置 `rollout.joint_actor_eval: true`，加载 `TianjiJointEvalWorker`）
- **关键参数**：
  - `runner.only_eval=true`，`runner.deploy_actor_updates=false`
  - `env.eval.override_cfg.hil_mode=inference`
  - `TIANJI_ACTION_SPACE=joint`，`TIANJI_RT_SCHEDULE=1`（通过 `/api/policy_schedule` 下发 16D 关节块）
- **示例命令（需先配置 `EMBODIED_PATH` 与模型路径环境变量）**：
  ```bash
  python -u evaluations/eval_embodied_agent.py \
    --config-path "$EMBODIED_PATH/config" \
    --config-name tianji2_joint_stage1 \
    env.eval.override_cfg.hil_mode=inference \
    runner.deploy_actor_updates=false
  ```

### 路径 B：Legacy Stage1 独立推理进程
- **入口**：`python -m rlinf.envs.realworld.tianji.stage1`（[stage1/__main__.py](../../rlinf/envs/realworld/tianji/stage1/__main__.py)）
- **说明**：使用 `hil_mode="stage1"` 与独立 TCP 服务（默认端口 `6110`），面板标题为 `HIL · Stage1 推理`，支持无机械臂 `check` 模式与物理实机 `execute` 模式。

---

## 2. HIL 数据收集（HIL Data Collection）

由人类操作员通过 PICO 遥操（在 LoopInfra 侧完成）或人工接管直接采集高质量示范/修正轨迹，**RLinf 不接入独立 Pico wrapper**（配置中 `use_pico: false` / 注释说明由 LoopInfra 透传）。

- **采集方式 1（纯人工 VR 采集）**：直接在 `LoopInfra` 采集器 UI（`http://127.0.0.1:8780/collector`）进行，PICO A 开始、B 结束并复位。
- **采集方式 2（RLinf 托管归档 `TianjiEpisodeArchive`）**：
  - 环境配置：[realworld_tianji.yaml](../../examples/embodiment/config/env/realworld_tianji.yaml) 配合 `override_cfg`：
    - `backend=loopinfra`
    - `hil_control=true`
    - **`hil_mode=training`**（关键：`tianji_env.py:203` 强制校验 `episode_archive and recorder_dir and hil_mode == "training"` 才启用完整轨迹归档）
    - `episode_archive=true`，`episode_archive_image_hz=15`（可选降采样写入流）
    - `recorder_dir=${runner.logger.log_path}/hil`
  - 若需同时触发 LoopInfra 侧原始录制，设置 `TIANJI_HIL_RAW_RECORDING=1`（调用 `POST /api/hil-recording`）。
- **人工接管检测**：[tianji_env.py](../../rlinf/envs/realworld/tianji/tianji_env.py) 在每步检测 LoopInfra `/api/status` 的 `human_active`、面板 `H/P` 锁存态（`human_latched`），以及相邻观测的关节变化量（`observed_action`），自动记录 `exec_action`、`intervene_action` 与 `iv_kind`。

---

## 3. DAgger 数据收集（DAgger Data Collection）

DAgger 的目标是在实机运行当前策略（VLA 或已有 checkpoint），当策略出现偏差时由人类通过 `H`/`P` 介入接管、调整子阶段并保存完整轨迹（包含策略前缀 + 人工接管动作 + 阶段重标），供后续离线 SFT / DAgger 训练使用。

### 当前实现现状与安全操作方式
- **现状说明**：当前仓库**没有独立的 `dagger_collect.yaml` 配置**。DAgger 采集复用 `AsyncEmbodiedRunner` 栈，通过开关控制是否部署新权重。
- **重要避坑（`hil_mode` 门控）**：
  - 在 `eval_embodied_agent.py` 中若仅传 `hil_mode=inference` + `episode_archive=true`（如旧版 car 脚本中的写法），`tianji_env.py:203` 会因 `hil_mode != "training"` 而**关闭 `TianjiEpisodeArchive`**（仅保留轻量 step 文本日志）。
  - 若要完整归档 `episodes/episode_NNNNNN/`（含三相机 `.bin`、策略输入输出 `.npz`、执行标签 `.npz` 与 `manifest.json`），必须以 **`hil_mode=training`** 运行。
- **安全执行 DAgger 采集（不把在线更新权重下发到真机）**：
  - 使用 `examples/embodiment/train_async.py` 配合 [tianji2_rlt_stage2_async.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_async.yaml)（待补齐缺失文件并关闭 sparse-checkout 后运行）：
    ```bash
    TIANJI_DEPLOY_ACTOR_UPDATES=0 \
    python -u examples/embodiment/train_async.py \
      --config-name tianji2_rlt_stage2_async \
      runner.deploy_actor_updates=false \
      env.train.override_cfg.hil_mode=training \
      env.train.override_cfg.episode_archive=true
    ```
  - 当 `runner.deploy_actor_updates=false` 时，GPU0 始终执行 `TIANJI_RLT_STAGE1_MODEL_PATH` 指定的基座策略（不接收 GPU1 Actor 的权重同步），同时完整触发 `TianjiEpisodeArchive` 归档每一条成功/失败轨迹（按“丢弃本条”则会删除 `.partial` 目录，不污染数据集）。

---

## 4. Online RL（异步双 GPU 在线强化学习）

在 GPU0 推理与真机交互的同时，GPU1 实时进行 Residual Actor-Critic（`rlt_ac`）更新，并将新权重按步同步回 GPU0。

- **入口**：`examples/embodiment/train_async.py`（`AsyncEmbodiedRunner`）
- **核心配置**：
  - 无触觉 / 默认 30Hz 关节版本：[tianji2_rlt_stage2_async.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_async.yaml)
  - 含 Xense 触觉版：[tianji2_rlt_stage2_tactile.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_tactile.yaml)
  - 早期 20D TCP 参考版：[tianji_rlt_stage2_async.yaml](../../examples/embodiment/config/tianji_rlt_stage2_async.yaml)
- **关键架构特性**：
  - **GPU 分配**：`cluster.component_placement` 将 `env: 0, rollout: 0`（冻结 4.1B VLA + 异步块推理，或通过 `TIANJI_PI05_SERVICE=1` 连常驻 `pi05_service`）与 `actor: 1`（FSDP Residual MLP Actor + Multi-head Q/V/Penalty Critic）隔离在两张 GPU 上。
  - **预加载门控**：`runner.init_before_start=true` 确保 Actor 初始化、断点恢复（`resume_dir`）与首次权重同步全部完成后，HIL 面板才允许开始第一条轨迹。
  - **训练信号**：
    - `takeover_penalty: -1.0`（`takeover_kinds: [2, 3]`）与 `F` 步失败按键注入 `-1` 惩罚；
    - 子阶段完成里程碑（按键 `1..9` / `N` 推进）注入正向里程碑奖励；
    - `direct_human_bc` 对人工接管片段直接施加 BC 监督（`bc_weight: 5.0`, `bc_human_weight_mult: 2.0`）。
  - **流式块调度**：`stream_chunks=true` + `rtc_guidance=true` + `prefetch_at=24`（60@30Hz）实现跨块平滑过渡与首块/恢复块关节斜坡混合（`first_chunk_ramp_steps=40`）。

---

## 5. HIL Web 控制面板（`:8879`）与快捷键

`TianjiHILControl` 在 `http://127.0.0.1:8879` 提供实时状态机控制：

- **主要阶段（`phase`）**：`waiting` → `running` ↔ `paused` → `ended` → `returning` → `waiting`。
- **控制模式（`control_mode`）**：`waiting` / `resetting` / `reset_failed` / `fault` / `policy` / `human` / `human_latched` / `paused` / `hold`。
- **快捷键（面板聚焦非输入框时）**：
  - `H`：人类接管（`human_start`，取消当前动作调度并锁存 `human_latched=true`，需按住 PICO 握持键移动）
  - `P`：交回 Policy（`human_stop`，以当前位置重置控制参考并恢复策略预取）
  - `F`：仅 `hil_mode=training` 可用，标记当前步失败（奖励 `-1`，机械臂继续执行）
  - `N`：推进到下一子阶段（`next_subtask`；在等待开始阶段若启用 Monitor 开关则 `N` 表示关闭下一条 Monitor）
  - `1`..`9`：切换至指定子阶段（`training` 模式下触发里程碑与回退重入记录）
  - `Y`：等待开始阶段开启下一条后台 Monitor（仅在配置 `tianji_shadow` 时生效）

---

## 6. Episode Archive 目录结构（`episodes/episode_NNNNNN/`）

当 `episode_archive=true` 且 `hil_mode=training` 时，每条轨迹在后台线程异步写入 `${recorder_dir}/episodes/.episode_NNNNNN.partial`，结束后原子重命名为 `episode_NNNNNN/`（若点击“丢弃本条”或 `outcome=discard` 则直接删除，除非设置 `TIANJI_KEEP_DISCARDED=1`）：

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
