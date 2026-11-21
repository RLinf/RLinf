# `RLinf-tianji-hil` ↔ `LoopInfra` 接口契约

本文档记录 `RLinf-tianji-hil` 与 `LoopInfra` 在当前 `tianji_hil` 分支下的完整通信协议、进程间接口、环境变量与跨仓库耦合点（代码依据：[tianji_backend.py](../../rlinf/envs/realworld/tianji/tianji_backend.py)、[tianji_reset.py](../../rlinf/envs/realworld/tianji/tianji_reset.py)、[tianji_image_stream.py](../../rlinf/envs/realworld/tianji/tianji_image_stream.py)、[tianji_episode_archive.py](../../rlinf/envs/realworld/tianji/tianji_episode_archive.py)、[tianji_joint.py](../../rlinf/envs/realworld/tianji/tianji_joint.py)）。

---

## 1. HTTP / JSON 与二进制包协议（`RLinf` → `LoopInfra`）

默认服务地址由 `LOOPINFRA_URL` 指定（默认 `http://127.0.0.1:8780`），可选鉴权头通过 `LOOPINFRA_TOKEN` 传递。所有代理会被显式绕过（`ProxyHandler({})`）。

| 协议常量 / Schema | 值 | 定义位置 |
| --- | --- | --- |
| 采集包协议 | `loopinfra.sdk.v1` | `tianji_backend.py`, `tianji_image_stream.py` |
| 位姿动作协议 | `loopinfra.policy_pose.v1` | `tianji_backend.py` |
| 关节目标协议 | `loopinfra.policy_joint_target.v1` | `tianji_joint.py` |
| 动作块与关节行调度协议 | `loopinfra.policy_chunk.v1` / `loopinfra.policy_chunk_joint_rows.v1` | `LoopInfra`, `tianji_backend.py` |
| 参考窗口与流式调度协议 | `loopinfra.policy_reference_window.v1` / `loopinfra.policy_reference_stream.v1` | `tianji_backend.py` |
| 运动学/跟踪预算协议 | `loopinfra.policy_motion_limits.v1` | `tianji_tracking.py` |
| 复位位姿协议 | `loopinfra.reset_pose.v1` | `tianji_reset.py` |
| 实时执行观测协议 | `loopinfra.policy_execution_observation.v1` | `tianji_actions.py`, `tianji_backend.py` |
| Xense Rectify 触觉包（`XTO1` / `XTR1`） | `loopinfra.xense_rectify.v1` / `loopinfra.xense_rectify_raw.v1` | `LoopInfra`（`/api/tianji/tactile*` 返回） |
| 原始/HIL 异步录制格式 | `loopinfra.raw_episode.v1` | `LoopInfra`（`/api/hil-recording` 写入） |

### 核心 HTTP 端点

| 方法与路径 | 调用方 | 用途与关键字段 |
| --- | --- | --- |
| `GET /api/status` | `LoopInfraPolicyBackend`, `tianji_reset.py` | 查询会话状态、机械臂连接、能力声明（`policy_pose_protocol`）、PICO 接管标志、当前租约（`lease_id`） |
| `GET /api/collection` | `LoopInfraPolicyBackend`, `tianji_hil.py` | 查询 `control_strategy`（校验 `wuji_compat`）及 Xense 设备在线数（`/api/xense-count` 轮询） |
| `GET /api/packet?tianji_stage1=1` | `LoopInfraPolicyBackend._packet()` | 拉取同步后的 3 路相机 JPEG + `tianji_stage1` 对齐的 14D 实测关节与末端状态 |
| `GET /api/packet?after=<id>` | `tianji_image_stream.py`（独立子进程） | 连续增量拉取全量同步帧写入 `images/stream.bin` + `stream.jsonl` |
| `GET /api/tianji/execution?current=1` | `LoopInfraPolicyBackend._execution_observation()` | `stream_chunks` / `sparse_execution_observation` 模式下仅拉取执行元数据（不拉取大图） |
| `GET /api/tianji/tactile?...` & `tactile_raw?...` | `LoopInfraPolicyBackend._read_tactile*()` | 拉取 Xense 触觉张量与原始图像帧（Rectify 模型与档案视频） |
| `POST /api/commands` | `LoopInfraPolicyBackend._send_pose_action()` | 非调度（`rt_schedule=false`）单步位姿/关节指令提交 |
| `POST /api/policy_schedule` | `LoopInfraPolicyBackend` | 动作块队列提交、追加（`append`）与取消（`operation: cancel`, `schedule_id`），支持 `row_space: joint` 与稠密行 |
| `GET /api/policy_schedule/row?...` | `LoopInfraPolicyBackend._schedule_row()` | 按索引读取调度行实测执行回执，供环境构造实际执行的动作标签与奖励 |
| `POST /api/robot` | `tianji_reset.py` | 发送 `reset` / `reset_renew`（携带 `lease_id`）触发或续租保存位姿回位 |
| `POST /api/hold` | `LoopInfraPolicyBackend.hold()`, `tianji_reset.py` | 暂停/异常或复位取消时立刻保持机械臂当前位姿 |
| `GET/POST /api/hil-recording` | `tianji_episode_archive.py` | 在 `TIANJI_HIL_RAW_RECORDING=1` 时查询状态并触发 LoopInfra 旁路原始 HIL 录制（`start` / `stop` / `stage`；要求 LoopInfra 启用 `mode: synchronized`、`raw_recording.hil: true` 与 `hil_storage_roots`） |

---

## 2. `RLinf-tianji-hil` 内部 IPC 与端口

| 组件 | 传输方式与默认端点 | 作用范围 |
| --- | --- | --- |
| **HIL 控制面板**（[tianji_hil.py](../../rlinf/envs/realworld/tianji/tianji_hil.py)） | HTTP `127.0.0.1:${TIANJI_HIL_PORT:-8879}`（`GET /`, `/api/status`, `/api/xense-count`; `POST /api/control`） | 仅写入当前环境的 episode/stage 状态机，不直接下发机械臂指令 |
| **常驻 VLA 服务**（[pi05_service.py](../../rlinf/models/embodiment/openpi_rlinf/pi05_service.py) / [pi05_client.py](../../rlinf/models/embodiment/openpi_rlinf/pi05_client.py)） | `$XDG_RUNTIME_DIR` 下 Unix Socket（`TIANJI_PI05_SERVICE=1` 启用） | 在 GPU0 常驻 4.1B pi0.5 + RLT 编码器，校验权重/代码/配置哈希后向 rollout 返回 `z_rl` 与 `ref_chunk` |
| **Legacy Stage1 服务**（[stage1/server.py](../../rlinf/envs/realworld/tianji/stage1/server.py) / [client.py](../../rlinf/envs/realworld/tianji/stage1/client.py)） | TCP `127.0.0.1:6110`（长度前缀帧） | 早期 Stage1 CPU/GPU 进程解耦推理服务 |
| **Car Monitor Tap**（[car_monitor_tap.py](../../rlinf/envs/realworld/tianji/car_monitor_tap.py)） | `AF_UNIX` `SOCK_DGRAM`（`TIANJI_CAR_MONITOR_SOCKET`） | 队列容量 4、非阻塞单向发送 NPZ（`capture_ns`, `episode`, `stage`, `stage_revision`, `control_revision`, `rlt[2048]`, `joint[16]`） |
| **Shadow Monitor 侧车**（[rlinf/tianji_shadow/](../../rlinf/tianji_shadow/__main__.py)） | 本地临时目录文件通道（`TIANJI_SHADOW_RUNTIME`）+ `LOOPINFRA_URL/api/tianji/tactile` | GPU1 独立进程实时计算阶段完成与 H 接管候选 |

---

## 3. 直接文件系统与 Python 紧耦合点（需在后续重构中解耦）

除了 HTTP 接口外，目前 `RLinf-tianji-hil` 存在两处对 `LoopInfra` 本地工作树的直接耦合：

1. **正运动学直接导包（[tianji_joint.py](../../rlinf/envs/realworld/tianji/tianji_joint.py)）**：
   - 将 `LOOPINFRA_ROOT` 加入 `sys.path`，直接执行 `from loopinfra.client.sdk_control_adapter import NativeKinematics`，并读取 `$LOOPINFRA_ROOT/third_party/TJ_FX_ROBOT_CONTRL_SDK` 计算 16D 关节 → 20D TCP（法兰 `+Z 160 mm`）映射。
   - **不一致隐患**：`tianji_joint.py` 默认回退路径为 `~/LoopInfra_joint`，而 `tianji_backend.py` 与 `tianji_episode_archive.py` 默认回退路径为 `~/LoopInfra_rtj`。
2. **PyAV 视频编码子进程（[tianji_episode_archive.py](../../rlinf/envs/realworld/tianji/tianji_episode_archive.py)）**：
   - 依赖读取 `$LOOPINFRA_ROOT/.loopinfra-python`（若不存在则回退到 `~/miniconda3/envs/loopinfra/bin/python`）来启动 Xense 触觉 `.mp4` 编码子进程（可通过 `TIANJI_XENSE_ENCODER_PYTHON` 显式覆盖）。

---

## 4. Python / Conda 环境隔离约定

- **RLinf 推理与训练侧**：使用训练/GPU 环境（如 `/data/loopinfra/tianji_progress_api_v1/runtime/miniconda3/envs/train/bin/python`，配合 `rlt_test/.venv-gpu/lib/python3.11/site-packages`），需要 PyTorch、CUDA、Ray、OpenPI 依赖。
- **LoopInfra 采集器侧**：使用 `/home/ubuntu/miniconda3/envs/loopinfra/bin/python`（记录在 `$LOOPINFRA_ROOT/.loopinfra-python`），拥有机器人 SDK、Orbbec/Realsense、Xense 与 PyAV 依赖。
- **隔离规则**：启动 `LoopInfra` 采集器或检查脚本时必须清除 `PYTHONPATH`（`env -u PYTHONPATH`），避免两个环境的包路径互相污染。

---

## 5. 关键环境变量清单

| 变量名 | 默认值 / 示例 | 作用 |
| --- | --- | --- |
| `LOOPINFRA_URL` | `http://127.0.0.1:8780` | LoopInfra 采集器 HTTP 服务地址 |
| `LOOPINFRA_TOKEN` | 可选 | LoopInfra HTTP 鉴权 Token |
| `LOOPINFRA_ROOT` | 建议显式设置（代码中分别回退 `~/LoopInfra_rtj` / `~/LoopInfra_joint`） | LoopInfra 检出路径（用于 `NativeKinematics` 与 `.loopinfra-python`） |
| `TIANJI_HIL_PORT` | `8879` | HIL Web 控制面板监听端口 |
| `TIANJI_ACTION_SPACE` | `eef` / `joint`（Stage2 实机为 `joint`） | 选择 20D TCP 或 16D 关节+夹爪动作空间 |
| `TIANJI_CONTROL_HZ` | `10`（YAML 默认）/ `30`（Stage2 实机） | 控制与相机同步频率（仅支持 `10`、`15`、`30`） |
| `TIANJI_ACTION_HORIZON` | `30` / `60` | VLA 原始预测块长度 |
| `TIANJI_STAGE2_CHUNK` / `TIANJI_STAGE2_PREFETCH_AT` / `RLT_REF_OFFSET` | 如 `42` / `24` / `17`（60@30Hz）或由调度表计算 | 异步预取窗口大小、发起预取步与参考偏移（`RLT_REF_OFFSET` 在导入时读取） |
| `TIANJI_RT_SCHEDULE` | `0` / `1` | 设为 `1` 时启用 `/api/policy_schedule` 流式行调度 |
| `TIANJI_ROW_SCALE` / `TIANJI_DENSE_ROWS` | `1` / `0` | 行时间重采样倍率与稠密行模式开关 |
| `TIANJI_DEPLOY_ACTOR_UPDATES` | `1` / `0` | 设为 `0` 时 actor 可训练存档但不下发更新到 rollout（执行纯 VLA 参考） |
| `TIANJI_PI05_SERVICE` | `0` / `1` | 设为 `1` 时 rollout 通过 Unix Socket 连接常驻 `pi05_service` |
| `TIANJI_SUBTASK_PROMPTS` | JSON 文件路径 | 分阶段提示词列表及阶段数配置 |
| `TIANJI_HIL_RAW_RECORDING` | `0` / `1` | 设为 `1` 时在 `episode_archive` 期间联动 `/api/hil-recording` |
| `TIANJI_CAR_1118` | `0` / `1` | 启用 15 阶段小车 1118v1 面板、修订号校验与 `MonitorTap` |
| `RLINF_REALWORLD_PLATFORM` / `RLINF_SKIP_REALWORLD_SETUP` | `tianji` / `1` | 跳过非 Tianji realworld 平台设备初始化 |
