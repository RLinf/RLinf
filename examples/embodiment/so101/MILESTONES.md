# SO-101 Development Milestones

## 总览

SO-101 机器人集成按以下 9 个模块顺序推进：

---

## ✅ Milestone 1: Hardware Registration
**状态：已完成并 commit (8b10061c)**

### 完成内容
- ✅ SO101Arm 硬件注册（`rlinf/envs/real/so101/base.py`）
- ✅ UVC 相机后端集成
- ✅ SO101LeaderDevice 遥操作设备（`rlinf/robotics/parts/teleop/so101_leader.py`）
- ✅ 配置文件验证 schema（`examples/embodiment/config/so101/validation_schema.yaml`）
- ✅ 单元测试（`tests/unit_tests/test_data.py`, `test_models.py`）

### 相关文件
- `rlinf/envs/real/so101/base.py`
- `rlinf/robotics/parts/teleop/so101_leader.py`
- `examples/embodiment/config/so101/validation_schema.yaml`

---

## ✅ Milestone 2: gRPC / Local Inference (推理功能)
**状态：开发完成，等待真机验证 (commit 5b0d75e1)**

### 背景
> 为什么不是先做 DAgger？因为如果要上真机验证，DAgger 必须要求策略能够推理，所以把推理提前了。

### 目标
实现可配置的**本地直接推理 / gRPC 协议走云端推理**

- **本地模式**：RLinf 原有模式，保持向后兼容
- **gRPC 模式**：LeRobot 的云端本地协同功能，支持没有 4090 工作站的初学者

### 已完成
- ✅ GRPCPolicyAdapter 实现（`rlinf/workers/rollout/grpc/grpc_policy_adapter.py`）
- ✅ MultiStepRolloutWorker 支持 gRPC backend
- ✅ gRPC server 实现（`examples/embodiment/so101/lerobot_grpc_policy_server.py`）
- ✅ 服务脚本（`lerobot_grpc_policy_service.sh`）
- ✅ 文档（`GRPC_IMPLEMENTATION.md`）
- ✅ 已对照 Franka 配置规划了 SO-101 配置结构
- ✅ 配置重构为 YAML 结构
  - 参考 Franka 的配置方式
  - 对齐命名：数采配置写 `joint` 而不是 `gello`
  - 创建标准 YAML 配置文件
  - 提取 rollout 配置组

### 待真机验证
- ⏳ 本地推理模式能跑通
- ⏳ gRPC 推理模式能跑通

### 相关文件
- `rlinf/workers/rollout/grpc/grpc_policy_adapter.py`
- `examples/embodiment/so101/lerobot_grpc_policy_server.py`
- `examples/embodiment/config/realworld_so101_eval_openpi.yaml`
- `examples/embodiment/config/realworld_so101_eval_openpi_grpc.yaml`
- `examples/embodiment/config/realworld_so101_collect_data.yaml`

---

## ✅ Milestone 3: HG-DAgger Logic Extensions（最关键）
**状态：已完成并 commit (d337bf88)**

### 目标
显式接管状态机扩展，支持 leader-follower 设备的安全操作

### 完成内容
- ✅ 添加 `teleop_intervention.mode` 配置（activity|explicit）
- ✅ explicit 模式绕过 PICO 设备检查
- ✅ SO101Leader 添加 `hold()` 方法返回当前关节位置
- ✅ 配置 3 秒缓冲期（可配置）用于操作员安全抓取/释放 leader arm
- ✅ 缓冲期内 episode 停止记录

### 核心改动
- `smooth_intervene.py`: 添加 mode 参数，只在 activity 模式检查 PICO
- `so101_leader.py`: 添加 `hold()` 方法（8 行）
- `composed.py`: 添加 gripper_position context getter
- `realworld_so101_dagger_openpi.yaml`: 配置 explicit mode + hold_buffer_seconds

### 安全机制
- **触发干预后 3 秒缓冲期**：等待操作员握稳 leader，避免机械臂砸向桌面
- **结束干预后 3 秒缓冲期**：等待操作员撤手，避机械臂突然移动夹到手

### 待真机验证
- ⏳ 缓冲期机制是否按预期工作
- ⏳ episode 在缓冲期内是否停止记录

### 相关文件
- `rlinf/utils/env_helpers/smooth_intervene.py`
- `rlinf/envs/wrappers/collect_episode.py`
- `rlinf/robotics/parts/teleop/so101_leader.py`
- `rlinf/envs/real/wrappers/teleop/composed.py`

---

## ✅ Milestone 4: OpenPI Model and Data Transforms
**状态：已完成并 commit (f1897981)**

### 目标
统一 SO101 的单位和 OpenPI transforms

### 完成内容
- ✅ SO101JointDataConfig 创建（`so101_dataconfig.py`）
  - 12 个关节配置
  - UVC 相机后端
  - 参考 Franka YAML 风格
- ✅ 注册 `pi05_so101_joint` 配置到 `dataconfig/__init__.py`
- ✅ SO101Policy 已存在并集成
- ✅ 数据加载器已就绪
  - `so101_sft_data_loader.py`
  - `so101_dagger_data_loader.py`

### 相关文件
- `rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py` (新增)
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py` (注册)
- `rlinf/models/embodiment/openpi/policies/so101_policy.py` (已存在)
- `rlinf/data/so101_sft_data_loader.py` (已存在)
- `rlinf/data/so101_dagger_data_loader.py` (已存在)

---

## ✅ Milestone 5: SFT / DAgger Training Configs
**状态：已完成并 commit (c42438f5)**

### 目标
标准化训练配置，对齐 Franka 风格

### 完成内容
- ✅ 标准化 YAML 格式（False/True → false/true）
- ✅ 添加 `_self_` 到 defaults 确保正确的覆盖顺序
- ✅ 简化 component_placement 为 `actor,rollout,env: all`
- ✅ 更新模型引用：pi0_5 → pi0_5_rlinf
- ✅ 补充 openpi 配置字段（action_horizon, discrete_state_input）
- ✅ 修正 action_dim: 8 → 12（SO-101 双臂各 6-DOF）
- ✅ 统一占位符注释格式

### 相关文件
- `examples/sft/config/realworld_sft_openpi_so101.yaml`
- `examples/embodiment/config/realworld_so101_dagger_openpi.yaml`
- `examples/embodiment/config/realworld_so101_collect_data_joint.yaml`

---

## ⏳ Milestone 6: Toolkits（选择性保留）
**状态：未开始**

### 目标
决定哪些独立 toolkit 保留

### 相关文件
- `toolkits/realworld_check/` 下的所有文件

---

## ⏳ Milestone 7: Docs
**状态：未开始**

### 目标
收敛文档结构

---

## ⏳ Milestone 8: Tests
**状态：未开始**

### 目标
收敛测试结构

---

## ⏳ Milestone 9: Entry Points / Dependency Wiring
**状态：未开始**

### 目标
收敛脚本结构和依赖关系

---

## 当前工作

**Milestone 5 开发完成 ✅**
- 配置文件标准化为统一 YAML 风格
- 对齐 Franka 配置结构
- 修正 action_dim 和补充缺失字段

**下一步：Milestone 6 - Toolkits（选择性保留）**

---

## 注意事项

1. 优先级顺序严格按照 Milestone 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
2. 每个 Milestone 完成后再开始下一个
3. Milestone 2 之所以提前，是因为 DAgger 上真机验证必须要求策略能够推理
