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

## 🔄 Milestone 2: gRPC / Local Inference (推理功能)
**状态：进行中 (~90%)**

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

### 待完成
- 🔄 **配置重构为 YAML 结构**（当前任务）
  - 参考 Franka 的配置方式
  - 对齐命名：数采配置要写 `joint` 而不是 `gello`
  - 删除 `.env.example`，用标准 YAML 配置
- ⏳ 真机验证推理流程能跑通

### 相关文件
- `rlinf/workers/rollout/grpc/grpc_policy_adapter.py`
- `examples/embodiment/so101/lerobot_grpc_policy_server.py`
- `examples/embodiment/config/realworld_so101_eval_openpi.yaml`
- `examples/embodiment/config/realworld_so101_eval_openpi_grpc.yaml`
- `examples/embodiment/config/realworld_so101_collect_data.yaml`

---

## ⏳ Milestone 3: HG-DAgger Logic Extensions（最关键）
**状态：未开始**

### 目标
显式接管状态机扩展

### 计划内容
- 显式接管状态机
- smooth_intervene 扩展
- episode recorder 集成

### 相关文件
- `rlinf/utils/env_helpers/smooth_intervene.py`
- `rlinf/envs/wrappers/collect_episode.py`

---

## ⏳ Milestone 4: OpenPI Model and Data Transforms
**状态：未开始**

### 目标
统一 SO101 的单位和 OpenPI transforms

### 计划内容
- 数据加载器放到正确位置
- SO101 dataconfig
- SO101 policy

### 相关文件
- `rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py`
- `rlinf/models/embodiment/openpi/policies/so101_policy.py`
- `rlinf/data/so101_sft_data_loader.py`
- `rlinf/data/so101_dagger_data_loader.py`

---

## ⏳ Milestone 5: SFT / DAgger Training Configs
**状态：未开始**

### 目标
修复训练配置与 norm stats

### 相关文件
- `examples/sft/config/realworld_sft_openpi_so101.yaml`
- `examples/embodiment/config/realworld_so101_dagger_openpi.yaml`

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

**正在做：Milestone 2 - 配置 YAML 化**
- 文档不用管了最后一起整理
- 把配置改成 yaml
- 看看 Franka 怎么做配置的

**进度：Milestone 2 约 85%**

---

## 注意事项

1. 优先级顺序严格按照 Milestone 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
2. 每个 Milestone 完成后再开始下一个
3. Milestone 2 之所以提前，是因为 DAgger 上真机验证必须要求策略能够推理
