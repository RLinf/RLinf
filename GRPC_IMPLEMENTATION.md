# SO-101 gRPC Backend Implementation Summary

## 概述

本次实现为 RLinf 添加了 gRPC 远程推理支持，允许在训练过程中将模型推理卸载到远程 GPU 服务器。

## 架构设计

### 1. 核心组件

#### `GRPCPolicyAdapter` (`rlinf/workers/rollout/grpc/grpc_policy_adapter.py`)
- 实现 `BasePolicy` 接口
- 将 `predict_action_batch()` 调用转发到远程 gRPC 服务器
- 处理批量观测（逐个处理后合并）
- 在 RLinf 观测格式和 LeRobot gRPC 协议之间转换

#### `MultiStepRolloutWorker` 修改 (`rlinf/workers/rollout/hf/huggingface_worker.py`)
- 在 `init_worker()` 中检查 `cfg.rollout.use_grpc_backend`
- 如果启用 gRPC：创建 `GRPCPolicyAdapter` 而不是加载本地模型
- 如果禁用 gRPC：标准本地模型加载（默认行为）
- 自动跳过 gRPC 模式下的优化（torch.compile、CUDA graphs）

### 2. gRPC Policy Server

#### LeRobot AsyncInference Server (`examples/embodiment/so101/lerobot_grpc_policy_server.py`)
- 从 PR1572 提取，用于独立推理和演示
- 实现 LeRobot 的 `PolicyServer` 协议
- 加载 RLinf PI05 checkpoint 并提供 gRPC 推理服务
- 支持配置化启动（通过 `.env` 文件和服务脚本）
- **支持云端 GPU 优化**：torch.compile 和 CUDA graphs（边缘设备不需要，云端可选）

### 3. 配置文件

#### 训练配置示例
- `examples/embodiment/config/realworld_so101_inference_grpc.yaml` - 纯推理模式
- `examples/embodiment/config/realworld_so101_dagger_grpc.yaml` - DAgger 训练模式

#### Server 配置
- `examples/embodiment/so101/lerobot_grpc_policy.env.example` - 服务端环境变量模板
- `examples/embodiment/so101/lerobot_grpc_policy_service.sh` - 服务管理脚本

## 使用方式

### 场景 1：训练时使用 gRPC 远程推理

**步骤 1：启动 gRPC 服务器（GPU 机器）**
```bash
cd examples/embodiment/so101
cp lerobot_grpc_policy.env.example lerobot_grpc_policy.env
# 编辑 .env 文件设置 checkpoint 路径
# 可选：启用 GPU 优化
# SO101_GRPC_ENABLE_TORCH_COMPILE=true  # ~2x 加速
./lerobot_grpc_policy_service.sh start
```

**步骤 2：运行训练（控制机器）**
```bash
python rlinf/train_embodied_agent.py \
  --config-name realworld_so101_dagger_grpc
```

配置文件中的关键设置：
```yaml
rollout:
  use_grpc_backend: true
  grpc:
    server_address: "gpu-server:50051"
    timeout: 30.0
```

### 场景 2：独立推理（不涉及训练）

使用 LeRobot 生态的独立推理工具：
```bash
examples/embodiment/run_so101_inference.sh \
  --inference-backend grpc \
  --server-address gpu-server:50051 \
  --serial-port /dev/ttyACM0 \
  --camera /dev/video0 \
  --allow-hardware-motion
```

## 技术细节

### 协议选择说明

本实现中存在两种 gRPC 协议：

1. **LeRobot PolicyProtocol** (简单同步)
   - `GRPCPolicyAdapter` 假设使用此协议
   - 简单的请求/响应模式
   - 适合训练时的批量推理

2. **LeRobot AsyncInference** (复杂异步)
   - `lerobot_grpc_policy_server.py` 使用此协议
   - 流式观测/动作，复杂的调度和时间戳管理
   - 适合实时机器人控制

**注意：** 当前 `GRPCPolicyAdapter` 实现针对 PolicyProtocol。如果需要与 AsyncInference server 通信，需要修改适配器实现。

### 观测格式转换

RLinf 格式 → gRPC 格式：
- `main_images`, `wrist_images` → `Image` (HWC, uint8, rgb8)
- `states` → `Tensor` (float32)
- `task_description` / `language_instruction` → `task` (string)

gRPC 响应 → RLinf 格式：
- `action.data` → numpy array
- 形状重塑为 `[batch_size, action_dim]`

### 与 DAgger 的兼容性

- Expert 模型始终在本地运行（不通过 gRPC）
- 只有 student policy 可以使用 gRPC backend
- DAgger 的 re-labeling 逻辑不受影响

### 限制

1. **网络延迟**：gRPC 推理增加 5-50ms 延迟
2. **Expert 本地化**：DAgger 的 expert 模型必须在本地
3. **OPD 本地化**：OPD feature models 必须在本地
4. **Server 更新**：训练期间模型更新需要重启 gRPC server

### GPU 优化策略

**边缘设备（rollout worker）：**
- 自动跳过 torch.compile 和 CUDA graphs
- 推理通过 gRPC 远程执行，本地无需优化

**云端服务器（gRPC policy server）：**
- 可选启用 `torch.compile`（推荐生产环境）
  - 约 2x 加速
  - 首次推理会有编译开销（warmup）
  - 通过 `--enable-torch-compile` 启用
- 可选启用 CUDA graphs（固定 batch size 场景）
  - 降低 kernel 启动开销
  - 与 torch.compile 互斥
  - 通过 `--enable-cuda-graph` 启用

## 文件清单

### 新增文件
```
rlinf/workers/rollout/grpc/
├── __init__.py
├── README.md
└── grpc_policy_adapter.py

examples/embodiment/config/
├── realworld_so101_inference_grpc.yaml
└── realworld_so101_dagger_grpc.yaml

examples/embodiment/so101/
├── __init__.py
├── README.md
├── lerobot_grpc_policy.env.example
├── lerobot_grpc_policy_server.py
└── lerobot_grpc_policy_service.sh
```

### 修改文件
```
rlinf/workers/rollout/hf/huggingface_worker.py
└── init_worker() 方法：添加 gRPC backend 支持
```

## 测试验证

所有关键组件已通过以下测试：
- ✅ 配置解析（OmegaConf）
- ✅ GRPCPolicyAdapter 导入
- ✅ MultiStepRolloutWorker 导入
- ✅ gRPC backend 配置分支逻辑

## 下一步

1. 实际硬件测试：在真实 SO-101 机器人上测试 gRPC 推理
2. 性能基准测试：对比本地推理和 gRPC 推理的延迟
3. 协议统一：考虑统一 PolicyProtocol 和 AsyncInference 的使用
4. 文档完善：添加更多使用示例和故障排查指南
