# RStar2 AgentLoop 架构更新说明

## 更新概述

本次更新将SGLangGenerateWorker和ToolAgentLoop修改为继承自Worker的类，并在Rstar2Runner中直接维护，避免了Worker嵌套的问题。

## 主要变化

### 1. 删除的文件
- `rlinf/workers/rollout/sglang/agentloop_sglang_worker.py` - 避免Worker嵌套

### 2. 修改的文件

#### SGLangGenerateWorker
- **位置**: `rlinf/workers/rollout/sglang/sglang_generate_worker.py`
- **变化**: 现在继承自Worker，支持分布式通信和日志记录
- **新增参数**: `parent_address`, `world_size`, `rank`

#### ToolAgentLoop
- **位置**: `rlinf/workers/agent_loop/tool_agent_loop.py`
- **变化**: 现在继承自Worker，支持分布式通信和日志记录
- **新增参数**: `parent_address`, `world_size`, `rank`

#### Rstar2Runner
- **位置**: `rlinf/runners/rstar2_runner.py`
- **新增功能**: 直接维护SGLangGenerateWorker和ToolAgentLoop实例
- **新增方法**:
  - `_init_agentloop_components()`: 初始化AgentLoop组件
  - `_get_agentloop_tools()`: 获取工具配置
  - `_async_generate_fn()`: 异步生成函数
  - `_run_agentloop_rollout()`: 运行AgentLoop rollout
- **修改方法**:
  - `init_workers()`: 初始化AgentLoop组件
  - `_sync_weights()`: 同步AgentLoop组件权重
  - `run()`: 根据模式选择rollout方式

#### Rollout Utils
- **位置**: `rlinf/workers/rollout/utils.py`
- **变化**: 更新了agentloop_sglang backend的处理逻辑

## 新架构优势

1. **避免Worker嵌套**: 不再有Worker包含其他Worker的情况
2. **统一管理**: 所有Worker交互都在Rstar2Runner中统一管理
3. **更好的可维护性**: 减少了中间层，逻辑更清晰
4. **分布式支持**: SGLangGenerateWorker和ToolAgentLoop现在都支持分布式通信
5. **保持兼容性**: 标准rollout模式不受影响

## 使用方式

### 配置AgentLoop模式
```yaml
rollout:
  rollout_backend: agentloop_sglang
  # AgentLoop specific configs
  max_user_turns: 3
  max_assistant_turns: 3
  max_parallel_calls: 2
  max_tool_response_length: 500
  tool_response_truncate_side: "right"
  response_length: 2048

tools:
  code_judge:
    host_addr: "localhost"
    host_port: 8088
    batch_size: 4
    concurrency: 2
    batch_timeout_seconds: 30.0
```

### 标准模式
```yaml
rollout:
  rollout_backend: sglang  # 或其他标准backend
```

## 测试

### 1. 语法测试
```bash
python test_syntax.py
```

### 2. 导入测试
```bash
python test_imports.py
```

### 3. 功能测试
```bash
python test_functionality.py
```

### 4. 完整测试
```bash
python test_worker_inheritance.py
```

## 迁移指南

### 从旧架构迁移

1. **配置文件**: 无需修改，配置格式保持不变
2. **代码调用**: 无需修改，接口保持不变
3. **工具集成**: 现在通过Rstar2Runner._get_agentloop_tools()方法添加

### 自定义工具

**旧方式** (已删除):
```python
# 在AgentLoopSGLangWorker中添加工具
self._agent_loop.tools["your_tool"] = custom_tool
```

**新方式**:
```python
# 在Rstar2Runner._get_agentloop_tools()中添加工具
def _get_agentloop_tools(self):
    tools = {}
    tools["your_tool"] = custom_tool
    return tools
```

## 注意事项

1. AgentLoop模式目前只支持collocated模式，不支持pipeline模式
2. 需要确保工具服务（如CodeJudge）正常运行
3. 异步rollout需要正确处理事件循环
4. 所有Worker现在都支持分布式通信，但需要正确的环境配置

## 故障排除

### 常见问题

1. **导入错误**: 确保所有依赖正确安装
2. **初始化错误**: 检查模型路径和tokenizer路径
3. **工具服务错误**: 确保CodeJudge等服务正常运行
4. **分布式错误**: 检查Worker环境配置

### 调试模式

启用调试输出：
```yaml
rollout:
  print_outputs: true
  detokenize: true
```

## 性能优化

1. **批处理大小**: 根据GPU内存调整rollout_batch_size
2. **并行调用**: 增加max_parallel_calls提高工具吞吐量
3. **响应长度**: 为您的用例设置适当的response_length
4. **SGLang设置**: 调整gpu_memory_utilization、max_running_requests等
