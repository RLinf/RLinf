# 架构更新说明

## 修改目标
将SGLangGenerateWorker和ToolAgentLoop修改为继承自Worker的类，并在Rstar2Runner中维护，避免Worker嵌套。

## 修改前架构
```
Rstar2Runner
├── SGLangWorker (标准rollout)
└── AgentLoopSGLangWorker (嵌套Worker)
    ├── SGLangGenerateWorker (非Worker类)
    └── ToolAgentLoop (继承自AgentLoopBase)
```

## 修改后架构
```
Rstar2Runner
├── SGLangWorker (标准rollout)
└── AgentLoop模式组件 (直接维护，无嵌套)
    ├── SGLangGenerateWorker (继承自Worker)
    └── ToolAgentLoop (继承自Worker)
```

## 主要修改

### 1. SGLangGenerateWorker
- **修改前**: 独立类，不继承Worker
- **修改后**: 继承自Worker，支持分布式通信和日志记录
- **位置**: `rlinf/workers/rollout/sglang/sglang_generate_worker.py`

### 2. ToolAgentLoop
- **修改前**: 继承自AgentLoopBase
- **修改后**: 继承自Worker，支持分布式通信和日志记录
- **位置**: `rlinf/workers/agent_loop/tool_agent_loop.py`

### 3. Rstar2Runner
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

### 4. 删除的文件
- `rlinf/workers/rollout/sglang/agentloop_sglang_worker.py`: 避免Worker嵌套

### 5. 更新的文件
- `rlinf/workers/rollout/utils.py`: 更新rollout backend选择逻辑

## 优势

1. **避免Worker嵌套**: 不再有Worker包含其他Worker的情况
2. **统一管理**: 所有Worker交互都在Rstar2Runner中统一管理
3. **更好的可维护性**: 减少了中间层，逻辑更清晰
4. **保持兼容性**: 标准rollout模式不受影响
5. **分布式支持**: SGLangGenerateWorker和ToolAgentLoop现在都支持分布式通信

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
运行测试脚本验证修改：
```bash
python test_worker_inheritance.py
```

## 注意事项

1. AgentLoop模式目前只支持collocated模式，不支持pipeline模式
2. 需要确保工具服务（如CodeJudge）正常运行
3. 异步rollout需要正确处理事件循环
