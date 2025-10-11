# RStar2 AgentLoop Examples

This directory contains examples for running RStar2 training with AgentLoop-based rollout using SGLang backend.

## Overview

The RStar2 AgentLoop implementation provides:
- **Tool Agent Loop**: Multi-turn conversation with tool calling capabilities
- **SGLang Backend**: High-performance inference engine
- **CodeJudgeTool Integration**: Code execution and evaluation tools
- **Compatible Interface**: Drop-in replacement for existing rollout workers

## Architecture

```
RStar2Runner (直接维护所有组件)
├── SGLangGenerateWorker (继承自Worker) → SGLang Engine
├── ToolAgentLoop (继承自Worker)
└── CodeJudgeTool (python_code_with_standard_io)
```

**新架构特点**:
- 避免Worker嵌套，所有组件由Rstar2Runner直接维护
- SGLangGenerateWorker和ToolAgentLoop都继承自Worker，支持分布式通信
- 统一的组件管理和交互

## Files

- `main_rstar2.py`: Main training script for RStar2 with AgentLoop
- `config/rstar2-agentloop-sglang.yaml`: Configuration file for AgentLoop SGLang backend
- `run_rstar2_agentloop.sh`: Shell script to run the training
- `README.md`: This documentation

## Configuration

Key configuration sections for AgentLoop:

### Rollout Backend
```yaml
rollout:
  rollout_backend: agentloop_sglang  # Use AgentLoop with SGLang
  
  # AgentLoop specific configs
  max_user_turns: 3
  max_assistant_turns: 3
  max_parallel_calls: 2
  max_tool_response_length: 500
  tool_response_truncate_side: "right"
  response_length: 2048
```

### Tools Configuration
```yaml
tools:
  code_judge:
    host_addr: "localhost"
    host_port: 8088
    batch_size: 4
    concurrency: 2
    batch_timeout_seconds: 30.0
```

### Chat Template
```yaml
data:
  apply_chat_template_kwargs: {}  # Additional args for chat template
```

## Usage

1. **Start CodeJudge Service** (if using code execution tools):
   ```bash
   # Start the code judge service on port 8088
   # (Implementation depends on your code judge service)
   ```

2. **Update Configuration**:
   - Set `rollout.model_dir` to your model path
   - Set `actor.tokenizer.tokenizer_model` to your tokenizer path
   - Set `data.train_data_paths` and `data.val_data_paths` to your dataset paths
   - Adjust `tools.code_judge` settings if needed

3. **Run Training**:
   ```bash
   cd examples/rstar2
   chmod +x run_rstar2_agentloop.sh
   ./run_rstar2_agentloop.sh
   ```

   Or with custom config:
   ```bash
   ./run_rstar2_agentloop.sh your-custom-config
   ```

## Key Features

### Multi-turn Conversation
- Supports multiple user/assistant turns
- Configurable turn limits (`max_user_turns`, `max_assistant_turns`)
- Tool calling between turns

### Tool Integration
- **CodeJudgeTool**: Execute Python code with standard I/O
- **Tool Parser**: Hermes-style `<tool_call>...</tool_call>` format
- **Parallel Execution**: Multiple tools can run concurrently
- **History Tracking**: Tools receive conversation history

### Performance
- **SGLang Backend**: Optimized inference with CUDA graphs
- **Async Generation**: Non-blocking tool execution
- **Memory Efficient**: Weight offloading and synchronization

## Differences from Standard Rollout

1. **Multi-turn Generation**: Instead of single-shot generation, AgentLoop supports multi-turn conversations
2. **Tool Calling**: Integrated tool execution during generation
3. **Conversation History**: Maintains context across turns
4. **Response Masking**: Distinguishes between LLM-generated and tool-generated tokens

## Architecture Changes

### 新架构优势

1. **避免Worker嵌套**: 不再有Worker包含其他Worker的情况，简化了架构
2. **统一管理**: 所有Worker交互都在Rstar2Runner中统一管理
3. **更好的可维护性**: 减少了中间层，逻辑更清晰
4. **分布式支持**: SGLangGenerateWorker和ToolAgentLoop现在都支持分布式通信

### 主要变化

- **删除了**: `AgentLoopSGLangWorker` 类
- **修改了**: `SGLangGenerateWorker` 现在继承自 `Worker`
- **修改了**: `ToolAgentLoop` 现在继承自 `Worker`
- **增强了**: `Rstar2Runner` 直接维护所有AgentLoop组件

## Troubleshooting

### Common Issues

1. **Tool Service Not Available**:
   - Ensure CodeJudge service is running on specified host/port
   - Check network connectivity

2. **Chat Template Mismatch**:
   - Verify tokenizer supports `<tool_call>` format
   - Check `apply_chat_template_kwargs` settings

3. **Memory Issues**:
   - Reduce `rollout_batch_size` or `max_parallel_calls`
   - Enable weight offloading in actor config

4. **Tool Execution Errors**:
   - Check tool service logs
   - Verify tool arguments format
   - Ensure proper tool registration

### Debug Mode

Enable debug output:
```yaml
rollout:
  print_outputs: true
  detokenize: true
```

## Advanced Configuration

### Custom Tools
To add custom tools, modify the `Rstar2Runner._get_agentloop_tools()` method:

```python
def _get_agentloop_tools(self):
    tools = {}
    
    # Add your custom tool
    custom_tool = YourCustomTool(name="your_tool", ...)
    tools["your_tool"] = custom_tool
    
    return tools
```

### Tool Parser
The default parser supports Hermes format. To use a different format, modify the tool parser in `ToolAgentLoop` initialization:

```python
# In Rstar2Runner._init_agentloop_components()
self._tool_agent_loop = ToolAgentLoop(
    config=self.cfg,
    tokenizer=tokenizer,
    # ... other parameters
    # The tool parser is set inside ToolAgentLoop.__init__()
)
```

## Performance Tuning

1. **Batch Size**: Adjust `rollout_batch_size` based on GPU memory
2. **Parallel Calls**: Increase `max_parallel_calls` for better tool throughput
3. **Response Length**: Set appropriate `response_length` for your use case
4. **SGLang Settings**: Tune `gpu_memory_utilization`, `max_running_requests`, etc.
