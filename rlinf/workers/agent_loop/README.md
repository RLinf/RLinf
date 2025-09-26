# RLinf Tool Agent Loop

这是一个在RLinf中实现的简单Tool Agent Loop，基于verl的agent loop设计，但进行了简化和适配。

## 功能特性

- 🤖 **多轮对话**: 支持用户和助手之间的多轮对话
- 🛠️ **工具调用**: 支持调用各种工具（计算器、搜索、天气查询等）
- 🔄 **异步处理**: 基于asyncio的异步实现
- 📝 **灵活配置**: 通过OmegaConf进行配置管理
- 🎯 **简单易用**: 简化的API设计，易于集成和使用

## 文件结构

```
rlinf/workers/agent_loop/
├── __init__.py              # 模块初始化
├── agent_loop.py            # 基础Agent Loop类
├── tool_agent_loop.py       # Tool Agent Loop实现
├── tool_parser.py           # 工具调用解析器
├── test_tool_agent.py       # 测试文件
├── example_usage.py         # 使用示例
└── README.md               # 说明文档
```

## 核心组件

### 1. AgentLoopBase
基础抽象类，定义了agent loop的基本接口。

### 2. ToolAgentLoop
具体的tool agent实现，支持：
- 多轮对话管理
- 工具调用解析和执行
- 响应生成和token管理

### 3. ToolParser
工具调用解析器，支持从模型响应中提取工具调用信息。

## 使用方法

### 基本使用

```python
import asyncio
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from rlinf.workers.agent_loop import ToolAgentLoop

async def main():
    # 1. 创建配置
    config = OmegaConf.create({
        "rollout": {
            "max_user_turns": 5,
            "max_assistant_turns": 5,
            "max_parallel_calls": 3,
            "max_tool_response_length": 500,
            "response_length": 1024
        },
        "data": {
            "apply_chat_template_kwargs": {}
        }
    })
    
    # 2. 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    
    # 3. 创建agent
    agent = ToolAgentLoop(config, tokenizer)
    
    # 4. 运行对话
    messages = [{"role": "user", "content": "请帮我计算 2+3 的结果"}]
    sampling_params = {"temperature": 0.7, "top_p": 0.9}
    
    result = await agent.run(
        sampling_params=sampling_params,
        raw_prompt=messages
    )
    
    print(f"响应: {result.response_text}")

# 运行
asyncio.run(main())
```

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_user_turns` | int | 5 | 最大用户轮数 |
| `max_assistant_turns` | int | 5 | 最大助手轮数 |
| `max_parallel_calls` | int | 3 | 最大并行工具调用数 |
| `max_tool_response_length` | int | 500 | 工具响应最大长度 |
| `tool_response_truncate_side` | str | "right" | 截断方向 |
| `response_length` | int | 1024 | 响应最大长度 |

## 内置工具

当前实现包含以下内置工具：

1. **计算器** (`calculator`): 执行数学计算
2. **搜索** (`search`): 模拟搜索功能
3. **天气查询** (`get_weather`): 模拟天气查询

### 工具调用格式

工具调用使用以下格式：

```xml
<tool_call>{"name": "calculator", "arguments": {"expression": "2+3"}}</tool_call>
```

## 运行测试

```bash
# 运行基本测试
python -m rlinf.workers.agent_loop.test_tool_agent

# 运行使用示例
python -m rlinf.workers.agent_loop.example_usage
```

## 扩展开发

### 添加新工具

1. 在`ToolAgentLoop`类的`__init__`方法中添加工具到`self.tools`字典
2. 实现对应的工具函数

```python
async def _new_tool(self, args: dict) -> str:
    """新工具的实现"""
    # 工具逻辑
    return "工具响应"
```

### 自定义工具解析器

继承`ToolParser`类并实现`extract_tool_calls`方法：

```python
@ToolParser.register("custom")
class CustomToolParser(ToolParser):
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        # 自定义解析逻辑
        pass
```

## 注意事项

1. 当前实现使用mock的LLM响应，实际使用时需要集成真实的LLM服务
2. 工具调用使用`eval()`函数，生产环境中应使用更安全的表达式求值方法
3. 错误处理机制需要根据实际需求进一步完善

## 与verl的差异

相比verl的agent loop实现，这个简化版本：

- 移除了Ray分布式支持
- 简化了配置管理
- 减少了外部依赖
- 专注于核心功能实现
- 更适合集成到RLinf框架中

## 许可证

Copyright 2025 The RLinf Authors. Licensed under the Apache License, Version 2.0.
