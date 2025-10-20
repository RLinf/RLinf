# Ray序列化错误修复说明

## 问题描述

在运行RStar2训练时，遇到以下Ray序列化错误：

```
TypeError: no default __reduce__ due to non-trivial __cinit__
Could not serialize the argument {'python_code_with_standard_io': <toolkits.rstar2.tools.code_judge_tool.PythonTool object at 0x7f627a3c5690>}
```

**根本原因**：`PythonTool` 对象包含不可序列化的组件（`aiohttp.ClientSession`、`uvloop.loop.Loop`、`RequestProcessor`），无法通过Ray传递给远程worker。

## 解决方案

修改代码架构，**不直接传递工具实例**，而是传递**工具配置**（可序列化），让每个Ray worker在本地初始化自己的工具实例。

## 修改的文件

### 1. `rlinf/workers/agent_loop/tool_agent_loop.py`

**主要改动**：
- 添加 `tool_config` 和 `_tools_initialized` 成员变量
- 修改 `run_agentloop_rollout()` 方法，接收 `tool_config`（配置）而不是 `tools`（实例）
- 新增 `_initialize_tools_locally()` 方法，在worker本地初始化工具

```python
async def run_agentloop_rollout(self, input_channel: Channel, output_channel: Channel, tool_config, rollout):
    """
    Args:
        tool_config: 工具配置（可序列化的配置路径或配置字典），而不是工具实例
    """
    self.tool_config = tool_config
    self.rollout = rollout
    
    # 在worker本地初始化工具
    if not self._tools_initialized:
        await self._initialize_tools_locally()
        self._tools_initialized = True
```

### 2. `rlinf/runners/rstar2_runner.py`

**主要改动**：
- 将 `_get_shared_tools()` 改为 `_get_tool_config()`
- 返回可序列化的配置（字符串路径或dict）而不是工具实例
- 在调用 `run_agentloop_rollout()` 时传递 `tool_config` 而不是 `tools`

```python
def _get_tool_config(self):
    """获取工具配置（可序列化），而不是工具实例"""
    tool_config_path = getattr(self.cfg, 'tool_config_path', None)
    # ...
    if tool_config_path is not None:
        return tool_config_path  # 返回路径字符串
    else:
        return OmegaConf.to_container(self.cfg.tools, resolve=True)  # 返回dict
```

### 3. `toolkits/rstar2/tools/shared_tool_manager.py`

**主要改动**：
- 修改 `_create_tools_from_config_object()` 方法签名，支持 `Union[DictConfig, Dict]`
- 添加统一的配置值获取逻辑，同时支持dict和DictConfig

## 配置方法

### 方式1：在训练配置文件中直接配置

在您的训练配置文件（如 `config.yaml`）中添加：

```yaml
tools:
  code_judge:
    host_addr: "localhost"
    host_port: 8088
    batch_size: 4
    concurrency: 2
    batch_timeout_seconds: 30.0
    enable_jupyter: false
```

### 方式2：使用独立的工具配置文件

在训练配置文件中指定工具配置文件路径：

```yaml
rollout:
  tool_config_path: "toolkits/rstar2/tools/tool_config/python_tool_config.yaml"
```

或者：

```yaml
tool_config_path: "toolkits/rstar2/tools/tool_config/python_tool_config.yaml"
```

## 工作原理

1. **主进程**：
   - 获取工具配置（路径字符串或dict）
   - 通过Ray远程调用传递配置（可序列化）

2. **Ray Worker**：
   - 接收工具配置
   - 在本地创建 `aiohttp.ClientSession`
   - 在本地初始化工具实例
   - 使用本地工具实例执行任务

3. **关键优势**：
   - 避免序列化网络连接和事件循环
   - 每个worker拥有独立的工具实例
   - 避免资源竞争

## 测试

修复后，您的训练应该能够正常启动，不再出现序列化错误。每个Ray worker会在首次需要时自动初始化自己的工具实例。

## 参考

- [Ray序列化文档](https://docs.ray.io/en/master/ray-core/objects/serialization.html)
- 相关配置文件：`toolkits/rstar2/tools/tool_config/python_tool_config.yaml`

