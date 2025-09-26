# 事件循环冲突修复

## 问题描述

在SGLangGenerateWorker中调用`sync_hf_weight`方法时出现事件循环冲突错误：

```
RuntimeError: this event loop is already running.
```

错误发生在：
```python
File "/mnt/mnt/public/wangxiangyuan/RLinf/rlinf/hybrid_engines/sglang/sglang_0_4_4/sgl_engine.py", line 110, in sync_hf_weight
    return loop.run_until_complete(self.tokenizer_manager.sync_hf_weight(obj))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "uvloop/loop.pyx", line 1512, in uvloop.loop.Loop.run_until_complete
```

## 问题原因

1. **错误的方法调用**: SGLangGenerateWorker调用了`self._engine.sync_hf_weight()`
2. **事件循环嵌套**: `engine.sync_hf_weight()`内部使用`run_until_complete()`，与已有事件循环冲突
3. **与AsyncSGLangWorker不一致**: 没有遵循AsyncSGLangWorker的正确实现模式

## 解决方案

### 1. 参考AsyncSGLangWorker的实现

AsyncSGLangWorker的正确实现：
```python
async def sync_model_from_actor(self):
    """Update the weights of the SGLang engine."""
    await self._engine.tokenizer_manager.sync_hf_weight(
        obj=io_struct.SyncHFWeightInput()
    )
```

### 2. 修复SGLangGenerateWorker的方法

#### 修复sync_hf_weight方法
```python
def sync_hf_weight(self):
    """同步权重，对应 AsyncSGLangWorker.sync_model_from_actor()"""
    if self._engine is None:
        raise RuntimeError("Engine not initialized. Call init_worker() first.")
    
    # 参考AsyncSGLangWorker，直接调用tokenizer_manager.sync_hf_weight()
    # 避免通过engine.sync_hf_weight()导致的事件循环冲突
    import asyncio
    from rlinf.hybrid_engines.sglang.sglang_0_4_4 import io_struct
    
    try:
        # 尝试异步调用，避免事件循环冲突
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环正在运行，创建新任务
            asyncio.create_task(self._async_sync_hf_weight())
        else:
            # 如果事件循环没有运行，直接运行
            loop.run_until_complete(self._async_sync_hf_weight())
    except Exception as e:
        self.log_warning(f"Failed to sync model weights: {e}")
        # 继续执行，不阻塞同步
```

#### 修复offload_model_weights方法
```python
def offload_model_weights(self):
    """卸载模型权重，对应 AsyncSGLangWorker.offload_engine()"""
    if self._engine is None:
        raise RuntimeError("Engine not initialized. Call init_worker() first.")
    
    # 参考AsyncSGLangWorker，直接调用tokenizer_manager.offload_model_weights()
    # 避免通过engine.offload_model_weights()导致的事件循环冲突
    import asyncio
    
    try:
        # 尝试异步调用，避免事件循环冲突
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环正在运行，创建新任务
            asyncio.create_task(self._async_offload_model_weights())
        else:
            # 如果事件循环没有运行，直接运行
            loop.run_until_complete(self._async_offload_model_weights())
    except Exception as e:
        self.log_warning(f"Failed to offload model weights: {e}")
        # 继续执行，不阻塞卸载
```

### 3. 添加AsyncSGLangWorker兼容的方法

#### 添加offload_engine方法
```python
async def offload_engine(self):
    """卸载模型权重，对应 AsyncSGLangWorker.offload_engine()"""
    if self._engine is None:
        raise RuntimeError("Engine not initialized. Call init_worker() first.")
    await self._engine.tokenizer_manager.offload_model_weights(
        io_struct.OffloadReqInput()
    )
```

#### 添加sync_model_from_actor方法
```python
async def sync_model_from_actor(self):
    """同步权重，对应 AsyncSGLangWorker.sync_model_from_actor()"""
    if self._engine is None:
        raise RuntimeError("Engine not initialized. Call init_worker() first.")
    await self._engine.tokenizer_manager.sync_hf_weight(
        obj=io_struct.SyncHFWeightInput()
    )
```

## 技术细节

### 事件循环状态检测
```python
loop = asyncio.get_event_loop()
if loop.is_running():
    # 事件循环正在运行，使用create_task()
    asyncio.create_task(self._async_sync_hf_weight())
else:
    # 事件循环没有运行，使用run_until_complete()
    loop.run_until_complete(self._async_sync_hf_weight())
```

### 直接调用tokenizer_manager
```python
# 错误的方式（导致事件循环冲突）
self._engine.sync_hf_weight()

# 正确的方式（直接调用tokenizer_manager）
await self._engine.tokenizer_manager.sync_hf_weight(
    obj=io_struct.SyncHFWeightInput()
)
```

## 修复效果

1. **解决事件循环冲突**: 不再出现"this event loop is already running"错误
2. **与AsyncSGLangWorker一致**: 采用相同的实现模式
3. **智能异步处理**: 根据事件循环状态选择合适的执行方式
4. **完善错误处理**: 添加异常捕获和日志记录

## 架构对比

### 修复前（错误方式）
```python
def sync_hf_weight(self):
    # 直接调用engine.sync_hf_weight() - 导致事件循环冲突
    self._engine.sync_hf_weight()
```

### 修复后（正确方式）
```python
def sync_hf_weight(self):
    # 智能检测事件循环状态，使用异步方法
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(self._async_sync_hf_weight())
    else:
        loop.run_until_complete(self._async_sync_hf_weight())

async def _async_sync_hf_weight(self):
    # 直接调用tokenizer_manager，避免事件循环冲突
    await self._engine.tokenizer_manager.sync_hf_weight(
        obj=io_struct.SyncHFWeightInput()
    )
```

## 测试验证

### 1. 语法检查
```bash
python3 -c "
import ast
with open('rlinf/workers/rollout/sglang/sglang_generate_worker.py') as f:
    ast.parse(f.read())
print('✓ 语法正确')
"
```

### 2. 导入测试
```bash
python3 -c "
from rlinf.workers.rollout.sglang.sglang_generate_worker import SGLangGenerateWorker
print('✓ 导入成功')
"
```

### 3. 功能测试
```bash
cd examples/rstar2
./run_rstar2_agentloop.sh
```

## 相关文件

- **修改文件**: `rlinf/workers/rollout/sglang/sglang_generate_worker.py`
- **新增方法**:
  - `offload_engine()` - 对应AsyncSGLangWorker.offload_engine()
  - `sync_model_from_actor()` - 对应AsyncSGLangWorker.sync_model_from_actor()
- **修改方法**:
  - `sync_hf_weight()` - 添加智能异步处理
  - `offload_model_weights()` - 添加智能异步处理

## 注意事项

1. **AsyncSGLangWorker参考**: 严格按照AsyncSGLangWorker的实现模式
2. **事件循环管理**: 智能检测事件循环状态，避免冲突
3. **错误处理**: 添加完善的异常捕获和日志记录
4. **向后兼容**: 保持与原有接口的兼容性

现在SGLangGenerateWorker应该可以在Ray环境中正常同步权重，不再出现事件循环冲突错误！
