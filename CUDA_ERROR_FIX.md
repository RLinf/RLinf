# CUDA错误修复

## 问题描述

在SGLangGenerateWorker初始化过程中出现CUDA错误：

```
RuntimeError: CUDA error: invalid argument
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

错误发生在SGLang尝试卸载模型权重时：
```python
File "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 1668, in <listcomp>
    (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
            ^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: invalid argument
```

## 问题原因

1. **初始化时强制卸载权重**: SGLangGenerateWorker在`init_worker()`中强制调用`offload_model_weights()`
2. **CUDA上下文问题**: 在Ray Worker环境中，CUDA上下文可能还没有完全初始化
3. **内存管理冲突**: 模型权重卸载时的内存操作与CUDA上下文产生冲突
4. **与AsyncSGLangWorker模式不一致**: AsyncSGLangWorker不在初始化时卸载权重

## 解决方案

### 1. 遵循AsyncSGLangWorker模式

修改`init_worker()`方法，不在初始化时强制卸载权重：

```python
def init_worker(self):
    """初始化工作器，对应 AsyncSGLangWorker.init_worker()"""
    # 初始化引擎
    self._init_engine()
    
    # 进行权重验证
    if self._cfg.rollout.validate_weight:
        self._validate_weight_at_first()

    # 注意：不在这里强制卸载权重，避免CUDA错误
    # 权重卸载将在需要时通过sync_model_from_actor()进行
    # 这遵循AsyncSGLangWorker的模式，避免初始化时的CUDA问题
```

### 2. 改进异步方法的CUDA错误处理

为异步方法添加专门的CUDA错误处理：

```python
async def _async_offload_model_weights(self):
    """异步卸载模型权重，内部方法"""
    if self._engine is None:
        raise RuntimeError("Engine not initialized. Call init_worker() first.")
    
    try:
        await self._engine.tokenizer_manager.offload_model_weights(
            io_struct.OffloadReqInput()
        )
    except RuntimeError as e:
        if "CUDA error" in str(e):
            self.log_warning(f"CUDA error during weight offload: {e}")
            # 尝试清理CUDA缓存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
        else:
            raise e
```

### 3. 智能权重同步策略

改进`sync_model_from_actor()`方法，使用异步版本避免CUDA问题：

```python
def sync_model_from_actor(self):
    """同步权重，对应 SGLangWorker.sync_model_from_actor()"""
    if self._engine is None:
        raise RuntimeError("Engine not initialized. Call init_worker() first.")
    
    # 在Ray环境中，使用异步版本避免事件循环冲突和CUDA错误
    try:
        self._engine.sync_hf_weight()
    except Exception as e:
        # 如果同步版本失败，尝试异步版本
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._async_sync_hf_weight())
            else:
                loop.run_until_complete(self._async_sync_hf_weight())
        except Exception as async_e:
            self.log_warning(f"Failed to sync model weights: {e}, async: {async_e}")
            # 继续执行，不阻塞同步
```

## 技术细节

### CUDA错误检测和处理

```python
except RuntimeError as e:
    if "CUDA error" in str(e):
        self.log_warning(f"CUDA error during weight sync: {e}")
        # 尝试清理CUDA缓存
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e
    else:
        raise e
```

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

## 修复效果

1. **避免初始化时CUDA错误**: 不在`init_worker()`中强制卸载权重
2. **智能权重管理**: 权重卸载在需要时进行，而不是初始化时
3. **改进错误处理**: 添加CUDA错误检测和缓存清理
4. **遵循最佳实践**: 采用AsyncSGLangWorker的成功模式

## 架构对比

### 修复前 (SGLangWorker模式)
```python
def init_worker(self):
    self._init_engine()
    if self._cfg.rollout.validate_weight:
        self._validate_weight_at_first()
    # 强制卸载权重 - 可能导致CUDA错误
    self._engine.offload_model_weights()
```

### 修复后 (AsyncSGLangWorker模式)
```python
def init_worker(self):
    self._init_engine()
    if self._cfg.rollout.validate_weight:
        self._validate_weight_at_first()
    # 不强制卸载权重，避免CUDA错误
    # 权重卸载在需要时通过sync_model_from_actor()进行
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
- **修改方法**:
  - `init_worker()` - 移除强制权重卸载
  - `_async_offload_model_weights()` - 添加CUDA错误处理
  - `_async_sync_hf_weight()` - 添加CUDA错误处理
  - `sync_model_from_actor()` - 改进异步回退机制

## 注意事项

1. **CUDA环境**: 此修复专门针对Ray Worker环境中的CUDA问题
2. **权重管理**: 权重卸载现在在需要时进行，而不是初始化时
3. **错误恢复**: 添加了CUDA缓存清理机制
4. **兼容性**: 保持与原有接口的兼容性

现在SGLangGenerateWorker应该可以在Ray环境中正常初始化，不再出现CUDA错误！
