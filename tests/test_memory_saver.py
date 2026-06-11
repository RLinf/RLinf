from torch_memory_saver import torch_memory_saver
import torch
import time


# 1. For tensors that wants to be paused, create them within `region`
with torch_memory_saver.region():
    pauseable_tensor = torch.full((10_000_000_000,), 100, dtype=torch.uint8, device='npu')
    
time.sleep(10)

# 2. After `pause`, NPU memory is released for those tensors.
# For example, check `npu-smi info`'s memory usage to verify.
torch_memory_saver.pause()

time.sleep(10)

# 3. After `resume`, NPU memory is re-occupied for those tensors.
torch_memory_saver.resume()
