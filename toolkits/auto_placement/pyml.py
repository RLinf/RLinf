import pynvml

def get_gpu_physical_memory(device_idx=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return info.used / 1024**2  

print(f"显卡实际已用: {get_gpu_physical_memory(0):.2f} MB")