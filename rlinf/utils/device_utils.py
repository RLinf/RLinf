import torch
import os

# Detect the backend architecture
if hasattr(torch, 'npu') and torch.npu.is_available():
    _BACKEND = torch.npu
    DEVICE_NAME = "npu"
    EVENT = torch.npu.Event
elif hasattr(torch, 'cuda') and torch.cuda.is_available():
    _BACKEND = torch.cuda
    DEVICE_NAME = "cuda"
    EVENT = torch.cuda.Event
else:
    # Fallback to CPU to prevent crashes if code runs locally
    _BACKEND = None
    DEVICE_NAME = "cpu"
    EVENT = None

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

def is_available():
    return _BACKEND is not None

def empty_cache():
    if _BACKEND:
        _BACKEND.empty_cache()

def synchronize(device=None):
    if _BACKEND:
        _BACKEND.synchronize(device)

def memory_allocated(device=None):
    if _BACKEND:
        return _BACKEND.memory_allocated(device)
    return 0

def memory_reserved(device=None):
    if _BACKEND:
        return _BACKEND.memory_reserved(device)
    return 0

def current_device():
    if _BACKEND:
        return _BACKEND.current_device()
    return 0

def set_device(device):
    if _BACKEND:
        _BACKEND.set_device(device)

def mem_get_info():
    if _BACKEND:
        return _BACKEND.mem_get_info()

def device_count():
    if _BACKEND:
        return _BACKEND.device_count()

def is_initialized():
    if _BACKEND:
        return _BACKEND.is_initialized()
    
def get_rng_state():
    if _BACKEND:
        return _BACKEND.get_rng_state()
    


# -----------------------------------------------------------------------------
# Device Object Helpers
# -----------------------------------------------------------------------------

def get_device_object(index=None):
    """
    Returns a torch.device object for the current backend.
    Example: returns torch.device("npu:0") or torch.device("cuda:0")
    """
    if index is None:
        return torch.device(DEVICE_NAME)
    return torch.device(f"{DEVICE_NAME}:{index}")

# -----------------------------------------------------------------------------
# Tensor Movement Helpers
# -----------------------------------------------------------------------------

def to_device(tensor_or_module):
    """
    Moves a tensor or model to the automatically detected device.
    Replaces: x.cuda() or x.npu()
    """
    return tensor_or_module.to(DEVICE_NAME)