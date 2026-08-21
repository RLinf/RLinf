"""Pinned-memory offload context manager for accelerator-agnostic model offload."""

from contextlib import contextmanager

import torch


@contextmanager
def pinned_offload_context():
    """Temporarily patch Tensor.to to use pinned memory for GPU->CPU offload.

    When a tensor on a non-CPU device is moved to CPU via .to('cpu') with no
    dtype change, the patch replaces the default pageable copy with:
      1. Allocate a pinned-CPU buffer (torch.empty_like(device='cpu', pin_memory=True))
      2. Synchronous copy GPU->pinned (non_blocking=False for correctness)
      3. Return the pinned buffer

    CPU->GPU direction is NOT patched: once data sits in pinned memory, the
    original Tensor.to(device) already uses full PCIe bandwidth.
    """
    orig_to = torch.Tensor.to

    def patched_to(self, *args, **kwargs):
        device = None
        dtype = None

        if args:
            device = args[0]
            if len(args) >= 2 and isinstance(args[1], (torch.dtype, type(None))):
                dtype = args[1]
        if device is None:
            device = kwargs.get("device")
        if dtype is None:
            dtype = kwargs.get("dtype")

        if (
            device is not None
            and dtype is None
            and torch.device(device).type == "cpu"
            and self.device.type != "cpu"
        ):
            buf = torch.empty_like(self, device="cpu", pin_memory=True)
            buf.copy_(self)
            return buf

        return orig_to(self, *args, **kwargs)

    torch.Tensor.to = patched_to
    try:
        yield
    finally:
        torch.Tensor.to = orig_to
