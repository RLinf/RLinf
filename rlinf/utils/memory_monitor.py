from __future__ import annotations

import argparse
import json
import sys
import time
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, Type

from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType, AcceleratorUtil


@dataclass(frozen=True)
class MemorySummary:
    accelerator_type: str
    device_index: int
    samples: int
    min_gib: float        # Minimum used memory detected
    max_gib: float        # Maximum used memory detected
    mean_gib: float       # Average used memory
    total_gib: float      # Total device capacity
    util: float # (max_gib / total_gib) * 100

class AcceleratorMonitor(ABC):
    """Abstract base class for extensible accelerator monitoring."""
    _registry: Dict[AcceleratorType, Type[AcceleratorMonitor]] = {}

    @classmethod
    def register(cls, atype: AcceleratorType):
        def decorator(subcls: Type[AcceleratorMonitor]):
            cls._registry[atype] = subcls
            return subcls
        return decorator

    @classmethod
    def create(cls, atype: AcceleratorType | None = None) -> AcceleratorMonitor:
        atype = atype or AcceleratorUtil.get_accelerator_type()
        if atype not in cls._registry:
            raise NotImplementedError(f"No monitor implementation for {atype}")
        return cls._registry[atype]()

    def __enter__(self): return self
    def __exit__(self, *args): pass

    @abstractmethod
    def read_usage(self, index: int) -> tuple[int, int, int]: 
        """Returns (used_bytes, total_bytes, free_bytes)."""

    def sample(self, index: int, interval: float, duration: float) -> MemorySummary:
        """Polls the device for a fixed duration and returns memory usage statistics."""
        max_bytes = 0
        min_bytes = float('inf')
        total_sum, count = 0, 0
        device_total_bytes = 0
        end_time = time.perf_counter() + duration

        with self:
            while time.perf_counter() < end_time:
                used, total, _ = self.read_usage(index)
                
                # Update rolling metrics
                if used > max_bytes: max_bytes = used
                if used < min_bytes: min_bytes = used
                
                total_sum += used
                count += 1
                device_total_bytes = total
                
                time.sleep(interval)

        actual_min = min_bytes if count > 0 else 0
        mean_bytes = total_sum / count if count > 0 else 0
        utilization = (max_bytes / device_total_bytes * 100) if device_total_bytes > 0 else 0

        return MemorySummary(
            accelerator_type=getattr(self, "type_name", "unknown"),
            device_index=index,
            samples=count,
            min_gib=round(actual_min / 2**30, 2),
            max_gib=round(max_bytes / 2**30, 2),
            mean_gib=round(mean_bytes / 2**30, 2),
            total_gib=round(device_total_bytes / 2**30, 2),
            util=round(utilization, 2)
        )

@AcceleratorMonitor.register(AcceleratorType.NV_GPU)
class NvidiaGPUMonitor(AcceleratorMonitor):
    type_name = "NV_GPU"
    
    def __init__(self):
        try:
            from ray._private.thirdparty import pynvml
        except ImportError:
            import pynvml
        self.pynvml = pynvml

    def __enter__(self):
        self.pynvml.nvmlInit()
        return self

    def __exit__(self, *args):
        try: self.pynvml.nvmlShutdown()
        except: pass

    def read_usage(self, index: int) -> tuple[int, int, int]:
        handle = self.pynvml.nvmlDeviceGetHandleByIndex(index)
        info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used, info.total, info.free

def spawn_monitor(index: int, interval: float, duration: float) -> MemorySummary:
    """Execute the monitor in a subprocess to isolate GPU driver contexts."""
    cmd = [
        sys.executable, "-m", __name__, 
        "--index", str(index), 
        "--interval", str(interval), 
        "--duration", str(duration)
    ]
    result = subprocess.check_output(cmd, text=True).splitlines()[-1]
    return MemorySummary(**json.loads(result))

def monitor():
    parser = argparse.ArgumentParser(description="Accelerator Memory Sampler (GiB)")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()

    monitor_inst = AcceleratorMonitor.create()
    summary = monitor_inst.sample(args.index, args.interval, args.duration)
    print(json.dumps(asdict(summary)))

if __name__ == "__main__":
    monitor()