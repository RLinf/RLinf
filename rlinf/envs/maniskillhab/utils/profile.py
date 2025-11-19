import os
import subprocess as sp
import time
from contextlib import contextmanager
from typing import Literal

import psutil

import torch


def flatten_dict_keys(d: dict, prefix=""):
    """Flatten a dict by expanding its keys recursively."""
    out = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_dict_keys(v, prefix + k + "/"))
        else:
            out[prefix + k] = v
    return out


class Profiler:
    """
    A simple class to help profile/benchmark simulator code
    """

    def __init__(self) -> None:
        self.stats = dict()

    @contextmanager
    def profile(self, name: str, total_steps: int, num_envs: int):
        print(f"start recording {name} metrics")
        process = psutil.Process(os.getpid())
        cpu_mem_use = process.memory_info().rss
        gpu_mem_use = torch.cuda.mem_get_info()
        torch.cuda.synchronize()
        stime = time.time()
        yield
        dt = time.time() - stime
        # dt: delta time (s)
        # fps: frames per second
        # psps: parallel steps per second (number of env.step calls per second)
        # NOTE (arth): per second stats include other code, e.g. sampling actions
        self.stats[name] = dict(
            dt=dt,
            fps=total_steps * num_envs / dt,
            psps=total_steps / dt,
            total_steps=total_steps,
            cpu_mem_use=cpu_mem_use,
            gpu_mem_use=gpu_mem_use[1] - gpu_mem_use[0],
        )
        torch.cuda.synchronize()
