# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Helper utils"""

import gc
import logging as builtin_logging
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import psutil
import torch
import torch.distributed as dist
import transformers
from torch import nn
from transformers import enable_full_determinism
from transformers import set_seed as set_seed_func

from ..distributed.parallel_state import get_parallel_state
from . import logging
from .count_flops import LingBotFlopsCounter
from .dist_utils import all_reduce
from .import_utils import is_torch_npu_available
from .seqlen_pos_transform_utils import culen2len, pos2culen

if is_torch_npu_available():
    import torch_npu  # noqa: F401 # type: ignore
    from torch_npu.contrib import transfer_to_npu  # noqa: F401 # type: ignore


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.get_logger(__name__)

CACHE_DIR = os.path.expanduser(
    os.getenv("CACHE_DIR", os.path.join("~/.cache", "lingbotvla"))
)


def _compute_seqlens(
    micro_batch: dict[str, "torch.Tensor"], rmpad: bool, rmpad_with_pos_ids: bool
) -> tuple[list[int], Optional[list[int]]]:
    """
    Computes the sequence lengths of the current batch.

    Args:
        micro_batch (Dict[str, Tensor]): The current batch.
        rmpad (bool): Whether to remove the padding tokens.
        rmpad_with_pos_ids (bool): Whether to remove the padding tokens using the position ids.
    """
    if "attention_mask" in micro_batch:
        attention_mask = micro_batch["attention_mask"]
        if rmpad:
            seqlens = culen2len(micro_batch["cu_seqlens"]).tolist()
            seqlens = seqlens[:-1] if (attention_mask == 0).any().item() else seqlens
        elif rmpad_with_pos_ids:
            seqlens = culen2len(pos2culen(micro_batch["position_ids"])).tolist()
            seqlens = seqlens[:-1] if (attention_mask == 0).any().item() else seqlens
        else:
            seqlens = attention_mask.sum(-1).tolist()
    elif "lang_masks" in micro_batch:
        attention_mask = micro_batch["lang_masks"]
        seqlens = attention_mask.sum(-1).tolist()

    return seqlens


class EnvironMeter:
    """
    Computes the metrics about the training efficiency.

    Args:
        config (PretrainedConfig): The configuration of the model.
        global_batch_size (int): The global batch size.
        rmpad (bool, optional): Whether to remove the padding tokens. Defaults to False.
        rmpad_with_pos_ids (bool, optional): Whether to remove the padding tokens using the position ids. Defaults to False.
        empty_cache_steps (int, optional): The number of steps to empty the cache. Defaults to 500.
    """

    def __init__(
        self,
        config: "PretrainedConfig",
        global_batch_size: int,
        rmpad: bool = False,
        rmpad_with_pos_ids: bool = False,
        empty_cache_steps: int = 500,
    ) -> None:
        self.config = config
        self.global_batch_size = global_batch_size
        self.rmpad = rmpad
        self.rmpad_with_pos_ids = rmpad_with_pos_ids
        self.empty_cache_steps = empty_cache_steps
        self.world_size = dist.get_world_size()
        self.consume_tokens = 0
        self.batch_seqlens = []
        self.image_seqlens = []

        self.estimate_flops = LingBotFlopsCounter(config).estimate_flops

    def state_dict(self) -> dict[str, Any]:
        state_dict = {"consume_tokens": self.consume_tokens}

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.consume_tokens = state_dict["consume_tokens"]

    def add(self, micro_batch: dict[str, "torch.Tensor"]) -> None:
        seqlens = _compute_seqlens(micro_batch, self.rmpad, self.rmpad_with_pos_ids)
        if "image_grid_thw" in micro_batch:
            image_grid_thw = micro_batch["image_grid_thw"]
            image_seqlens = torch.repeat_interleave(
                image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
            )
            self.image_seqlens.extend(image_seqlens.tolist())
        elif "lang_masks" in micro_batch:
            self.image_seqlens = [768] * micro_batch["images"].size(0)
            self.state_action_seqlens = [51] * micro_batch["images"].size(0)
        self.batch_seqlens.extend(seqlens)

    def step(self, delta_time: float, global_step: int) -> dict[str, Any]:
        if len(self.image_seqlens) > 0:
            flops_achieved, flops_promised = self.estimate_flops(
                self.batch_seqlens, delta_time, image_seqlens=self.image_seqlens
            )
        else:
            flops_achieved, flops_promised = self.estimate_flops(
                self.batch_seqlens, delta_time
            )

        flops_achieved, batch_tokens, real_global_batch_size = all_reduce(
            (flops_achieved, sum(self.batch_seqlens), len(self.batch_seqlens)),
            op="sum",
            group=get_parallel_state().dp_group,
        )
        flops_promised = flops_promised * self.world_size
        mfu = flops_achieved / flops_promised

        # calculate average effective len and tokens per second
        avg_effective_len = batch_tokens / self.global_batch_size
        avg_sample_seq_len = batch_tokens / real_global_batch_size
        tokens_per_second = batch_tokens / delta_time
        self.consume_tokens += batch_tokens

        # cuda memory
        allocated_memory = torch.cuda.max_memory_allocated()
        reserved_memory = torch.cuda.max_memory_reserved()
        num_alloc_retries = torch.cuda.memory_stats()["num_alloc_retries"]
        allocated_memory, reserved_memory, num_alloc_retries = all_reduce(
            (allocated_memory, reserved_memory, num_alloc_retries), op="max"
        )

        # cpu memory
        cpu_memory_info = psutil.virtual_memory()

        metrics = {
            "flops_achieved(T)": flops_achieved,
            "flops_promised(T)": flops_promised,
            "mfu": mfu,
            "training/avg_effective_len": avg_effective_len,
            "training/avg_sample_seq_len": avg_sample_seq_len,
            "tokens_per_second(M)": tokens_per_second / 1e6,
            "consume_tokens(M)": self.consume_tokens / 1e6,
            "consume_tokens(B)": self.consume_tokens / 1e9,
            "max_memory_allocated(GB)": allocated_memory / (1024**3),
            "max_memory_reserved(GB)": reserved_memory / (1024**3),
            "cpu_used_memory(GB)": cpu_memory_info.used / (1024**3),
            "cpu_available_memory(GB)": cpu_memory_info.available / (1024**3),
            "cpu_memory_usage(%)": cpu_memory_info.percent,
            "num_alloc_retries": num_alloc_retries,
        }

        if self.empty_cache_steps > 0 and global_step % self.empty_cache_steps == 0:
            empty_cache()

        self.batch_seqlens = []
        self.image_seqlens = []

        return metrics


def enable_high_precision_for_bf16():
    """
    Set high accumulation dtype for matmul and reduction.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def set_seed(seed: int, full_determinism: bool = False) -> None:
    """
    Sets a manual seed on all devices.
    """
    if full_determinism:
        enable_full_determinism(seed)
    else:
        set_seed_func(seed)


def create_logger(name: Optional[str] = None) -> "logging._Logger":
    """
    Creates a pretty logger for the third-party program.
    """
    logger = builtin_logging.getLogger(name)
    formatter = builtin_logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    handler = builtin_logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(builtin_logging.INFO)
    logger.propagate = False
    return logger


def enable_third_party_logging() -> None:
    """
    Enables explicit logger of the third-party libraries.
    """
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()


def print_device_mem_info(prompt: str = "VRAM usage") -> None:
    """
    Logs VRAM info.
    """
    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info_rank0(
        f"{prompt}: cur {memory_allocated:.2f}GB, max {max_memory_allocated:.2f}GB."
    )


def print_cpu_memory_info():
    cpu_usage = psutil.cpu_percent(interval=1)  # 1 秒间隔
    logger.info_rank0(f"CPU Usage: {cpu_usage}%")

    memory_info = psutil.virtual_memory()
    logger.info_rank0(f"Total Memory: {memory_info.total / (1024**3):.2f} GB")
    logger.info_rank0(f"Available Memory: {memory_info.available / (1024**3):.2f} GB")
    logger.info_rank0(f"Used Memory: {memory_info.used / (1024**3):.2f} GB")
    logger.info_rank0(f"Memory Usage: {memory_info.percent}%")


def empty_cache() -> None:
    """
    Collects system memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_cache_dir(path: Optional[str] = None) -> str:
    """
    Returns the cache directory for the given path.
    """
    if path is None:
        return CACHE_DIR

    path = os.path.normpath(path)
    if not os.path.splitext(path)[-1]:  # is a dir
        path = os.path.join(path, "")

    path = os.path.split(os.path.dirname(path))[-1]
    return os.path.join(CACHE_DIR, path, "")  # must endswith os.path.sep


@lru_cache
def get_dtype_size(dtype: "torch.dtype") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/v0.4.5/bindings/python/py_src/safetensors/torch.py#L350
    """
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    _SIZE = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
        _float8_e4m3fn: 1,
        _float8_e5m2: 1,
    }
    return _SIZE[dtype]


def unwrap_model(model: "nn.Module") -> "nn.Module":
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Taken from: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_utils.py#L4808
    """
    if hasattr(model, "module"):
        return unwrap_model(getattr(model, "module"))
    else:
        return model


def print_example(example: dict[str, "torch.Tensor"], rank: int) -> None:
    """
    Logs a single example to screen.
    """
    for key, value in example.items():
        if isinstance(value, torch.Tensor):
            logger.info(
                f"[rank {rank}]: {key}'s shape: {value.shape}, device: {value.device}, {value}"
            )
        else:
            logger.info(f"[rank {rank}]: {key}: {value}")
        # logger.info(f"[rank {rank}]: {key}'s shape: {value.shape}, device: {value.device},")


def dict2device(input_dict: dict):
    """
    Move a dict of Tensor to GPUs.
    """
    output_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            output_dict[k] = v.cuda()
        elif isinstance(v, dict):
            output_dict[k] = dict2device(v)
        else:
            output_dict[k] = v
    return output_dict


def make_list(item):
    if isinstance(item, list) or isinstance(item, np.ndarray):
        return item
    return [item]


def create_profiler(
    start_step: int,
    end_step: int,
    trace_dir: str,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
):
    """
    Creates a profiler to record the CPU and CUDA activities. Default export to trace.json.
    Profile steps in [start_step, end_step).

    Args:
        start_step (int): The step to start recording.
        end_step (int): The step to end recording.
        trace_dir (str): The path to save the profiling result.
        record_shapes (bool): Whether to record the shapes of the tensors.
        profile_memory (bool): Whether to profile the memory usage.
        with_stack (bool): Whether to include the stack trace.
    """

    def handler_fn(p):
        torch.profiler.tensorboard_trace_handler(trace_dir)(p)
        logger.info(f"Profiling result saved at {trace_dir}.")

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    warmup = 0 if start_step == 1 else 1
    wait = start_step - warmup - 1
    active = end_step - start_step
    logger.info(
        f"build profiler schedule - wait: {wait}, warmup: {warmup}, active: {active}."
    )
    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=1,
        ),
        on_trace_ready=handler_fn,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_modules=True,
        with_stack=with_stack,
    )
    return profiler
