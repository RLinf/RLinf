# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import gc
import importlib
import os
import random
import sys
from contextlib import contextmanager
from functools import partial, wraps
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from rlinf.utils.metric_utils import compute_loss_mask


def clear_memory(sync=True):
    from rlinf.scheduler.worker.worker import Worker

    if sync:
        Worker.torch_platform.synchronize()
    gc.collect()
    Worker.torch_platform.ipc_collect()
    Worker.torch_platform.empty_cache()


def apply_func_to_dict(func, dictionary):
    return {k: func(v) for k, v in dictionary.items()}


def move_to_device_if_tensor(device, item):
    if torch.is_tensor(item):
        item = item.to(device)
    return item


cuda_dict = partial(apply_func_to_dict, partial(move_to_device_if_tensor, "cuda"))
cpu_dict = partial(apply_func_to_dict, partial(move_to_device_if_tensor, "cpu"))
_UINT32_MOD = 2**32


def materialize_tensor(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    """Materialize a DTensor into a dense tensor, or return tensors as-is."""
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    assert isinstance(tensor, torch.Tensor), "Expected a torch.Tensor or DTensor"
    return tensor


def normalize_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Normalize string dtype aliases into torch.dtype values."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        key = dtype.lower()
        if key in mapping:
            return mapping[key]
    raise TypeError(f"Unsupported dtype: {dtype}")


def normalize_device(device: torch.device | str | None) -> torch.device:
    """Convert a device string into torch.device, defaulting to the worker device."""
    if device is None:
        from rlinf.scheduler.worker.worker import Worker

        device = Worker.torch_device_type
    return device if isinstance(device, torch.device) else torch.device(device)


def collect_param_names_need_sync(module: torch.nn.Module) -> list[str]:
    """Collect trainable parameters and persistent buffers for selective sync."""
    trainable_param_names = [
        name
        for name, param in module.named_parameters(remove_duplicate=False)
        if param.requires_grad
    ]

    persistent_buffer_names: list[str] = []
    for module_name, submodule in module.named_modules(remove_duplicate=False):
        non_persistent_buffers = getattr(
            submodule, "_non_persistent_buffers_set", set()
        )
        for buffer_name, _ in submodule.named_buffers(
            recurse=False, remove_duplicate=False
        ):
            if buffer_name in non_persistent_buffers:
                continue
            full_name = (
                buffer_name if not module_name else f"{module_name}.{buffer_name}"
            )
            persistent_buffer_names.append(full_name)

    return trainable_param_names + persistent_buffer_names


def synchronize_pending_accel_copies(copy_devices: set[torch.device]) -> None:
    """Wait for queued accelerator copies before host-side consumption."""
    if not copy_devices:
        return

    events: list[torch.Event] = []
    for device in copy_devices:
        from rlinf.scheduler.worker.worker import Worker

        event = Worker.torch_platform.Event()
        event.record(Worker.torch_platform.current_stream(device))
        events.append(event)

    for event in events:
        event.synchronize()


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and PyTorch RNGs."""
    normalized_seed = int(seed)
    numpy_seed = normalized_seed % _UINT32_MOD

    random.seed(normalized_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(normalized_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(normalized_seed)
        torch.cuda.manual_seed_all(normalized_seed)

    return normalized_seed


def retrieve_model_state_dict_in_cpu(model, offloaded_buffer=None):
    """get a copy of the model states in CPU"""
    if offloaded_buffer is None:
        offloaded_buffer = {}

    for name, item in model.state_dict().items():
        if isinstance(item, torch.Tensor):
            if name in offloaded_buffer:
                offloaded_buffer[name].copy_(item.detach(), non_blocking=True)
            else:
                item = (
                    item.detach()
                    .to(device="cpu", non_blocking=True, copy=True)
                    .pin_memory()
                )
                offloaded_buffer[name] = item
        else:
            offloaded_buffer[name] = item
    from rlinf.scheduler.worker.worker import Worker

    Worker.torch_platform.synchronize()
    return offloaded_buffer


@torch.no_grad()
def swap_dict(
    resident_model, cpu_weights, offload_onto_cpu=True, offloaded_buffer=None
):
    """swap the state dict with a specified state dict, and offload the current state dict onto CPU
    if needed
    """
    if offloaded_buffer is None:
        offloaded_buffer = {}

    if offload_onto_cpu:
        offloaded_buffer = retrieve_model_state_dict_in_cpu(
            resident_model, offloaded_buffer
        )

    resident_model.load_state_dict(cpu_weights)
    return offloaded_buffer


@contextmanager
def cpu_weight_swap(resident_model, cpu_weights, offloaded_buffer=None):
    """swap the weights into GPU, and then swap it out once return"""
    offloaded_buffer = swap_dict(
        resident_model, cpu_weights, offloaded_buffer=offloaded_buffer
    )

    try:
        yield

    finally:
        swap_dict(resident_model, offloaded_buffer, offload_onto_cpu=False)


def _get_nvtx_module():
    try:
        return importlib.import_module("nvtx")
    except ImportError:
        return None


@contextmanager
def nvtx_range(name: str, color: str | int | None = None):
    """Annotate a code range for Nsight or other NVTX-aware profilers."""
    nvtx_module = _get_nvtx_module()
    if nvtx_module is not None:
        annotate_kwargs = {"message": name}
        if color is not None:
            annotate_kwargs["color"] = color
        with nvtx_module.annotate(**annotate_kwargs):
            yield
        return

    from rlinf.utils.logging import get_logger

    get_logger().warning(
        "nvtx_range: NVTX module not found, NVTX annotations are disabled. "
        "Using torch.cuda.nvtx instead",
    )

    if hasattr(torch.cuda, "nvtx") and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
        return
    get_logger().warning(
        "nvtx_range: torch.cuda.nvtx is not available, NVTX annotations are disabled."
    )
    yield


def configure_batch_sizes(rank, mbs, gbs, dp=1):
    from megatron.core.num_microbatches_calculator import (
        reconfigure_num_microbatches_calculator,
    )

    reconfigure_num_microbatches_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        data_parallel_size=dp,
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis=None):
    """Compute mean of tensor with a masked values."""
    if mask is None:
        return values.mean(axis=axis)
    elif (~mask).all():
        return (values * mask).sum(axis=axis)
    else:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis=None):
    """Compute sum of tensor with a masked values."""
    return (values * mask).sum(axis=axis)


def seq_mean_token_sum(values: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    seq_losses = torch.sum(values * mask, dim=-1)  # token-sum
    loss = torch.mean(seq_losses)  # seq-mean
    return loss


def seq_mean_token_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    seq_losses = torch.sum(values * mask, dim=-1) / torch.sum(
        mask, dim=-1
    )  # token-mean
    loss = torch.mean(seq_losses)  # seq-mean
    return loss


def masked_mean_ratio(
    values: torch.Tensor, mask: torch.Tensor, loss_mask_ratio: torch.Tensor
):
    # for embodied tasks
    return (values / loss_mask_ratio * mask).mean()


def get_loss_agg_func(
    loss_agg: str,
) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
    """
    Get loss aggregation function based on the loss_agg string.

    Args:
        loss_agg (str): The loss aggregation method. Options are:
            - "seq-mean-token-sum": Sequence mean of token sums.
            - "seq-mean-token-mean": Sequence mean of token means.
            - "token-mean": Mean over tokens.

    Returns:
        Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]: A function that takes values, mask, and dim as inputs and returns the aggregated
    """
    if loss_agg == "seq-mean-token-sum":
        return seq_mean_token_sum
    elif loss_agg == "seq-mean-token-mean":
        return seq_mean_token_mean
    elif loss_agg == "token-mean":
        return masked_mean
    else:
        raise ValueError(f"Unsupported loss aggregation method: {loss_agg}")


def reshape_entropy(
    entropy: Optional[torch.Tensor],
    entropy_type: str,
    action_dim: int = 7,
    batch_size: int = 1,
) -> Optional[torch.Tensor]:
    """
    Reshape entropy based on the entropy type.If entropy is None, return None.
    If entropy_type is "action_level", reshape entropy to [batch_size, seq_len] by summing over action_dim.
    If entropy_type is "chunk_level", reshape entropy to [batch_size, seq_len]

    Args:
        entropy(Optional[torch.Tensor]): [B, seq_len * action_dim] or [B, seq_len] or None
        entropy_type(str): "action_level" or "chunk_level"
        action_dim(int): action dimension, default is 7

    Returns:
        entropy(Optional[torch.Tensor]): reshaped entropy or None
    """
    if entropy is not None:
        if entropy_type == "action_level":
            entropy = entropy.reshape(batch_size, -1, action_dim).sum(dim=-1)
        elif entropy_type == "chunk_level":
            entropy = entropy.sum(dim=-1)
    return entropy


def logprobs_from_logits_flash_attn(
    logits: torch.Tensor, labels: torch.Tensor, inplace_backward: bool = True
) -> torch.Tensor:
    """
    Compute logprobs by logits using flash-attn's cross_entropy_loss.

    Args:
        logits(torch.Tensor): [B*seq-len, vocab-size]
        labels(torch.Tensor): [B*seq-len]
        inplace_backward(bool): whether to use inplace backward to save memory

    Returns:
        logprobs(torch.Tensor): [B*seq-len]
    """
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    return -output[0]


def logprobs_from_logits_liger_kernel(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute logprobs by logits using liger-kernel's cross_entropy_loss.

    Args:
        logits(torch.Tensor): [B*seq-len, vocab-size]
        labels(torch.Tensor): [B*seq-len]

    Returns:
        logprobs(torch.Tensor): [B*seq-len]
    """
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

    loss_func = LigerCrossEntropyLoss(reduction="none")
    logprobs = -loss_func(logits, labels)
    return logprobs


def compute_logprobs_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    op_type: Literal["torch", "flash_attn", "liger_kernel"] = "torch",
) -> torch.Tensor:
    """
    Compute logprobs by logits.

    Args:
        logits(torch.Tensor): [B, seq-len, vocab-size]
        target(torch.Tensor): [B, seq-len]
        op_type(str): the type of logprobs computation method, options are "torch", "flash_attn", "liger_kernel"
            default is "torch".
            flash_attn calc in fp32, torch and liger_kernel calc in logits.dtype.

    Returns:
        logprobs(torch.Tensor): [B, seq-len]
    """
    batch_dim = logits.shape[:-1]
    last_dim = logits.shape[-1]
    logits = logits.reshape(-1, last_dim)
    labels = target.reshape(-1)

    assert op_type in ["torch", "flash_attn", "liger_kernel"], (
        f"Unsupported op_type: {op_type} for logprobs computation. Supported types are 'torch', 'flash_attn', 'liger_kernel'."
    )
    if op_type == "liger_kernel":
        # liger_kernel will use input dtype to compute logprobs.
        logprobs = logprobs_from_logits_liger_kernel(logits, labels)
    elif op_type == "flash_attn":
        # flash_attn will use fp32 to compute logprobs.
        logprobs = logprobs_from_logits_flash_attn(logits, labels)
    elif op_type == "torch":
        # torch will use input dtype to compute logprobs.
        logprobs = -F.cross_entropy(logits, labels, reduction="none")

    # reshape back to [B, seq-len]
    logprobs = logprobs.view(*batch_dim).float()
    return logprobs


def compute_entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy by logits,formula: H(X) = - sum(p(x) * log(p(x)))
    In case logits are too small to cause numerical instability(like downflow to zero after softmax),
    we use log_softmax to compute(it will automatically stabilize the computation) logp.

    Args:
        - logits(torch.Tensor): [B,seq-len,vocab-size]
        - dim(int): the dimension to compute entropy
    Returns:
        - entropy(torch.Tensor): [B, seq-len]
    """
    logp = F.log_softmax(logits, dim=dim)
    p = logp.exp()
    # if some p are zero, p*logp will be nan, we set those terms to zero
    entropy_term = torch.where(p > 0, p * logp, 0.0)
    entropy = -entropy_term.sum(dim=dim)
    return entropy


class DualOutput:
    def __init__(self, file, terminal):
        self.file = file
        self.terminal = terminal

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()  # Flush immediately to ensure the data is written.

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def fileno(self):
        # Return the terminal's fileno to maintain expected behavior
        return self.terminal.fileno()

    def isatty(self):
        return self.terminal.isatty()

    def close(self):
        self.flush()
        self.file.close()

    def readable(self):
        return False

    def writable(self):
        return True

    def seekable(self):
        return False


def output_redirector(func):
    @wraps(func)
    def wrapper(cfg, *args, **kwargs):
        log_path = os.path.join(
            cfg.runner.output_dir, cfg.runner.experiment_name, "log", "main.log"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        f = open(log_path, "w", encoding="utf-8", buffering=1)

        def close():
            dual_out.flush()
            dual_err.flush()
            f.flush()
            f.close()

        atexit.register(close)

        dual_out = DualOutput(f, sys.stdout)
        dual_err = DualOutput(f, sys.stderr)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = dual_out
            sys.stderr = dual_err
            return func(cfg, *args, **kwargs)

        except Exception as e:
            import traceback

            error_msg = f"\nException occurred: {e}\n{traceback.format_exc()}\n"
            dual_err.write(error_msg)
            dual_err.flush()
            f.flush()
            raise

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return wrapper


def warmup_optimizer_state(optimizer: Optimizer) -> None:
    """
    pre initialize optimizer.state to avoid KeyError during subsequent load_state_dict/set_optimizer_state_dict.
    This function does not modify parameter values (by temporarily setting lr to zero + using zero gradients
    to achieve this).
    Suitable for mainstream optimizers such as Adam/AdamW/SGD/RMSprop/Adagrad/Adamax/Adadelta (including fused/foreach variants).
    Not suitable for LBFGS (requires closure and multiple forward/backward passes);
    if using LBFGS, please manually initialize or switch to another optimizer and then switch back.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        raise RuntimeError("fake_optimizer_step does not support LBFGS")

    def zero_grad_like(p):
        if getattr(p, "is_meta", False) or (
            hasattr(p, "device") and p.device.type == "meta"
        ):
            return None  # skip meta
        try:
            if p.layout is torch.strided and not isinstance(p, DTensor):
                return torch.zeros_like(p, memory_format=torch.preserve_format)
        except Exception:
            pass
        return p.detach().new_zeros(p.shape)

    # backup every param group's lr
    saved_lrs = []
    for g in optimizer.param_groups:
        saved_lrs.append(g.get("lr", None))
        g["lr"] = 0.0

    # backup every param's grad, and fill zero grad (ensure every param will init state during step())
    saved_grads = {}
    all_params = []
    for g in optimizer.param_groups:
        for p in g.get("params", []):
            if p is None:
                continue
            all_params.append(p)
            saved_grads[p] = p.grad  # may be None, save as is
            if p.grad is None:
                p.grad = zero_grad_like(p)

    # step to create optimizer.state entries
    # use torch.no_grad to avoid any unexpected side effects from custom optimizers
    with torch.no_grad():
        optimizer.step()

    # restore every param group's lr
    for g, lr in zip(optimizer.param_groups, saved_lrs):
        if lr is not None:
            g["lr"] = lr

    for p in all_params:
        p.grad = saved_grads[p]


def get_rng_state() -> dict:
    """
    Get the current RNG state for both CPU and CUDA (if available).

    Returns:
        dict: A dictionary containing the RNG states("cpu", "numpy", "random", and optionally "cuda").
    """
    rng_state = {
        "cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    from rlinf.scheduler.worker.worker import Worker

    if Worker.torch_platform.is_available():
        rng_state[Worker.torch_device_type] = Worker.torch_platform.get_rng_state()
    return rng_state


def set_rng_state(rng_state: dict) -> None:
    """
    Set the RNG state for both CPU and CUDA (if available) from the provided state dictionary.

    Args:
        rng_state (dict): A dictionary containing the RNG states("cpu", "numpy", "random", and optionally "cuda").
    """
    required_keys = ["cpu", "numpy", "random"]
    assert set(required_keys).issubset(rng_state.keys()), (
        f"rng_state must contain the keys: {required_keys}"
    )
    torch.set_rng_state(rng_state["cpu"])
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["random"])
    from rlinf.scheduler.worker.worker import Worker

    if Worker.torch_platform.is_available() and Worker.torch_device_type in rng_state:
        Worker.torch_platform.set_rng_state(rng_state[Worker.torch_device_type])


PIPELINE_BATCH_KEY_SEPARATOR = "::"


def pack_batch(batch: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    packed_batch: dict[str, Any] = {}
    for key, value in batch.items():
        packed_key = (
            key if not prefix else f"{prefix}{PIPELINE_BATCH_KEY_SEPARATOR}{key}"
        )
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            packed_batch[packed_key] = value
        elif isinstance(value, dict):
            packed_batch.update(pack_batch(value, prefix=packed_key))
        else:
            raise ValueError(
                f"Unsupported value type in batch: {type(value)} for key: {key}"
            )
    return packed_batch


def unpack_batch(batch: dict[str, Any]) -> dict[str, Any]:
    unpack_batch_dict: dict[str, Any] = {}
    for packed_key, value in batch.items():
        cursor = unpack_batch_dict
        key_parts = packed_key.split(PIPELINE_BATCH_KEY_SEPARATOR)
        for key_part in key_parts[:-1]:
            cursor = cursor.setdefault(key_part, {})
        cursor[key_parts[-1]] = value
    return unpack_batch_dict


def flatten_embodied_batch(
    batch: dict[str, Any], shuffle_id: torch.Tensor
) -> dict[str, Any]:
    # here to flatten the batch for embodied data, which is originally
    # in shape [T, B, ...] to [T*B, ...], and also shuffle the batch according to shuffle_id
    ret_dict: dict[str, Any] = {}
    for key, value in batch.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            # could we just ignore it?
            ret_dict[key] = None
        elif isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = flatten_embodied_batch(value, shuffle_id)
    return ret_dict


def merge_rollout_epochs(batch: dict[str, Any], rollout_epoch: int) -> dict[str, Any]:
    ret_dict: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # here to merge the batch for embodied data
            # which is originally in shape [rollout_epoch, B, ...] to [rollout_epoch*B, ...]
            new_value = value.reshape(rollout_epoch, -1, *value.shape[1:]).transpose(
                0, 1
            )
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = merge_rollout_epochs(value, rollout_epoch)
        else:
            raise ValueError(
                f"Unsupported value type in batch: {type(value)} for key: {key}"
            )
    return ret_dict


def preprocess_embodied_batch(
    batch: dict[str, Any],
    *,
    rollout_epoch: int,
    auto_reset: bool,
    ignore_terminations: bool,
    reward_type: str,
    filter_rewards: bool,
    group_size: int,
    rewards_lower_bound: float | None = None,
    rewards_upper_bound: float | None = None,
) -> dict[str, torch.Tensor]:
    batch = merge_rollout_epochs(batch, rollout_epoch)

    if not auto_reset and not ignore_terminations:
        dones = batch["dones"]
        loss_mask, loss_mask_sum = compute_loss_mask(dones)

        if reward_type == "chunk_level":
            loss_mask = loss_mask.any(dim=-1, keepdim=True)
            loss_mask_sum = loss_mask_sum[..., -1:]

        batch["loss_mask"] = loss_mask
        batch["loss_mask_sum"] = loss_mask_sum

    if filter_rewards:
        rewards = batch["rewards"]
        if batch.get("loss_mask", None) is not None:
            rewards = rewards * batch["loss_mask"]
        n_chunk_step, batch_size, _ = rewards.shape

        assert batch_size % group_size == 0, (
            f"batch {batch_size} not divisible by group_size {group_size}"
        )
        n_prompts = batch_size // group_size

        rewards = rewards.transpose(0, 1).reshape(rewards.shape[1], -1)
        reward_matrix = rewards.reshape(n_prompts, group_size, rewards.shape[-1])
        reward_matrix = reward_matrix.sum(dim=-1)
        mean_reward_in_group = reward_matrix.mean(dim=1)

        reward_filter_mask = (mean_reward_in_group >= rewards_lower_bound) & (
            mean_reward_in_group <= rewards_upper_bound
        )
        reward_filter_mask = reward_filter_mask.repeat_interleave(group_size)
        reward_filter_mask = (
            reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
        )

        if batch.get("loss_mask", None) is not None:
            batch["loss_mask"] = reward_filter_mask & batch["loss_mask"]
        else:
            batch["loss_mask"] = reward_filter_mask

    return batch
