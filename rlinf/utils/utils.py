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
import os
import random
import sys
import uuid
from contextlib import contextmanager
from functools import partial, wraps
from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


def clear_memory(sync=True):
    if sync:
        torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def apply_func_to_dict(func, dictionary):
    return {k: func(v) for k, v in dictionary.items()}


def move_to_device_if_tensor(device, item):
    if torch.is_tensor(item):
        item = item.to(device)
    return item


cuda_dict = partial(apply_func_to_dict, partial(move_to_device_if_tensor, "cuda"))
cpu_dict = partial(apply_func_to_dict, partial(move_to_device_if_tensor, "cpu"))


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

    torch.cuda.synchronize()
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
    """Compute mean of tensor with a masked values."""
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
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state()
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
    if torch.cuda.is_available() and "cuda" in rng_state:
        torch.cuda.set_rng_state(rng_state["cuda"])


def get_model_weights_id(model, k=128):
    first_p = None
    last_p = None

    for _, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        if first_p is None:
            first_p = p
        last_p = p

    if first_p is None or last_p is None:
        return None

    def tensor_fingerprint(p):
        flat = p.detach().view(-1)
        sample = flat[:k] if flat.numel() >= k else flat
        return sample.to(dtype=torch.float32).cpu().numpy().tobytes()

    name_bytes = tensor_fingerprint(first_p) + tensor_fingerprint(last_p)
    name_str = name_bytes.hex()

    return uuid.uuid5(uuid.NAMESPACE_DNS, name_str)
