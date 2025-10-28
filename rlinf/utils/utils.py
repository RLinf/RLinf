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
import sys
from contextlib import contextmanager
from functools import partial, wraps

import numpy as np
import torch
import torch.nn.functional as F


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


def retrieve_model_state_dict_in_cpu(model):
    """get a copy of the model states in CPU"""
    cpu_dict = {}

    for name, item in model.state_dict().items():
        if isinstance(item, torch.Tensor):
            item = item.detach().to(device="cpu", non_blocking=True, copy=True)

        cpu_dict[name] = item

    torch.cuda.synchronize()
    return cpu_dict


@torch.no_grad()
def swap_dict(resident_model, cpu_weights, offload_onto_cpu=True):
    """swap the state dict with a specified state dict, and offload the current state dict onto CPU
    if needed
    """
    offloaded_weights = {}

    if offload_onto_cpu:
        offloaded_weights = retrieve_model_state_dict_in_cpu(resident_model)

    resident_model.load_state_dict(cpu_weights)
    return offloaded_weights


@contextmanager
def cpu_weight_swap(resident_model, cpu_weights):
    """swap the weights into GPU, and then swap it out once return"""
    cpu_dict = swap_dict(resident_model, cpu_weights)

    try:
        yield

    finally:
        swap_dict(resident_model, cpu_dict, offload_onto_cpu=False)


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


def reshape_entropy(entropy, entropy_type, action_dim=7, batch_size=1):
    if entropy is not None:
        if entropy_type == "action_level":
            entropy = entropy.reshape(batch_size, -1, action_dim).sum(dim=-1)
        elif entropy_type == "chunk_level":
            entropy = entropy.sum(dim=-1)
    return entropy


def logprobs_from_logits_flash_attn(logits, labels, inplace_backward=True):
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    return -output[0]


def compute_logprobs_from_logits(logits, target, task_type="embodied"):
    if task_type == "embodied":
        logprobs = -F.cross_entropy(
            logits, target=target, reduction="none"
        )  # [B, action-dim]
        return logprobs
    batch_dim = logits.shape[:-1]
    last_dim = logits.shape[-1]
    logits = logits.reshape(-1, last_dim)
    labels = target.reshape(-1)
    logprobs = logprobs_from_logits_flash_attn(
        logits, labels=labels, inplace_backward=False
    )
    logprobs = logprobs.view(*batch_dim)
    return logprobs


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def compute_entropy_from_logits(logits, epsilon=1e-10, task_type="embodied"):
    """
    Compute entropy by logits.

    Args:
        logits: [B, vocab-size, seq-len]
    Returns:
        entropy: [B, seq-len]
    """
    if task_type == "embodied":
        all_probs = F.softmax(logits, dim=1)  # [B, vocab-size, seq-len]
        all_log_probs = torch.log(all_probs + epsilon)
        entropy = -torch.sum(all_probs * all_log_probs, dim=1)  # [B, seq-len]
        return entropy
    return entropy_from_logits(logits=logits)


def quat_2_euler(quat: np.ndarray):
    """Convert quaternion to euler angles (xyz order)."""
    from scipy.spatial.transform import Rotation as R

    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat(xyz: np.ndarray):
    """Convert euler angles (xyz order) to quaternion."""
    from pyquaternion import Quaternion

    yaw, pitch, roll = xyz
    yaw = np.pi - yaw
    yaw_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0, 0, 1.0],
        ]
    )
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    roll_matrix = np.array(
        [
            [1.0, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
    return Quaternion(matrix=rot_mat).elements


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
