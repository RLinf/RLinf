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

from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any

import numpy as np
import torch

# Keys that we have already warned about in concat_batch, so each missing key
# only produces a single warning per process (avoid log spam in the replay /
# demo batch pipeline).
_CONCAT_BATCH_WARNED_KEYS: set[str] = set()


def update_nested_cfg(base_cfg, override_cfg):
    for key, value in override_cfg.items():
        if (
            key in base_cfg
            and isinstance(base_cfg[key], dict)
            and isinstance(value, dict)
        ):
            update_nested_cfg(base_cfg[key], value)
        else:
            base_cfg[key] = value
    return base_cfg


def copy_dict_tensor(next_extracted_obs: dict):
    """
    Recursively clones all torch tensors in a dict.
    """
    ret = {}
    for key, value in next_extracted_obs.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value.clone()
        elif isinstance(value, dict):
            ret[key] = copy_dict_tensor(value)
        else:
            ret[key] = value
    return ret


def clone_nested_to_cpu(value: Any):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone_nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested_to_cpu(item) for item in value)
    return value


def put_tensor_device(data, device, *, detach=False):
    """Move nested tensors to a device, optionally detaching autograd state."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        if detach and (data.requires_grad or data.grad_fn is not None):
            data = data.detach()
        return data.to(device=device).contiguous()
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = put_tensor_device(value, device, detach=detach)
        return data
    if isinstance(data, list):
        for index, value in enumerate(data):
            data[index] = put_tensor_device(value, device, detach=detach)
        return data
    if isinstance(data, tuple):
        return tuple(put_tensor_device(value, device, detach=detach) for value in data)
    return data


def _split_list_by_sizes(value: list, split_sizes: list[int] | int) -> list[list]:
    if isinstance(split_sizes, int):
        chunks = split_sizes
        k, m = divmod(len(value), chunks)
        split_sizes = [k + (1 if i < m else 0) for i in range(chunks)]
    out, i = [], 0
    for n in split_sizes:
        out.append(value[i : i + n])
        i += n
    return out


def _chunk_bounds(length: int, num_chunks: int, index: int) -> tuple[int, int]:
    # Same sizes as torch.tensor_split: the first length % num_chunks chunks
    # get one extra element.
    size, remainder = divmod(length, num_chunks)
    start = index * size + min(index, remainder)
    return start, start + size + (1 if index < remainder else 0)


def _dict_chunk(data: dict, num_chunks: int, index: int, dim: int) -> dict:
    chunk = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            start, end = _chunk_bounds(value.shape[dim], num_chunks, index)
            chunk[key] = value.narrow(dim, start, end - start).contiguous()
        elif isinstance(value, list):
            assert dim == 0, f"List field only supports dim=0, got {dim}."
            start, end = _chunk_bounds(len(value), num_chunks, index)
            chunk[key] = value[start:end]
        elif value is None:
            chunk[key] = None
        elif isinstance(value, dict):
            chunk[key] = _dict_chunk(value, num_chunks, index, dim)
        else:
            raise ValueError(f"{key=}, {type(value)} is not supported.")
    return chunk


def iter_dict_chunks(data: dict, num_chunks: int, dim: int = 0) -> Iterator[dict]:
    """Yield the chunks of :func:`split_dict_to_chunk` one at a time.

    Each chunk is built when requested, so a consumer that drops a chunk
    before asking for the next keeps no reference to it.

    Args:
        data: Nested dict of tensors, lists, and ``None`` values.
        num_chunks: Number of chunks to yield.
        dim: Tensor dimension to split along.

    Yields:
        Nested dicts with the same keys as ``data``, in order.
    """
    assert num_chunks > 0, f"Expected a positive number of chunks, got {num_chunks}."
    for index in range(num_chunks):
        yield _dict_chunk(data, num_chunks, index, dim)


def split_dict_to_chunk(data: dict, split_size, dim=0):
    return list(iter_dict_chunks(data, split_size, dim))


def _alloc_on_device(data, device, copies: list):
    """Mirror ``data`` with empty device tensors and queue the pinned sources."""
    if isinstance(data, torch.Tensor):
        if data.device.type != "cpu":
            return data.to(device=device).contiguous()
        src = data.contiguous()
        if not src.is_pinned():
            src = src.pin_memory()
        dst = torch.empty_like(src, device=device)
        copies.append((dst, src))
        return dst
    if isinstance(data, dict):
        return {
            key: _alloc_on_device(value, device, copies) for key, value in data.items()
        }
    if isinstance(data, (list, tuple)):
        return type(data)(_alloc_on_device(value, device, copies) for value in data)
    return data


def prefetch_to_device(
    batches: Iterable[Any],
    device: torch.device | int | str,
    torch_platform: Any | None,
) -> Iterator[Any]:
    """Yield ``put_tensor_device(batch, device)`` for each batch, one batch ahead.

    With an accelerator ``torch_platform`` (``Worker.torch_platform``), the copy
    of the next batch is issued on a side stream before the current batch is
    yielded, so it overlaps the caller's work on the current batch. CPU tensors
    that are not pinned are pinned first, because only pinned copies run
    asynchronously. Destination memory is allocated on the current stream, which
    waits for a batch's copy before the batch is yielded, so the tensors need no
    ``record_stream`` and can be used on the current stream like any other.
    Without a platform or for a CPU device, batches are moved one at a time.

    Args:
        batches: Nested tensor containers, as accepted by ``put_tensor_device``.
        device: Target device, as accepted by ``Tensor.to``.
        torch_platform: Accelerator module such as ``torch.cuda``, or ``None``.

    Yields:
        Each batch on ``device``, in order. Device tensors are contiguous.
    """
    on_cpu = not isinstance(device, int) and torch.device(device).type == "cpu"
    if torch_platform is None or on_cpu:
        # map keeps no reference to a yielded batch, so the caller can free it.
        yield from map(partial(put_tensor_device, device=device), batches)
        return

    copy_stream = torch_platform.Stream(device=device)
    current_stream = torch_platform.current_stream(device)

    def start_copy(batch):
        copies = []
        staged = _alloc_on_device(batch, device, copies)
        # Destination blocks may have been freed by work still queued on the
        # current stream.
        copy_stream.wait_stream(current_stream)
        try:
            with torch_platform.stream(copy_stream):
                for dst, src in copies:
                    dst.copy_(src, non_blocking=True)
                done = torch_platform.Event()
                done.record(copy_stream)
        except BaseException:
            # Queued copies must finish before their destinations are freed.
            copy_stream.synchronize()
            raise
        return staged, done

    batch_iter = iter(batches)
    pending = None
    try:
        for batch in batch_iter:
            pending = start_copy(batch)
            del batch
            break
        while pending is not None:
            ready = [pending[0]]
            current_stream.wait_event(pending[1])
            pending = None
            for batch in batch_iter:
                pending = start_copy(batch)
                del batch
                break
            # Yield without keeping a reference, so the caller can free the batch.
            yield ready.pop()
    finally:
        # A batch dropped mid-copy must not hand its memory back early.
        if pending is not None:
            current_stream.wait_event(pending[1])


def concat_batch(data1, data2):
    batch = {}
    for key, value in data1.items():
        if isinstance(value, torch.Tensor):
            if key not in data2:
                # NOTE: NO WARNING FOR THE CASE THAT DATA2 DOES NOT CONTAIN SOME KEYS IN DATA1
                continue
            batch[key] = torch.cat([data1[key], data2[key]], dim=0)
        elif isinstance(value, dict):
            # NOTE: added this for dealing with different keys in demo data.
            if key not in data2:
                if key not in _CONCAT_BATCH_WARNED_KEYS:
                    _CONCAT_BATCH_WARNED_KEYS.add(key)
                    # Lazy import to avoid pulling rlinf.scheduler.worker (and
                    # its heavy deps) at module import time. This only runs
                    # once per missing key, inside a worker where that import
                    # is essentially free.
                    from rlinf.utils.logging import get_logger

                    get_logger().warning(
                        "concat_batch: key '%s' not found in data2 (value type: %s), "
                        "skipping. This warning is only emitted once per key.",
                        key,
                        type(value).__name__,
                    )
                continue
            batch[key] = concat_batch(data1[key], data2[key])
    return batch


def stack_list_of_dict_tensor(list_of_dict: list, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = {}
    for key in keys:
        _v0 = list_of_dict[0][key]
        if isinstance(_v0, torch.Tensor):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = torch.stack(v_list, dim=dim)
        elif isinstance(_v0, dict):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = stack_list_of_dict_tensor(v_list)
        elif _v0 is None:
            pass
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")
    return ret


def cat_list_of_dict_tensor(list_of_dict: list, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = {}
    for key in keys:
        _v0 = list_of_dict[0][key]
        if _v0 is None:
            continue

        v_list = [d[key] for d in list_of_dict]

        if isinstance(_v0, torch.Tensor):
            ret[key] = torch.cat(v_list, dim=dim)
        elif isinstance(_v0, np.ndarray):
            ret[key] = np.concatenate([v for v in v_list if v is not None], axis=dim)
        elif isinstance(_v0, list):
            assert dim == 0, f"{key=} is list, dim !=0 is not supported!"
            ret[key] = [item for sub in v_list if sub is not None for item in sub]
        elif isinstance(_v0, dict):
            ret[key] = cat_list_of_dict_tensor(v_list, dim=dim)
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")

    return ret


def split_dict(
    batch: dict[str, Any],
    split_sizes: list[int],
    dim: int = 0,
) -> list[dict[str, Any]]:
    """Split one batch dict into size-specified sub-batches.

    Tensor values are chunked on ``dim``; list values are sliced proportionally;
    nested dict values are split recursively.

    Args:
        batch: Dict.
        split_sizes: Batch sizes for each destination rank.
        dim: Tensor dimension to split. Defaults to 0.

    Returns:
        A list of splited batches, one item per destination rank.
    """
    count = len(split_sizes)
    total_size = sum(split_sizes)
    splitted_batches = [{} for _ in range(count)]
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.shape[dim] == total_size, (
                f"Tensor field '{key}' expected split dim size {total_size}, "
                f"got {value.shape[dim]} on dim {dim}."
            )
            splitted_values = torch.split(value, split_sizes, dim=dim)
            for i in range(count):
                splitted_batches[i][key] = splitted_values[i].contiguous()
        elif isinstance(value, list):
            assert dim == 0, f"List field '{key}' only supports dim=0, got {dim}."
            length = len(value)
            assert length == total_size, (
                f"List field '{key}' expected length {total_size}, got {length}."
            )
            begin = 0
            for i, size in enumerate(split_sizes):
                splitted_batches[i][key] = value[begin : begin + size]
                begin += size
        elif isinstance(value, dict):
            splitted_sub_batches = split_dict(value, split_sizes, dim=dim)
            for i in range(count):
                splitted_batches[i][key] = splitted_sub_batches[i]
        else:
            for i in range(count):
                splitted_batches[i][key] = value

    return splitted_batches


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id, pin_memory=False):
    """Flatten ``[T, B, ...]`` tensors to ``[T * B, ...]`` and reorder by ``shuffle_id``.

    With ``pin_memory``, CPU results are gathered straight into pinned host
    memory, so later host-to-device copies can run asynchronously.
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            value = value.reshape(-1, *value.shape[2:])
            if pin_memory and value.device.type == "cpu":
                out = torch.empty(
                    (shuffle_id.numel(), *value.shape[1:]),
                    dtype=value.dtype,
                    pin_memory=True,
                )
                ret_dict[key] = torch.index_select(value, 0, shuffle_id, out=out)
            else:
                ret_dict[key] = value[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id, pin_memory)
    return ret_dict


def trim_nested_tensor_time_dim(value, target_steps: int, key_path=()):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        assert value.shape[0] in {target_steps, target_steps + 1}, (
            f"Cannot trim field {'.'.join(key_path)!r} with shape "
            f"{tuple(value.shape)} to {target_steps} OPD training steps."
        )
        return value[:target_steps]
    if isinstance(value, dict):
        return {
            key: trim_nested_tensor_time_dim(
                nested_value, target_steps, (*key_path, key)
            )
            for key, nested_value in value.items()
        }
    raise TypeError(
        f"Unsupported field {'.'.join(key_path)!r} type {type(value)} for OPD trimming."
    )


def flatten_nested_tensor_time_batch(value, key_path=()):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        assert value.dim() >= 2, (
            f"Cannot flatten field {'.'.join(key_path)!r} with shape "
            f"{tuple(value.shape)} across time and batch."
        )
        return value.reshape(-1, *value.shape[2:])
    if isinstance(value, dict):
        return {
            key: flatten_nested_tensor_time_batch(nested_value, (*key_path, key))
            for key, nested_value in value.items()
        }
    raise TypeError(
        f"Unsupported field {'.'.join(key_path)!r} type {type(value)} for flattening."
    )
