"""Lossless, chunked CPU compression for actor-bound trajectories."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory


@dataclass(frozen=True)
class TrajectoryCompressionConfig:
    """Static settings for the Actor trajectory transfer codec."""

    enabled: bool = False
    level: int = 1
    chunk_steps: int = 16

    def __post_init__(self) -> None:
        if not 0 <= self.level <= 9:
            raise ValueError("Trajectory compression level must be in [0, 9].")
        if self.chunk_steps <= 0:
            raise ValueError("Trajectory compression chunk_steps must be positive.")


@dataclass(frozen=True)
class CompressedTrajectoryBlock:
    """One compressed time slice of a trajectory tensor leaf."""

    path: tuple[str, ...]
    full_shape: tuple[int, ...]
    dtype: str
    start: int
    stop: int
    offset: int
    length: int


@dataclass(frozen=True)
class CompressedTrajectory:
    """Actor-bound lossless representation of one trajectory shard."""

    max_episode_length: int
    model_weights_id: str
    blocks: tuple[CompressedTrajectoryBlock, ...]
    payload: torch.Tensor


def compress_trajectory(
    trajectory: Trajectory, config: TrajectoryCompressionConfig
) -> CompressedTrajectory:
    """Compress every trajectory tensor independently in bounded time blocks."""
    if not config.enabled:
        raise ValueError("Cannot compress with a disabled trajectory codec.")

    blocks = []
    payloads = []
    offset = 0
    for path, tensor in _tensor_leaves(trajectory):
        if tensor.device.type != "cpu":
            raise ValueError("Actor-bound trajectories must be CPU tensors.")
        if tensor.dim() == 0:
            ranges = ((0, 1),)
            full_shape = ()
        else:
            ranges = _chunk_ranges(tensor.shape[0], config.chunk_steps)
            full_shape = tuple(tensor.shape)
        for start, stop in ranges:
            value = tensor if tensor.dim() == 0 else tensor[start:stop]
            raw = value.contiguous().numpy().tobytes()
            encoded = zlib.compress(raw, level=config.level)
            length = len(encoded)
            blocks.append(
                CompressedTrajectoryBlock(
                    path=path,
                    full_shape=full_shape,
                    dtype=_dtype_name(tensor.dtype),
                    start=start,
                    stop=stop,
                    offset=offset,
                    length=length,
                )
            )
            payloads.append(
                torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8).copy())
            )
            offset += length
    return CompressedTrajectory(
        max_episode_length=trajectory.max_episode_length,
        model_weights_id=trajectory.model_weights_id,
        blocks=tuple(blocks),
        payload=torch.cat(payloads) if payloads else torch.empty(0, dtype=torch.uint8),
    )


def decompress_trajectory(compressed: CompressedTrajectory) -> Trajectory:
    """Restore one compressed trajectory shard and validate every block range."""
    trajectory = Trajectory(
        max_episode_length=compressed.max_episode_length,
        model_weights_id=compressed.model_weights_id,
    )
    tensors: dict[tuple[str, ...], torch.Tensor] = {}
    ranges: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for block in compressed.blocks:
        tensor = tensors.get(block.path)
        if tensor is None:
            shape = block.full_shape or ()
            tensor = torch.empty(shape, dtype=_dtype_from_name(block.dtype))
            tensors[block.path] = tensor
            ranges[block.path] = []
        if tuple(tensor.shape) != block.full_shape or tensor.dtype != _dtype_from_name(
            block.dtype
        ):
            raise ValueError(
                f"Inconsistent compressed trajectory block for {block.path}."
            )
        encoded = compressed.payload.narrow(0, block.offset, block.length)
        raw = zlib.decompress(encoded.numpy().tobytes())
        value = torch.from_numpy(np.frombuffer(raw, dtype=tensor.numpy().dtype).copy())
        if tensor.dim() == 0:
            if block.start != 0 or block.stop != 1:
                raise ValueError(f"Invalid scalar block range for {block.path}.")
            tensor.copy_(value.reshape(()))
        else:
            expected_shape = (block.stop - block.start, *tensor.shape[1:])
            if value.numel() != int(np.prod(expected_shape)):
                raise ValueError(f"Compressed block size does not match {block.path}.")
            tensor[block.start : block.stop] = value.reshape(expected_shape)
        ranges[block.path].append((block.start, block.stop))

    for path, tensor in tensors.items():
        if tensor.dim() > 0 and _chunk_ranges(tensor.shape[0], 1) != _expand_ranges(
            ranges[path]
        ):
            raise ValueError(f"Compressed trajectory is incomplete for {path}.")
        _set_tensor(trajectory, path, tensor)
    return trajectory


def _tensor_leaves(trajectory: Trajectory):
    for field in fields(trajectory):
        value = getattr(trajectory, field.name)
        if isinstance(value, torch.Tensor):
            yield (field.name,), value
        elif isinstance(value, dict):
            yield from _dict_tensor_leaves(value, (field.name,))


def _dict_tensor_leaves(value: dict[str, Any], prefix: tuple[str, ...]):
    for key, nested_value in value.items():
        path = (*prefix, key)
        if isinstance(nested_value, torch.Tensor):
            yield path, nested_value
        elif isinstance(nested_value, dict):
            yield from _dict_tensor_leaves(nested_value, path)
        else:
            raise TypeError(f"Trajectory field {'.'.join(path)!r} must be a tensor.")


def _set_tensor(
    trajectory: Trajectory, path: tuple[str, ...], tensor: torch.Tensor
) -> None:
    if len(path) == 1:
        setattr(trajectory, path[0], tensor)
        return
    target = getattr(trajectory, path[0])
    for key in path[1:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = tensor


def _chunk_ranges(length: int, chunk_steps: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, min(start + chunk_steps, length))
        for start in range(0, length, chunk_steps)
    )


def _expand_ranges(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (index, index + 1) for start, stop in ranges for index in range(start, stop)
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as error:
        raise ValueError(
            f"Unsupported compressed trajectory dtype: {name!r}."
        ) from error
