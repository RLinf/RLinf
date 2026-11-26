# Copyright 2026 The RLinf Authors.
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

"""Transfer a release SGLang processor result to its requesting rollout worker."""

import asyncio
import hashlib
import os
import re
import stat
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any

import torch
from safetensors.torch import load_file, save

from rlinf.utils.image_replay import (
    REPLAY_FORMAT,
    replay_positions,
    split_image_pixels,
    store_pixels,
)

CAPTURE_KEY = "rlinf_image_capture"
EXTERNAL_MM_PROCESSOR_PACKAGE = "rlinf.hybrid_engines.sglang.multimodal"
MIN_CAPTURE_SGLANG_VERSION = "0.5.12.post1"


def ensure_image_capture_env(sglang_version: object) -> None:
    """Validate SGLang version and register the RLinf multimodal processor."""
    from packaging.version import parse

    if parse(str(sglang_version)) < parse(MIN_CAPTURE_SGLANG_VERSION):
        raise ValueError(
            f"Multimodal input capture requires SGLang >= {MIN_CAPTURE_SGLANG_VERSION}, "
            f"got {sglang_version}."
        )
    current = os.environ.get("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE")
    if not current:
        os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"] = (
            EXTERNAL_MM_PROCESSOR_PACKAGE
        )
    elif current != EXTERNAL_MM_PROCESSOR_PACKAGE:
        raise ValueError(
            f"SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE is set to {current!r}, "
            f"but multimodal input capture requires {EXTERNAL_MM_PROCESSOR_PACKAGE!r}."
        )


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash a tensor's dtype, shape and exact CPU bytes."""
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tensor.dtype, tuple(tensor.shape))).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


async def generate_with_capture(
    generate: Callable[..., Awaitable[dict[str, Any]]],
    prompt_ids: list[int],
    images: Sequence[Any] | None,
    sampling_params: dict[str, Any],
    capture: dict[str, Any],
    return_logprob: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Use a private local handoff between the tokenizer process and rollout worker."""
    spans = capture.get("image_spans")
    if not spans or not images or len(spans) != len(images):
        raise ValueError(
            "Image capture requires non-empty image_spans and images of equal length: "
            f"got {len(spans or [])} spans and {len(images or [])} images."
        )
    previous = capture.get("history")
    prefix = previous["length"] if previous is not None else 0
    if previous is not None and (
        prefix > len(prompt_ids)
        or tensor_digest(torch.tensor(prompt_ids[:prefix], dtype=torch.long))
        != previous["tokens"]
    ):
        raise ValueError("Prompt no longer starts with the sampled token history.")
    with TemporaryDirectory(prefix="rlinf-qwen-") as directory:
        ticket = Path(directory).name
        params = dict(sampling_params)
        params["custom_params"] = {
            **(params.get("custom_params") or {}),
            CAPTURE_KEY: {
                "image_spans": spans,
                "prefix_length": prefix,
                "ticket": ticket,
            },
        }
        result = await generate(
            input_ids=prompt_ids,
            image_data=images,
            sampling_params=params,
            return_logprob=return_logprob,
        )
        entry, history = await asyncio.to_thread(
            finish_capture,
            capture_file(ticket, result["meta_info"]["id"]),
            result["output_ids"],
            previous,
            capture.get("store_dir"),
        )
        return result, entry, history


def finish_capture(
    path: Path | str,
    output_ids: Sequence[int],
    previous: dict[str, Any] | None,
    store_dir: str | os.PathLike[str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the preceding turn and prepare the image row off the event loop."""
    entry = load_capture(path)
    image_rows = split_image_pixels(entry)
    fingerprints = [
        [span, grid, tensor_digest(pixels)]
        for span, grid, pixels in zip(
            entry["image_spans"],
            entry["image_grid_thw"].tolist(),
            image_rows,
            strict=True,
        )
    ]
    if previous is not None and (
        [image for image in fingerprints if image[0][0] < previous["length"]]
        != previous["images"]
        or tensor_digest(entry["mrope_positions"][:, : previous["length"]])
        != previous["positions"]
    ):
        raise ValueError(
            "Earlier image spans, pixels, grids or M-RoPE positions changed."
        )
    ids = entry["prompt_token_ids"] + list(output_ids)
    history = {
        "length": len(ids),
        "tokens": tensor_digest(torch.tensor(ids, dtype=torch.long)),
        "positions": tensor_digest(replay_positions(entry, ids)),
        "images": fingerprints,
    }
    if store_dir is not None:
        store_pixels(entry, Path(store_dir))
    return entry, history


def capture_file(ticket: str, rid: str) -> Path:
    """Resolve and validate the private per-request capture file path."""
    if not re.fullmatch(r"rlinf-qwen-[a-z0-9_]{8}", ticket) or not re.fullmatch(
        r"[a-fA-F0-9]{32}", rid
    ):
        raise ValueError(
            f"Invalid SGLang capture ticket or request ID: ticket={ticket!r}, rid={rid!r}."
        )
    return Path(gettempdir()) / ticket / f"{rid}.safetensors"


def save_capture(path: Path, output: Any) -> None:
    """Write captured image tensors into the worker-owned private capture directory."""
    items = output.mm_items
    grids = torch.cat([item.image_grid_thw.reshape(-1, 3) for item in items])
    spans = [item.offsets[0] for item in items]
    tensors = {
        "pixel_values": torch.cat([item.feature.cpu() for item in items]),
        "image_grid_thw": grids.cpu(),
        "image_spans": torch.tensor(
            [(start, end + 1) for start, end in spans], dtype=torch.long
        ),
        "prompt_token_ids": torch.tensor(output.input_ids, dtype=torch.long),
        "mrope_positions": output.mrope_positions.cpu().contiguous(),
        "mrope_position_delta": torch.as_tensor(output.mrope_position_delta)
        .cpu()
        .reshape(1),
    }
    # Only write inside a private directory owned by this worker. Never follow
    # user-supplied links or overwrite the result of an earlier attempt.
    directory = os.open(path.parent, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        owner = os.fstat(directory)
        if (
            owner.st_uid != os.getuid()
            or (stat.S_IMODE(owner.st_mode) & ~stat.S_ISGID) != 0o700
        ):
            raise ValueError("Invalid image capture directory owner or permissions.")
        fd = os.open(
            path.name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=directory
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(save(tensors))
    finally:
        os.close(directory)


def load_capture(path: Path | str) -> dict[str, Any]:
    """Load the captured safetensors entry written by the SGLang processor."""
    capture_path = Path(path)
    if not capture_path.is_file():
        raise ValueError(
            f"SGLang did not write image capture file {capture_path}; verify a "
            "Qwen3.5 model, SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE="
            "rlinf.hybrid_engines.sglang.multimodal, and a shared TMPDIR between "
            "rollout worker and SGLang tokenizer."
        )
    entry = load_file(str(capture_path))
    entry["format"] = REPLAY_FORMAT
    entry["prompt_token_ids"] = entry["prompt_token_ids"].tolist()
    entry["image_spans"] = entry["image_spans"].tolist()
    return entry
