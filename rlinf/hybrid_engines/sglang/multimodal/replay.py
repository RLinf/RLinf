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

"""Transfer official SGLang processor outputs to the requesting rollout worker."""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import os
import re
import shutil
import stat
import threading
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import torch
from safetensors.torch import load_file, save

from rlinf.utils.image_replay import REPLAY_FORMAT, replay_positions, store_pixels

CAPTURE_KEY = "rlinf_image_capture"
_LOCAL_IMAGE_CACHE_MAX = 2048
_LOCAL_IMAGE_LOCK = threading.Lock()
_LOCAL_IMAGE_DIR: Path | None = None
_LOCAL_IMAGE_LRU: OrderedDict[str, Path] = OrderedDict()


def _cleanup_local_image_dir() -> None:
    if _LOCAL_IMAGE_DIR is not None:
        shutil.rmtree(_LOCAL_IMAGE_DIR, ignore_errors=True)


def stage_local_images(images: list | None) -> list | None:
    """Stage raw PNG bytes once on local tmpfs so SGLang avoids per-turn Base64 JSON."""
    if not images:
        return images
    global _LOCAL_IMAGE_DIR
    staged: list = []
    with _LOCAL_IMAGE_LOCK:
        if _LOCAL_IMAGE_DIR is None:
            cache_dir = Path(gettempdir()) / f"rlinf-mm-imgs-{os.getpid()}"
            cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            _LOCAL_IMAGE_DIR = cache_dir
            atexit.register(_cleanup_local_image_dir)
        for item in images:
            if not isinstance(item, (bytes, bytearray, memoryview)):
                staged.append(item)
                continue
            key = hashlib.blake2b(item, digest_size=16).hexdigest()
            path = _LOCAL_IMAGE_LRU.get(key)
            if path is not None and path.is_file():
                _LOCAL_IMAGE_LRU.move_to_end(key)
                staged.append(str(path))
                continue
            path = _LOCAL_IMAGE_DIR / f"{key}.png"
            if not path.is_file():
                tmp = (
                    _LOCAL_IMAGE_DIR
                    / f".{key}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                tmp.write_bytes(item)
                tmp.replace(path)
            _LOCAL_IMAGE_LRU[key] = path
            _LOCAL_IMAGE_LRU.move_to_end(key)
            while len(_LOCAL_IMAGE_LRU) > _LOCAL_IMAGE_CACHE_MAX:
                _, evicted = _LOCAL_IMAGE_LRU.popitem(last=False)
                evicted.unlink(missing_ok=True)
            staged.append(str(path))
    return staged


def tensor_digest(tensor: torch.Tensor) -> str:
    """Stable SHA-256 fingerprint of a CPU tensor's dtype, shape, and bytes."""
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(f"{t.dtype}:{tuple(t.shape)}".encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


async def generate_with_capture(
    generate, prompt_ids, images, sampling_params, capture, return_logprob
):
    """Run SGLang generation and capture exact expanded tokens, pixels, and M-RoPE."""
    previous = capture.get("history")
    prefix = previous["length"] if previous is not None else 0
    if previous is not None and (
        prefix > len(prompt_ids)
        or tensor_digest(torch.tensor(prompt_ids[:prefix])) != previous["tokens"]
    ):
        raise ValueError("Prompt no longer starts with the sampled token history")
    with TemporaryDirectory(prefix="rlinf-qwen-") as directory:
        ticket = Path(directory).name
        params = dict(sampling_params)
        params["custom_params"] = {
            **(params.get("custom_params") or {}),
            CAPTURE_KEY: {
                "image_spans": capture["image_spans"],
                "prefix_length": prefix,
                "ticket": ticket,
                "token_budget": capture.get("token_budget"),
            },
        }
        staged_images = await asyncio.to_thread(stage_local_images, images)
        result = await generate(
            input_ids=prompt_ids,
            image_data=staged_images,
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


def finish_capture(path, output_ids, previous, store_dir):
    """Validate prefix stability and optionally spill pixels to disk."""
    entry = load_capture(path)
    image_rows = entry["pixel_values"].split(
        entry["image_grid_thw"].prod(dim=1).tolist()
    )
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
            "Earlier image spans, pixels, grids or M-RoPE positions changed"
        )
    ids = entry["prompt_token_ids"] + list(output_ids)
    history = {
        "length": len(ids),
        "tokens": tensor_digest(torch.tensor(ids)),
        "positions": tensor_digest(replay_positions(entry, ids)),
        "images": fingerprints,
    }
    if store_dir is not None:
        store_pixels(entry, Path(store_dir))
    return entry, history


def capture_file(ticket: str, rid: str) -> Path:
    if not re.fullmatch(r"rlinf-qwen-[a-z0-9_]{8}", ticket) or not re.fullmatch(
        r"[a-fA-F0-9]{32}", rid
    ):
        raise ValueError("Invalid SGLang capture ticket or request ID")
    return Path(gettempdir()) / ticket / f"{rid}.safetensors"


def save_capture(path: Path, output) -> None:
    items = output.mm_items
    grids = torch.cat([item.image_grid_thw.reshape(-1, 3) for item in items])
    spans = [item.offsets[0] for item in items]
    tensors = {
        "pixel_values": torch.cat([item.feature.cpu() for item in items]),
        "image_grid_thw": grids.cpu(),
        "image_spans": torch.tensor([(start, end + 1) for start, end in spans]),
        "prompt_token_ids": torch.tensor(output.input_ids),
        "mrope_positions": output.mrope_positions.cpu().contiguous(),
        "mrope_position_delta": torch.as_tensor(output.mrope_position_delta)
        .cpu()
        .reshape(1),
    }
    directory = os.open(path.parent, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        owner = os.fstat(directory)
        if owner.st_uid != os.getuid() or stat.S_IMODE(owner.st_mode) != 0o700:
            raise ValueError("Invalid image capture directory owner or permissions")
        fd = os.open(
            path.name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=directory
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(save(tensors))
    finally:
        os.close(directory)


def load_capture(path: Path) -> dict:
    entry = load_file(str(path))
    entry["format"] = REPLAY_FORMAT
    entry["prompt_token_ids"] = entry["prompt_token_ids"].tolist()
    entry["image_spans"] = entry["image_spans"].tolist()
    return entry
