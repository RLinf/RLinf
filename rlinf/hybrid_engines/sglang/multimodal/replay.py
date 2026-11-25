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
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, gettempdir

import torch
from safetensors.torch import load, load_file, save

from .token_space import tensor_digest

REPLAY_FORMAT = "rlinf-qwen-image-inputs-v1"
CAPTURE_KEY = "rlinf_image_capture"


async def generate_with_capture(
    generate, prompt_ids, images, sampling_params, capture, return_logprob
):
    """Use a private local handoff between the tokenizer process and rollout worker."""
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


def finish_capture(path, output_ids, previous, store_dir):
    """Validate the preceding turn and prepare the image row off the event loop."""
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


def save_capture(path, output):
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
    # Only write inside a private directory owned by this worker. Never follow
    # user-supplied links or overwrite the result of an earlier attempt.
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


def load_capture(path):
    entry = load_file(str(path))
    entry["format"] = REPLAY_FORMAT
    entry["prompt_token_ids"] = entry["prompt_token_ids"].tolist()
    entry["image_spans"] = entry["image_spans"].tolist()
    return entry


def replay_positions(entry, ids):
    prompt = entry["prompt_token_ids"]
    positions = entry["mrope_positions"]
    if ids[: len(prompt)] != prompt or tuple(positions.shape) != (3, len(prompt)):
        raise ValueError("Training prompt differs from the captured SGLang prompt")
    extension = torch.arange(len(prompt), len(ids)) + entry["mrope_position_delta"]
    return torch.cat([positions, extension.expand(3, -1)], dim=1)


def store_pixels(entry, directory):
    """Publish immutable per-image tensors for pending rows and checkpoint replay."""
    directory.mkdir(parents=True, exist_ok=True)
    pixels = entry["pixel_values"]
    references = []
    for image in pixels.split(entry["image_grid_thw"].prod(dim=1).tolist()):
        data = save({"pixels": image.contiguous()})
        digest = hashlib.sha256(data).hexdigest()
        target = directory / (digest + ".safetensors")
        if not target.exists():
            with NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                temporary.write_bytes(data)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        references.append({"path": str(target.resolve()), "sha256": digest})
    entry["pixel_refs"] = references
    entry["pixel_values_bytes"] = pixels.numel() * 2
    del entry["pixel_values"]


def replay_pixels(entry):
    if "pixel_refs" not in entry:
        return entry["pixel_values"].to(torch.bfloat16)
    images = []
    for ref in entry["pixel_refs"]:
        data = Path(ref["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("Captured image tensor checksum mismatch")
        images.append(load(data)["pixels"].to(torch.bfloat16))
    return torch.cat(images)
