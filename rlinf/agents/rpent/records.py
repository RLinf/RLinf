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

"""Persist sampled tokens and image references for training recovery."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from rlinf.agents.rpent.episode import Episode
from rlinf.workers.agent.agent_loop import AgentLoopOutput


def write_training_turn(
    output_dir: str,
    episode: Episode,
    turn: AgentLoopOutput,
    generated: dict,
    images: list,
    started_at: float,
    timings: dict[str, float],
) -> None:
    """Save the sampled turn and media contract before prefix rows are merged."""
    media = turn.response_images or {}
    grid = media.get("image_grid_thw")
    if isinstance(grid, torch.Tensor):
        grid = grid.tolist()
    image_hashes = (
        [reference["sha256"] for reference in media["image_refs"]]
        if "image_refs" in media
        else [hashlib.sha256(image).hexdigest() for image in images]
    )
    if media.get("format") == "rlinf-qwen-image-inputs-v1":
        directory = Path(output_dir) / episode.episode_id / "training_images"
        directory.mkdir(parents=True, exist_ok=True)
        for image, digest in zip(images, image_hashes, strict=True):
            target = directory / digest
            if not target.exists():
                with NamedTemporaryFile(dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(image)
                try:
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
    audit = {
        "episode_id": episode.episode_id,
        "task": episode.task,
        "turn_id": turn.extra_fields["turn_id"],
        "chain_id": episode.history.chain_id,
        "started_at": started_at,
        "finished_at": time.time(),
        "input_ids": turn.prompt_ids,
        "output_ids": turn.response_ids,
        "response_mask": turn.response_mask or [1] * len(turn.response_ids),
        "behavior_logprobs": generated.get("logprobs"),
        "weight_version_start": generated.get("version_start"),
        "weight_version_end": generated.get("version_end"),
        "token_weight_versions": generated.get("weight_versions"),
        "image_sha256": image_hashes,
        "served_prompt_ids_match": generated.get("prompt_token_ids") == turn.prompt_ids,
        "multimodal_contract": {
            **media,
            "image_grid_thw": grid,
        }
        if media.get("format")
        else None,
        "multimodal_format": media.get("format"),
        "processor_fingerprint": media.get("processor_fingerprint"),
        "image_grid_thw": grid,
        "finish_reason": generated["finish_reason"],
        "tool_names": turn.extra_fields["tool_names"],
        **timings,
    }
    directory = Path(output_dir) / episode.episode_id / "training_audit"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"turn-{turn.extra_fields['turn_id']:05d}.json"
    path.write_text(json.dumps(audit, default=lambda value: value.tolist()))
