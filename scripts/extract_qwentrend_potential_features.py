#!/usr/bin/env python3
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

"""Extract Qwen hidden features for scalar reward-head training."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from examples.reward.preprocess_qwentrend_state_value_dataset import (
    _extract_extra_view_image,
    _to_uint8_rgb,
)
from examples.reward.preprocess_qwentrend_state_value_potential_dataset import (
    potential_prompt,
)
from rlinf.data.datasets.vlm import QwenTrendProgressSFTDataset
from rlinf.models.embodiment.reward.vlm_reward_model import VLMRewardModel


def read_rows(path: Path, sample_type: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [
        row for row in rows if row["segment_metadata"]["sample_type"] == sample_type
    ]


def load_frames(path: str) -> tuple[list[Any], list[Any]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["main_frames"], payload["extra_view_frames"]


@functools.lru_cache(maxsize=1)
def load_episode(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_history_frames(
    row: dict[str, Any], end_step: int, history_size: int
) -> tuple[list[Any], list[Any]]:
    """Load a left-padded dual-view history ending at an original rollout step."""
    source_window = int(row["segment_metadata"]["window_size"])
    sample_type = row["segment_metadata"]["sample_type"]
    if history_size == source_window and row["pkl_path"] != row["source_episode_path"]:
        main, extra = load_frames(row["pkl_path"])
        if sample_type == "progress":
            source_end = int(row["segment_metadata"]["end_step"])
            if end_step == source_end:
                return main[source_window:], extra[source_window:]
            return main[:source_window], extra[:source_window]
        return main, extra

    episode = load_episode(row["source_episode_path"])
    observations = episode["observations"]
    if end_step < 0 or end_step >= len(observations):
        raise IndexError(
            f"end_step {end_step} outside episode with {len(observations)} steps"
        )
    indices = list(range(max(0, end_step - history_size + 1), end_step + 1))
    indices = [indices[0]] * (history_size - len(indices)) + indices
    main_frames = []
    extra_frames = []
    for index in indices:
        observation = observations[index]
        main = observation.get("main_images")
        extra = observation.get("third_view_images")
        if extra is None:
            extra = _extract_extra_view_image(observation.get("extra_view_images"))
        if main is None or extra is None:
            raise ValueError(
                f"Missing dual-view image at {row['source_episode_path']}:{index}"
            )
        main_frames.append(_to_uint8_rgb(main))
        extra_frames.append(_to_uint8_rgb(extra))
    return main_frames, extra_frames


@torch.no_grad()
def encode(
    model: VLMRewardModel,
    prompts: list[str],
    videos: list[list[Any]],
) -> torch.Tensor:
    _, inputs, _ = QwenTrendProgressSFTDataset.process_inputs(
        processor=model._processor,
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=[[prompt] for prompt in prompts],
        videos=videos,
        answer_text=None,
    )
    inputs = {
        key: value.to(model._model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    outputs = model._model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]
    attention_mask = inputs["attention_mask"].bool()
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0)
    last_positions = positions.masked_fill(~attention_mask, -1).amax(dim=1)
    batch_ids = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_ids, last_positions].float().cpu()


def extract_potential(
    model: VLMRewardModel,
    rows: list[dict[str, Any]],
    batch_size: int,
    history_size: int,
) -> dict[str, Any]:
    features = []
    targets = []
    successes = []
    source_paths = []
    end_steps = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        videos = []
        for row in batch:
            end_step = int(row["segment_metadata"]["end_step"])
            main, extra = load_history_frames(row, end_step, history_size)
            videos.append([main, extra])
        prompts = [potential_prompt(row["task"], history_size, 10) for row in batch]
        features.append(encode(model, prompts, videos))
        targets.extend(float(row["supervision"]["teacher_value"]) for row in batch)
        successes.extend(bool(row["segment_metadata"]["success"]) for row in batch)
        source_paths.extend(row["source_episode_path"] for row in batch)
        end_steps.extend(int(row["segment_metadata"]["end_step"]) for row in batch)
    return {
        "features": torch.cat(features).to(torch.float16),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "successes": torch.tensor(successes, dtype=torch.bool),
        "source_paths": source_paths,
        "end_steps": torch.tensor(end_steps, dtype=torch.int32),
    }


def extract_progress(
    model: VLMRewardModel,
    rows: list[dict[str, Any]],
    batch_size: int,
    history_size: int,
) -> dict[str, Any]:
    pair_features = []
    deltas = []
    labels = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = []
        videos = []
        for row in batch:
            metadata = row["segment_metadata"]
            clip_size = int(metadata["window_size"])
            end_step = int(metadata["end_step"])
            earlier_end = int(metadata["start_step"]) + clip_size - 1
            earlier = load_history_frames(row, earlier_end, history_size)
            current = load_history_frames(row, end_step, history_size)
            prompt = potential_prompt(row["task"], history_size, 10)
            prompts.extend([prompt, prompt])
            videos.extend([[earlier[0], earlier[1]], [current[0], current[1]]])
            deltas.append(float(row["supervision"]["teacher_delta"]))
            labels.append(row["answer"])
        encoded = encode(model, prompts, videos)
        pair_features.append(encoded.reshape(len(batch), 2, -1))
    return {
        "features": torch.cat(pair_features).to(torch.float16),
        "teacher_deltas": torch.tensor(deltas, dtype=torch.float32),
        "labels": labels,
    }


def main(args: argparse.Namespace) -> None:
    rows = read_rows(Path(args.manifest), args.sample_type)
    # Keep every window from an episode on one rank. The episode pickle is large,
    # so row-wise sharding makes every rank deserialize nearly every episode.
    rows = [
        row
        for row in rows
        if int(
            hashlib.sha256(row["source_episode_path"].encode()).hexdigest()[:16],
            16,
        )
        % args.world_size
        == args.rank
    ]
    rows.sort(
        key=lambda row: (
            row["source_episode_path"],
            row["segment_metadata"]["end_step"],
        )
    )
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    cfg = OmegaConf.create(
        {
            "model_path": args.model_path,
            "lora_path": args.checkpoint,
            "precision": "bf16",
            "inference_mode": "generate",
            "input_builder_name": "qwentrend_potential_input_builder",
            "input_builder_params": {"history_buffer_names": ["history_window"]},
            "reward_parser_name": "base_reward_parser",
            "reward_parser_params": {},
        }
    )
    model = VLMRewardModel(cfg)
    model._model.to(args.device).eval()
    if args.sample_type == "potential":
        payload = extract_potential(model, rows, args.batch_size, args.history_size)
    else:
        payload = extract_progress(model, rows, args.batch_size, args.history_size)
    payload["metadata"] = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "sample_type": args.sample_type,
        "rank": args.rank,
        "world_size": args.world_size,
        "num_samples": len(rows),
        "history_size": args.history_size,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps(payload["metadata"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sample-type", choices=("potential", "progress"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
