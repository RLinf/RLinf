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

"""Extract frozen VLM prompt features for scalar-head training.

Example:
    python examples/reward/extract_vlm_trend_potential_features.py \\
        --model-path /path/to/Qwen3-VL-4B-Instruct \\
        --checkpoint /path/to/potential/global_step_N/actor/model_state_dict/full_weights.pt \\
        --manifest logs/xxx/potential_data/train/segments.jsonl \\
        --sample-type potential \\
        --output logs/xxx/features/train_potential_0.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from examples.reward.vlm_trend_data import (
    load_dual_view_sample,
    potential_prompt,
    sample_source_hash,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


@torch.no_grad()
def encode_feature_batch(
    model: Any,
    prompts: list[str],
    videos: list[list[Any]],
    video_fps: float,
) -> torch.Tensor:
    """Pool Qwen features using the same processor path as VLM Trend SFT."""
    from rlinf.data.datasets.vlm import VLMTrendRewardSFTDataset
    from rlinf.models.embodiment.reward.vlm_trend_success_potential_reward_model import (
        extract_prompt_features,
    )

    _, inputs, _ = VLMTrendRewardSFTDataset.process_inputs(
        processor=model._processor,
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=[[prompt] for prompt in prompts],
        videos=videos,
        answer_text=None,
        video_fps=video_fps,
    )
    inputs = {
        key: value.to(model._model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    return extract_prompt_features(model._model, inputs).cpu()


def feature_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load and shard manifest rows for this worker."""
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [
        row
        for row in rows
        if row["segment_metadata"]["sample_type"] == args.sample_type
        and sample_source_hash(row) % args.world_size == args.rank
    ]
    rows.sort(
        key=lambda row: (
            row["source_episode_path"],
            row["segment_metadata"]["end_step"],
        )
    )
    return rows if args.max_samples is None else rows[: args.max_samples]


def extract_features(
    model: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Encode one feature shard from a potential or progress manifest."""
    feature_batches = []
    targets = []
    labels = []
    successes = []
    source_paths = []
    end_steps = []
    for start in tqdm(
        range(0, len(rows), args.batch_size),
        desc=f"Extracting {args.sample_type} features",
    ):
        batch = rows[start : start + args.batch_size]
        prompts = []
        videos = []
        for index, row in enumerate(batch, start=start):
            main_frames, extra_view_frames = load_dual_view_sample(row, index)
            source_videos = [main_frames, extra_view_frames]
            window_size = int(row["segment_metadata"]["window_size"])
            if window_size != args.history_size:
                raise ValueError(
                    "--history-size must match the preprocessed manifest window size"
                )
            prompt = potential_prompt(row["task"], args.history_size, args.num_bins)
            if args.sample_type == "potential":
                prompts.append(prompt)
                videos.append(source_videos)
                targets.append(float(row["supervision"]["teacher_value"]))
                successes.append(bool(row["segment_metadata"]["success"]))
                source_paths.append(row["source_episode_path"])
                end_steps.append(int(row["segment_metadata"]["end_step"]))
            else:
                main, extra = source_videos
                if len(main) != 2 * window_size or len(extra) != 2 * window_size:
                    raise ValueError("Progress samples must contain two full windows")
                prompts.extend([prompt, prompt])
                videos.extend(
                    [
                        [main[:window_size], extra[:window_size]],
                        [main[window_size:], extra[window_size:]],
                    ]
                )
                targets.append(float(row["supervision"]["teacher_delta"]))
                labels.append(row["answer"])
        encoded = encode_feature_batch(model, prompts, videos, args.video_fps)
        if args.sample_type == "progress":
            encoded = encoded.reshape(len(batch), 2, -1)
        feature_batches.append(encoded)

    payload: dict[str, Any] = {
        "features": torch.cat(feature_batches).to(torch.float16),
    }
    if args.sample_type == "potential":
        payload.update(
            targets=torch.tensor(targets, dtype=torch.float32),
            successes=torch.tensor(successes, dtype=torch.bool),
            source_paths=source_paths,
            end_steps=torch.tensor(end_steps, dtype=torch.int32),
        )
    else:
        payload.update(
            teacher_deltas=torch.tensor(targets, dtype=torch.float32),
            labels=labels,
        )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen VLM Trend potential features."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Full path to Potential VLM SFT full_weights.pt "
            "(typically .../actor/model_state_dict/full_weights.pt), or a "
            "PEFT adapter directory that contains adapter_config.json."
        ),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sample-type", choices=("potential", "progress"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--video-fps", type=float, default=24.0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.reward.vlm_reward_model import VLMRewardModel
    from rlinf.models.embodiment.reward.vlm_trend_success_potential_reward_model import (
        load_lora_adapter,
    )

    args = parse_args(argv)
    rows = feature_rows(args)
    if not rows:
        raise ValueError("No manifest rows selected for this feature shard")
    cfg = OmegaConf.create(
        {
            "model_path": args.model_path,
            "precision": "bf16",
            "subprocessor_kwargs": {"video_processor": {"do_sample_frames": True}},
            "input_builder_name": "base_vlm_input_builder",
            "input_builder_params": {},
            "reward_parser_name": "base_reward_parser",
            "reward_parser_params": {},
        }
    )
    model = VLMRewardModel(cfg)
    model._model = load_lora_adapter(model._model, args.checkpoint)
    model._model.to(args.device).eval()
    payload = extract_features(model, rows, args)
    payload["metadata"] = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "sample_type": args.sample_type,
        "rank": args.rank,
        "world_size": args.world_size,
        "num_samples": len(rows),
        "history_size": args.history_size,
        "num_bins": args.num_bins,
        "video_fps": args.video_fps,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    logger.info("%s", json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
