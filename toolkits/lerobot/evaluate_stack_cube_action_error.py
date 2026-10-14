# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Measure Stack Cube H1/H5/H10 action errors with fixed validation samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from openpi.shared import normalize
from openpi.transforms import Normalize

from rlinf.data.storage.lerobot import episode_boundaries
from rlinf.models import get_model
from toolkits.lerobot.evaluate_stack_cube_openpi import preprocessing_protocol


def select_frames(actions: np.ndarray, uniform_count: int = 32) -> np.ndarray:
    """Select uniform coverage plus every gripper transition and its ±2 frames."""
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError("Expected a nonempty sequence of 7D actions")
    if uniform_count < 1:
        raise ValueError("uniform_count must be positive")
    uniform = np.linspace(
        0, len(actions) - 1, min(uniform_count, len(actions)), dtype=int
    )
    transitions = np.flatnonzero(actions[1:, -1] != actions[:-1, -1]) + 1
    neighbors = np.clip(transitions[:, None] + np.arange(-2, 3), 0, len(actions) - 1)
    return np.unique(np.concatenate([uniform, neighbors.reshape(-1)]))


def sample_noise_seed(seed: int, episode: int, frame: int) -> int:
    """Derive flow noise independently of sample iteration order."""
    return int(
        np.random.SeedSequence([seed, episode, frame]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def evaluation_protocol(frames_per_episode: int = 32, noise_seed: int = 0) -> dict:
    """Describe the fixed offline inputs and comparison metrics."""
    return {
        "version": 1,
        "primary_metric": "h5.normalized_mae",
        "uniform_frames_per_episode": frames_per_episode,
        "gripper_transition_neighbors": [-2, -1, 0, 1, 2],
        "selection": "sorted_unique_with_episode_boundary_clipping",
        "flow_noise": "numpy.SeedSequence([seed, episode_index, frame_index]) uint64 -> torch CPU Generator",
        "noise_seed": noise_seed,
        "eval_sft_image_crop": True,
        "action_horizon": 10,
        "execution_chunk": 5,
        "denoising_steps": 5,
        "action_dimensions": 7,
        "normalization": "train_only_quantile",
        "targets": "relative_actions_without_extra_delta_or_episode_padding",
    }


def evaluate(
    dataset_root: Path,
    checkpoint: Path,
    norm_stats: Path,
    *,
    frames_per_episode: int = 32,
    noise_seed: int = 0,
    device: str = "cuda:0",
) -> dict:
    """Evaluate relative actions without delta subtraction or padded targets.

    Observations cover uniform samples and gripper-transition neighborhoods.
    Flow noise is fixed per episode and frame. Normalization must come from
    the matching train split.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if frames_per_episode < 1:
        raise ValueError("frames_per_episode must be positive")
    marker = checkpoint / "actor/checkpoint_metadata.json"
    if not (checkpoint / "model.safetensors").exists():
        if not marker.exists() or not json.loads(marker.read_text()).get("complete"):
            raise ValueError("SFT checkpoint must have complete export metadata")
    config_dir = Path(__file__).resolve().parents[2] / "examples/sft/config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="embodichain_stack_cube_sft_openpi_pi05_smoke")
    cfg = config.actor.model
    cfg.model_path = str(checkpoint)
    cfg.openpi.task = "eval"
    cfg.openpi.eval_sft_image_crop = True
    cfg.openpi.action_chunk = 10
    cfg.num_action_chunks = 10
    cfg.openpi_data.norm_stats_path = str(norm_stats)
    torch.set_num_threads(4)
    model = get_model(cfg).to(device).eval()
    dataset = LeRobotDataset(str(dataset_root))
    starts, ends = episode_boundaries(dataset)
    normalization = Normalize(normalize.load(norm_stats.parent), use_quantiles=True)
    protocol = evaluation_protocol(frames_per_episode, noise_seed)
    preprocessing = preprocessing_protocol(eval_sft_image_crop=True)
    selected_samples = []
    prompt = "Stack the red block on the blue block, then stack the green block on the red block"
    rows = []
    for episode, (start, end) in enumerate(zip(starts, ends, strict=True)):
        start, end = int(start), int(end)
        table = pd.read_parquet(dataset_root / dataset.meta.get_data_file_path(episode))
        actions = np.stack(table["action"])
        if len(actions) != end - start:
            raise ValueError("Parquet episode size does not match LeRobot boundaries")
        selected = select_frames(actions, frames_per_episode)
        for relative_frame in selected:
            frame = start + int(relative_frame)
            selected_samples.append([episode, int(relative_frame)])
            sample = dataset[frame]
            obs = {
                "main_images": (
                    sample["observation.images.front"].permute(1, 2, 0) * 255
                ).to(torch.uint8)[None],
                "wrist_images": (
                    sample["observation.images.wrist"].permute(1, 2, 0) * 255
                ).to(torch.uint8)[None],
                "states": sample["observation.state"][None],
                "task_descriptions": [prompt],
            }
            generator = torch.Generator(device="cpu").manual_seed(
                sample_noise_seed(noise_seed, episode, int(relative_frame))
            )
            noise = torch.randn((1, 10, 32), generator=generator).to(device)
            predicted, _ = model.predict_action_batch(obs, mode="eval", noise=noise)
            predicted = np.asarray(
                predicted.detach().cpu() if torch.is_tensor(predicted) else predicted
            )[0, :, :7]
            if predicted.shape != (10, 7) or not np.isfinite(predicted).all():
                raise ValueError(f"Invalid model action horizon: {predicted.shape}")
            target = actions[relative_frame : relative_frame + 10]
            for horizon in (1, 5, 10):
                count = min(horizon, len(target))
                p, t = predicted[:count], target[:count]
                error = (
                    normalization({"actions": p})["actions"]
                    - normalization({"actions": t})["actions"]
                )
                if not np.isfinite(error).all():
                    raise ValueError(
                        f"Non-finite action error at episode {episode}, frame {frame}"
                    )
                rows.append(
                    {
                        "episode": episode,
                        "frame": int(frame - start),
                        "horizon": horizon,
                        "target_steps": count,
                        "normalized_mae": float(np.abs(error).mean()),
                        "raw_mae_per_dim": np.abs(p - t).mean(axis=0).tolist(),
                        "gripper_error_rate": float(
                            np.mean(np.sign(p[:, -1]) != np.sign(t[:, -1]))
                        ),
                    }
                )
    summary = {}
    for horizon in (1, 5, 10):
        part = [row for row in rows if row["horizon"] == horizon]
        summary[f"h{horizon}"] = {
            "normalized_mae": float(np.mean([row["normalized_mae"] for row in part])),
            "raw_mae_per_dim": np.mean(
                [row["raw_mae_per_dim"] for row in part], axis=0
            ).tolist(),
            "gripper_error_rate": float(
                np.mean([row["gripper_error_rate"] for row in part])
            ),
        }
    return {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "norm_stats": str(norm_stats),
        "norm_stats_sha256": hashlib.sha256(norm_stats.read_bytes()).hexdigest(),
        "noise_seed": noise_seed,
        "frames_per_episode": frames_per_episode,
        "episodes": len(starts),
        "eval_sft_image_crop": True,
        "preprocessing_protocol": preprocessing,
        "preprocessing_protocol_sha256": hashlib.sha256(
            json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "selected_samples": selected_samples,
        "selection_sha256": hashlib.sha256(
            json.dumps(selected_samples).encode()
        ).hexdigest(),
        "summary": summary,
        "samples": rows,
    }


def main() -> None:
    """Write reproducible auxiliary action errors, without running simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-episode", type=int, default=32)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = evaluate(
        args.dataset,
        args.checkpoint,
        args.norm_stats,
        frames_per_episode=args.frames_per_episode,
        noise_seed=args.noise_seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
