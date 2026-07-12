# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Build dual-view terminal-success SFT data from collected episode pickles."""

import argparse
import hashlib
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROMPT = (
    "Estimate task-conditioned success potential for this robot manipulation "
    "state. Task: {task}. The two synchronized videos show the same {window}-frame "
    "history from two camera views."
)


def _array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.uint8)


def _extra_view(observation: dict[str, Any]) -> Any | None:
    value = observation.get("third_view_images")
    if value is not None:
        return value
    value = observation.get("extra_view_images")
    if value is None:
        return None
    if torch.is_tensor(value):
        return value[0] if value.ndim == 4 else value
    value = np.asarray(value)
    return value[0] if value.ndim == 4 else value


def _is_success(payload: dict[str, Any], path: Path) -> bool:
    if "success" in payload:
        return bool(payload["success"])
    infos = payload.get("infos", [])
    for info in reversed(infos):
        if isinstance(info, dict) and "success" in info:
            value = info["success"]
            if torch.is_tensor(value):
                return bool(value.detach().cpu().any().item())
            return bool(np.asarray(value).any())
    return path.stem.endswith("_success")


def _split(path: Path, eval_ratio: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    return "eval" if score < eval_ratio else "train"


def _window_sample(
    path: Path,
    payload: dict[str, Any],
    start: int,
    end: int,
    answer: str,
    target_type: str,
    task: str,
) -> dict[str, Any] | None:
    observations = payload.get("observations", [])
    selected = observations[start : end + 1]
    main = [obs.get("main_images") for obs in selected]
    extra = [_extra_view(obs) for obs in selected]
    if len(selected) != end - start + 1 or any(x is None for x in main + extra):
        return None
    return {
        "source": str(path),
        "answer": answer,
        "target_type": target_type,
        "start_step": start,
        "end_step": end,
        "task": task,
        "main_frames": [_array(x) for x in main],
        "extra_view_frames": [_array(x) for x in extra],
    }


def build_samples(
    paths: list[Path],
    window_size: int,
    min_failure_steps: int,
    hard_negative_margin: int,
    task_description: str,
) -> list[dict[str, Any]]:
    """Build positives, terminal failures, and nonterminal hard negatives."""
    samples: list[dict[str, Any]] = []
    for path in paths:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        observations = payload.get("observations", [])
        if len(observations) < window_size:
            continue
        task = str(
            payload.get("task") or payload.get("task_description") or task_description
        )
        success = _is_success(payload, path)
        if not success and len(observations) < min_failure_steps:
            continue
        end = len(observations) - 1
        terminal = _window_sample(
            path,
            payload,
            end - window_size + 1,
            end,
            "1" if success else "0",
            "success_terminal" if success else "failure_terminal",
            task,
        )
        if terminal is not None and (success or len(observations) >= min_failure_steps):
            samples.append(terminal)

        latest_hard_end = end - hard_negative_margin if success else end - 1
        if latest_hard_end >= window_size - 1:
            hard_end = latest_hard_end
            hard = _window_sample(
                path,
                payload,
                hard_end - window_size + 1,
                hard_end,
                "0",
                "nonterminal_hard_negative",
                task,
            )
            if hard is not None:
                samples.append(hard)
    return samples


def _balance(
    samples: list[dict[str, Any]], negative_ratio: float, seed: int
) -> list[dict[str, Any]]:
    positives = [sample for sample in samples if sample["answer"] == "1"]
    negatives = [sample for sample in samples if sample["answer"] == "0"]
    random.Random(seed).shuffle(negatives)
    keep = min(len(negatives), round(len(positives) * negative_ratio))
    merged = positives + negatives[:keep]
    random.Random(seed).shuffle(merged)
    return merged


def write_dataset(
    samples: list[dict[str, Any]], output: Path, eval_ratio: float, seed: int
) -> None:
    counters: Counter[str] = Counter()
    handles = {}
    try:
        for split in ("train", "eval"):
            split_dir = output / split
            (split_dir / "pkl").mkdir(parents=True, exist_ok=True)
            handles[split] = (split_dir / "segments.jsonl").open("w")
        for index, sample in enumerate(samples):
            source = Path(sample["source"])
            split = _split(source, eval_ratio, seed)
            sample_path = output / split / "pkl" / f"sample_{index:08d}.pkl"
            with sample_path.open("wb") as stream:
                pickle.dump(
                    {
                        "main_frames": sample.pop("main_frames"),
                        "extra_view_frames": sample.pop("extra_view_frames"),
                    },
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            prompt = PROMPT.format(
                task=sample["task"],
                window=sample["end_step"] - sample["start_step"] + 1,
            )
            record = {
                "question": prompt,
                "prompt": prompt,
                "answer": sample["answer"],
                "pkl_path": str(sample_path.resolve()),
                "source_episode_path": sample["source"],
                "segment_metadata": {
                    "start_step": sample["start_step"],
                    "end_step": sample["end_step"],
                    "target_name": "terminal_success",
                    "target_type": sample["target_type"],
                },
            }
            handles[split].write(json.dumps(record) + "\n")
            counters[f"{split}_{sample['answer']}"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    (output / "dataset_info.json").write_text(
        json.dumps(dict(counters), indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-description", default="robot manipulation task")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-failure-steps", type=int, default=50)
    parser.add_argument("--hard-negative-margin", type=int, default=5)
    parser.add_argument("--negative-positive-ratio", type=float, default=3.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = sorted(
        {
            path.resolve()
            for root in args.raw_data_path
            for path in Path(root).rglob("*.pkl")
        }
    )
    samples = build_samples(
        paths,
        args.window_size,
        args.min_failure_steps,
        args.hard_negative_margin,
        args.task_description,
    )
    samples = _balance(samples, args.negative_positive_ratio, args.seed)
    if not samples:
        raise ValueError("No valid dual-view success samples were produced")
    write_dataset(samples, Path(args.output_dir), args.eval_ratio, args.seed)


if __name__ == "__main__":
    main()
