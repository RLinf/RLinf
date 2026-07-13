#!/usr/bin/env python3
"""Build terminal success data from QwenTrend rollout episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _as_bool(value: Any) -> bool:
    """Convert scalar tensor/array metadata to bool."""
    if value is None:
        return False
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def inspect_episode(path: Path, window_size: int) -> dict[str, Any] | None:
    """Read the metadata needed to sample one rollout episode."""
    with path.open("rb") as stream:
        episode = pickle.load(stream)
    observations = episode.get("observations", [])
    actions = episode.get("actions", [])
    if len(observations) < window_size or not actions:
        return None
    terminated = episode.get("terminated", [])
    truncated = episode.get("truncated", [])
    success = bool(episode.get("success", False))
    is_complete = (
        success
        or bool(terminated and terminated[-1])
        or bool(truncated and truncated[-1])
    )
    end_step = min(len(observations) - 1, len(actions))
    success_steps = [
        index
        for index, info in enumerate(episode.get("infos", [])[: end_step + 1])
        if isinstance(info, dict) and _as_bool(info.get("success"))
    ]
    if success and not success_steps:
        success_steps = [end_step]
    return {
        "path": str(path.resolve()),
        "success": success,
        "end_step": end_step,
        "success_steps": success_steps,
        "is_complete": is_complete,
        "task": str(
            episode.get("task")
            or episode.get("task_description")
            or "Pick up the red cube and place it on the green spot on the table."
        ),
        "source_run": path.parent.parent.name,
    }


def split_for(path: str, val_split: float) -> str:
    """Assign an episode to a stable source-level split."""
    fraction = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) / 2**32
    return "eval" if fraction < val_split else "train"


def make_row(
    item: dict[str, Any],
    window_size: int,
    end_step: int,
    label_value: bool,
    target_type: str,
) -> dict[str, Any]:
    """Create one manifest row referencing an uncopied source episode."""
    label = "1" if label_value else "0"
    prompt = (
        "Estimate task-conditioned success potential for this robot manipulation "
        f"state. Task: {item['task']}. The two synchronized videos show the same "
        f"{window_size}-frame history from two camera views."
    )
    return {
        "task": item["task"],
        "prompt": prompt,
        "question": prompt,
        "answer": label,
        "pkl_path": item["path"],
        "source_episode_path": item["path"],
        "source_run": item["source_run"],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": label}]},
        ],
        "segment_metadata": {
            "start_step": max(0, end_step - window_size + 1),
            "end_step": end_step,
            "window_size": window_size,
            "progress_gap_steps": None,
            "success": label_value,
            "sample_type": "potential",
            "target_name": "terminal_success",
            "is_complete": item["is_complete"],
            "target_type": target_type,
            "source_run": item["source_run"],
        },
        "supervision": {
            "score_name": "terminal_success",
            "teacher_value": float(label_value),
            "teacher_delta": 0.0,
        },
    }


def build_rows(
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Inspect, split, sample, and balance all source episodes."""
    paths = sorted(
        path for root in args.raw_data_path for path in Path(root).glob("*.pkl")
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspected = list(
            executor.map(lambda path: inspect_episode(path, args.window_size), paths)
        )
    items = [item for item in inspected if item is not None]
    rng = random.Random(args.seed)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {"input_episodes": len(paths), "splits": {}}
    online_interval = int(getattr(args, "online_interval", 0))
    if online_interval > 0:
        for split in ("train", "eval"):
            rows: list[dict[str, Any]] = []
            positive_count = 0
            split_items = [
                item
                for item in items
                if split_for(item["path"], args.val_split) == split
            ]
            for item in split_items:
                success_steps = set(item["success_steps"])
                for end_step in range(
                    args.window_size,
                    item["end_step"] + 1,
                    online_interval,
                ):
                    label = end_step in success_steps
                    positive_count += int(label)
                    rows.append(
                        make_row(
                            item,
                            args.window_size,
                            end_step,
                            label,
                            "success_observed" if label else "online_negative",
                        )
                    )
            rng.shuffle(rows)
            rows_by_split[split] = rows
            stats["splits"][split] = {
                "positive": positive_count,
                "negative": len(rows) - positive_count,
                "online_interval": online_interval,
            }
        stats["complete_episodes"] = sum(item["is_complete"] for item in items)
        stats["partial_episodes"] = sum(not item["is_complete"] for item in items)
        return rows_by_split, stats
    for split in ("train", "eval"):
        split_items = [
            item for item in items if split_for(item["path"], args.val_split) == split
        ]
        positive = [
            item for item in split_items if item["is_complete"] and item["success"]
        ]
        terminal_negative = [
            item for item in split_items if item["is_complete"] and not item["success"]
        ]
        positive = positive[: args.max_positive]
        rng.shuffle(positive)
        positive_rows: list[dict[str, Any]] = []
        rng.shuffle(terminal_negative)
        hard_negative: list[tuple[dict[str, Any], int]] = []
        for item in split_items:
            candidates = list(range(args.window_size - 1, item["end_step"] + 1))
            if item["success_steps"]:
                candidates = [
                    end_step
                    for end_step in candidates
                    if all(
                        abs(end_step - success_step) > args.success_exclusion_steps
                        for success_step in item["success_steps"]
                    )
                ]
            else:
                candidates = candidates[:-1]
            if len(candidates) > args.hard_negatives_per_episode:
                candidates = rng.sample(candidates, args.hard_negatives_per_episode)
            hard_negative.extend((item, end_step) for end_step in candidates)
        rng.shuffle(hard_negative)
        for item in positive:
            success_step = rng.choice(item["success_steps"])
            positive_rows.append(
                make_row(item, args.window_size, success_step, True, "success_observed")
            )
            candidates = list(
                range(
                    max(
                        args.window_size - 1,
                        success_step - args.success_positive_lead_steps,
                    ),
                    success_step,
                )
            )
            rng.shuffle(candidates)
            for end_step in candidates[: args.near_terminal_positives_per_episode]:
                positive_rows.append(
                    make_row(
                        item, args.window_size, end_step, True, "success_near_observed"
                    )
                )
        target_negative_count = int(
            round(len(positive_rows) * args.negative_positive_ratio)
        )
        terminal_negative = terminal_negative[:target_negative_count]
        hard_limit = max(0, target_negative_count - len(terminal_negative))
        hard_negative = hard_negative[:hard_limit]
        rows = list(positive_rows)
        rows.extend(
            make_row(
                item,
                args.window_size,
                item["end_step"],
                False,
                "failure_terminal",
            )
            for item in terminal_negative
        )
        rows.extend(
            make_row(
                item,
                args.window_size,
                end_step,
                False,
                "nonterminal_hard_negative",
            )
            for item, end_step in hard_negative
        )
        rng.shuffle(rows)
        rows_by_split[split] = rows
        stats["splits"][split] = {
            "positive": len(positive_rows),
            "terminal_negative": len(terminal_negative),
            "hard_negative": len(hard_negative),
            "negative_positive_ratio": (
                (len(terminal_negative) + len(hard_negative))
                / max(1, len(positive_rows))
            ),
        }
    stats["complete_episodes"] = sum(item["is_complete"] for item in items)
    stats["partial_episodes"] = sum(not item["is_complete"] for item in items)
    return rows_by_split, stats


def main(args: argparse.Namespace) -> None:
    """Write manifests and dataset statistics."""
    rows_by_split, stats = build_rows(args)
    output_dir = Path(args.output_dir)
    for split, rows in rows_by_split.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "segments.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    (output_dir / "dataset_info.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-positive", type=int, default=5000)
    parser.add_argument("--negative-positive-ratio", type=float, default=4.0)
    parser.add_argument("--hard-negatives-per-episode", type=int, default=3)
    parser.add_argument("--success-exclusion-steps", type=int, default=10)
    parser.add_argument("--near-terminal-positives-per-episode", type=int, default=1)
    parser.add_argument("--success-positive-lead-steps", type=int, default=4)
    parser.add_argument("--online-interval", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
