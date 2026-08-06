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

"""Terminal-success sampling helpers for the dual-line VLM Trend pipeline."""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rlinf.data.datasets.vlm_trend_io import inspect_episode, split_for
from rlinf.utils.logging import get_logger

logger = get_logger()


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


def _items_for_split(
    items: list[dict[str, Any]], split: str, val_split: float
) -> list[dict[str, Any]]:
    return [item for item in items if split_for(item["path"], val_split) == split]


def _build_online_split_rows(
    split_items: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample fixed-interval windows for one split (online-matched labels)."""
    rows: list[dict[str, Any]] = []
    positive_count = 0
    for item in split_items:
        success_steps = set(item["success_steps"])
        for end_step in range(
            args.window_size,
            item["end_step"] + 1,
            args.online_interval,
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
    return rows, {
        "positive": positive_count,
        "negative": len(rows) - positive_count,
        "online_interval": args.online_interval,
    }


def _hard_negative_candidates(
    item: dict[str, Any], args: argparse.Namespace, rng: random.Random
) -> list[tuple[dict[str, Any], int]]:
    """Pick hard-negative end steps far from success for one episode."""
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
    return [(item, end_step) for end_step in candidates]


def _build_balanced_split_rows(
    split_items: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Balance terminal positives with terminal and hard negatives."""
    positive = [item for item in split_items if item["is_complete"] and item["success"]]
    terminal_negative = [
        item for item in split_items if item["is_complete"] and not item["success"]
    ]
    positive = positive[: args.max_positive]
    rng.shuffle(positive)
    rng.shuffle(terminal_negative)

    hard_negative: list[tuple[dict[str, Any], int]] = []
    for item in split_items:
        hard_negative.extend(_hard_negative_candidates(item, args, rng))
    rng.shuffle(hard_negative)

    positive_rows: list[dict[str, Any]] = []
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
        make_row(item, args.window_size, item["end_step"], False, "failure_terminal")
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
    return rows, {
        "positive": len(positive_rows),
        "terminal_negative": len(terminal_negative),
        "hard_negative": len(hard_negative),
        "negative_positive_ratio": (
            (len(terminal_negative) + len(hard_negative)) / max(1, len(positive_rows))
        ),
    }


def build_rows(
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Inspect, split, sample, and balance all source episodes."""
    paths = sorted(
        path for root in args.raw_data_path for path in Path(root).glob("*.pkl")
    )

    def _inspect(path: Path) -> dict[str, Any] | None:
        item = inspect_episode(str(path), args.window_size)
        if item is None:
            logger.warning("Skipping unreadable episode %s", path)
        return item

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspected = list(executor.map(_inspect, paths))
    items = [item for item in inspected if item is not None]
    rng = random.Random(args.seed)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {"input_episodes": len(paths), "splits": {}}
    online_interval = int(getattr(args, "online_interval", 0))

    for split in ("train", "eval"):
        split_items = _items_for_split(items, split, args.val_split)
        if online_interval > 0:
            rows, split_stats = _build_online_split_rows(split_items, args, rng)
        else:
            rows, split_stats = _build_balanced_split_rows(split_items, args, rng)
        rows_by_split[split] = rows
        stats["splits"][split] = split_stats

    stats["complete_episodes"] = sum(item["is_complete"] for item in items)
    stats["partial_episodes"] = sum(not item["is_complete"] for item in items)
    return rows_by_split, stats


__all__ = ["build_rows", "make_row"]
