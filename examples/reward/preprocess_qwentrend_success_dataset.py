#!/usr/bin/env python3
"""Compatibility entry point for terminal QwenTrend success preprocessing."""

from examples.reward.preprocess_qwentrend_terminal_success_dataset import (
    build_rows,
    inspect_episode,
    main,
    make_row,
    parse_args,
    split_for,
)

__all__ = [
    "build_rows",
    "inspect_episode",
    "main",
    "make_row",
    "parse_args",
    "split_for",
]

if __name__ == "__main__":
    main(parse_args())
