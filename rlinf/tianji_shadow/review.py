"""Shared human and Monitor event log for episode review."""

import json
from pathlib import Path

_HUMAN_KINDS = {
    "start": "episode_start",
    "first_advance": "subtask",
    "backtrack": "subtask",
    "revisit": "subtask",
    "human_start": "h",
    "monitor_stage_requested": "control",
    "monitor_stage_expired": "control",
    "monitor_s4_candidate": "control",
    "monitor_pause": "control",
    "monitor_recovered": "control",
    "monitor_manual": "control",
    "manual_stage_correction": "control",
    "human_stop": "takeover_end",
    "success": "episode_end",
    "failure": "episode_end",
    "discard": "episode_end",
    "fault": "episode_end",
    "reset_failed": "episode_end",
}


def episode_file(run: Path, episode: int, start_ns: int) -> Path:
    """Return the shared path for an accepted episode start."""
    return run / "episode_reviews" / f"episode_{episode:04d}_{start_ns}" / "review.jsonl"


def append_review(path: Path, row: dict) -> None:
    """Append a complete line with a single append-mode write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with path.open("ab", buffering=0) as log:
        if log.write(line) != len(line):
            raise OSError("Short review event write")


def human_row(event: dict) -> dict | None:
    """Convert an accepted operator action to a review row."""
    kind = _HUMAN_KINDS.get(event["name"])
    if kind is None:
        return None
    row = dict(event, source=event.get("source", "human"), kind=kind)
    if kind == "subtask":
        row["to_stage"] = event["stage"]
    if event["name"] == "success":
        row["completed_stage"] = event["stage"]
    return row


def monitor_row(episode: int, prediction: dict, model_input: str | None) -> dict:
    """Convert a Monitor output to a review row."""
    name = prediction["name"]
    row = dict(prediction, source="monitor", episode=episode,
               kind="subtask" if name == "complete" else "h")
    if name == "complete":
        row["completed_stage"] = prediction["stage"]
        row["to_stage"] = prediction["stage"] + 1 if prediction["stage"] < 4 else None
    if model_input is not None:
        row["input"] = f"monitor_shadow/{model_input}"
    return row
