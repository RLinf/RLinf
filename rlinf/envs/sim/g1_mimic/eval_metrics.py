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

"""Summary metrics of the G1 evaluation tasks.

:class:`~rlinf.envs.sim.g1_mimic.g1_mimic_eval_env.G1MimicEvalEnv` returns
one row per evaluation job, with ``<task>.<column>`` keys. The functions here
turn the rows every env rank returned into scalar metrics and plain tables.
"""

from __future__ import annotations

import math

import torch

from rlinf.envs.sim.g1_mimic.push_recovery import (
    OUTCOMES,
    REASONS,
    summarize_trials,
)

TRACKING = "tracking"
PUSH_RECOVERY = "push_recovery"
EVAL_TASKS = (TRACKING, PUSH_RECOVERY)


def check_job_coverage(rows: dict[str, torch.Tensor], num_jobs: int) -> None:
    """Every job of the evaluation table must come back in exactly one valid row."""
    jobs = rows["job"][rows["valid"] > 0].long()
    if not torch.equal(torch.sort(jobs).values, torch.arange(num_jobs)):
        missing = sorted(set(range(num_jobs)) - set(jobs.tolist()))
        raise RuntimeError(
            f"expected one valid row for each of the {num_jobs} evaluation jobs, "
            f"found {jobs.numel()} valid rows; missing jobs {missing[:10]}"
        )


def task_rows(rows: dict[str, torch.Tensor], task: str) -> dict[str, torch.Tensor]:
    """Valid rows of one task, with the ``<task>.`` prefix removed."""
    keep = (rows["task"] == EVAL_TASKS.index(task)) & (rows["valid"] > 0)
    prefix = f"{task}."
    return {
        key[len(prefix) :]: value[keep]
        for key, value in rows.items()
        if key.startswith(prefix)
    }


def _mean(values: torch.Tensor) -> float:
    return float(values.float().mean()) if values.numel() else float("nan")


def summarize_tracking(rows: dict[str, torch.Tensor]) -> dict[str, float]:
    """Unweighted mean over clips; one row per clip, so batching does not matter."""
    fail = rows["fail"].float()
    fail_count = fail.sum().clamp_min(1.0)
    return {
        "normalized_return": _mean(rows["normalized_return"]),
        "body_pos_err": _mean(rows["body_pos_err"]),
        "torso_pos_err": _mean(rows["torso_pos_err"]),
        "torso_ori_err": _mean(rows["torso_ori_err"]),
        "free_root_pos_err": _mean(rows["free_root_pos_err"]),
        "free_root_ori_err": _mean(rows["free_root_ori_err"]),
        "free_root_xy_drift": _mean(rows["free_root_xy_drift"]),
        "free_root_yaw_drift": _mean(rows["free_root_yaw_drift"]),
        "fail_rate": _mean(fail),
        "fail_step": float((rows["fail_step"].float() * fail).sum() / fail_count),
        "fail_time_s": float(rows["fail_time_s"].sum() / fail_count),
        "episode_length_s": _mean(rows["episode_length_s"]),
        "clip_end_rate": _mean(rows["clip_end"]),
        "steps": _mean(rows["steps"]),
        "num_clips": float(rows["clip_id"].shape[0]),
    }


def _optional(value: float) -> float | None:
    return None if not math.isfinite(value) else value


def push_trial_records(
    rows: dict[str, torch.Tensor], frame: str, push_mode: str, seed: int
) -> list[dict]:
    """One record per push trial, ordered by trial index."""
    order = torch.argsort(rows["trial"])
    records = []
    for i in order.tolist():
        outcome = OUTCOMES[int(rows["outcome"][i])]
        records.append(
            {
                "trial": int(rows["trial"][i]),
                "speed_m_s": float(rows["speed_m_s"][i]),
                "direction_x": float(rows["direction_x"][i]),
                "direction_y": float(rows["direction_y"][i]),
                "frame": frame,
                "push_mode": push_mode,
                "seed": int(seed),
                "outcome": outcome,
                "reason": REASONS[int(rows["reason"][i])],
                "recovered": int(outcome == "recovered"),
                "fallen": int(outcome == "fallen"),
                "timeout": int(outcome == "timeout"),
                "initialization_failed": int(outcome == "initialization_failed"),
                **{
                    key: _optional(float(rows[key][i]))
                    for key in (
                        "recovery_time_s",
                        "stable_onset_time_s",
                        "world_push_x",
                        "world_push_y",
                        "measured_dvx",
                        "measured_dvy",
                    )
                },
            }
        )
    return records


def summarize_push(
    records: list[dict], speeds: list[float]
) -> tuple[dict[str, float], dict]:
    """Logger metrics and the nested summary of :func:`summarize_trials`."""
    stats = summarize_trials(records, speeds)

    def flat(prefix: str, group: dict) -> dict[str, float]:
        return {
            f"{prefix}/{key}": float("nan") if value is None else float(value)
            for key, value in group.items()
            if key != "speed_m_s"
        }

    metrics = flat("push", stats["overall"])
    for group in stats["by_speed"]:
        metrics.update(flat(f"push/speed_{group['speed_m_s']:g}", group))
    return metrics, stats


def labeled_rows(
    rows: dict[str, torch.Tensor], labels: list[str], label_key: str
) -> list[dict]:
    """One dict per row; a vector column becomes ``<key>_<i>`` columns."""
    out = []
    for index, label in enumerate(labels):
        row = {label_key: label}
        for key, value in rows.items():
            item = value[index]
            if item.numel() == 1:
                row[key] = item.item()
            else:
                for j, part in enumerate(item.tolist()):
                    row[f"{key}_{j}"] = part
        out.append(row)
    return out


def _sorted(rows: dict[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    order = torch.argsort(rows[key])
    return {name: value[order] for name, value in rows.items()}


def trajectory_records(rows: dict[str, torch.Tensor]) -> list[dict]:
    """Per-step push trajectories: phase, root position, root velocity, tilt."""
    names = ("phase", "root_x", "root_y", "root_z", "vel_x", "vel_y", "vel_z", "tilt")
    records = []
    for trial, frames in zip(rows["trial"].tolist(), rows["trajectory"], strict=True):
        for step, frame in enumerate(frames.tolist()):
            records.append(
                {"trial": int(trial), "step": step, **dict(zip(names, frame))}
            )
    return records


def summarize_rows(
    rows: dict[str, torch.Tensor], meta: dict
) -> tuple[dict[str, float], dict[str, list[dict]], dict]:
    """Metrics, export tables and nested summaries of every task in ``rows``.

    ``rows`` holds the rows of all env ranks on CPU and ``meta`` is
    ``G1MimicEvalEnv.metadata()``. Tables are keyed by their file stem.
    """
    check_job_coverage(rows, int(meta["num_jobs"]))
    metrics: dict[str, float] = {}
    tables: dict[str, list[dict]] = {}
    nested: dict = {}
    tasks = meta["tasks"]
    if TRACKING in tasks:
        part = _sorted(task_rows(rows, TRACKING), "clip_id")
        metrics.update(summarize_tracking(part))
        names = [meta["clip_names"][int(i)] for i in part["clip_id"].tolist()]
        tables["per_clip"] = labeled_rows(part, names, "clip_name")
    if PUSH_RECOVERY in tasks:
        part = _sorted(task_rows(rows, PUSH_RECOVERY), "trial")
        records = push_trial_records(
            part, meta["push_frame"], meta["push_mode"], meta["push_seed"]
        )
        push_metrics, nested["push_recovery"] = summarize_push(
            records, meta["push_speeds"]
        )
        metrics.update(push_metrics)
        tables["per_trial"] = records
        if "trajectory" in part:
            tables["trajectory"] = trajectory_records(part)
    return metrics, tables, nested
