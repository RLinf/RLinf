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

"""Evaluation of SuperMimic G1 policies and export of the result tables."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.sim.g1_mimic.eval_metrics import PUSH_RECOVERY, summarize_rows
from rlinf.envs.sim.g1_mimic.push_recovery import (
    PAPER_PROTOCOL,
    PROTOCOL_NAME,
    REFERENCE_POSE,
)
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.utils.nested_dict_process import cat_list_of_dict_tensor


def merge_rank_rows(results: list[dict | None]) -> dict[str, torch.Tensor]:
    """Concatenate the result rows every env rank returned."""
    parts = [rows for rows in results if rows]
    if not parts:
        raise RuntimeError("no env rank returned evaluation rows")
    return cat_list_of_dict_tensor(parts)


def _json_ready(metrics: dict[str, float]) -> dict[str, float | None]:
    """Strict-JSON metrics: an undefined value (NaN) becomes ``null``."""
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in metrics.items()
    }


def export_eval_results(
    out_dir: str | os.PathLike,
    cfg: DictConfig,
    meta: dict,
    metrics: dict[str, float],
    tables: dict[str, list[dict]],
    nested: dict,
) -> Path:
    """Write ``summary.json``, one CSV per table and the resolved config."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {
        "checkpoint": cfg.runner.get("ckpt_path", None),
        "experiment": cfg.runner.logger.experiment_name,
        **{key: value for key, value in meta.items() if key != "clip_names"},
    }
    if PUSH_RECOVERY in meta["tasks"]:
        info.update(
            protocol=PROTOCOL_NAME, reference_pose=REFERENCE_POSE, paper=PAPER_PROTOCOL
        )
    summary = {"meta": info, "metrics": _json_ready(metrics), **nested}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    )
    for name, rows in tables.items():
        if not rows:
            continue
        with open(out / f"{name}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    OmegaConf.save(cfg, out / "config.yaml")
    return out


def eval_export_dir(cfg: DictConfig, step: int | None = None) -> str | None:
    """Export directory of ``runner.eval_export``; ``None`` when export is off."""
    export = cfg.runner.get("eval_export", None)
    if export is None or not export.get("enabled", False):
        return None
    base = export.get("output_dir", None) or os.path.join(
        cfg.runner.logger.log_path, cfg.runner.logger.experiment_name, "eval_results"
    )
    return base if step is None else os.path.join(base, f"global_step_{step}")


class SuperMimicEvalRunner(EmbodiedEvalRunner):
    """Run the G1 evaluation tasks and summarize the rows of every env rank.

    The env workers return one row per job; this runner computes the logged
    metrics and, with ``runner.eval_export.enabled``, writes the tables.
    """

    def evaluate(self) -> dict[str, float]:
        meta = self.env.eval_metadata().wait()[0]
        env_handle = self.env.evaluate(
            input_channel=self.env_channel, rollout_channel=self.rollout_channel
        )
        rollout_handle = self.rollout.evaluate(
            input_channel=self.rollout_channel, output_channel=self.env_channel
        )
        rows = merge_rank_rows(env_handle.wait())
        rollout_handle.wait()
        metrics, tables, nested = summarize_rows(rows, meta)
        out_dir = eval_export_dir(self.cfg)
        if out_dir is not None:
            export_eval_results(out_dir, self.cfg, meta, metrics, tables, nested)
            self.logger.info(f"Wrote SuperMimic evaluation results to {out_dir}")
        return metrics
