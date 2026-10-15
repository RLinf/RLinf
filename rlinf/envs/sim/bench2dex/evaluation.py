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

"""Bench2Dex's four-channel scene protocol and native result reporting."""

import copy
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

CHANNELS = ("none", "cov", "inv", "inv_cov")
RESAMPLE_GROUPS = {
    "cov_only": {"object_pose", "table_height"},
    "inv_only": {"background", "table_surface", "light", "clutter", "camera"},
}


def normalize_profile(profile: str) -> str:
    """Resolve the channel aliases accepted by Bench2Dex's native evaluator."""
    aliases = {
        "": "none",
        "none": "none",
        "baseline": "none",
        "cov": "cov_only",
        "cov_only": "cov_only",
        "cov-only": "cov_only",
        "inv": "inv_only",
        "inv_only": "inv_only",
        "inv-only": "inv_only",
        "inv_cov": "inv_cov",
        "inv+cov": "inv_cov",
        "full": "inv_cov",
    }
    try:
        return aliases[str(profile or "none").strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Expected a Bench2Dex channel, got {profile!r}") from exc


def configure_profile(raw: dict, profile: str) -> dict:
    """Apply the axis switches and perturbation ranges from native run_policy.py."""
    profile = normalize_profile(profile)
    config = copy.deepcopy(raw)

    def mapping(*path):
        value = config
        for key in path:
            if not isinstance(value.get(key), dict):
                value[key] = {}
            value = value[key]
        return value

    invariant = [
        mapping("appearance", "background"),
        mapping("appearance", "table_surface"),
        mapping("appearance", "light"),
        mapping("clutter", "tabletop"),
        mapping("spatial", "camera"),
    ]
    object_pose = mapping("spatial", "object_pose")
    table_height = mapping("spatial", "table_height")
    covariant = [object_pose, table_height]
    for axis in invariant + covariant:
        axis["enabled"] = False
    if profile in ("inv_only", "inv_cov"):
        for axis in invariant:
            axis["enabled"] = True
            axis["asset_split"] = "unseen"
    if profile in ("cov_only", "inv_cov"):
        for axis in covariant:
            axis["enabled"] = True
        object_pose["position_jitter_cm"] = 2.0
        object_pose["yaw_jitter_deg"] = 10.0
        table_height["offset_m_range"] = [-0.05, 0.05]
    if profile != "none":
        config["asset_split"] = "unseen"
    return config


def apply_asset_split(raw: dict, split: str) -> dict:
    """Select the same asset-backed splits as the official evaluator."""
    if split not in ("seen", "unseen", "all"):
        raise ValueError(f"Expected seen, unseen or all asset split, got {split!r}")
    config = copy.deepcopy(raw)
    config["asset_split"] = split
    for category, axis in (
        ("appearance", "background"),
        ("appearance", "table_surface"),
        ("clutter", "tabletop"),
    ):
        config[category][axis]["asset_split"] = split
    return config


class Bench2DexResults:
    """Write native per-episode and aggregate results for one evaluation process."""

    def __init__(self, source_path: Path, output_dir: Path, policy_name: str):
        if str(source_path) not in sys.path:
            sys.path.append(str(source_path))
        from benchmark.metrics import EpisodeResult
        from benchmark.results import initialize_incremental_results

        self.result_type = EpisodeResult
        self.field_names = {field.name for field in fields(EpisodeResult)}
        self.output_dir = output_dir
        self.policy_name = policy_name
        self.results = []
        initialize_incremental_results(output_dir)

    def append(
        self, info: dict[str, Any], policy_queries: int, query_at_success: int | None
    ) -> None:
        """Record one terminal transition with the native success definition."""
        from benchmark.results import append_episode_result

        metrics = info["metrics"]
        early_stop = info["early_stop"]
        values = {
            key: value for key, value in metrics.items() if key in self.field_names
        }
        values.update(
            scene=info["scene"],
            episode_index=info["episode_index"],
            seed=info["seed"],
            episode_seed=info["seed"],
            base_seed=info["base_seed"],
            seed_policy=info["seed_policy"],
            scene_generalization_sample=info["scene_generalization_sample"],
            success=bool(
                metrics["stable_success"]
                if early_stop
                else metrics["at_end_success_budget"]
            ),
            steps=info["physics_steps"],
            robot_key=info["robot_key"],
            task_family=info["task_family"],
            perturbation_axis={
                "none": "none",
                "cov_only": "cov",
                "inv_only": "inv",
                "inv_cov": "inv+cov",
            }[info["generalization_profile"]],
            policy_query_count=policy_queries,
            policy_queries_to_stable_success=query_at_success,
            early_stop=early_stop,
            max_episode_steps=info["max_control_steps"] * info["control_decimation"],
            dwell_time_s=info["dwell_time_s"],
        )
        result = self.result_type(**values)
        self.results.append(result)
        append_episode_result(self.output_dir, self.policy_name, result, self.results)
