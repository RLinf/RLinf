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

"""Configuration and episode selection without importing Isaac Sim."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from rlinf.data.schema.embodied_types import JointPositionTargets

from .evaluation import normalize_profile


@dataclass(frozen=True)
class Bench2DexSpec:
    """Task and embodiment contract shared with the simulator child."""

    source_path: Path
    assets_path: Path
    task_path: Path
    robot_key: str
    action_dim: int
    full_dof: int
    collect_config: Path
    main_camera: str
    extra_cameras: tuple[str, ...]
    wrist_cameras: tuple[str, ...]
    physics_dt: float
    control_decimation: int
    max_control_steps: int
    active_joint_names: tuple[str, ...]
    full_joint_names: tuple[str, ...]
    active_indices: tuple[int, ...]

    @property
    def camera_names(self) -> tuple[str, ...]:
        return (self.main_camera, *self.extra_cameras, *self.wrist_cameras)

    @classmethod
    def from_config(
        cls, config: dict, *, fallback_control_steps: int | None = None
    ) -> Bench2DexSpec:
        """Resolve a task and read DOF metadata from the selected checkout."""
        source = Path(config["source_path"]).expanduser().resolve()
        assets = Path(config["assets_path"]).expanduser().resolve()
        task = Path(config["task"])
        if not task.is_absolute():
            task = source / "scenes" / task.with_suffix(".yaml")
        with task.open() as stream:
            task_data = yaml.safe_load(stream)
        with (source / "robots/active_dof_maps.yml").open() as stream:
            robots = yaml.safe_load(stream)["robots"]
        robot_key = config["embodiment"]
        if robot_key not in robots:
            raise ValueError(
                f"Unknown Bench2Dex embodiment {robot_key!r}; choose {sorted(robots)}"
            )
        dof = robots[robot_key]
        full_names = tuple(dof["full_joint_names"])
        active_indices = tuple(int(index) for index in dof["active_indices"])
        if (
            len(full_names) != int(dof["full_dof"])
            or len(set(full_names)) != len(full_names)
            or len(active_indices) != int(dof["active_dof"])
            or len(set(active_indices)) != len(active_indices)
            or any(index < 0 or index >= len(full_names) for index in active_indices)
        ):
            raise ValueError(f"Invalid named joint layout for {robot_key!r}")
        dt = float(config["physics_dt"])
        decimation = int(config["control_decimation"])
        if dt <= 0 or decimation <= 0:
            raise ValueError("physics_dt and control_decimation must be positive")
        budget = config.get("max_control_steps")
        if budget is None:
            expert_steps = task_data["metrics"].get("expert_time_step")
            budget = (
                math.ceil(1.5 * expert_steps / decimation)
                if expert_steps is not None
                else fallback_control_steps
            )
            if budget is None:
                raise ValueError(
                    f"Set max_episode_steps or init_params.max_control_steps for "
                    f"{task.stem}, which has no expert time-step budget."
                )
        if int(budget) <= 0:
            raise ValueError(f"Expected a positive control-step budget, got {budget}")
        cameras = config["cameras"]
        collect = Path(config["collect_config"])
        if not collect.is_absolute():
            collect = source / collect
        spec = cls(
            source,
            assets,
            task.resolve(),
            robot_key,
            int(dof["active_dof"]),
            int(dof["full_dof"]),
            collect,
            cameras["main"],
            tuple(cameras.get("extra", [])),
            tuple(cameras.get("wrist", [])),
            dt,
            decimation,
            int(budget),
            tuple(full_names[index] for index in active_indices),
            full_names,
            active_indices,
        )
        if len(set(spec.camera_names)) != len(spec.camera_names):
            raise ValueError(f"Camera roles must be unique, got {spec.camera_names}")
        return spec


def joint_targets_in_runtime_order(
    actions,
    *,
    active_joint_names: tuple[str, ...],
    full_joint_names: tuple[str, ...],
) -> tuple[np.ndarray, str]:
    """Validate absolute joint targets and reorder named commands for execution."""
    layout = "active"
    names = active_joint_names
    if isinstance(actions, JointPositionTargets):
        if len(actions.joint_names) != 1:
            raise ValueError("Expected named targets for one simulator environment")
        layout = actions.joint_layout
        names = tuple(actions.joint_names[0])
        values = actions.joint_positions
        if values.ndim == 3:
            values = values[0]
    else:
        values = actions
    expected = active_joint_names if layout == "active" else full_joint_names
    values = np.asarray(values, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape != (values.shape[0], len(names))
        or not len(values)
    ):
        raise ValueError(
            f"Expected nonempty [C,{len(names)}] joint targets, got {values.shape}"
        )
    if len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError(f"Expected joint names {expected}, got {names}")
    if not np.isfinite(values).all():
        raise ValueError("Bench2Dex joint targets must be finite")
    order = [names.index(name) for name in expected]
    return values[:, order], layout


@dataclass(frozen=True)
class Bench2DexEpisode:
    """An optional scene anchor and an explicit seed supplied by a manifest."""

    anchor_hdf5: Path | None
    seed: int | None = None


def _anchor_pool(anchor_dir: str) -> list[Bench2DexEpisode]:
    path = Path(anchor_dir).expanduser().resolve()
    main, variants = [], []
    for anchor in path.glob("episode_*.hdf5"):
        match = re.fullmatch(r"episode_(\d+)(_1)?\.hdf5", anchor.name)
        if match:
            (variants if match[2] else main).append((int(match[1]), anchor))
    main.sort()
    variants.sort()
    selected = [p for i, p in main if i < 25] + [p for i, p in variants if i >= 25]
    selected = selected or [p for _, p in main]
    if not selected:
        raise ValueError(f"Bench2Dex anchor directory has no episode files: {path}")
    return [Bench2DexEpisode(p) for p in selected]


def load_episodes(config: dict) -> list[Bench2DexEpisode]:
    """Resolve the official anchor pool or a file/inline seeded episode manifest.

    Directory anchors follow the native main-0-24 and variant-25+ ordering.
    The fully resampled channel uses no anchor. All returned paths are absolute.
    """
    profile = normalize_profile(config.get("generalization_profile", "none"))
    if profile == "inv_cov":
        return [Bench2DexEpisode(None)]
    if config.get("anchor_dir"):
        return _anchor_pool(config["anchor_dir"])
    manifest = config.get("episode_manifest")
    if isinstance(manifest, list):
        entries = manifest
        base = Path.cwd()
    elif manifest:
        path = Path(manifest).expanduser().resolve()
        with path.open() as stream:
            entries = json.load(stream)
        base = path.parent
    else:
        if not config.get("anchor_hdf5"):
            raise ValueError(
                f"Bench2Dex {profile} requires anchor_dir, anchor_hdf5 or episode_manifest"
            )
        entries = [{"anchor_hdf5": config["anchor_hdf5"]}]
        base = Path.cwd()
    if not isinstance(entries, list) or not entries:
        raise ValueError("episode_manifest must contain a nonempty JSON list")
    episodes = []
    for entry in entries:
        anchor = Path(entry["anchor_hdf5"]).expanduser()
        anchor = anchor if anchor.is_absolute() else base / anchor
        seed = int(entry["seed"]) if "seed" in entry else None
        if seed is not None and not 0 <= seed < 2**32:
            raise ValueError(f"Expected seed in [0, 2**32), got {seed}")
        if not anchor.is_file():
            raise FileNotFoundError(f"Bench2Dex anchor not found: {anchor}")
        episodes.append(Bench2DexEpisode(anchor.resolve(), seed))
    return episodes


def validate_bench2dex_env_cfg(
    cfg: DictConfig,
    env_cfg: DictConfig,
    model_cfg: DictConfig,
    env_world_size: int,
    split: str,
    *,
    actor_world_size: int | None,
) -> None:
    """Validate model and simulator geometry before workers start."""
    spec = Bench2DexSpec.from_config(
        OmegaConf.to_container(env_cfg.init_params, resolve=True),
        fallback_control_steps=env_cfg.get("max_episode_steps"),
    )
    if cfg.rollout.pipeline_stage_num != 1:
        raise ValueError("Bench2Dex requires pipeline_stage_num=1")
    if (
        env_world_size < 1
        or env_cfg.total_num_envs < 1
        or env_cfg.total_num_envs % env_world_size
    ):
        raise ValueError(
            "Bench2Dex total_num_envs must be positive and divisible by "
            f"env worker count {env_world_size}, got {env_cfg.total_num_envs}"
        )
    if cfg.env.get("enable_offload", False):
        raise ValueError(
            "Set Bench2Dex offload per split with env.train.enable_offload "
            "or env.eval.enable_offload, not env.enable_offload"
        )
    if env_cfg.get("enable_offload", False):
        if env_cfg.auto_reset or (
            env_cfg.max_steps_per_rollout_epoch < spec.max_control_steps
        ):
            raise ValueError(
                "Bench2Dex offload requires auto_reset=false and a rollout "
                "budget covering the full episode"
            )
        if split == "train" and (
            cfg.runner.get("use_training_pipeline", False)
            or cfg.runner.get("enable_decoupled_mode", False)
            or cfg.runner.get("overlap_env_bootstrap", False)
            or cfg.algorithm.get("loss_type") == "decoupled_actor_critic"
        ):
            raise ValueError(
                "Bench2Dex environment offload requires synchronous, "
                "non-pipelined training without bootstrap prefetch"
            )
    if env_cfg.get("ignore_terminations", False):
        raise ValueError(
            "Bench2Dex uses ignore_terminations=false; select early_stop in init_params"
        )
    if env_cfg.get("skip_intermediate_renders", False):
        needs_per_action_observations = (
            OmegaConf.select(env_cfg, "video_cfg.save_video", default=False)
            or OmegaConf.select(env_cfg, "data_collection.enabled", default=False)
            or OmegaConf.select(env_cfg, "init_params.tactile.enabled", default=False)
            or (
                cfg.get("reward", {}).get("use_reward_model", False)
                and not cfg.get("reward", {}).get("standalone_realworld", False)
            )
        )
        if needs_per_action_observations:
            raise ValueError(
                "Bench2Dex video, data collection, tactile sensors and external "
                "reward models require skip_intermediate_renders=false"
            )
    if split == "train":
        group_size = int(env_cfg.get("group_size", 1))
        if group_size != int(cfg.algorithm.group_size) or group_size < 1:
            raise ValueError(
                "Expected positive matching Bench2Dex env/algorithm "
                f"group_size, got {group_size}/{cfg.algorithm.group_size}"
            )
        if group_size > 1:
            if not env_cfg.use_fixed_reset_state_ids or env_cfg.auto_reset:
                raise ValueError(
                    "Grouped Bench2Dex training requires "
                    "use_fixed_reset_state_ids=true and auto_reset=false"
                )
            for name, world_size in (
                ("env", env_world_size),
                ("actor", actor_world_size),
            ):
                if (
                    world_size is None
                    or world_size < 1
                    or env_cfg.total_num_envs % (world_size * group_size)
                ):
                    raise ValueError(
                        "Expected complete Bench2Dex groups per "
                        f"{name} worker, got total_num_envs="
                        f"{env_cfg.total_num_envs}, {name}_world_size="
                        f"{world_size}, group_size={group_size}"
                    )
        if (
            cfg.algorithm.adv_type == "grpo"
            and env_cfg.max_steps_per_rollout_epoch < spec.max_control_steps
        ):
            raise ValueError(
                "Bench2Dex GRPO rollout must cover the task control-step budget"
            )
    if int(model_cfg.action_dim) != spec.action_dim:
        raise ValueError(
            f"{spec.robot_key} needs {spec.action_dim} active joint coordinates, model declares {model_cfg.action_dim}"
        )
    if int(env_cfg.max_episode_steps) != spec.max_control_steps:
        raise ValueError(
            "env.max_episode_steps must match the Bench2Dex task control-step budget"
        )
    if env_cfg.max_steps_per_rollout_epoch < spec.max_control_steps and split == "eval":
        raise ValueError(
            "Bench2Dex evaluation rollout must cover the task control-step budget"
        )
    chunk_length = int(model_cfg.num_action_chunks)
    if chunk_length < 1:
        raise ValueError(
            f"Expected positive Bench2Dex action chunk length, got {chunk_length}"
        )
    if split == "train" and chunk_length > 1:
        if env_cfg.auto_reset:
            raise ValueError(
                "Bench2Dex chunk training requires auto_reset=false so "
                "the actor masks transitions after episode termination"
            )
        if (
            cfg.algorithm.reward_type != "chunk_level"
            or cfg.algorithm.logprob_type != "chunk_level"
        ):
            raise ValueError(
                "Bench2Dex chunk training requires chunk_level reward_type "
                "and logprob_type"
            )
    if model_cfg.model_type == "openpi":
        horizon = model_cfg.openpi.get("action_horizon")
        if horizon is not None and chunk_length > int(horizon):
            raise ValueError(
                f"Expected Bench2Dex execution chunk <= action_horizon "
                f"{horizon}, got {chunk_length}"
            )
        if model_cfg.openpi.config_name != "pi05_bench2dex":
            raise ValueError("Bench2Dex OpenPI requires config_name=pi05_bench2dex")
        if int(model_cfg.openpi.model_action_dim) < spec.action_dim:
            raise ValueError(
                "openpi.model_action_dim must cover the robot's active DOFs"
            )
        expected_keys = (
            "base_0_rgb",
            "base_1_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        )
        if tuple(model_cfg.openpi.get("image_keys", ())) != expected_keys:
            raise ValueError(f"Bench2Dex pi05 requires image_keys={expected_keys}")
        if len(spec.extra_cameras) != 1 or len(spec.wrist_cameras) != 2:
            raise ValueError("Bench2Dex pi05 uses two stereo and two wrist images")
