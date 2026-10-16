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

"""Bench2Dex scene lifecycle, owned exclusively by the simulator child.

The lifecycle follows Bench2Dex run_policy.py at fd90dcc. Its MIT notice is
in third_party/Bench2Dex.LICENSE. Model loading and CLI code stay upstream.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from rlinf.data.schema.embodied_types import JointPositionTargets
from rlinf.utils.logging import get_logger

from .evaluation import (
    RESAMPLE_GROUPS,
    apply_asset_split,
    configure_profile,
    normalize_profile,
)
from .spec import (
    Bench2DexEpisode,
    Bench2DexSpec,
    joint_targets_in_runtime_order,
    load_episodes,
)


def make_bench2dex_runtime(
    config: dict, seed: int, shard: int, shards: int
) -> Bench2DexRuntime:
    """Launch Isaac in the simulator child and return its scene/app owner."""
    spec = Bench2DexSpec.from_config(config)
    # Resolve caller-relative paths before entering the upstream working directory.
    episodes = load_episodes(config)
    os.environ.pop("DISPLAY", None)
    os.environ["DEX2BENCH_DATASET_ROOT"] = str(spec.assets_path / "dex2bench_dataset")
    # Upstream uses top-level packages and relative resources inside this child.
    sys.path.insert(0, str(spec.source_path))
    os.chdir(spec.source_path)
    from argparse import Namespace

    from isaaclab.app import AppLauncher
    from utils.isaac_rendering import configure_headless_camera_parity_experience

    if config.get("output_dir"):
        kit_root = (
            Path(config["output_dir"]).expanduser().resolve() / "kit" / str(shard)
        )
        kit_root.mkdir(parents=True, exist_ok=True)
    else:
        kit_root = Path(tempfile.mkdtemp(prefix="bench2dex-kit-"))
    sdk = importlib.util.find_spec("isaacsim")
    sdk_root = Path(next(iter(sdk.submodule_search_locations))) / "kit"
    for user_config in sdk_root.glob("data/Kit/*/*/user.config.json"):
        target = kit_root / user_config.relative_to(sdk_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(user_config, target)

    launcher_args = Namespace(
        headless=True,
        enable_cameras=True,
        device=config.get("device", "cuda:0"),
        kit_args=f"--portable-root {kit_root}",
    )
    configure_headless_camera_parity_experience(launcher_args, log_prefix="bench2dex")
    app = AppLauncher(launcher_args).app
    try:
        return Bench2DexRuntime(config, spec, app, episodes, seed, shard, shards)
    except BaseException:
        get_logger().exception("Failed to initialize Bench2Dex simulator")
        app.close()
        raise


class Bench2DexRuntime:
    """Run one scene with upstream robot controllers and task metrics."""

    def __init__(
        self,
        config: dict,
        spec: Bench2DexSpec,
        app,
        episodes: list[Bench2DexEpisode],
        base_seed: int,
        shard: int,
        shards: int,
    ):
        import isaaclab.sim as sim_utils
        from build.generalization import resolve_sample_paths
        from robots import ROBOT_SPAWNERS
        from utils.seed_policy import (
            resolve_task_seed_id,
            seed_everything,
            task_base_seed,
        )

        self.config, self.spec, self.app = config, spec, app
        self.episodes = episodes
        self.profile = normalize_profile(config.get("generalization_profile", "none"))
        inv_cov_seed = config.get("inv_cov_seed")
        self.namespace_base_seed = (
            (200_000_000 if inv_cov_seed is None else int(inv_cov_seed))
            if self.profile == "inv_cov"
            else base_seed
        )
        self.episode_index = int(config.get("start_episode", 1)) - 1 + shard
        self.episode_stride = shards
        if self.episode_index < 0:
            raise ValueError("Bench2Dex start_episode must be one-indexed")
        self.device = config.get("device", "cuda:0")
        self.early_stop = bool(config.get("early_stop", True))
        self.runtime = self.camera_rig = self.contact_reader = self.tactile_rig = None
        self._last_transition = None
        self._closed = False
        with spec.task_path.open() as stream:
            self.task = resolve_sample_paths(
                yaml.safe_load(stream), str(spec.assets_path)
            )
        self._load_generalization_config()
        # Upstream spawners expose a relative USD constant; bind it only in this child.
        module_name, _ = ROBOT_SPAWNERS[spec.robot_key]
        spawner = importlib.import_module(f"robots.{module_name}")
        spawner.ROBOT_USD_PATH = Path(
            resolve_sample_paths(str(spawner.ROBOT_USD_PATH), str(spec.assets_path))
        )
        seed_everything(
            task_base_seed(
                self.namespace_base_seed,
                resolve_task_seed_id(str(spec.task_path), config.get("task_seed_id")),
            )
        )
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=spec.physics_dt,
                device=self.device,
                physx=sim_utils.PhysxCfg(
                    enable_ccd=True,
                    enable_stabilization=True,
                    bounce_threshold_velocity=0.01,
                    gpu_max_rigid_contact_count=2**23,
                    gpu_max_rigid_patch_count=2**22,
                ),
            )
        )
        self.sim.set_camera_view([0.0, 0.0, 2.20], [0.0, 0.0, 0.82])

    def _load_generalization_config(self) -> None:
        from build import (
            merge_scene_generalization_overrides,
            parse_scene_generalization_config,
        )

        path = Path(
            self.config.get(
                "generalization_config", "configs/scene/generalization.yaml"
            )
        )
        path = path if path.is_absolute() else self.spec.source_path / path
        with path.open() as stream:
            raw = yaml.safe_load(stream) or {}
        overrides = self.task.get("generalization_overrides")
        if overrides:
            raw = merge_scene_generalization_overrides(raw, overrides)
        raw = configure_profile(raw, self.profile)
        if overrides:
            raw = merge_scene_generalization_overrides(raw, overrides)
        raw = apply_asset_split(raw, self.config.get("generalization_split", "unseen"))
        from build.generalization import resolve_sample_paths

        raw = resolve_sample_paths(raw, str(self.spec.assets_path))
        self.generalization_cfg = parse_scene_generalization_config(
            raw, config_path=str(path)
        )

    def _sample_scene(self, episode: Bench2DexEpisode):
        import h5py
        from build import (
            collect_task_asset_codes,
            dict_to_generalization_sample,
            merge_generalization_samples,
            sample_scene_generalization,
            validate_resolved_object_placement_keys,
        )
        from build.generalization import resolve_sample_paths
        from robots import get_robot_keys

        anchor = None
        if episode.anchor_hdf5 is not None:
            with h5py.File(episode.anchor_hdf5, "r") as stream:
                value = stream["meta/scene_generalization_sample"][()]
                raw = json.loads(
                    value.decode() if isinstance(value, bytes) else str(value)
                )
            raw = resolve_sample_paths(raw, str(self.spec.assets_path))
            validate_resolved_object_placement_keys(
                raw, (obj["id"] for obj in self.task["objects"]), required=True
            )
            anchor = dict_to_generalization_sample(raw)
        if self.profile == "none":
            sample = copy.deepcopy(anchor)
        else:
            sample = sample_scene_generalization(
                self.generalization_cfg,
                enabled=True,
                task_asset_codes=collect_task_asset_codes(self.task),
                asset_split="unseen",
                available_robot_keys=get_robot_keys(),
            )
            if self.profile in RESAMPLE_GROUPS:
                sample = merge_generalization_samples(
                    copy.deepcopy(anchor), sample, RESAMPLE_GROUPS[self.profile]
                )
        sample.robot_key = self.spec.robot_key
        return sample

    def _prepare_episode_scene(self, seed: int | None):
        from build import scene_generalization_sample_to_dict
        from utils.seed_policy import (
            episode_seed_context,
            resolve_task_seed_id,
            seed_everything,
        )

        anchor_index = self.episode_index % len(self.episodes)
        episode = self.episodes[anchor_index]
        self.current_episode_index = self.episode_index + 1
        self.episode_index += self.episode_stride
        context = episode_seed_context(
            seed_namespace="eval",
            base_seed=self.namespace_base_seed,
            task_seed_id=resolve_task_seed_id(
                str(self.spec.task_path), self.config.get("task_seed_id")
            ),
            episode_index=self.current_episode_index,
        )
        self.base_seed = context.task_base_seed
        self.episode_seed = int(
            seed
            if seed is not None
            else episode.seed
            if episode.seed is not None
            else context.episode_seed
        )
        self.seed_policy = (
            context.seed_policy
            if episode.seed is None
            else "explicit episode_manifest seed"
        )
        seed_everything(self.episode_seed)
        scene_sample = self._sample_scene(episode)
        sample = scene_generalization_sample_to_dict(scene_sample)
        self.scene_metadata = {
            "_generalization_profile": self.profile,
            "_generalization_split": self.generalization_cfg.asset_split,
        }
        if episode.anchor_hdf5 is not None:
            self.scene_metadata["_anchor_source_index"] = anchor_index
        sample.update(self.scene_metadata)
        sample.get("appearance", {}).pop("light", None)
        self.anchor = (
            str(episode.anchor_hdf5) if episode.anchor_hdf5 is not None else None
        )
        return scene_sample, sample

    def reset(self, seed: int | None = None, env_ids=None) -> tuple[dict, dict]:
        """Rebuild the scene from the next anchor, excluding warmup from metrics."""
        from benchmark.metric_tracker import MetricTracker
        from build import (
            build_scene,
        )
        from collector import load_collect_config
        from collector.cameras import CameraRig
        from collector.state_reader import read_joint_limits, read_joint_state
        from robots.active_dof_utils import get_active_dof_info_for_runtime
        from utils.episode_runtime import (
            initialize_scene_runtime_state,
            move_controlled_articulations_home,
        )
        from utils.runtime_helpers import classify_interactive_objects

        if env_ids is not None and list(np.asarray(env_ids.cpu()).reshape(-1)) != [0]:
            raise ValueError("A Bench2Dex process contains only environment 0")
        self._clear_scene()
        scene_sample, sample = self._prepare_episode_scene(seed)
        self.runtime = build_scene(
            self.task,
            str(self.spec.task_path.parent),
            generalization_enabled=True,
            generalization_cfg=self.generalization_cfg,
            generalization_sample=scene_sample,
            robot_key=self.spec.robot_key,
            existing_robot_runtime=None,
        )
        self.scene_sample = dict(self.runtime["scene_generalization_sample"])
        self.scene_sample.update(self.scene_metadata)
        self.objects = self.runtime["interactive_objects"]
        self.object_ids = sorted(self.runtime.get("asset_local_bbox", {})) or sorted(
            name for name in self.objects if name != "global_robot"
        )
        self.hooks = list(
            (self.runtime.get("robot_runtime") or {}).get("pre_step_hooks", [])
        )
        collect = load_collect_config(
            str(self.spec.collect_config), robot_key=self.spec.robot_key
        )
        cameras = {camera.camera_id: camera for camera in collect.cameras}
        render_names = tuple(
            self.config["cameras"].get("render", self.spec.camera_names)
        )
        missing = set(self.spec.camera_names) - set(render_names)
        unknown = set(render_names) - set(cameras)
        if missing or unknown or len(set(render_names)) != len(render_names):
            raise ValueError(
                "Expected unique render cameras covering all policy views; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}, got={render_names}"
            )
        self.camera_rig = CameraRig(
            self.sim,
            [cameras[name] for name in render_names],
            enable_rgb=True,
            enable_depth=False,
            robot_articulation=classify_interactive_objects(
                self.objects
            ).robot_articulation,
            camera_generalization_sample=sample,
        )
        tactile_config = self.config.get("tactile", {})
        if tactile_config.get("enabled", False):
            from collector.tacmap_rig import TacMapRig, build_tacmap_raycast_targets

            paths, body_types = build_tacmap_raycast_targets(
                self.runtime.get("object_prim_paths", {}),
                self.runtime.get("object_body_types", {}),
            )
            self.tactile_rig = TacMapRig(
                robot_key=self.spec.robot_key,
                object_prim_paths=paths,
                object_body_types=body_types,
                resolution_step=int(tactile_config.get("resolution_step", 1)),
                max_distance=float(tactile_config.get("max_distance_m", 0.015)),
                sensor_map_dir=tactile_config.get("sensor_map_dir"),
            )
        groups, self.controlled, _, _, self.targets = initialize_scene_runtime_state(
            sim=self.sim,
            physics_dt=self.spec.physics_dt,
            interactive_objects=self.objects,
            object_prim_paths=self.runtime.get("object_prim_paths", {}),
            object_display_colors=self.runtime.get("object_display_colors", {}),
            collector=None,
            app_running_state_fn=self.app.is_running,
        )
        self.robot = groups.robot_articulation
        if self.robot is None:
            raise RuntimeError("Bench2Dex scene did not create global_robot")
        self.dof = get_active_dof_info_for_runtime(
            self.spec.robot_key, list(self.robot.data.joint_names)
        )
        with (self.spec.source_path / "robots/active_dof_maps.yml").open() as stream:
            mimic_count = yaml.safe_load(stream)["robots"][self.spec.robot_key].get(
                "mimic_dof", 0
            )
        if len(self.dof.mimic_rules) != mimic_count:
            raise ValueError(
                f"Missing mimic rules for {self.spec.robot_key}: expected {mimic_count}, got {len(self.dof.mimic_rules)}"
            )
        self.camera_rig.set_robot_articulation(self.robot)
        self.camera_rig.initialize_after_reset()
        if self.tactile_rig is not None:
            self.tactile_rig.initialize_after_reset(self.robot)
        self.joint_limits = read_joint_limits(self.robot)
        metrics = copy.deepcopy(self.task["metrics"])
        metrics.setdefault("safety", {})["table_z"] = self.runtime["table_z"]
        if isinstance(metrics.get("grasp"), dict):
            metrics["grasp"]["table_z"] = self.runtime["table_z"]
        height = sample.get("spatial", {}).get("table_height", {})
        self.metrics = MetricTracker(
            metrics,
            dt=self.spec.physics_dt,
            table_height_offset=float(height.get("height_offset_m", 0))
            if height.get("enabled")
            else 0.0,
            robot_key=self.spec.robot_key,
        )
        if not self.metrics.available:
            raise ValueError(f"Task {self.spec.task_path.stem} has no terminal metric")
        self.metrics.record_joint_limit_baseline(
            "post_reset",
            robot_state=read_joint_state(self.robot),
            joint_limits=self.joint_limits,
        )
        pairs = self.runtime.get("contact_pairs", [])
        if pairs:
            from collector.contact_sensor_reader import ContactSensorReader

            self.contact_reader = ContactSensorReader(
                pairs, object_prim_paths=self.runtime.get("object_prim_paths", {})
            )
        for _ in range(5):
            self.sim.render()
        move_controlled_articulations_home(self.controlled, self.targets, teleport=True)
        for _ in range(int(self.config.get("warmup_steps", 60))):
            self._advance_physics()
        self.metrics.record_joint_limit_baseline(
            "post_home",
            robot_state=read_joint_state(self.robot),
            joint_limits=self.joint_limits,
        )
        self.control_steps = self.physics_steps = 0
        self.success_once = self.terminated = self.truncated = False
        self.episode_return = 0.0
        self._last_transition = None
        # Native evaluation captures preflight before rendering policy frame zero.
        self._capture_camera_frames()
        observation = self._observe()
        return observation, self._info()

    def _advance_physics(self) -> None:
        from utils.runtime_helpers import update_sim_objects, write_articulation_targets

        if not self.app.is_running():
            raise RuntimeError("Isaac Sim closed during Bench2Dex execution")
        for hook in self.hooks:
            hook()
        write_articulation_targets(self.controlled, self.targets)
        self.sim.step(render=False)
        update_sim_objects(self.objects.values(), self.spec.physics_dt)

    def _capture_camera_frames(self) -> dict:
        """Render and validate the complete rig independently of policy views."""
        self.camera_rig.sync_mounted_camera_poses()
        self.sim.render()
        for _ in range(5):
            frames = self.camera_rig.capture(dt=self.spec.physics_dt)
            if all(
                name in frames
                and frames[name] is not None
                and frames[name].rgb is not None
                for name in self.camera_rig.camera_ids
            ):
                break
            self.sim.render()
        else:
            raise RuntimeError(
                f"Incomplete Bench2Dex cameras: {self.camera_rig.last_capture_errors}"
            )
        return frames

    def _observe(self) -> dict:
        from robots.active_dof_utils import select_active

        frames = self._capture_camera_frames()
        images = {
            name: np.asarray(frames[name].rgb, dtype=np.uint8).copy()
            for name in self.spec.camera_names
        }
        qpos = self.robot.data.joint_pos[0].detach().cpu().numpy()
        tactile = {
            "site_names": [],
            "tacmap": np.zeros((0, 0, 0), dtype=np.uint8),
            "distance_along_normal_m": np.zeros((0, 0, 0), dtype=np.float32),
            "contact_mask": np.zeros((0, 0, 0), dtype=bool),
            "valid": np.zeros(0, dtype=bool),
            "depth_valid": np.zeros(0, dtype=bool),
            "contact_valid": np.zeros(0, dtype=bool),
            "max_distance_m": np.float32(0),
            "resolution_step": np.int64(0),
        }
        if self.tactile_rig is not None:
            capture = self.tactile_rig.capture(dt=self.spec.physics_dt)
            sites = tuple(self.tactile_rig.site_names)
            shape = (self.tactile_rig.image_size, self.tactile_rig.image_size)
            tactile["site_names"] = list(sites)
            for field, dtype in (
                ("tacmap", np.uint8),
                ("distance_along_normal_m", np.float32),
                ("contact_mask", bool),
            ):
                maps = capture[field]
                tactile[field] = np.stack(
                    [
                        np.asarray(maps.get(site, np.zeros(shape)), dtype=dtype)
                        for site in sites
                    ]
                ).copy()
            tactile["valid"] = np.array([site in capture["tacmap"] for site in sites])
            tactile["depth_valid"] = np.array(
                [site in capture["distance_along_normal_m"] for site in sites]
            )
            tactile["contact_valid"] = np.array(
                [site in capture["contact_mask"] for site in sites]
            )
            tactile["max_distance_m"] = np.float32(self.tactile_rig.max_distance)
            tactile["resolution_step"] = np.int64(self.tactile_rig.resolution_step)
        return {
            "images": images,
            "state": select_active(qpos, self.dof).astype(np.float32),
            "full_joint_positions": qpos.astype(np.float32, copy=True),
            "active_joint_names": list(self.dof.active_joint_names),
            "full_joint_names": list(self.dof.full_joint_names),
            "robot_key": self.spec.robot_key,
            "tactile": tactile,
            "instruction": self.task["description"],
        }

    def _info(self) -> dict:
        return {
            "task_id": self.spec.task_path.stem,
            "scene": str(self.spec.task_path.relative_to(self.spec.source_path))
            if self.spec.task_path.is_relative_to(self.spec.source_path)
            else str(self.spec.task_path),
            "episode_index": self.current_episode_index,
            "base_seed": self.base_seed,
            "seed_policy": self.seed_policy,
            "generalization_profile": self.profile,
            "scene_generalization_sample": self.scene_sample,
            "early_stop": self.early_stop,
            "control_decimation": self.spec.control_decimation,
            "task_family": self.metrics.task_family,
            "dwell_time_s": self.metrics.dwell_time_s,
            "robot_key": self.spec.robot_key,
            "anchor_hdf5": self.anchor,
            "render_camera_names": tuple(self.camera_rig.camera_ids),
            "seed": self.episode_seed,
            "active_joint_names": tuple(self.dof.active_joint_names),
            "control_steps": self.control_steps,
            "physics_steps": self.physics_steps,
            "max_control_steps": self.spec.max_control_steps,
            "success_once": self.success_once,
            "episode_return": self.episode_return,
        }

    def step(self, action) -> tuple:
        """Execute one active/full joint target through the existing step RPC."""
        import torch
        from collector.state_reader import read_joint_state, read_object_states
        from robots.active_dof_utils import expand_to_full, select_active

        if not isinstance(action, JointPositionTargets):
            action = np.asarray(action).reshape(1, -1)
        actions, layout = joint_targets_in_runtime_order(
            action,
            active_joint_names=tuple(self.dof.active_joint_names),
            full_joint_names=tuple(self.dof.full_joint_names),
        )
        if self.terminated or self.truncated:
            obs, _, terminated, truncated, info = self._last_transition
            return obs, 0.0, terminated, truncated, dict(info, action_steps=0)
        action = actions[0]
        target = expand_to_full(action, self.dof) if layout == "active" else action
        self.targets[id(self.robot)][0, :] = torch.as_tensor(
            target, device=self.robot.device
        )
        command = {
            "raw_active_action": select_active(target, self.dof),
            "expanded_full_target": target,
            "shielded_full_target": target,
            "shield_applied": False,
            "active_joint_names": self.dof.active_joint_names,
            "full_joint_names": self.dof.full_joint_names,
            "mimic_source_by_joint": {
                self.dof.full_joint_names[i]: self.dof.full_joint_names[j]
                for i, j, _, _ in self.dof.mimic_rules
            },
        }
        for _ in range(self.spec.control_decimation):
            self._advance_physics()
            states = read_object_states(self.objects, self.object_ids)
            if self.contact_reader is not None and self.contact_reader.available:
                self.contact_reader.update(self.spec.physics_dt)
                for name, forces in self.contact_reader.read().items():
                    if name in states:
                        states[name]["contact_forces"] = forces
            self.metrics.update(
                states,
                sim_step=self.physics_steps,
                dt=self.spec.physics_dt,
                robot_state=read_joint_state(self.robot),
                joint_limits=self.joint_limits,
                joint_command=command,
            )
            self.physics_steps += 1
            if self.metrics.success and self.early_stop:
                break
        self.control_steps += 1
        success = bool(self.metrics.success)
        reward = float(success and not self.success_once)
        self.success_once |= success
        self.episode_return += reward
        self.terminated = self.early_stop and success
        self.truncated = (
            self.control_steps >= self.spec.max_control_steps and not self.terminated
        )
        info = dict(self._info(), action_steps=1)
        if self.terminated or self.truncated:
            info["metrics"] = dataclasses.asdict(
                self.metrics.finalize(
                    steps=self.physics_steps,
                    terminated_reason="stable_success"
                    if self.terminated
                    else "max_steps",
                    max_steps=self.spec.max_control_steps
                    * self.spec.control_decimation,
                    policy_stride=self.spec.control_decimation,
                    evaluation_protocol="reach_and_stop"
                    if self.early_stop
                    else "fixed_horizon",
                )
            )
        self._last_transition = (
            self._observe(),
            reward,
            self.terminated,
            self.truncated,
            info,
        )
        return self._last_transition

    def _clear_scene(self) -> None:
        from utils.episode_runtime import clear_scene_prims_preserve_robot
        from utils.usd_prims import delete_prim_compat, get_current_stage_compat

        if self.camera_rig is not None:
            self.camera_rig.close()
            self.camera_rig = None
        if self.tactile_rig is not None:
            self.tactile_rig.close()
            self.tactile_rig = None
        self.contact_reader = None
        if self.runtime is not None:
            self.runtime = self.objects = self.robot = self.controlled = (
                self.targets
            ) = None
            self.hooks = []
            clear_scene_prims_preserve_robot(sim=self.sim, preserve_robot=False)
            stage = get_current_stage_compat()
            sensors = stage.GetPrimAtPath("/World/Sensors")
            if sensors.IsValid():
                for child in list(sensors.GetChildren()):
                    delete_prim_compat(str(child.GetPath()), stage=stage)

    def close(self) -> None:
        """Release the scene and owned app once, including exceptional exits."""
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._clear_scene()
            finally:
                try:
                    self.sim.clear_all_callbacks()
                finally:
                    self.sim.clear_instance()
        finally:
            self.app.close()
