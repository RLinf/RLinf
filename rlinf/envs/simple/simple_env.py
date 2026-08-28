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

"""Single-process SIMPLE Teleop environment for Psi0."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit

from rlinf.envs.simple.controllers import (
    SimpleController,
    SimpleTeleopController,
    extract_psi0_state,
)

_SUPPORTED_TASK_IDS = frozenset(
    {
        "simple/G1WholebodyCloseDoorTeleop-v0",
        "simple/G1WholebodyOpenFaucetTeleop-v0",
    }
)


def _resolve_hssd_scene_dir(state: dict[str, Any]) -> Path | None:
    """Resolve the HSSD scene directory used by relative USD references."""
    scene = state.get("dr_state_dict", {}).get("scene", {})
    if not str(scene.get("uid", "")).startswith("hssd:"):
        return None

    data_dir = scene.get("data_dir")
    if not isinstance(data_dir, str):
        raise ValueError("SIMPLE HSSD reset state is missing scene.data_dir.")

    from simple.utils import resolve_data_path

    scene_dir = Path(resolve_data_path(data_dir, auto_download=True))
    props_dir = scene_dir / "props"
    if not props_dir.is_dir():
        raise FileNotFoundError(
            f"SIMPLE HSSD material assets are missing under {props_dir}."
        )
    return scene_dir


def _anchor_hssd_material_references(scene_dir: Path) -> Any:
    """Anchor HSSD material references to their containing scene layer."""
    from pxr import Sdf

    scene_path = scene_dir / f"{scene_dir.name}.usd"
    layer = Sdf.Layer.FindOrOpen(str(scene_path))
    if layer is None:
        raise FileNotFoundError(f"Cannot open SIMPLE HSSD scene layer {scene_path}.")
    props_dir = scene_dir / "props"
    material_files = {
        path.relative_to(props_dir).as_posix().casefold(): path.relative_to(
            props_dir
        ).as_posix()
        for path in props_dir.rglob("*")
        if path.is_file()
    }
    material_refs = [
        asset_path
        for asset_path in layer.GetCompositionAssetDependencies()
        if "/props/" in asset_path
    ]
    if not material_refs:
        raise ValueError(f"SIMPLE HSSD scene has no material references: {scene_path}.")
    for asset_path in material_refs:
        requested_path = asset_path.rsplit("/props/", 1)[1]
        actual_path = material_files.get(requested_path.casefold())
        if actual_path is None:
            raise FileNotFoundError(
                f"SIMPLE HSSD material {requested_path} is missing under {props_dir}."
            )
        anchored_path = f"./props/{actual_path}"
        if not layer.UpdateCompositionAssetDependency(asset_path, anchored_path):
            raise RuntimeError(
                f"Cannot update SIMPLE HSSD material reference {asset_path}."
            )
    return layer


class SimpleLeRobotResetDataset:
    """Load fixed SIMPLE reset states from LeRobot episode metadata."""

    def __init__(
        self,
        path: str | Path,
        *,
        dr_level: int,
        episode_start: int,
        num_episodes: int,
    ):
        dataset_path = Path(path).expanduser()
        if dataset_path.suffix == ".zip":
            raise ValueError("Extract the SIMPLE reset-state zip before evaluation.")
        if episode_start < 0 or num_episodes <= 0:
            raise ValueError("episode_start must be >= 0 and num_episodes must be > 0.")

        level_name = f"level-{dr_level}"
        metadata_files = [
            path
            for path in dataset_path.rglob("meta/episodes.jsonl")
            if path.parent.parent.name == level_name
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f"Expected one {level_name}/meta/episodes.jsonl under "
                f"{dataset_path}, found {len(metadata_files)}."
            )
        with metadata_files[0].open() as file:
            episodes = [json.loads(line) for line in file if line.strip()]
        episodes.sort(key=lambda episode: int(episode["episode_index"]))
        self._episodes = episodes[episode_start : episode_start + num_episodes]
        if len(self._episodes) != num_episodes:
            raise ValueError(
                f"Requested {num_episodes} reset episodes from index {episode_start}, "
                f"but found {len(self._episodes)} in {metadata_files[0]}."
            )

    def __len__(self) -> int:
        return len(self._episodes)

    def load(self, index: int) -> tuple[dict[str, Any], Any]:
        episode = self._episodes[index]
        encoded_config = episode.get("environment_config")
        if not isinstance(encoded_config, str):
            raise ValueError(
                "SIMPLE LeRobot episode metadata is missing environment_config."
            )
        return json.loads(encoded_config), episode


class SimpleEnv(gym.Env):
    """Expose one fixed SIMPLE Teleop runtime through RLinf's chunk API."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        del worker_info
        if num_envs != 1:
            raise ValueError("SIMPLE requires one env per EnvWorker.")
        if total_num_processes < 1:
            raise ValueError("total_num_processes must be positive.")
        task_id = str(cfg.init_params.task_id)
        if task_id not in _SUPPORTED_TASK_IDS:
            raise ValueError(f"Unsupported Psi0 SIMPLE Teleop task: {task_id}.")
        if cfg.init_params.controller_mode != "decoupled_wbc":
            raise ValueError("Psi0 SIMPLE Teleop requires decoupled_wbc.")
        if cfg.auto_reset:
            raise ValueError(
                "Psi0 SIMPLE requires auto_reset=false to preserve one episode "
                "per rollout epoch."
            )

        self.cfg = cfg
        self.num_envs = 1
        self.seed = int(cfg.seed) + int(seed_offset)
        self.auto_reset = bool(cfg.auto_reset)
        self.ignore_terminations = bool(cfg.ignore_terminations)
        self.is_start = True
        self._elapsed_steps = 0
        self._return = 0.0
        self._last_observation: dict[str, Any] | None = None
        self._terminal_episode: dict[str, torch.Tensor] | None = None
        if cfg.reset_dataset.get("format", "lerobot") != "lerobot":
            raise ValueError("The fixed SIMPLE Eval reset dataset uses LeRobot format.")
        self._dataset = SimpleLeRobotResetDataset(
            cfg.reset_dataset.path,
            dr_level=int(cfg.reset_dataset.dr_level),
            episode_start=int(cfg.reset_dataset.episode_start),
            num_episodes=int(cfg.reset_dataset.num_episodes),
        )
        initial_dataset_index = int(seed_offset) % len(self._dataset)
        self._dataset_index = initial_dataset_index - 1
        initial_state, _ = self._dataset.load(initial_dataset_index)
        self._hssd_scene_dir = _resolve_hssd_scene_dir(initial_state)
        self._hssd_scene_layer = None
        self._raw_env, self._native_env, self._controller = self._create_runtime()
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(36,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({})

    @staticmethod
    def _make_sonic_config() -> dict[str, Any]:
        import tyro
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

        config = tyro.cli(
            SimLoopConfig,
            config=(tyro.conf.ConsolidateSubcommandArgs,),
            args=[],
        )
        sonic_config = config.load_wbc_yaml()
        sonic_config["ENV_NAME"] = "simple"
        return sonic_config

    def _create_runtime(self) -> tuple[Any, Any, SimpleController]:
        """Initialize native libraries before Isaac and create one runtime."""
        import simple.envs  # noqa: F401

        sonic_config = self._make_sonic_config()
        sim_mode = str(self.cfg.init_params.sim_mode)
        raw_env = gym.make(
            str(self.cfg.init_params.task_id),
            sim_mode=sim_mode,
            headless=bool(self.cfg.init_params.headless),
            sonic_config=sonic_config,
            render_hz=int(self.cfg.init_params.render_hz),
            dr_level=int(self.cfg.reset_dataset.dr_level),
        )
        native_env = raw_env.unwrapped
        if sim_mode == "mujoco_isaac" and not native_env.task.metadata.get(
            "debug", False
        ):
            # SIMPLE otherwise renders and discards the MuJoCo camera frames.
            native_env._render_frame = native_env.isaac.render
        task_horizon = int(native_env.task.metadata["max_episode_steps"])
        configured_horizon = int(self.cfg.max_episode_steps)
        if configured_horizon != task_horizon:
            raise ValueError(
                "SIMPLE Teleop horizon differs from the fixed task metadata: "
                f"config={configured_horizon}, task={task_horizon}."
            )
        raw_env = TimeLimit(raw_env, max_episode_steps=configured_horizon)
        controller = SimpleTeleopController(native_env.task.robot, sonic_config)
        return raw_env, native_env, controller

    def _wrap_observation(
        self,
        observation: dict[str, Any],
        *,
        reset: bool,
    ) -> dict[str, Any]:
        image = np.asarray(observation["head_stereo_left"])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                "SIMPLE head_stereo_left must be an RGB/HWC uint8 image, "
                f"got {image.shape}/{image.dtype}."
            )
        state = extract_psi0_state(
            observation["joint_qpos"], self._controller.last_base_height
        )
        instruction = str(self._native_env.task.instruction)
        return {
            "main_images": torch.from_numpy(np.ascontiguousarray(image))[None],
            "wrist_images": None,
            "extra_view_images": None,
            "states": torch.from_numpy(state)[None, None],
            "task_descriptions": [instruction],
            "reset_mask": torch.tensor([reset], dtype=torch.bool),
        }

    def _next_dataset_index(self) -> int:
        self._dataset_index = (self._dataset_index + 1) % len(self._dataset)
        return self._dataset_index

    def update_reset_state_ids(self) -> None:
        """Keep reset-state ownership local to the SIMPLE dataset cursor."""

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.seed = int(seed)
        state, _ = self._dataset.load(self._next_dataset_index())
        scene_dir = _resolve_hssd_scene_dir(state)
        if scene_dir != self._hssd_scene_dir:
            raise ValueError(
                "A SIMPLE Eval worker cannot switch HSSD scenes after USD startup: "
                f"initial={self._hssd_scene_dir}, requested={scene_dir}."
            )
        if scene_dir is not None and self._hssd_scene_layer is None:
            self._hssd_scene_layer = _anchor_hssd_material_references(scene_dir)
        observation, info = self._raw_env.reset(options={"state_dict": state})
        self._controller.reset()

        control_dt = self._controller.control_dt
        for _ in range(int(self.cfg.init_params.stabilization_steps)):
            if self._controller.is_stabilized():
                break
            step_start = time.monotonic()
            low_level_action = self._controller.stabilize(observation)
            observation, *_, info = self._native_env.step(low_level_action)
            self._native_env.update_viewer()
            self._native_env.update_reward()
            remaining = control_dt - (time.monotonic() - step_start)
            if remaining > 0:
                time.sleep(remaining)
        self._controller.finish_stabilization()

        self._elapsed_steps = 0
        self._return = 0.0
        self._terminal_episode = None
        self.is_start = False
        self._last_observation = observation
        return self._wrap_observation(observation, reset=True), info

    def chunk_step(self, chunk_actions):
        """Synchronously execute a valid prefix of one 24-step Psi0 response."""
        actions = np.asarray(chunk_actions, dtype=np.float32)
        if actions.shape != (1, 24, 36):
            raise ValueError(
                f"SIMPLE chunk actions must have shape (1, 24, 36), got {actions.shape}."
            )
        if self._last_observation is None:
            raise RuntimeError("SIMPLE environment must be reset before chunk_step().")

        if self._terminal_episode is not None:
            terminal_observation = self._wrap_observation(
                self._last_observation, reset=False
            )
            infos_list = [{} for _ in range(24)]
            infos_list[-1] = {
                "episode": self._terminal_episode,
                "executed_mask": torch.zeros((1, 24), dtype=torch.bool),
            }
            return (
                [terminal_observation] * 24,
                torch.zeros((1, 24), dtype=torch.float32),
                torch.zeros((1, 24), dtype=torch.bool),
                torch.zeros((1, 24), dtype=torch.bool),
                infos_list,
            )

        obs_list = []
        infos_list = []
        rewards = torch.zeros((1, 24), dtype=torch.float32)
        raw_terminations = torch.zeros((1, 24), dtype=torch.bool)
        raw_truncations = torch.zeros((1, 24), dtype=torch.bool)
        executed_mask = torch.zeros((1, 24), dtype=torch.bool)
        terminal_observation = None
        terminal_info = None

        observation = self._last_observation
        for index, high_level_action in enumerate(actions[0]):
            low_level_action = self._controller.action(observation, high_level_action)
            observation, reward, terminated, truncated, info = self._raw_env.step(
                low_level_action
            )
            self._elapsed_steps += 1
            self._return += float(reward)
            executed_mask[0, index] = True
            rewards[0, index] = float(reward)
            raw_terminations[0, index] = bool(terminated)
            raw_truncations[0, index] = bool(truncated)
            obs_list.append(self._wrap_observation(observation, reset=False))
            infos_list.append(dict(info))
            if terminated or truncated:
                terminal_observation = obs_list[-1]
                terminal_info = dict(info)
                break

        done = terminal_observation is not None
        if done:
            success = bool(self._native_env._success)
            episode = {
                "success": torch.tensor([success]),
                "return": torch.tensor([self._return], dtype=torch.float32),
                "episode_len": torch.tensor([self._elapsed_steps]),
            }
            self._terminal_episode = episode
            terminal_info["episode"] = episode
            terminal_info["executed_mask"] = executed_mask
            while len(obs_list) < 24:
                obs_list.append(terminal_observation)
                infos_list.append({})
            infos_list[-1] = terminal_info
        else:
            infos_list[-1]["executed_mask"] = executed_mask

        self._last_observation = observation
        terminations = torch.zeros_like(raw_terminations)
        truncations = torch.zeros_like(raw_truncations)
        terminations[:, -1] = raw_terminations.any(dim=1)
        truncations[:, -1] = raw_truncations.any(dim=1)
        return obs_list, rewards, terminations, truncations, infos_list

    def capture_image(self) -> np.ndarray | None:
        if self._last_observation is None:
            return None
        return np.asarray(self._last_observation["head_stereo_left"])[None]

    def close(self):
        self._raw_env.close()
        super().close()
