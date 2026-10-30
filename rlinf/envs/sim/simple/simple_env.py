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

"""SIMPLE Teleop runtime and Ray proxy for Psi0."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
from contextlib import nullcontext
from typing import Any
from uuid import uuid4

import gymnasium as gym
import numpy as np
import ray
import torch
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from rlinf.envs.sim.simple.controller import (
    SimpleController,
    SimpleTeleopController,
)
from rlinf.envs.sim.simple.dataset import SimpleLeRobotResetDataset
from rlinf.envs.sim.simple.runtime import (
    IsaacFrameGate,
    anchor_hssd_material_references,
    install_articulation_reuse_guard,
    link_kit_cache,
    resolve_hssd_scene_dir,
    use_pip_isaac_sim,
)
from rlinf.envs.sim.simple.utils import (
    SIMPLE_ACTION_CHUNK_SHAPE,
    SIMPLE_ACTION_CHUNK_SIZE,
    SIMPLE_ACTION_DIM,
    extract_psi0_state,
)

_SUPPORTED_TASK_IDS = frozenset(
    {
        "simple/G1WholebodyXMovePickTeleop-v0",
        "simple/G1WholebodyHandoverTeleop-v0",
        "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
        "simple/G1WholebodyXMoveBendPickTeleop-v0",
        "simple/G1WholebodyCloseDoorTeleop-v0",
        "simple/G1WholebodyOpenOvenTeleop-v0",
        "simple/G1WholebodyOpenFaucetTeleop-v0",
        "simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0",
    }
)


@ray.remote(
    num_cpus=1, num_gpus=0, max_concurrency=1, max_restarts=0, max_task_retries=0
)
class SimpleProcess(gym.Env):
    """Own one simulator and controller on a synchronous Ray actor's main thread."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self, cfg: DictConfig, num_envs: int, seed_offset: int, total_num_processes: int
    ) -> None:
        if num_envs != 1:
            raise ValueError("SIMPLE requires one env per EnvWorker.")
        use_pip_isaac_sim()
        if total_num_processes < 1:
            raise ValueError("total_num_processes must be positive.")
        task_id = str(cfg.init_params.task_id)
        if task_id not in _SUPPORTED_TASK_IDS:
            raise ValueError(f"Unsupported Psi0 SIMPLE task: {task_id}.")

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
            raise ValueError("The SIMPLE reset dataset uses LeRobot format.")
        self._dataset = SimpleLeRobotResetDataset(
            cfg.reset_dataset.path,
            dr_level=int(cfg.reset_dataset.dr_level),
            episode_start=int(cfg.reset_dataset.episode_start),
            num_episodes=int(cfg.reset_dataset.num_episodes),
        )
        initial_dataset_index = int(seed_offset) % len(self._dataset)
        self._dataset_index = initial_dataset_index - 1
        initial_state, _ = self._dataset.load(initial_dataset_index)
        self._hssd_scene_dir = resolve_hssd_scene_dir(initial_state)
        self._hssd_scene_layer = None
        self._kit_cache_lock = None
        self._frames: IsaacFrameGate | None = None
        self._raw_env, self._native_env, self._controller = self._create_runtime()
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(SIMPLE_ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({})

    def ready(self) -> None:
        """Confirm initialization without advancing the reset dataset cursor."""

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
        original_argv = sys.argv
        try:
            if "isaac" in sim_mode:
                # SimulationApp reads Kit CLI options during gym.make(). Give
                # each process its own writable config/logs and a warm cache
                # slot; keep SIMPLE source assets shared. Preserve logs after a
                # crash for diagnosis.
                runtime_dir = tempfile.mkdtemp(prefix="rlinf-simple-isaac-")
                self._kit_cache_lock = link_kit_cache(runtime_dir)
                sys.argv = [*original_argv, "--portable-root", runtime_dir]
            raw_env = gym.make(
                str(self.cfg.init_params.task_id),
                sim_mode=sim_mode,
                headless=bool(self.cfg.init_params.headless),
                sonic_config=sonic_config,
                render_hz=int(self.cfg.init_params.render_hz),
                dr_level=int(self.cfg.reset_dataset.dr_level),
            )
        finally:
            sys.argv = original_argv
        native_env = raw_env.unwrapped
        if sim_mode == "mujoco_isaac":
            install_articulation_reuse_guard(native_env)
            if not native_env.task.metadata.get("debug", False):
                # SIMPLE otherwise renders and discards the MuJoCo camera frames.
                native_env._render_frame = native_env.isaac.render
            if self.cfg.init_params.get("render_policy_frames_only", True):
                self._frames = IsaacFrameGate(native_env)
        task_horizon = int(native_env.task.metadata["max_episode_steps"])
        configured_horizon = int(self.cfg.max_episode_steps)
        if configured_horizon <= 0 or configured_horizon > task_horizon:
            raise ValueError(
                "SIMPLE Teleop horizon must not exceed the task metadata: "
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

    def _defer_rendering(self) -> Any:
        """Skip Isaac mirroring and rendering for observations RLinf drops."""
        return nullcontext() if self._frames is None else self._frames.deferred()

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
        scene_dir = resolve_hssd_scene_dir(state)
        if scene_dir != self._hssd_scene_dir:
            raise ValueError(
                "A SIMPLE worker cannot switch HSSD scenes after USD startup: "
                f"initial={self._hssd_scene_dir}, requested={scene_dir}."
            )
        if scene_dir is not None and self._hssd_scene_layer is None:
            self._hssd_scene_layer = anchor_hssd_material_references(scene_dir)
        observation, info = self._raw_env.reset(options={"state_dict": state})
        self._controller.reset()

        # The controller follows simulated time, so this loop need not keep
        # pace with wall time.
        with self._defer_rendering():
            for _ in range(int(self.cfg.init_params.stabilization_steps)):
                if self._controller.is_stabilized():
                    break
                low_level_action = self._controller.stabilize(observation)
                observation, *_, info = self._native_env.step(low_level_action)
                self._native_env.update_viewer()
                self._native_env.update_reward()
        if self._frames is not None:
            observation = self._frames.observe()
        self._controller.finish_stabilization()

        self._elapsed_steps = 0
        self._return = 0.0
        self._terminal_episode = None
        self.is_start = False
        self._last_observation = observation
        return self._wrap_observation(observation, reset=True), info

    def _handle_auto_reset(
        self,
        terminal_observation: dict[str, Any],
        terminal_info: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset after a terminal chunk step and keep Gymnasium final_* fields."""
        reset_obs, reset_info = self.reset()
        infos = dict(reset_info)
        infos["final_observation"] = copy.deepcopy(terminal_observation)
        infos["final_info"] = copy.deepcopy(terminal_info)
        infos["_final_observation"] = np.array([True])
        infos["_final_info"] = np.array([True])
        infos["_elapsed_steps"] = np.array([True])
        infos["executed_mask"] = terminal_info["executed_mask"]
        return reset_obs, infos

    def chunk_step(self, chunk_actions):
        """Synchronously execute a valid prefix of one Psi0 action chunk."""
        actions = np.asarray(chunk_actions, dtype=np.float32)
        expected_shape = (1, *SIMPLE_ACTION_CHUNK_SHAPE)
        if actions.shape != expected_shape:
            raise ValueError(
                "SIMPLE chunk actions must have shape "
                f"{expected_shape}, got {actions.shape}."
            )
        if self._last_observation is None:
            raise RuntimeError("SIMPLE environment must be reset before chunk_step().")

        chunk_mask_shape = (1, SIMPLE_ACTION_CHUNK_SIZE)
        # Finished-episode padding: only used when auto_reset=false, until the
        # worker calls reset() at the next epoch boundary.
        if not self.auto_reset and self._terminal_episode is not None:
            terminal_observation = self._wrap_observation(
                self._last_observation, reset=False
            )
            infos_list = [{} for _ in range(SIMPLE_ACTION_CHUNK_SIZE)]
            infos_list[-1] = {
                "episode": self._terminal_episode,
                "executed_mask": torch.zeros(chunk_mask_shape, dtype=torch.bool),
            }
            return (
                [terminal_observation] * SIMPLE_ACTION_CHUNK_SIZE,
                torch.zeros(chunk_mask_shape, dtype=torch.float32),
                torch.zeros(chunk_mask_shape, dtype=torch.bool),
                torch.zeros(chunk_mask_shape, dtype=torch.bool),
                infos_list,
            )

        obs_list = []
        infos_list = []
        rewards = torch.zeros(chunk_mask_shape, dtype=torch.float32)
        raw_terminations = torch.zeros(chunk_mask_shape, dtype=torch.bool)
        raw_truncations = torch.zeros(chunk_mask_shape, dtype=torch.bool)
        executed_mask = torch.zeros(chunk_mask_shape, dtype=torch.bool)
        terminal_observation = None
        terminal_info = None

        observation = self._last_observation
        last_index = SIMPLE_ACTION_CHUNK_SIZE - 1
        for index, high_level_action in enumerate(actions[0]):
            low_level_action = self._controller.action(observation, high_level_action)
            with self._defer_rendering():
                observation, reward, terminated, truncated, info = self._raw_env.step(
                    low_level_action
                )
            self._elapsed_steps += 1
            self._return += float(reward)
            executed_mask[0, index] = True
            rewards[0, index] = float(reward)
            raw_terminations[0, index] = bool(terminated)
            raw_truncations[0, index] = bool(truncated)
            if self._frames is None:
                obs_list.append(self._wrap_observation(observation, reset=False))
            elif terminated or truncated or index == last_index:
                observation = self._frames.observe()
                obs_list.append(self._wrap_observation(observation, reset=False))
            else:
                # RLinf reads only the final or terminal observation of a chunk;
                # RecordVideo.record_video_in_result filters out None entries.
                obs_list.append(None)
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
            terminal_info["episode"] = episode
            terminal_info["executed_mask"] = executed_mask
            while len(obs_list) < SIMPLE_ACTION_CHUNK_SIZE:
                obs_list.append(terminal_observation)
                infos_list.append({})
            infos_list[-1] = terminal_info
            if self.auto_reset:
                obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                    terminal_observation, terminal_info
                )
            else:
                self._terminal_episode = episode
                self._last_observation = observation
        else:
            infos_list[-1]["executed_mask"] = executed_mask
            self._last_observation = observation

        terminations = torch.zeros_like(raw_terminations)
        truncations = torch.zeros_like(raw_truncations)
        terminations[:, -1] = raw_terminations.any(dim=1)
        truncations[:, -1] = raw_truncations.any(dim=1)
        return obs_list, rewards, terminations, truncations, infos_list

    def close(self):
        self._raw_env.close()
        super().close()


class SimpleEnv(gym.Env):
    """Proxy one SIMPLE runtime while keeping native imports out of EnvWorker."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        del worker_info
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = int(cfg.seed) + int(seed_offset)
        self.auto_reset = bool(cfg.auto_reset)
        self.ignore_terminations = bool(cfg.ignore_terminations)
        self.is_start = True
        self._process = None
        self._seed_offset = int(seed_offset)
        self._total_num_processes = int(total_num_processes)
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(SIMPLE_ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({})
        self._start_process()

    def _start_process(self) -> None:
        """Start the simulator actor in the worker's Ray namespace."""
        # RLinf assigns GPU visibility independently of Ray GPU resources.
        env_vars = {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "MUJOCO_EGL_DEVICE_ID",
                "EGL_DEVICE_ID",
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
            )
            if key in os.environ
        }
        env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        self._process = SimpleProcess.options(
            name=f"SimpleProcess-{uuid4().hex}",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=False
            ),
            # Ray inherits the parent's Python/runtime environment and merges env_vars.
            runtime_env={"env_vars": env_vars},
        ).remote(self.cfg, self.num_envs, self._seed_offset, self._total_num_processes)
        try:
            ray.get(self._process.ready.remote())
        except BaseException:
            # The constructor may have only partially initialized native libraries.
            self._kill_process()
            raise

    def _kill_process(self) -> None:
        process, self._process = self._process, None
        if process is not None and ray.is_initialized():
            try:
                ray.kill(process, no_restart=True)
            except ray.exceptions.RayActorError:
                pass

    def _ensure_process(self) -> Any:
        if self._process is None:
            self._start_process()
        return self._process

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset once in the simulator process."""
        observation, info = ray.get(
            self._ensure_process().reset.remote(seed=seed, options=options)
        )
        if seed is not None:
            self.seed = int(seed)
        self.is_start = False
        return observation, info

    def chunk_step(
        self, chunk_actions: np.ndarray | torch.Tensor
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Execute a whole chunk in one RPC, including any terminal padding."""
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        result = ray.get(self._ensure_process().chunk_step.remote(chunk_actions))
        _, _, _, _, infos = result
        executed_mask = infos[-1].get("executed_mask")
        if (
            not isinstance(executed_mask, torch.Tensor)
            or executed_mask.dtype != torch.bool
            or executed_mask.shape != (1, SIMPLE_ACTION_CHUNK_SIZE)
        ):
            raise RuntimeError(
                "SIMPLE must return a bool executed_mask of shape "
                f"[1, {SIMPLE_ACTION_CHUNK_SIZE}]."
            )
        return result

    def update_reset_state_ids(self) -> None:
        """Leave reset cursor advancement to the process's next reset call."""

    def offload(self) -> None:
        """Release the simulator actor so the worker can free its GPU."""
        self.close()

    def close(self) -> None:
        """Close the simulator, then kill its actor.

        Isaac ignores a graceful close, so that mode is torn down by process kill.
        """
        if self._process is None:
            return
        try:
            if ray.is_initialized() and "isaac" not in str(
                self.cfg.init_params.sim_mode
            ):
                try:
                    ray.get(self._process.close.remote())
                except ray.exceptions.RayActorError:
                    pass
        finally:
            self._kill_process()
