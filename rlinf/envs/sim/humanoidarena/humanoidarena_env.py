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

"""Evaluation environment backed by the original SONIC batch executable."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.sim.humanoidarena.bridge import ArenaBridge
from rlinf.envs.sim.humanoidarena.runtime import RuntimeProcess


class HumanoidArenaEnv(gym.Env):
    """Expose one upstream episode at action-chunk boundaries for evaluation.

    The upstream executable owns reset, SONIC control and termination. Only
    terminal success rewards are exposed; intermediate observations are absent.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        del total_num_processes, worker_info
        if num_envs != 1 or cfg.auto_reset or cfg.ignore_terminations:
            raise ValueError(
                "HumanoidArena requires one env, auto_reset=false and ignore_terminations=false"
            )
        self.cfg, self.num_envs = cfg, num_envs
        self.seed = int(cfg.seed) + seed_offset
        self.is_start = True
        self.chunk_size = int(cfg.chunk_size)
        self.repeat = -1
        self._episode_count = int(cfg.rollout_epoch)
        self._pending_observation = None
        self._process = None
        self._bridge = None
        self._obs = None
        self._terminal = None
        self._output = (
            Path(cfg.output_dir).expanduser().resolve()
            / f"env-{seed_offset}-{uuid4().hex}"
        )
        self._output.mkdir(parents=True)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (40,), np.float32)
        self.observation_space = gym.spaces.Dict({})

    def _wait_observation(self) -> dict | None:
        deadline = time.monotonic() + float(self.cfg.timeout)
        while time.monotonic() < deadline:
            if self._bridge.error is not None:
                raise RuntimeError("Arena HTTP request failed") from self._bridge.error
            if self._result_path.exists():
                # Upstream writes JSON directly, so wait for a complete document.
                try:
                    result = json.loads(self._result_path.read_text())
                except json.JSONDecodeError:
                    continue
                if result.get("failure_reason") not in ("success", "fall", "timeout"):
                    raise RuntimeError(f"Arena episode failed: {result}")
                log = self._process.log_path.read_text(errors="replace")
                if any(
                    marker in log
                    for marker in (
                        "get_action error:",
                        "GEAR-SONIC inference error:",
                        "Traceback (most recent call last)",
                    )
                ):
                    raise RuntimeError(
                        f"Arena reported runtime errors; see {self._process.log_path}"
                    )
                self._terminal = result
                return None
            try:
                if self._pending_observation is None:
                    item = self._bridge.observations.get(timeout=0.1)
                else:
                    item, self._pending_observation = self._pending_observation, None
                # The next episode can reach HTTP before we notice this result.
                if self._result_path.exists():
                    self._pending_observation = item
                    continue
                rgb, state, policy_seed = item
                if policy_seed is not None:
                    raise ValueError(
                        "Expected the original unseeded final policy reset; check the Arena revision"
                    )
                return {
                    "main_images": torch.from_numpy(rgb[None]),
                    "states": torch.from_numpy(state[None]),
                    "task_descriptions": [str(self.cfg.prompt)],
                    "reset_mask": torch.tensor([self.is_start], dtype=torch.bool),
                }
            except queue.Empty:
                if (
                    self._process.process.poll() is not None
                    and self._result_path.exists()
                ):
                    continue
                self._process.check()
        raise TimeoutError(f"Timed out waiting for Arena; see {self._process.log_path}")

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        """Advance within one original persistent task/seed batch."""
        del options
        if self.repeat >= 0 and self._terminal is None:
            raise RuntimeError("Finish the current HumanoidArena episode before reset")
        if seed is not None and int(seed) != self.seed:
            if self.repeat >= 0:
                raise ValueError("A persistent HumanoidArena batch keeps one seed")
            self.seed = int(seed)
        if self.repeat + 1 >= self._episode_count:
            raise RuntimeError("HumanoidArena reset exceeds configured rollout_epoch")
        self.repeat += 1
        self._terminal = None
        self.is_start = True
        output = self._output / f"episode-{self.repeat}"
        output.mkdir(exist_ok=True)
        self._result_path = output / "result.json"
        if self.repeat > 0:
            try:
                self._obs = self._wait_observation()
                if self._obs is None:
                    raise RuntimeError("Arena terminated before its policy observation")
                self.is_start = False
                return self._obs, {}
            except BaseException:
                self.close()
                raise
        task = str(self.cfg.task_id)
        specs = []
        for repeat in range(self._episode_count):
            episode_output = self._output / f"episode-{repeat}"
            episode_output.mkdir(exist_ok=True)
            digest = hashlib.sha256(f"{task}|{self.seed}|{repeat}".encode()).digest()
            specs.append(
                {
                    "seed": self.seed,
                    "repeat_idx": repeat,
                    "episode_seed": int.from_bytes(digest[-4:], "big") & 0x7FFFFFFF,
                    "episode_index": self.seed * 20 + repeat,
                    "result_json": str(episode_output / "result.json"),
                    "success_video_dir": str(episode_output / "videos/success"),
                    "failure_video_dir": str(episode_output / "videos/failure"),
                    "recording_save_dir": str(episode_output / "recordings"),
                    "max_steps": int(self.cfg.max_episode_steps),
                    "video_fps": 30,
                    "post_termination_record_steps": int(
                        self.cfg.get("native_post_termination_steps", 0)
                    ),
                    "model_label": "wbwam",
                }
            )
        batch_path = self._output / "episode_batch.json"
        batch_path.write_text(json.dumps({"episodes": specs}, indent=2))
        self._bridge = ArenaBridge(self.chunk_size, float(self.cfg.timeout))
        arena = Path(self.cfg.arena_root).expanduser().resolve()
        repo = Path(self.cfg.wbwam_repo).expanduser().resolve()
        command = [
            "bash",
            str(repo / "benchmark/humanoidarena/isaac_native_python.sh"),
            "-u",
            "script/eval_scripts/sonic_pi05/sim_eval_vla.py",
            "--task",
            task,
            "--env_config_yaml",
            str(self.cfg.env_config),
            "--seed",
            str(self.seed),
            "--episode_batch_json",
            str(batch_path),
            "--max_steps",
            str(self.cfg.max_episode_steps),
            "--sonic_encoder_path",
            str(Path(self.cfg.sonic_root) / "model_encoder.onnx"),
            "--sonic_decoder_path",
            str(Path(self.cfg.sonic_root) / "model_decoder.onnx"),
            "--sonic_vla_root_rot6d_layout",
            "row",
            "--sonic_vla_root_max_delta_deg",
            "26.0",
            "--lerobot_server_url",
            self._bridge.url,
            "--lerobot_server_timeout",
            str(self.cfg.timeout),
            "--robot_type",
            "unitree_g1_refpose_v3_1",
            "--result_json",
            str(self._result_path),
            "--success_video_dir",
            str(output / "videos/success"),
            "--failure_video_dir",
            str(output / "videos/failure"),
            "--recording_save_dir",
            str(output / "recordings"),
            "--record_video_every_n",
            "1" if self.cfg.get("native_record_video", False) else "0",
            "--post_termination_record_steps",
            str(self.cfg.get("native_post_termination_steps", 0)),
            "--device",
            "cpu",
            "--headless",
            "--kit_args",
            f"--/renderer/activeGpu={self.cfg.simulation_gpu} --/physics/cudaDevice={self.cfg.simulation_gpu} --/renderer/multiGpu/enabled=false",
        ]
        env = dict(os.environ)
        # Isaac selects physical devices through Kit; Ray's mask is CUDA-local.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(
            {
                "HA_SIMULATION_PYTHON": str(self.cfg.simulation_python),
                "HA_ARENA_ROOT": str(arena),
                "HA_DEPENDENCIES_ROOT": str(self.cfg.dependencies_root),
                "HA_ISAAC_HOME": str(output / "isaac-home"),
                "HA_ISAAC_CACHE": str(self._output / "isaac-cache"),
            }
        )
        try:
            self._process = RuntimeProcess(
                command, output / "sim.log", cwd=arena / "isaaclab_twist2_g1", env=env
            )
            self._obs = self._wait_observation()
            if self._obs is None:
                raise RuntimeError(
                    f"Arena terminated before its first policy observation: {self._terminal}"
                )
            self.is_start = False
            return self._obs, {}
        except BaseException:
            self.close()
            raise

    def chunk_step(self, chunk_actions: np.ndarray | torch.Tensor) -> tuple:
        """Execute a chunk and report the upstream terminal success/fall/timeout."""
        if self._obs is None or self._terminal is not None:
            raise RuntimeError("Reset HumanoidArena before stepping a new episode")
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        try:
            self._bridge.send_action(chunk_actions)
            observation = self._wait_observation()
            if observation is not None:
                self._obs = observation
            size = self.chunk_size
            rewards = torch.zeros((1, size), dtype=torch.float32)
            terminated = torch.zeros((1, size), dtype=torch.bool)
            truncated = torch.zeros_like(terminated)
            infos = [{} for _ in range(size)]
            if self._terminal is not None:
                result = self._terminal
                success = bool(result["success"])
                rewards[0, -1] = float(success)
                truncated[0, -1] = result["failure_reason"] == "timeout"
                terminated[0, -1] = not truncated[0, -1]
                infos[-1]["episode"] = {
                    "success": torch.tensor([success]),
                    "return": torch.tensor([float(success)]),
                    "episode_len": torch.tensor([int(result["episode_steps"])]),
                }
                if self.repeat == self._episode_count - 1:
                    self.close()
            # Evaluation only: last policy observation is held at termination.
            return (
                [None] * (size - 1) + [self._obs],
                rewards,
                terminated,
                truncated,
                infos,
            )
        except BaseException:
            self.close()
            raise

    def update_reset_state_ids(self) -> None:
        """Advance the repeat index only on reset."""

    def close(self) -> None:
        """Release the simulator and local bridge; keep episode artifacts."""
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        if self._process is not None:
            self._process.close()
            self._process = None
