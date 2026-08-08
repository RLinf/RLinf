# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay one LeRobot joint-space episode through the real-world env path."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pyarrow.parquet as pq
from omegaconf import open_dict

from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
from rlinf.envs.realworld.franka.ros2_controller import (
    FR3_JOINT_LIMITS_LOWER,
    FR3_JOINT_LIMITS_UPPER,
)
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker

_ARM_INDICES = np.array([*range(7), *range(8, 15)])
_GRIPPER_INDICES = np.array([7, 15])


@dataclass(frozen=True)
class ReplayEpisode:
    """Joint states and actions for one LeRobot episode."""

    root: Path
    episode_index: int
    fps: float
    frame_indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray


def load_replay_episode(dataset_path: str, episode_index: int) -> ReplayEpisode:
    """Load one episode without decoding its videos."""
    root = resolve_lerobot_dataset_root(dataset_path)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    with info_path.open(encoding="utf-8") as file_obj:
        info = json.load(file_obj)

    rows: list[tuple[int, list[float], list[float]]] = []
    required = {"episode_index", "frame_index", "state", "actions"}
    for parquet_path in sorted((root / "data").rglob("*.parquet")):
        schema = pq.read_schema(parquet_path)
        if not required.issubset(schema.names):
            continue
        table = pq.read_table(parquet_path, columns=sorted(required))
        episode_values = table["episode_index"].to_pylist()
        frame_values = table["frame_index"].to_pylist()
        state_values = table["state"].to_pylist()
        action_values = table["actions"].to_pylist()
        rows.extend(
            (int(frame), state, action)
            for ep, frame, state, action in zip(
                episode_values,
                frame_values,
                state_values,
                action_values,
                strict=True,
            )
            if int(ep) == episode_index
        )

    if not rows:
        raise ValueError(f"Episode {episode_index} not found under {root}")
    rows.sort(key=lambda row: row[0])
    frame_indices = np.asarray([row[0] for row in rows], dtype=np.int64)
    states = np.asarray([row[1] for row in rows], dtype=np.float32)
    actions = np.asarray([row[2] for row in rows], dtype=np.float32)
    if states.shape != (len(rows), 16) or actions.shape != (len(rows), 16):
        raise ValueError(
            "Replay requires 16-D dual-Franka joint state/actions, got "
            f"states={states.shape}, actions={actions.shape}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Episode contains non-finite state/action values")
    if not np.array_equal(frame_indices, np.arange(len(rows))):
        raise ValueError(f"Episode frame indices are not contiguous: {frame_indices}")

    joint_low = np.concatenate([FR3_JOINT_LIMITS_LOWER] * 2)
    joint_high = np.concatenate([FR3_JOINT_LIMITS_UPPER] * 2)
    joint_actions = actions[:, _ARM_INDICES]
    if np.any(joint_actions < joint_low) or np.any(joint_actions > joint_high):
        raise ValueError("Episode actions exceed FR3 joint limits")
    if np.any(np.abs(actions[:, _GRIPPER_INDICES]) > 1.0):
        raise ValueError("Episode gripper actions must be in [-1, 1]")

    return ReplayEpisode(
        root=root,
        episode_index=episode_index,
        fps=float(info["fps"]),
        frame_indices=frame_indices,
        states=states,
        actions=actions,
    )


def _error_stats(error: np.ndarray) -> dict[str, float]:
    absolute = np.abs(error)
    return {
        "rmse_rad": float(np.sqrt(np.mean(np.square(error)))),
        "mean_abs_rad": float(np.mean(absolute)),
        "max_abs_rad": float(np.max(absolute)),
    }


class LeRobotEpisodeReplayer(Worker):
    """Run recorded actions through the same env path used by policy inference."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.episode = load_replay_episode(
            cfg.runner.replay_dataset_path,
            int(cfg.runner.replay_episode_index),
        )
        first_state = self.episode.states[0]
        with open_dict(self.cfg):
            self.cfg.env.eval.use_gello_joint = False
            self.cfg.env.eval.keyboard_reward_wrapper = None
            self.cfg.env.eval.override_cfg.teleop_direct_stream = False
            self.cfg.env.eval.override_cfg.joint_action_mode = "absolute"
            self.cfg.env.eval.override_cfg.step_frequency = self.episode.fps
            self.cfg.env.eval.override_cfg.max_num_steps = len(self.episode.actions) + 200
            self.cfg.env.eval.override_cfg.joint_reset_qpos = [
                first_state[:7].tolist(),
                first_state[8:15].tolist(),
            ]
        self.env = RealWorldEnv(
            self.cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

    @staticmethod
    def _state_array(obs) -> np.ndarray:
        states = obs["states"]
        if hasattr(states, "detach"):
            states = states.detach().cpu().numpy()
        return np.asarray(states, dtype=np.float32)[0]

    def run(self) -> dict[str, object]:
        try:
            obs, _ = self.env.reset()
            start_target = self.episode.actions[0].copy()
            start_target[_ARM_INDICES] = self.episode.states[0, _ARM_INDICES]

            deadline = time.monotonic() + 5.0
            start_error = float("inf")
            while time.monotonic() < deadline:
                obs, *_ = self.env.step(start_target[None])
                current = self._state_array(obs)
                start_error = float(
                    np.max(
                        np.abs(
                            current[_ARM_INDICES]
                            - self.episode.states[0, _ARM_INDICES]
                        )
                    )
                )
                if start_error <= 0.03:
                    break
            if start_error > 0.03:
                raise RuntimeError(
                    f"Failed to reach episode start state: max error={start_error:.4f} rad"
                )

            actual_states = []
            step_starts = []
            for index, action in enumerate(self.episode.actions):
                step_starts.append(time.perf_counter())
                obs, *_ = self.env.step(action[None])
                actual_states.append(self._state_array(obs))
                if index % 30 == 0 or index + 1 == len(self.episode.actions):
                    self.log_info(
                        f"Replay {index + 1}/{len(self.episode.actions)} frames"
                    )

            actual = np.stack(actual_states)
            tracking_error = (
                actual[:, _ARM_INDICES]
                - self.episode.actions[:, _ARM_INDICES]
            )
            reproduction_error = (
                actual[:-1, _ARM_INDICES]
                - self.episode.states[1:, _ARM_INDICES]
            )
            intervals = np.diff(np.asarray(step_starts))
            result = {
                "episode_index": self.episode.episode_index,
                "frames": len(self.episode.actions),
                "fps": self.episode.fps,
                "mean_actual_fps": (
                    float(1.0 / intervals.mean()) if len(intervals) else None
                ),
                "target_tracking": _error_stats(tracking_error),
                "trajectory_reproduction": _error_stats(reproduction_error),
            }
            self.log_info(json.dumps(result, indent=2))
            return result
        finally:
            self.env.close()


_SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("EMBODIED_PATH", str(_SCRIPT_DIR))
os.environ.setdefault("RLINF_TASK_DESCRIPTION", "LeRobot episode replay")


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_collect_data_ros2_gello_dual_franka_pnp",
)
def main(cfg) -> None:
    dataset_path = os.environ.get("RLINF_REPLAY_DATASET")
    if not dataset_path:
        raise ValueError("Set RLINF_REPLAY_DATASET to a LeRobot dataset root")
    episode_index = int(os.environ.get("RLINF_REPLAY_EPISODE", "0"))
    episode = load_replay_episode(dataset_path, episode_index)
    max_joint_step = float(
        np.max(np.abs(np.diff(episode.actions[:, _ARM_INDICES], axis=0)))
    )
    print(
        f"Dataset: {episode.root}\n"
        f"Episode: {episode_index}, frames: {len(episode.actions)}, "
        f"fps: {episode.fps:g}, max joint step: {max_joint_step:.4f} rad"
    )
    if os.environ.get("RLINF_REPLAY_VALIDATE_ONLY") == "1":
        print("Validation only; no hardware was opened.")
        return
    try:
        import rclpy  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ROS 2 is not sourced. Run `source /opt/ros/humble/setup.bash` and "
            "`source /home/shang/RLinf/ros2_ws/install/setup.bash` first."
        ) from exc
    if input(f"Type REPLAY {episode_index} to start hardware replay: ").strip() != (
        f"REPLAY {episode_index}"
    ):
        print("Replay cancelled.")
        return

    with open_dict(cfg):
        cfg.runner.replay_dataset_path = str(episode.root)
        cfg.runner.replay_episode_index = episode_index
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ComponentPlacement(cfg, cluster).get_strategy("env")
    replayer = LeRobotEpisodeReplayer.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement,
    )
    result = replayer.run().wait()[0]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
