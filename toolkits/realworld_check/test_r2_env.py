#!/usr/bin/env python3
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

"""Drive or verify a Robbyant R2 robot through ``R2ReachEnv``.

Examples::

    # Run against fake RFrame SDKs (no hardware required)
    python -m toolkits.realworld_check.test_r2_env --mock --steps 5

    # Connect to a real R2 robot and inspect observations without moving
    python -m toolkits.realworld_check.test_r2_env --no-move

    # Interactive CLI on a real or mock R2 robot
    python -m toolkits.realworld_check.test_r2_env --mock --interactive
"""

from __future__ import annotations

import argparse
import contextlib
import signal
from pathlib import Path
from typing import Any

import numpy as np

HELP = """
  left <j1..j7>    step left arm joint (1..7) by +step deg, e.g. "left 2 +1"
  right <j1..j7>   step right arm joint (1..7) by +step deg, e.g. "right 2 -1"
  open / close     open or close both grippers
  home             return active arms to the reset configuration
  where            print current joint angles and gripper openings
  quit
"""


@contextlib.contextmanager
def _deferred_interrupts() -> Any:
    """Hold Ctrl-C until teardown completes."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _describe_obs(obs: dict[str, Any]) -> str:
    parts: list[str] = []
    for group in ("state", "frames"):
        sub = obs.get(group)
        if not isinstance(sub, dict):
            continue
        items = ", ".join(f"{k}:{tuple(v.shape)}/{v.dtype}" for k, v in sub.items())
        parts.append(f"{group}={{{items}}}")
    return " ".join(parts)


def _run_batch_steps(
    env: Any,
    *,
    steps: int,
    delta_rad: float,
    reset: bool,
    no_move: bool,
) -> None:
    if no_move:
        obs = env._get_observation()
        assert env.observation_space.contains(obs), (
            "observation does not match env.observation_space"
        )
        print(f"Observation: {_describe_obs(obs)}")
        joints = obs["state"]["arm_joint_position"]
        print(f"Joints (deg): {np.round(np.rad2deg(joints), 2).tolist()}")
        return

    if reset:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), (
            "reset() observation does not match env.observation_space"
        )
        print(f"Reset observation: {_describe_obs(obs)}")
    else:
        obs = env._get_observation()

    base_joints = np.asarray(obs["state"]["arm_joint_position"], dtype=np.float32)
    for step_idx in range(steps):
        sign = 1.0 if (step_idx % 2 == 0) else -1.0
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        offset = 0
        for idx, _side in enumerate(env.config.active_arms):
            q_slice = slice(idx * 7, (idx + 1) * 7)
            action[offset : offset + 7] = base_joints[q_slice] + sign * delta_rad
            offset += 7
            if env._ee_dim > 0:
                action[offset : offset + env._ee_dim] = 1.0
                offset += env._ee_dim
        obs, reward, terminated, truncated, _info = env.step(action)
        assert env.observation_space.contains(obs), (
            f"step {step_idx} observation does not match env.observation_space"
        )
        q0 = float(obs["state"]["arm_joint_position"][0])
        print(
            f"step={step_idx + 1}/{steps} reward={reward:.4f} "
            f"terminated={terminated} truncated={truncated} "
            f"left_j1={np.rad2deg(q0):+.2f} deg"
        )


def _run_interactive(env: Any, *, step_rad: float, reset: bool) -> None:
    if reset:
        env.reset()
    obs = env._get_observation()
    joints = np.array(obs["state"]["arm_joint_position"], dtype=np.float32)
    grip = 1.0
    print(HELP)

    while True:
        try:
            words = input("r2> ").split()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not words:
            continue
        cmd = words[0].lower()
        if cmd in ("quit", "exit", "q"):
            return
        if cmd in ("help", "?"):
            print(HELP)
            continue
        if cmd == "where":
            obs = env._get_observation()
            joints = np.array(obs["state"]["arm_joint_position"], dtype=np.float32)
            print("  joints (deg):", np.round(np.rad2deg(joints), 1).tolist())
            if "gripper_position" in obs["state"]:
                print(
                    "  end_effector:",
                    np.round(obs["state"]["gripper_position"], 2).tolist(),
                )
            continue
        if cmd == "home":
            obs, _ = env.reset()
            joints = np.array(obs["state"]["arm_joint_position"], dtype=np.float32)
            print("  joints (deg):", np.round(np.rad2deg(joints), 1).tolist())
            continue
        if cmd == "open":
            grip = 1.0
        elif cmd == "close":
            grip = 0.0
        elif cmd in ("left", "right") and len(words) >= 2:
            try:
                joint_idx = int(words[1]) - 1
                mult = float(words[2]) if len(words) >= 3 else 1.0
            except ValueError:
                print("Usage: left|right <1..7> [multiplier]")
                continue
            if not (0 <= joint_idx < 7):
                print("Joint index must be 1..7")
                continue
            arm_offset = 0 if cmd == "left" else 7
            if arm_offset + joint_idx < joints.size:
                joints[arm_offset + joint_idx] += float(mult * step_rad)
        else:
            print(f"Unknown command {cmd!r}; type help")
            continue

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        offset = 0
        for idx, _side in enumerate(env.config.active_arms):
            q_slice = slice(idx * 7, (idx + 1) * 7)
            action[offset : offset + 7] = joints[q_slice]
            offset += 7
            if env._ee_dim > 0:
                action[offset : offset + env._ee_dim] = grip
                offset += env._ee_dim
        obs, _reward, _term, _trunc, _info = env.step(action)
        joints = np.array(obs["state"]["arm_joint_position"], dtype=np.float32)
        print("  joints (deg):", np.round(np.rad2deg(joints), 1).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run against fake RFrame SDKs, no hardware required",
    )
    parser.add_argument(
        "--body-dof",
        type=str,
        default="auto",
        choices=("auto", "20", "21"),
        help="R2 whole-body DoF (auto, 20, or 21; default auto)",
    )
    parser.add_argument(
        "--end-effector",
        type=str,
        default="auto",
        choices=("auto", "gripper", "hand", "none"),
        help="end-effector type on both arms (auto, gripper, hand, or none; default auto)",
    )
    parser.add_argument(
        "--enable-base",
        action="store_true",
        help="include the omnidirectional wheeled chassis",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=["head", "wrist_left", "wrist_right"],
        help="R2 camera names to subscribe to",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=2.0,
        help="degrees per move (default 2.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="number of batch steps when not running interactively (default 5)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="control updates per second (default 20)",
    )
    parser.add_argument(
        "--reset-speed",
        type=float,
        default=30.0,
        help="maximum reset joint speed in degrees per second (default 30)",
    )
    parser.add_argument(
        "--no-reset",
        dest="reset",
        action="store_false",
        help="start from current arm pose instead of resetting first",
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="only connect and print observation without sending joint commands",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="enter interactive command prompt",
    )
    parser.add_argument(
        "--enable-camera-player",
        action="store_true",
        help="enable the camera preview window (disabled by default)",
    )
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if not np.isfinite(args.reset_speed) or args.reset_speed <= 0:
        parser.error("--reset-speed must be finite and positive")

    if args.mock:
        import sys

        sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))
        from robot_mocks import mocked_sdks

        context: Any = mocked_sdks()
    else:
        context = contextlib.nullcontext()

    with context:
        from rlinf.envs.real.r2 import R2ReachEnv
        from rlinf.robotics import R2Config
        from rlinf.robotics.discovery import RobotDiscovery

        cfg = R2Config(
            node_rank=0,
            body_dof=None if args.body_dof == "auto" else int(args.body_dof),
            end_effector_type=(
                None if args.end_effector == "auto" else args.end_effector
            ),
            enable_base=args.enable_base,
            cameras=list(args.cameras),
        )
        discovery = RobotDiscovery.registry["R2"].discovery_cls
        resources = discovery.enumerate(node_rank=0, configs=[cfg])
        resolved_cfg = resources.infos[0].config
        print(
            f"R2 hardware config: body_dof={resolved_cfg.body_dof}, "
            f"end_effector_type={resolved_cfg.end_effector_type!r}"
        )

        env = R2ReachEnv(
            {
                "enable_camera_player": args.enable_camera_player,
                "step_frequency": args.fps,
                "reset_joint_speed": float(np.deg2rad(args.reset_speed)),
                "reset_on_init": False,
            },
            robot_info=resources.infos[0],
        )
        try:
            if args.interactive:
                _run_interactive(
                    env,
                    step_rad=float(np.deg2rad(args.step)),
                    reset=args.reset,
                )
            else:
                _run_batch_steps(
                    env,
                    steps=args.steps,
                    delta_rad=float(np.deg2rad(args.step)),
                    reset=args.reset,
                    no_move=args.no_move,
                )
        finally:
            with _deferred_interrupts():
                env.close()


if __name__ == "__main__":
    main()
