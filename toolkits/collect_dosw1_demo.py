#!/usr/bin/env python3
"""Collect DOSW1 pick demonstrations via leader-arm teleop.

Saves data in ``TrajectoryReplayBuffer`` checkpoint format so it can be
loaded directly by ``demo_buffer.load_path`` in the training config.

Usage (on the robot machine):
    python toolkits/collect_dosw1_demo.py \
        --robot-url 127.0.0.1 \
        --camera-serials 419622072872 243722075219 244222070702 \
        --target-grasp-joint -0.47 -1.13 0.65 1.41 -0.60 -1.09 \
        --target-lift-joint  -0.51 -0.94 0.61 1.40 -0.63 -1.12 \
        --num-episodes 10 \
        --save-dir /mlp_vepfs/share/wyf/RLinf-open/demo_data

Keyboard controls:
    FreeTeleop (before each episode):
        Move leader arms freely to position the robot, press 's' to start.
    During episode (all control comes from leader arm):
        p  -- pause (hold current position)
        t  -- resume teleop (from pause)
        r  -- abort current episode, return to FreeTeleop
"""

from __future__ import annotations

import argparse
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.envs.realworld.dosw1.dosw1_env import ControlMode
from rlinf.envs.realworld.dosw1.tasks.pick import PickEnv

_ACTION_DIM = 14


def _force_teleop_mode(env: PickEnv) -> None:
    """Switch env to TELEOP right after reset so every step reads from leader arm."""
    env._control_mode = ControlMode.TELEOP
    env._snapshot_teleop_init()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect DOSW1 pick demos via teleop")
    p.add_argument("--robot-url", type=str, default="127.0.0.1")
    p.add_argument("--left-arm-port", type=int, default=50051)
    p.add_argument("--right-arm-port", type=int, default=50053)
    p.add_argument("--left-lead-port", type=int, default=50050)
    p.add_argument("--right-lead-port", type=int, default=50052)
    p.add_argument(
        "--camera-serials", nargs="*", default=None,
        help="RealSense serial numbers; omit to auto-discover",
    )
    p.add_argument("--main-image-key", type=str, default="cam_left")
    p.add_argument(
        "--target-grasp-joint", nargs=6, type=float,
        default=[-0.75, 0.0, 0.0, 1.57, 0.0, -1.57],
    )
    p.add_argument(
        "--target-lift-joint", nargs=6, type=float,
        default=[-0.75, -0.15, 0.0, 1.57, 0.0, -1.57],
    )
    p.add_argument("--step-frequency", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument(
        "--save-dir", type=str,
        default="/mlp_vepfs/share/wyf/RLinf-open/demo_data",
    )
    return p.parse_args()


def _raw_obs_to_model_obs(
    raw_obs: dict,
    main_image_key: str,
) -> dict[str, torch.Tensor]:
    """Convert DOSW1Env raw obs to model-facing tensor dict.

    Mirrors ``RealWorldEnv._get_observation``:
      state keys sorted alphabetically + concatenated -> ``states``
      main camera frame                               -> ``main_images``
      remaining camera frames stacked                 -> ``extra_view_images``
    """
    raw_state = OrderedDict(sorted(raw_obs["state"].items()))
    states = np.concatenate(list(raw_state.values()), axis=-1)

    frames = OrderedDict(sorted(raw_obs["frames"].items()))
    main_image = frames.pop(main_image_key)

    obs: dict[str, torch.Tensor] = {
        "states": torch.from_numpy(states.copy()),
        "main_images": torch.from_numpy(main_image.copy()),
    }
    if frames:
        extra = np.stack(list(frames.values()), axis=0)
        obs["extra_view_images"] = torch.from_numpy(extra.copy())
    return obs


def _read_actual_action(env: PickEnv) -> np.ndarray:
    """Read the robot's current joint state as the executed action."""
    state = env._robot_state
    actual = np.empty(_ACTION_DIM, dtype=np.float64)
    actual[:6] = state.left_joint_positions
    actual[6] = state.left_gripper
    actual[7:13] = state.right_joint_positions
    actual[13] = state.right_gripper
    return actual


def _build_trajectory(
    obs_list: list[dict[str, torch.Tensor]],
    next_obs_list: list[dict[str, torch.Tensor]],
    actions: list[np.ndarray],
    rewards: list[float],
    terminations: list[bool],
    truncations: list[bool],
) -> Trajectory:
    """Assemble one episode into a Trajectory with shape (T, 1, ...)."""
    T = len(actions)

    action_t = torch.from_numpy(np.stack(actions)).unsqueeze(1).float()   # (T, 1, 14)
    reward_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)    # (T, 1)
    term_t = torch.tensor(terminations, dtype=torch.bool).unsqueeze(1)    # (T, 1)
    trunc_t = torch.tensor(truncations, dtype=torch.bool).unsqueeze(1)    # (T, 1)
    done_t = (term_t | trunc_t)                                           # (T, 1)
    intervene_t = torch.ones_like(action_t, dtype=torch.bool)             # (T, 1, 14)

    def _stack_obs(seq: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        stacked: dict[str, torch.Tensor] = {}
        for k in seq[0]:
            stacked[k] = torch.stack([o[k] for o in seq]).unsqueeze(1)  # (T, 1, ...)
        return stacked

    versions = torch.zeros(T, 1, dtype=torch.float32)
    model_weights_id = str(
        uuid.uuid5(uuid.NAMESPACE_DNS, versions.numpy().tobytes().hex())
    )

    return Trajectory(
        max_episode_length=T,
        model_weights_id=model_weights_id,
        actions=action_t,
        intervene_flags=intervene_t,
        rewards=reward_t,
        terminations=term_t,
        truncations=trunc_t,
        dones=done_t,
        versions=versions,
        curr_obs=_stack_obs(obs_list),
        next_obs=_stack_obs(next_obs_list),
    )


def main() -> None:
    args = _parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = PickEnv(
        override_cfg={
            "robot_url": args.robot_url,
            "left_arm_port": args.left_arm_port,
            "right_arm_port": args.right_arm_port,
            "left_lead_port": args.left_lead_port,
            "right_lead_port": args.right_lead_port,
            "camera_serials": args.camera_serials,
            "target_grasp_joint": args.target_grasp_joint,
            "target_lift_joint": args.target_lift_joint,
            "step_frequency": args.step_frequency,
            "max_num_steps": args.max_steps,
            "enable_human_in_loop": True,
            "enable_data_persistence": False,
            "is_dummy": False,
        },
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )

    max_cache = args.num_episodes + 10
    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=max_cache,
        sample_window_size=max_cache,
        auto_save=False,
        trajectory_format="pt",
    )

    print(f"[collect] Will collect {args.num_episodes} episodes -> {save_dir}")
    print("[collect] Controls: FreeTeleop -> press 's' to start | "
          "during episode: p(pause) t(resume teleop) r(abort)")

    collected = 0
    while collected < args.num_episodes:
        print(f"\n===== Episode {collected + 1}/{args.num_episodes} =====")
        raw_obs, _ = env.reset()
        _force_teleop_mode(env)
        print("[collect] Teleop active -- leader arm controls follower")

        obs_list: list[dict[str, torch.Tensor]] = []
        next_obs_list: list[dict[str, torch.Tensor]] = []
        action_list: list[np.ndarray] = []
        reward_list: list[float] = []
        term_list: list[bool] = []
        trunc_list: list[bool] = []

        curr_model_obs = _raw_obs_to_model_obs(raw_obs, args.main_image_key)
        done = False
        aborted = False
        step_count = 0

        grasp_joint = np.asarray(args.target_grasp_joint, dtype=np.float64)
        lift_joint = np.asarray(args.target_lift_joint, dtype=np.float64)

        while not done:
            placeholder = np.zeros(_ACTION_DIM, dtype=np.float64)
            raw_next_obs, reward, terminated, truncated, info = env.step(placeholder)
            done = terminated or truncated

            if done and env._in_free_teleop:
                aborted = True
                break

            left_joint = env._robot_state.left_joint_positions
            if env._phase == "reach":
                target = grasp_joint
                tag = "reach->grasp"
            else:
                target = lift_joint
                tag = "lift->target"
            diff = left_joint - target
            dist = float(np.linalg.norm(diff))
            per_joint = " ".join(f"{d:+.3f}" for d in diff)
            grip_info = ""
            if env._teleop_target_left_gripper is not None:
                grip_info = f" | grip_target={env._teleop_target_left_gripper:.4f}"
            print(f"  [{tag}] dist={dist:.4f} delta=[{per_joint}]{grip_info} r={reward:.3f}")

            actual_action = _read_actual_action(env)
            next_model_obs = _raw_obs_to_model_obs(raw_next_obs, args.main_image_key)

            obs_list.append(curr_model_obs)
            next_obs_list.append(next_model_obs)
            action_list.append(actual_action)
            reward_list.append(reward)
            term_list.append(terminated)
            trunc_list.append(truncated)

            curr_model_obs = next_model_obs
            step_count += 1

        if aborted or step_count == 0:
            print("[collect] Episode aborted or empty, skipping")
            continue

        traj = _build_trajectory(
            obs_list, next_obs_list, action_list,
            reward_list, term_list, trunc_list,
        )
        buffer.add_trajectories([traj])
        collected += 1
        print(f"[collect] Episode {collected} done ({step_count} steps, "
              f"reward_sum={sum(reward_list):.3f})")

    buffer.save_checkpoint(str(save_dir))
    env.close()

    print(f"\n[collect] {collected} episodes saved to {save_dir}")
    print(f"[collect] Training config: algorithm.demo_buffer.load_path: \"{save_dir}\"")


if __name__ == "__main__":
    main()
