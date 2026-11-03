# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the pinned RLinf IsaacLab Stack Cube task on frozen scene requests.

Run with the isolated Isaac Sim 5.1 interpreter. Model inference is spawned
with --model-python so the established OpenPI environment remains unchanged.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from toolkits.lerobot.evaluate_stack_cube_openpi import (
    _hash,
    aggregate_reports,
    receive_policy,
    start_policy_process,
    stop_policy_process,
    summarize_episode,
)

FORK_COMMIT = "4246b6b4f4a3e74ee20e002ed7536b1c788d39f4"
ENV_ID = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0"
PROMPT = (
    "Stack the red block on the blue block, then stack the green block on the red block"
)
ARM_QPOS = [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952]


def policy_observation(raw: dict) -> dict:
    """Expose dataset Euler XYZ state, deliberately distinct from legacy rotvec."""
    from scipy.spatial.transform import Rotation

    obs = raw["policy"]
    quaternion = obs["eef_quat"].detach().cpu().numpy()[:, [1, 2, 3, 0]]
    euler = Rotation.from_quat(quaternion).as_euler("xyz")
    state = np.concatenate(
        (obs["eef_pos"].cpu().numpy(), euler, obs["gripper_pos"].cpu().numpy()), axis=-1
    )
    return {
        "states": state.astype(np.float32),
        "main_images": obs["table_cam"].cpu().numpy()[..., :3].astype(np.uint8),
        "wrist_images": obs["wrist_cam"].cpu().numpy()[..., :3].astype(np.uint8),
        "task_descriptions": [PROMPT],
    }


def reset_request(env, request: dict) -> tuple[dict, dict]:
    """Set paired XY/yaw/joints, retaining the native USD asset's reset height."""
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz

    if request.get("visual") is not None:
        raise ValueError(
            "This evaluator does not map EmbodiChain light units to IsaacLab."
        )
    env.reset(seed=int(request["seed"]))
    robot = env.scene["robot"]
    qpos = robot.data.default_joint_pos.clone()
    arm_ids, _ = robot.find_joints(
        [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True
    )
    hand_ids, _ = robot.find_joints(
        ["panda_finger_joint1", "panda_finger_joint2"], preserve_order=True
    )
    qpos[:, arm_ids] = torch.tensor(ARM_QPOS, device=env.device) + torch.tensor(
        request["arm_joint_offsets"], device=env.device
    )
    qpos[:, hand_ids] = 0.04
    limits = robot.data.soft_joint_pos_limits
    if torch.any(qpos < limits[..., 0]) or torch.any(qpos > limits[..., 1]):
        raise ValueError("Requested joints violate the IsaacLab limits")
    desired = {}
    for name, color in (("cube_1", "blue"), ("cube_2", "red"), ("cube_3", "green")):
        obj = env.scene[name]
        state = obj.data.default_root_state.clone()
        state[:, :2] = torch.tensor(
            request["objects"][color]["position"][:2], device=env.device
        )
        state[:, :3] += env.scene.env_origins
        yaw = torch.tensor([request["objects"][color]["yaw_rad"]], device=env.device)
        state[:, 3:7] = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        state[:, 7:] = 0
        desired[name] = state
        lifted = state.clone()
        lifted[:, 2] += 0.1
        obj.write_root_state_to_sim(lifted)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    for name, state in desired.items():
        env.scene[name].write_root_state_to_sim(state)
    robot.write_joint_state_to_sim(qpos, torch.zeros_like(qpos))
    robot.set_joint_position_target(qpos)
    env.action_manager.reset()
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    env.sim.render()
    raw = env.observation_manager.compute()
    initial = {
        "qpos": robot.data.joint_pos[0].cpu().tolist(),
        "qvel": robot.data.joint_vel[0].cpu().tolist(),
        "joint_names": robot.joint_names,
        "object_root_states_w": {
            name: env.scene[name].data.root_state_w[0].cpu().tolist()
            for name in desired
        },
        "position_frame": "world; environment origin added to requested robot-base XY",
        "height_policy": "native USD asset default Z (0.0203 m), not EmbodiChain center Z",
    }
    return raw, initial


def physical_metrics(env) -> dict:
    """Use the pinned task's success test and measure independent stable dwell."""
    import torch
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    objects = [env.scene[name] for name in ("cube_1", "cube_2", "cube_3")]
    positions = torch.stack([obj.data.root_pos_w[0] for obj in objects])
    offsets = positions[1:] - positions[:-1]
    xy = offsets[:, :2].norm(dim=-1)
    height_error = (offsets[:, 2] - 0.0468).abs()
    linear = [float(obj.data.root_lin_vel_w[0].norm()) for obj in objects]
    angular = [float(obj.data.root_ang_vel_w[0].norm()) for obj in objects]
    baseline = bool(mdp.cubes_stacked(env)[0])
    layers = (xy < 0.04) & (offsets[:, 2] > 0) & (offsets[:, 2] - 0.0468 < 0.005)
    return {
        "positions": positions.cpu().tolist(),
        "xy_errors": xy.cpu().tolist(),
        "height_errors": height_error.cpu().tolist(),
        "layers": layers.cpu().tolist(),
        "linear_speeds": linear,
        "angular_speeds": angular,
        "baseline_success": baseline,
        "stable_frame": baseline
        and bool((height_error < 0.005).all())
        and max(linear) <= 0.02
        and max(angular) <= 0.5,
    }


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    """Write original-rate RGB frames without importing another simulator."""
    height, width = frames[0].shape[:2]
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "20",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        input=b"".join(frame.tobytes() for frame in frames),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sim-gpu",
        type=int,
        required=True,
        help="Physical GPU index in PCI/nvidia-smi order.",
    )
    parser.add_argument("--model-gpu", required=True)
    parser.add_argument("--max-steps", type=int, default=450)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "-C", str(args.isaaclab_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != FORK_COMMIT:
        parser.error(f"IsaacLab must use the inspected fork commit {FORK_COMMIT}")
    sim_version = importlib.metadata.version("isaacsim")
    if not sim_version.startswith("5.1."):
        parser.error(f"Expected isolated Isaac Sim 5.1, got {sim_version}")
    requests = json.loads(args.requests.read_text())["requests"]
    if not requests or args.max_steps < 1:
        parser.error("requests must not be empty and max-steps must be positive")
    marker = args.checkpoint / "actor/checkpoint_metadata.json"
    if not (args.checkpoint / "model.safetensors").is_file() and (
        not marker.is_file() or not json.loads(marker.read_text()).get("complete")
    ):
        parser.error("SFT checkpoint requires a complete export marker")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.pop("DISPLAY", None)
    for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks"):
        sys.path.insert(0, str(args.isaaclab_root / "source" / package))
    connection, process = start_policy_process(
        args.checkpoint.resolve(),
        args.norm_stats.resolve(),
        args.model_gpu,
        True,
        args.model_python,
    )
    env = app = None
    reports = []
    try:
        ready = receive_policy(connection, process)
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            headless=True,
            enable_cameras=True,
            device=f"cuda:{args.sim_gpu}",
            multi_gpu=False,
        ).app
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import torch
        from isaaclab.managers import TerminationTermCfg
        from isaaclab_tasks.utils import load_cfg_from_registry

        cfg = load_cfg_from_registry(ENV_ID, "env_cfg_entry_point")
        cfg.seed = int(requests[0]["seed"])
        cfg.scene.num_envs = 1
        cfg.sim.device = f"cuda:{args.sim_gpu}"
        torch.cuda.set_device(args.sim_gpu)
        for camera in (cfg.scene.wrist_cam, cfg.scene.table_cam):
            camera.height = camera.width = 256
        # Fixed manifests own initialization. Keep the native assets/cameras/
        # controller, and disable stochastic reset events and automatic resets.
        cfg.events = None
        for name, term in vars(cfg.terminations).items():
            if isinstance(term, TerminationTermCfg):
                setattr(cfg.terminations, name, None)
        env = gym.make(ENV_ID, cfg=cfg, render_mode="rgb_array").unwrapped
        if abs(env.step_dt - 0.05) > 1e-8:
            raise ValueError("IsaacLab task must run at 20 Hz")
        source_files = [
            args.isaaclab_root
            / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_visumotor_rewarded_env_cfg.py",
            args.isaaclab_root
            / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_visuomotor_env_cfg.py",
            args.isaaclab_root
            / "source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
        ]
        metadata = {
            "source_sha256": {str(path): _hash(path) for path in source_files},
            "isaaclab_commit": commit,
            "isaacsim_version": sim_version,
            "sim_physical_gpu": args.sim_gpu,
            "renderer_multi_gpu": False,
            "env_id": ENV_ID,
            "model_config": ready["model_config"],
            "request_sha256": _hash(args.requests),
            "stats_sha256": _hash(args.norm_stats),
            "max_steps": args.max_steps,
            "action_source": "learned_policy",
            "state_rotation": "Euler XYZ, dataset protocol",
            "overrides": {
                "reset_events": "frozen manifest",
                "termination_terms": "disabled for fixed-budget independent scoring",
                "camera_resolution": [256, 256],
                "visuals": "native defaults; no randomization",
            },
        }
        (args.output / "run_manifest.json").write_text(json.dumps(metadata, indent=2))
        for index, request in enumerate(requests):
            raw, initial = reset_request(env, request)
            connection.send({"command": "reset", "seed": request["seed"]})
            receive_policy(connection, process)
            measurements, actions, states = [], [], []
            images = {"main_images": [], "wrist_images": []}
            while len(measurements) < args.max_steps:
                observation = policy_observation(raw)
                connection.send({"command": "predict", "observation": observation})
                prediction = receive_policy(connection, process)["actions"]
                for action in prediction[0]:
                    if len(measurements) >= args.max_steps:
                        break
                    observation = policy_observation(raw)
                    states.append(observation["states"][0])
                    for key in images:
                        images[key].append(observation[key][0])
                    raw, _, terminated, truncated, _ = env.step(
                        torch.as_tensor(action[None], device=env.device)
                    )
                    if terminated.any() or truncated.any():
                        raise RuntimeError(
                            "Unexpected automatic reset during fixed-budget evaluation"
                        )
                    actions.append(action)
                    measurements.append(physical_metrics(env))
            report = summarize_episode(measurements) | {
                "request": request,
                "initial_state": initial,
            }
            directory = args.output / f"episode_{index:03d}"
            directory.mkdir()
            for key, frames in images.items():
                write_video(directory / f"{key}.mp4", frames)
            np.savez_compressed(
                directory / "trajectory.npz", states=states, actions=actions
            )
            (directory / "measurements.json").write_text(json.dumps(measurements))
            (directory / "report.json").write_text(json.dumps(report, indent=2))
            reports.append(report)
            summary = aggregate_reports(reports, len(requests)) | {
                "episodes": reports,
                "learned_policy_evaluation": True,
            }
            (args.output / "partial_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    except BaseException:
        (args.output / "failure.json").write_text(
            json.dumps(
                {"completed_episodes": len(reports), "error": traceback.format_exc()},
                indent=2,
            )
        )
        raise
    finally:
        try:
            stop_policy_process(connection, process)
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if app is not None:
                    app.close()


if __name__ == "__main__":
    main()
