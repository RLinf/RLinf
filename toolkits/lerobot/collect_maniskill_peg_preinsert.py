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

"""Collect and independently replay held-peg preinsert demonstrations on GPU."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.utils.logging import get_logger

LOG = get_logger()
DEFAULT_CONFIG = "examples/embodiment/config/env/maniskill_peg_preinsert.yaml"


def expert_action(task: Any) -> torch.Tensor:
    """Align the held peg, then advance its head at most 4 mm per control step."""
    from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles
    from mani_skill.utils.structs.pose import Pose

    head = task.box_hole_pose.inv() * task.peg_head_pose
    peg_hole = task.box_hole_pose.inv() * task.peg.pose
    rotation = matrix_to_euler_angles(
        peg_hole.to_transformation_matrix()[:, :3, :3], "XYZ"
    )
    aligned = (head.p[:, 1:].norm(dim=-1) < 0.002) & (rotation.abs().amax(-1) < 0.02)
    target_head = head.p.clone()
    target_head[:, 1:] = 0
    target_head[:, 0] = torch.minimum(
        head.p[:, 0] + aligned.float() * 0.004,
        torch.full_like(head.p[:, 0], -0.005),
    )
    target_peg = (
        task.box_hole_pose
        * Pose.create_from_pq(target_head)
        * task.peg_head_offsets.inv()
    )
    target_tcp = target_peg * task.peg.pose.inv() * task.agent.tcp.pose
    robot_inv = task.agent.robot.pose.inv()
    target = (robot_inv * target_tcp).to_transformation_matrix()
    current = (robot_inv * task.agent.tcp.pose).to_transformation_matrix()
    delta_pos = (2 * (target[:, :3, 3] - current[:, :3, 3])).clamp(-0.004, 0.004)
    delta_rot = matrix_to_euler_angles(
        target[:, :3, :3] @ current[:, :3, :3].transpose(-1, -2), "XYZ"
    )
    delta_rot = delta_rot / (delta_rot.norm(dim=-1, keepdim=True) / 0.025).clamp_min(1)
    arm = task.agent.controller.controllers["arm"]
    if task.control_mode == "pd_joint_delta_pos":
        from rlinf.envs.sim.maniskill.peg_preinsert import preinsert_kinematics

        qpos = task.agent.robot.qpos
        target_qpos = preinsert_kinematics(task).compute_ik(
            pose=torch.cat((delta_pos, delta_rot), dim=-1),
            q0=qpos,
            is_delta_pose=True,
            current_pose=robot_inv * task.agent.tcp.pose,
            solver_config={"type": "levenberg_marquardt", "alpha": 1.0},
        )
        delta = target_qpos - qpos[:, arm.active_joint_indices]
        action = (
            2
            * (delta - arm.action_space_low)
            / (arm.action_space_high - arm.action_space_low)
            - 1
        )
        return torch.cat((action.clamp(-1, 1), -torch.ones_like(action[:, :1])), -1)
    pos = (
        2
        * (delta_pos - arm.action_space_low[:3])
        / (arm.action_space_high[:3] - arm.action_space_low[:3])
        - 1
    )
    return torch.cat(
        (pos, delta_rot / arm.config.rot_lower, -torch.ones_like(pos[:, :1])), -1
    )


def encode_action(task: Any, action: torch.Tensor) -> np.ndarray:
    """Encode controller commands in the configured SFT representation."""
    if task.control_mode == "pd_joint_delta_pos":
        return action[0].cpu().numpy().copy()
    from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix

    from rlinf.utils.rot6d import matrix_to_rot6d

    scaled = task.agent.controller.controllers["arm"]._clip_and_scale_action(
        action[:, :6]
    )
    rot6d = matrix_to_rot6d(euler_angles_to_matrix(scaled[:, 3:], "XYZ").cpu().numpy())
    return np.concatenate(
        (action[:, :3].cpu().numpy(), rot6d, action[:, -1:].cpu().numpy()), -1
    )[0]


def frame(obs: dict[str, Any], label: np.ndarray) -> dict[str, Any]:
    """Copy the observation before its corresponding action."""
    return {
        "image": obs["main_images"][0].cpu().numpy().copy(),
        "wrist_image": obs["wrist_images"][0].cpu().numpy().copy(),
        "state": obs["states"][0].cpu().numpy().copy(),
        "actions": label.copy(),
        "task": obs["task_descriptions"][0],
    }


def diagnostics(task: Any) -> dict[str, Any]:
    """Measure insertion geometry and physical contact for the replay manifest."""
    head = (task.box_hole_pose.inv() * task.peg_head_pose).p[0]
    return {
        "head_hole_m": head.cpu().tolist(),
        "gap_to_front_m": float(-task.peg_half_sizes[0, 0] - head[0]),
        "grasped": bool(task.agent.is_grasping(task.peg)[0])
        if task.elapsed_steps[0] > 0
        else None,
        "tcp_peg_m": (task.peg.pose.inv() * task.agent.tcp.pose).p[0].cpu().tolist(),
        "success": bool(task.evaluate()["success"][0]),
    }


def video_frame(env: Any, obs: dict[str, Any], seed: int, step: int) -> np.ndarray:
    """Render a side view plus the model's two cameras and geometric diagnostics."""
    from PIL import Image, ImageDraw

    side = env.env.unwrapped.render_rgb_array()[0].cpu().numpy()
    main = obs["main_images"][0].cpu().numpy()
    wrist = obs["wrist_images"][0].cpu().numpy()
    panel = np.concatenate((main, wrist), axis=1)
    canvas = Image.fromarray(np.concatenate((side, panel), axis=0))
    draw = ImageDraw.Draw(canvas)
    d = diagnostics(env.env.unwrapped)
    text = f"seed {seed} | step {step} | gap {d['gap_to_front_m'] * 1000:.1f} mm | success {d['success']}"
    draw.rectangle((0, 0, canvas.width, 24), fill="black")
    draw.text((8, 6), text, fill="white")
    return np.asarray(canvas)


def main() -> None:
    """Write only demonstrations whose saved actions replay successfully."""
    import imageio.v3 as iio

    from rlinf.data.storage.lerobot import add_frame_to_dataset
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv
    from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
        validate_eef_gpu_controller,
    )
    from rlinf.models.embodiment.openpi.policies.maniskill_policy import (
        ManiSkillEEFRot6dOutputs,
        ManiSkillOutputs,
    )
    from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import _create_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=310000)
    parser.add_argument("--render-backend", default=None)
    args = parser.parse_args()
    if args.num_episodes <= 0 or args.max_attempts < args.num_episodes:
        parser.error("Expected 0 < num_episodes <= max_attempts.")
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    (out / "videos").mkdir()
    (out / "traces").mkdir()
    cfg = OmegaConf.load(args.config)
    if cfg.init_params.get("preinsert") is None:
        raise ValueError("Expected init_params.preinsert in the environment config.")
    if args.render_backend:
        cfg.init_params.render_backend = args.render_backend
    OmegaConf.save(cfg, out / "env.yaml")
    torch.set_num_threads(4)
    env = ManiskillRLTEnv(cfg, 1, 0, 1, None, record_metrics=False)
    joint = cfg.wrap_obs_mode == "rlt_openpi_joint"
    state_dim, action_dim = (9, 8) if joint else (10, 10)
    decoder = (
        ManiSkillOutputs(output_action_dim=8) if joint else ManiSkillEEFRot6dOutputs()
    )
    dataset = None
    entries, attempts = [], []
    try:
        task = env.env.unwrapped
        if joint:
            from rlinf.envs.sim.maniskill.peg_preinsert import (
                validate_joint_gpu_controller,
            )

            contract = validate_joint_gpu_controller(env.env)
        else:
            contract = validate_eef_gpu_controller(env.env)
        for seed in range(args.seed, args.seed + args.max_attempts):
            obs, _ = env.reset(seed=seed, options={"reconfigure": True})
            initial = diagnostics(task)
            frames, labels, states = [], [], []
            for _ in range(cfg.max_episode_steps):
                command = expert_action(task)
                label = encode_action(task, command)
                decoded = decoder({"actions": label[None]})["actions"]
                np.testing.assert_allclose(decoded, command.cpu().numpy(), atol=2e-6)
                frames.append(frame(obs, label))
                labels.append(label)
                states.append(obs["states"][0].cpu().numpy().copy())
                obs, _, terminated, truncated, info = env.step(
                    torch.as_tensor(decoded, device=env.device), auto_reset=False
                )
                if info["success"].any() or terminated.any() or truncated.any():
                    break
            collected = bool(info["success"][0])
            attempt = {"seed": seed, "collection_success": collected}
            attempts.append(attempt)
            if not collected:
                LOG.info("Collection failed seed=%s %s", seed, diagnostics(task))
                continue
            states.append(obs["states"][0].cpu().numpy().copy())
            obs, _ = env.reset(seed=seed, options={"reconfigure": True})
            video = [video_frame(env, obs, seed, 0)]
            replay_states, replay_diagnostics = [], [diagnostics(task)]
            for step, label in enumerate(labels, 1):
                replay_states.append(obs["states"][0].cpu().numpy().copy())
                decoded = decoder({"actions": label[None]})["actions"]
                obs, _, _, _, info = env.step(
                    torch.as_tensor(decoded, device=env.device), auto_reset=False
                )
                video.append(video_frame(env, obs, seed, step))
                replay_diagnostics.append(diagnostics(task))
            replay_states.append(obs["states"][0].cpu().numpy().copy())
            error = float(np.max(np.abs(np.array(states) - np.array(replay_states))))
            success = bool(info["success"][0])
            attempt.update(replay_success=success, replay_state_max_error=error)
            if not success or error > 0.002:
                LOG.warning("Replay rejected: %s", attempt)
                continue
            if dataset is None:
                dataset = _create_dataset(
                    repo_id=str(out / "dataset"),
                    image_shape=frames[0]["image"].shape,
                    wrist_image_shape=frames[0]["wrist_image"].shape,
                    fps=cfg.init_params.sim_config.control_freq,
                    image_writer_threads=4,
                    image_writer_processes=0,
                    state_dim=state_dim,
                    action_dim=action_dim,
                )
            for item in frames:
                add_frame_to_dataset(dataset, item)
            dataset.save_episode()
            stem = f"seed_{seed}"
            iio.imwrite(
                out / "videos" / f"{stem}.mp4",
                np.stack(video),
                fps=cfg.init_params.sim_config.control_freq,
            )
            iio.imwrite(out / "videos" / f"{stem}_initial.png", video[0])
            np.savez_compressed(
                out / "traces" / f"{stem}.npz",
                actions=np.stack(labels),
                states=np.stack(states),
                replay_states=np.stack(replay_states),
            )
            entry = {
                **attempt,
                "steps": len(labels),
                "initial": initial,
                "replay": replay_diagnostics,
            }
            entries.append(entry)
            manifest = {
                **contract,
                "config": "env.yaml",
                "episodes": entries,
                "attempts": attempts,
                "success_filtered": True,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "fps": cfg.init_params.sim_config.control_freq,
            }
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            LOG.info("Saved %d/%d: %s", len(entries), args.num_episodes, attempt)
            if len(entries) == args.num_episodes:
                break
        if len(entries) != args.num_episodes:
            raise RuntimeError(
                f"Expected {args.num_episodes} successful replays, got {len(entries)}; attempts={attempts}"
            )
    finally:
        if dataset is not None and dataset.image_writer is not None:
            dataset.image_writer.wait_until_done()
        env.env.close()


if __name__ == "__main__":
    main()
