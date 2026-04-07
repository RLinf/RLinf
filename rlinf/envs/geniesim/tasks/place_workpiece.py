# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
GeneSim RL task: place_workpiece

Right-arm only EE control.  The model receives only right-arm state (26-dim)
and outputs right-arm actions (7-dim).  Internally, the task env maps them
to/from the full SHM layout (52-dim state / 14-dim action).

Reward (dense, per-step):
    All positional reward terms are based on the **workpiece → workspace**
    relative pose.  The target relative pose is a fixed constant derived
    from successful human demonstrations.

    Components:
      r_xy          – always: penalise xy distance to target slot
      r_z_approach  – above pre-place height: penalise excess height
      r_z_insert    – between pre-place and target-z AND xy aligned:
                      positive reward for descending onto the rivet
      r_below       – below target-z: strong penalty (workpiece jammed)
      r_lateral_low – below pre-place but xy not aligned: penalty (rivet collision)
      r_orient      – always: penalise orientation error vs. target quaternion
      r_ee_speed    – EE speed > threshold: penalty (only term using EE state)
      r_still       – near target & workpiece stationary: continuous positive
      r_success     – held near target for N steps: one-time large bonus

    Termination:
        Episodes end via truncation at max_episode_steps.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from rlinf.envs.geniesim import register_geniesim_env
from rlinf.envs.geniesim.geniesim_env import GenieSimBaseEnv

_EE_L_RESET_POS = np.array([0.3407, 0.2927, 0.5810], dtype=np.float32)
_EE_L_RESET_RPY = np.array([3.0708, 0.4881, 3.1416], dtype=np.float32)
_EE_R_RESET_POS = np.array([0.4833, 0.0051, 1.2548], dtype=np.float32)
_EE_R_RESET_RPY = np.array([2.5633, 0.0261, 1.5791], dtype=np.float32)

_POS_SCALE = np.float32(0.015)
_RPY_SCALE = np.float32(0.05)

_POS_HALF_RANGE = 0.15
_RPY_HALF_RANGE = 0.2

_SAFETY_LOW = np.concatenate([
    _EE_R_RESET_POS - _POS_HALF_RANGE,
    _EE_R_RESET_RPY - _RPY_HALF_RANGE,
]).astype(np.float32)

_SAFETY_HIGH = np.concatenate([
    _EE_R_RESET_POS + _POS_HALF_RANGE,
    _EE_R_RESET_RPY + _RPY_HALF_RANGE,
]).astype(np.float32)

# Full-SHM state layout (52-dim):
#   [0:7]   arm_l joint pos      [7:14]  arm_r joint pos
#   [14:21] arm_l joint vel      [21:28] arm_r joint vel
#   [28:31] ee_l pos             [31:34] ee_l rpy
#   [34:37] ee_l lin_vel         [37:40] ee_l ang_vel
#   [40:43] ee_r pos             [43:46] ee_r rpy
#   [46:49] ee_r lin_vel         [49:52] ee_r ang_vel
#
# Right-arm EE-only state (12-dim):
#   [0:3]   ee_r pos              <- full[40:43]
#   [3:6]   ee_r rpy              <- full[43:46]
#   [6:9]   ee_r lin_vel          <- full[46:49]
#   [9:12]  ee_r ang_vel          <- full[49:52]
_STATE_INDICES = np.arange(40, 52).astype(np.intp)

_TARGET_REL_POS = np.array([-0.073, 0.007, 1.185], dtype=np.float32)
_TARGET_WP_QUAT = np.array([0.1807, 0.6802, 0.6847, 0.1896], dtype=np.float32)
_TARGET_WP_QUAT /= np.linalg.norm(_TARGET_WP_QUAT)

_RIVET_HEIGHT = 0.01
_PRE_PLACE_DZ = _TARGET_REL_POS[2] + _RIVET_HEIGHT

_STILL_SPEED_THRESH = 0.02
_STILL_STEPS_REQUIRED = 5
_XY_TOLERANCE = 0.02
_Z_TOLERANCE = 0.01
_ORIENT_TOLERANCE = 0.15

_TERM_XY_DIST = 0.12
_TERM_Z_DROP = -0.04
_TERM_Z_HIGH = 0.15
_TERM_ORIENT_DIFF = 2.50
_TERM_EE_SPEED = 5.0

_IDLE_DIST_THRESH = 0.002
_IDLE_STEPS_LIMIT = 100


def _quat_angle_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1_n = q1 / (torch.linalg.norm(q1, dim=-1, keepdim=True) + 1e-8)
    q2_n = q2 / (torch.linalg.norm(q2, dim=-1, keepdim=True) + 1e-8)
    dot = (q1_n * q2_n).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


@register_geniesim_env("geniesim_place_workpiece")
class PlaceWorkpieceEnv(GenieSimBaseEnv):
    """
    RLinf RL env for the GenieSim pick-and-place task (right-arm only).

    Observation:
        images: right wrist camera (wrist_images, via wrist_cam_prim config)
        states (12-dim): EE pose + EE velocity

    Action (7-dim): [pos_r(3), rpy_r(3), grip_r(1)]

    Safety box: ±0.15 m (xyz), ±0.2 rad (rpy) around reset EE pose.
    """

    spacemouse_wrapper_kwargs = dict(
        action_dim=7,
        r_pos_action=slice(0, 3),
        r_rot_action=slice(3, 6),
        r_grip_action=6,
        s_r_ee_pos=slice(0, 3),
        s_r_ee_rot=slice(3, 6),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._still_counter = np.zeros(self.num_envs, dtype=np.int32)
        self._prev_wp_pos = None
        self._step_counter = np.zeros(self.num_envs, dtype=np.int32)
        self._idle_counter = np.zeros(self.num_envs, dtype=np.int32)
        self._prev_ee_pos = None
        self._custom_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._last_extracted_states = np.zeros(
            (self.num_envs, len(_STATE_INDICES)), dtype=np.float32,
        )
        self._ee_target = np.tile(
            np.concatenate([_EE_R_RESET_POS, _EE_R_RESET_RPY]),
            (self.num_envs, 1),
        ).astype(np.float32)

    _OUT_SIZE = 128
    _CROP_SIZE = 256

    @staticmethod
    def _center_crop(img, crop_h, crop_w):
        _, _, H, W = img.shape
        y0 = (H - crop_h) // 2
        x0 = (W - crop_w) // 2
        return img[:, :, y0:y0 + crop_h, x0:x0 + crop_w]

    def _extract_images(self, obs_dict: dict) -> dict:
        main = obs_dict.get("main_images")
        if main is not None:
            if main.dim() == 5:
                main = main[:, 0]
            main = main.permute(0, 3, 1, 2).float()
            main = self._center_crop(main, self._CROP_SIZE, self._CROP_SIZE)
            main = F.interpolate(main, size=(self._OUT_SIZE, self._OUT_SIZE), mode="bilinear", align_corners=False)
            main = main.permute(0, 2, 3, 1).to(torch.uint8)
            obs_dict["main_images"] = main
        extra = obs_dict.get("extra_view_images")
        if extra is not None:
            B, N, H, W, C = extra.shape
            flat = extra.reshape(B * N, H, W, C).permute(0, 3, 1, 2).float()
            flat = self._center_crop(flat, self._CROP_SIZE, self._CROP_SIZE)
            flat = F.interpolate(flat, size=(self._OUT_SIZE, self._OUT_SIZE), mode="bilinear", align_corners=False)
            flat = flat.permute(0, 2, 3, 1).to(torch.uint8)
            obs_dict["extra_view_images"] = flat.reshape(B, N, self._OUT_SIZE, self._OUT_SIZE, C)
        return obs_dict

    def _extract_states(self, states: np.ndarray) -> np.ndarray:
        extracted = states[..., _STATE_INDICES]
        self._last_extracted_states = extracted
        return extracted

    def _expand_actions(self, actions: np.ndarray) -> np.ndarray:
        n = actions.shape[0]
        full = np.zeros((n, 14), dtype=np.float32)
        full[:, 0:3] = _EE_L_RESET_POS
        full[:, 3:6] = _EE_L_RESET_RPY
        full[:, 6:9] = actions[:, 0:3]
        full[:, 9:12] = actions[:, 3:6]
        full[:, 12] = 0.0
        full[:, 13] = actions[:, 6]
        return full

    def _randomized_reset_ee(self, n: int = 1) -> np.ndarray:
        base = np.concatenate([_EE_R_RESET_POS, _EE_R_RESET_RPY])
        noise = np.zeros((n, 6), dtype=np.float32)
        noise[:, :3] = np.random.uniform(-0.015, 0.015, size=(n, 3))
        noise[:, 3:] = np.random.uniform(-0.05, 0.05, size=(n, 3))
        return self._clip_safety_box(base[None, :] + noise)

    def reset(self, seed=None, env_ids=None, **kwargs):
        obs, info = super().reset(seed=seed, env_ids=env_ids, **kwargs)
        if env_ids is None:
            n = self.num_envs
            self._still_counter[:] = 0
            self._prev_wp_pos = None
            self._step_counter[:] = 0
            self._idle_counter[:] = 0
            self._prev_ee_pos = None
            self._custom_returns[:] = 0.0
            self._ee_target[:] = self._randomized_reset_ee(n)
        else:
            ids = np.asarray(env_ids)
            self._still_counter[ids] = 0
            self._step_counter[ids] = 0
            self._idle_counter[ids] = 0
            # Reset prev positions to NaN so the first step of the new episode
            # does not inherit stale terminal-state values.  NaN propagates
            # safely: displacement/speed will be NaN → comparison is False →
            # idle_counter and still_counter stay at 0.
            if self._prev_ee_pos is not None:
                self._prev_ee_pos[ids, :] = np.nan
            if self._prev_wp_pos is not None:
                self._prev_wp_pos[ids, :] = float("nan")
            self._custom_returns[ids] = 0.0
            self._ee_target[ids] = self._randomized_reset_ee(len(ids))
        return obs, info

    @staticmethod
    def _clip_safety_box(abs_pose: np.ndarray) -> np.ndarray:
        return np.clip(abs_pose, _SAFETY_LOW, _SAFETY_HIGH)

    def _delta_to_absolute(self, delta: np.ndarray, _unused=None) -> np.ndarray:
        delta = np.clip(delta, -1.0, 1.0)
        self._ee_target[:, 0:3] += delta[:, 0:3] * _POS_SCALE
        self._ee_target[:, 3:6] += delta[:, 3:6] * _RPY_SCALE
        self._ee_target[:] = self._clip_safety_box(self._ee_target)
        return np.concatenate([self._ee_target.copy(), delta[:, 6:7]], axis=-1)

    def step(self, actions, auto_reset: bool = True):
        if isinstance(actions, torch.Tensor):
            a = actions.cpu().numpy()
        else:
            a = np.asarray(actions, dtype=np.float32)

        abs_actions = self._delta_to_absolute(a, None)

        obs, _rewards, terminated, truncated, infos = super().step(
            abs_actions, auto_reset=False
        )
        self._step_counter += 1
        states = obs["states"] if isinstance(obs, dict) else obs
        rewards = self._compute_reward(infos, states)
        self._custom_returns += rewards.numpy()

        if "episode" in infos:
            infos["episode"]["return"] = torch.from_numpy(
                self._custom_returns.copy()
            )
            infos["episode"]["reward"] = torch.from_numpy(
                np.where(
                    self._elapsed_steps > 0,
                    self._custom_returns / np.maximum(self._elapsed_steps, 1),
                    0.0,
                ).astype(np.float32)
            )

        infos["reward_detail"] = self._last_reward_detail

        n = self.num_envs

        term = torch.zeros(n, dtype=torch.bool)

        success_mask = torch.from_numpy(self._still_counter >= _STILL_STEPS_REQUIRED)

        drop_mask = self._last_dist_z < _TERM_Z_DROP
        term = term | drop_mask

        z_high_mask = self._last_dist_z > _TERM_Z_HIGH
        term = term | z_high_mask

        xy_far_mask = self._last_dist_xy > _TERM_XY_DIST
        term = term | xy_far_mask

        orient_bad_mask = self._last_orient_diff > _TERM_ORIENT_DIFF
        term = term | orient_bad_mask

        if isinstance(states, torch.Tensor):
            ee_lin_vel = states[:, 6:9]
            ee_pos_t = states[:, 0:3]
        else:
            s_t = torch.from_numpy(np.asarray(states))
            ee_lin_vel = s_t[:, 6:9]
            ee_pos_t = s_t[:, 0:3]
        ee_speed = torch.linalg.norm(ee_lin_vel, dim=-1)
        speed_mask = ee_speed > _TERM_EE_SPEED
        term = term | speed_mask

        ee_pos_np = ee_pos_t.detach().cpu().numpy()
        if self._prev_ee_pos is not None:
            ee_displacement = np.linalg.norm(ee_pos_np - self._prev_ee_pos, axis=-1)
            not_near_target = (self._last_dist_xy > _XY_TOLERANCE).numpy() | (self._last_dist_z.abs() > _Z_TOLERANCE).numpy()
            idle_inc = (ee_displacement < _IDLE_DIST_THRESH) & not_near_target
            self._idle_counter[idle_inc] += 1
            self._idle_counter[~idle_inc] = 0
        self._prev_ee_pos = ee_pos_np.copy()

        idle_mask = torch.from_numpy(self._idle_counter >= _IDLE_STEPS_LIMIT)
        term = term | idle_mask

        warmup = torch.from_numpy(self._step_counter < 150)
        term = term & ~warmup

        if not getattr(self, "_collecting", False):
            term = term | success_mask

        for i in range(n):
            if term[i] and not terminated[i]:
                reasons = []
                if success_mask[i]:
                    reasons.append("success")
                if drop_mask[i]:
                    reasons.append(f"drop(dz={self._last_dist_z[i]:.4f})")
                if z_high_mask[i]:
                    reasons.append(f"z_high(dz={self._last_dist_z[i]:.4f})")
                if xy_far_mask[i]:
                    reasons.append(f"xy_far(d={self._last_dist_xy[i]:.4f})")
                if orient_bad_mask[i]:
                    reasons.append(f"orient(diff={self._last_orient_diff[i]:.4f}rad)")
                if speed_mask[i]:
                    reasons.append(f"ee_speed({ee_speed[i]:.4f}m/s)")
                if idle_mask[i]:
                    reasons.append(f"idle({self._idle_counter[i]}steps)")
                rd = self._last_reward_detail
                logger.debug(
                    "[TERM] env_%d (step %d): %s | "
                    "r_alive=%.4f r_below=%.4f r_success=%.4f "
                    "d3d=%.4f dxy=%.4f dz=%.4f odiff=%.4f",
                    i, self._step_counter[i], ", ".join(reasons),
                    rd["r_alive"][i], rd["r_below"][i], rd["r_success"][i],
                    rd["dist_3d"][i], rd["dist_xy"][i],
                    rd["diff_z"][i], rd["orient_diff"][i],
                )

        terminated = terminated | term

        fail_term = term & ~success_mask
        rewards[success_mask & term] = 5.0

        done = terminated | truncated
        if done.any():
            done_np = done.numpy() if isinstance(done, torch.Tensor) else np.asarray(done)
            self._step_counter[done_np] = 0
            self._still_counter[done_np] = 0
            self._idle_counter[done_np] = 0
            self._ee_target[done_np] = self._randomized_reset_ee(int(done_np.sum()))

        if done.any() and auto_reset and self.auto_reset:
            if "episode" not in infos:
                infos = self._record_metrics(rewards, terminated, infos)
            obs, infos = self._handle_auto_reset(done, obs, infos)

        return obs, rewards, terminated, truncated, infos

    def _compute_reward(self, infos, states) -> torch.Tensor:
        n = self.num_envs
        body_poses = infos.get("body_poses")
        if body_poses is None:
            self._last_dist_xy = torch.zeros(n)
            self._last_dist_z = torch.zeros(n)
            self._last_orient_diff = torch.zeros(n)
            return torch.full((n,), 0.0, dtype=torch.float32)

        wp = body_poses.get("workpiece_r")
        ws = body_poses.get("/World/workspace01")
        if wp is None or ws is None:
            self._last_dist_xy = torch.zeros(n)
            self._last_dist_z = torch.zeros(n)
            self._last_orient_diff = torch.zeros(n)
            return torch.full((n,), 0.0, dtype=torch.float32)

        wp_pos = torch.from_numpy(wp[:, :3].copy())
        wp_quat = torch.from_numpy(wp[:, 3:7].copy())
        ws_pos = torch.from_numpy(ws[:, :3].copy())

        target_rel = torch.from_numpy(_TARGET_REL_POS).unsqueeze(0)
        target_quat = torch.from_numpy(_TARGET_WP_QUAT).unsqueeze(0)

        rel_pos = wp_pos - ws_pos
        target_xy = target_rel[:, :2]
        target_z = target_rel[:, 2]

        dist_xy = torch.linalg.norm(rel_pos[:, :2] - target_xy, dim=-1)
        diff_z = rel_pos[:, 2] - target_z
        dist_3d = torch.sqrt(dist_xy ** 2 + diff_z ** 2)

        orient_diff = _quat_angle_diff(wp_quat, target_quat)

        r_alive = 5.0 * torch.exp(-10.0 * dist_3d) * torch.exp(-5.0 * orient_diff)

        overshoot = torch.clamp(-diff_z - 0.01, min=0.0)
        r_below = -20.0 * overshoot

        if self._prev_wp_pos is not None:
            wp_speed = torch.linalg.norm(wp_pos - self._prev_wp_pos, dim=-1) * 30.0
        else:
            wp_speed = torch.zeros(n)
        self._prev_wp_pos = wp_pos.clone()

        xy_aligned = dist_xy < _XY_TOLERANCE
        orient_ok = orient_diff < _ORIENT_TOLERANCE
        near_target = xy_aligned & (diff_z.abs() < _Z_TOLERANCE) & orient_ok
        still_ok = wp_speed < _STILL_SPEED_THRESH

        for i in range(n):
            if near_target[i] and still_ok[i]:
                self._still_counter[i] += 1
            else:
                self._still_counter[i] = 0

        success = torch.from_numpy(self._still_counter >= _STILL_STEPS_REQUIRED)
        just_succeeded = torch.from_numpy(self._still_counter == _STILL_STEPS_REQUIRED)
        r_success = torch.where(just_succeeded, 10.0, 0.0)

        reward = r_alive + r_below + r_success

        self._last_dist_xy = dist_xy
        self._last_dist_z = diff_z
        self._last_orient_diff = orient_diff
        self._last_reward_detail = {
            "r_alive": r_alive,
            "r_below": r_below,
            "r_success": r_success,
            "dist_3d": dist_3d,
            "dist_xy": dist_xy,
            "diff_z": diff_z,
            "orient_diff": orient_diff,
        }

        return reward
