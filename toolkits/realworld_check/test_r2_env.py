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
import os
import signal
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rlinf.robotics.parts.arms.r2 import (
    R2_LEFT_ARM_LIMITS_HIGH,
    R2_LEFT_ARM_LIMITS_LOW,
    R2_LEFT_HOME_QPOS,
    R2_LEFT_READY_QPOS,
    R2_NECK_LIMITS_HIGH,
    R2_NECK_LIMITS_LOW,
    R2_RIGHT_ARM_LIMITS_HIGH,
    R2_RIGHT_ARM_LIMITS_LOW,
    R2_RIGHT_HOME_QPOS,
    R2_RIGHT_READY_QPOS,
    R2_WAIST_LIMITS_HIGH,
    R2_WAIST_LIMITS_LOW,
    compute_r2_arm_cartesian_step,
    compute_r2_arm_fk_and_jacobian,
)

HELP = """
  === Arm 3D Cartesian Control (moves hand in base frame, default 3 cm) ===
  l|r <up|down|fwd|back|left|right> [cm]    e.g. "r fwd 4", "r up 4", "l left 3"

  === Arm Intuitive Joint Control (symmetric left/right, default --step deg) ===
  l|r <raise|lower> [mult]                  raise/lower shoulder forward/back (J1)
  l|r <out|in> [mult]                       open/close shoulder outward/inward (J2)
  l|r <twist_out|twist_in> [mult]           rotate upper arm outward/inward (J3)
  l|r <bend|extend> [mult]                  bend/straighten elbow (J4)
  l|r <wrist_up|wrist_down> [mult]          pitch wrist up/down (J6)
  l|r <roll_cw|roll_ccw> [mult]             roll wrist clockwise/counter-clockwise (J7)
  left|right <1..7> [mult]                  raw joint step (J1..J7)

  === Gripper Control ===
  open / close                              open or close both grippers
  l|r <open|close>                          open or close left or right gripper only

  === Waist Control (5-DoF lower-body & torso) ===
  waist <down|up> [mult]                    squat down / stand up (keeps torso upright)
  waist <fwd|back> [mult]                   bow torso forward / lean backward (HipPitch)
  waist <left|right> [mult]                 lean torso left / right (HipRoll)
  waist <turn_left|turn_right> [mult]       twist torso yaw left / right (HipYaw)
  waist <1..5> [mult]                       step waist joint 1..5 directly
  waist home                                return waist to upright zero pose

  === Head / Neck Control (2-DoF) ===
  head <left|right|up|down|home> [mult]     turn or pitch head, or return to zero

  === Mobile Base Control (omnidirectional wheeled chassis) ===
  base <fwd|back|left|right> [mult] [spd]   drive/strafe chassis (mult scales distance/time)
  base <turn_left|turn_right> [mult] [spd]  rotate chassis (mult scales angle/time)
  base stop                                 stop chassis immediately

  === VR Teleoperation (PICO) ===
  pico [zmq_addr]                           run PICO VR teleoperation loop until Ctrl-C
                                            (grip: arms, trigger/X/Y/A/B: grippers, thumbsticks: base)

  === Postures, Diagnostics & Status ===
  ready / pose ready                        move active arms to bent-elbow ready posture
  home  / pose home                         return arms, waist, and head to home posture
  where                                     print joints, TCP xyz, grippers, waist, head, base
  diag                                      print hardware fault and health diagnosis
  recover                                   clear faults, re-enable body axes, and restore ServoJ
  help / quit
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


def _get_connection_host(env: Any) -> Any:
    if getattr(env, "robot", None) is None:
        return None
    arm = env.robot.child(env.config.active_arms[0]).child("arm")
    return getattr(arm, "_host", getattr(arm, "owner", None))


def _fmt_vec(arr: Any, decimals: int = 1) -> list[float]:
    return np.round(np.asarray(arr, dtype=np.float64), decimals).tolist()


def _print_status(env: Any, obs: dict[str, Any], host: Any) -> None:
    measured = np.asarray(obs["state"]["arm_joint_position"], dtype=np.float64)
    for idx, side in enumerate(env.config.active_arms):
        q_arm = measured[idx * 7 : (idx + 1) * 7]
        tcp = obs["state"].get("tcp_pose")
        if tcp is not None and len(tcp) >= (idx + 1) * 6:
            xyz = np.asarray(tcp[idx * 6 : idx * 6 + 3], dtype=np.float64)
        else:
            xyz, _, _, _ = compute_r2_arm_fk_and_jacobian(side, q_arm)
        print(
            f"  {side:5s} arm (deg): {_fmt_vec(np.rad2deg(q_arm), 1)} "
            f"| tcp xyz (m): {_fmt_vec(xyz, 3)}"
        )
    if "gripper_position" in obs["state"]:
        print("  end_effector:   ", _fmt_vec(obs["state"]["gripper_position"], 2))
    if host is not None:
        try:
            rstate = host.get_state()
            print(
                "  waist (deg):    ",
                _fmt_vec(np.rad2deg(rstate.waist_positions), 1),
                "| head (deg):",
                _fmt_vec(np.rad2deg(rstate.neck_positions), 1),
            )
            print(
                "  base pose:      ",
                _fmt_vec(rstate.base_pose, 3),
                "| base vel:",
                _fmt_vec(rstate.base_velocity, 3),
            )
        except Exception:
            pass
    frames = obs.get("frames")
    if isinstance(frames, dict) and frames:
        cam_summary = ", ".join(
            f"{k}: {tuple(v.shape)} (mean={float(np.mean(v)):.1f})"
            for k, v in frames.items()
            if isinstance(v, np.ndarray)
        )
        if cam_summary:
            print(f"  cameras:         {cam_summary}")


def _smooth_alpha(step_i: int, n_interp: int, *, smooth: bool) -> float:
    progress = min(1.0, float(step_i) / float(max(n_interp, 1)))
    if not smooth:
        return progress
    return 0.5 * (1.0 - float(np.cos(np.pi * progress)))


def _build_env_action(env: Any, q_joints: np.ndarray, grips: np.ndarray) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    offset = 0
    for idx, _side in enumerate(env.config.active_arms):
        q_slice = slice(idx * 7, (idx + 1) * 7)
        action[offset : offset + 7] = q_joints[q_slice]
        offset += 7
        if env._ee_dim > 0:
            g_val = float(grips[idx]) if idx < grips.size else float(grips[0])
            action[offset : offset + env._ee_dim] = g_val
            offset += env._ee_dim
    return action


def _step_arm_smooth(
    env: Any,
    current_joints: np.ndarray,
    target_joints: np.ndarray,
    grips: np.ndarray,
    max_delta_rad: float,
    *,
    gripper_changed: bool = False,
) -> dict[str, Any]:
    diff = target_joints - current_joints
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    is_real_rate = not env.config.is_dummy and float(env.config.step_frequency) < 1000.0
    host = _get_connection_host(env) if is_real_rate else None

    if is_real_rate and host is not None:
        active_arms = tuple(env.config.active_arms)
        l_target = (
            target_joints[
                active_arms.index("left") * 7 : (active_arms.index("left") + 1) * 7
            ]
            if "left" in active_arms
            else None
        )
        r_target = (
            target_joints[
                active_arms.index("right") * 7 : (active_arms.index("right") + 1) * 7
            ]
            if "right" in active_arms
            else None
        )
        l_grip = (
            float(grips[active_arms.index("left")])
            if (gripper_changed and env._ee_dim == 1 and "left" in active_arms)
            else None
        )
        r_grip = (
            float(
                grips[
                    active_arms.index("right")
                    if active_arms.index("right") < grips.size
                    else 0
                ]
            )
            if (gripper_changed and env._ee_dim == 1 and "right" in active_arms)
            else None
        )
        host.move_body_smooth(
            left_arm=l_target,
            right_arm=r_target,
            left_gripper=l_grip,
            right_gripper=r_grip,
            max_velocity=0.15,
        )
        return env._get_observation()

    step_cap = (
        min(max(max_delta_rad, 0.01), 0.018)
        if is_real_rate
        else max(max_delta_rad, 0.03)
    )
    n_interp = max(1, int(np.ceil(max_abs / step_cap)))
    hold_steps = (30 if gripper_changed else 5) if is_real_rate else 0
    total_steps = n_interp + hold_steps
    for step_i in range(1, total_steps + 1):
        alpha = _smooth_alpha(step_i, n_interp, smooth=is_real_rate)
        q_interp = current_joints + alpha * diff
        env.step(_build_env_action(env, q_interp, grips))
    return env._get_observation()


def _apply_arm_command(
    side: str,
    sub_words: Sequence[str],
    q_arm: np.ndarray,
    q_waist: np.ndarray,
    step_rad: float,
) -> Optional[np.ndarray]:
    if not sub_words:
        return None
    op = sub_words[0].lower()
    arg = float(sub_words[1]) if len(sub_words) >= 2 else None
    lo = R2_LEFT_ARM_LIMITS_LOW if side == "left" else R2_RIGHT_ARM_LIMITS_LOW
    hi = R2_LEFT_ARM_LIMITS_HIGH if side == "left" else R2_RIGHT_ARM_LIMITS_HIGH

    cartesian_dirs: dict[str, np.ndarray] = {
        "fwd": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "forward": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "back": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "backward": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "left": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "right": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "down": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    if op in cartesian_dirs:
        cm = 3.0 if arg is None else float(arg)
        delta_xyz = cartesian_dirs[op] * float(cm * 0.01)
        return compute_r2_arm_cartesian_step(
            side, q_arm, delta_xyz, q_waist=q_waist, max_step_rad=0.35
        )

    mult = 1.0 if arg is None else float(arg)
    delta = float(mult * step_rad)
    q_new = q_arm.copy()
    out_sign = 1.0 if side == "left" else -1.0
    bend_sign = -1.0 if side == "left" else 1.0

    semantic_map: dict[str, tuple[int, float]] = {
        "raise": (0, -1.0),
        "lower": (0, 1.0),
        "out": (1, out_sign),
        "in": (1, -out_sign),
        "twist_out": (2, 1.0),
        "twist_in": (2, -1.0),
        "bend": (3, bend_sign),
        "extend": (3, -bend_sign),
        "wrist_up": (5, -1.0),
        "wrist_down": (5, 1.0),
        "roll_cw": (6, 1.0),
        "roll_ccw": (6, -1.0),
    }
    if op in semantic_map:
        j_idx, sign = semantic_map[op]
        q_new[j_idx] = float(np.clip(q_new[j_idx] + sign * delta, lo[j_idx], hi[j_idx]))
        return q_new

    try:
        joint_idx = int(op) - 1
    except ValueError:
        return None
    if not (0 <= joint_idx < 7):
        print("Joint index must be 1..7")
        return None
    q_new[joint_idx] = float(
        np.clip(q_new[joint_idx] + delta, lo[joint_idx], hi[joint_idx])
    )
    return q_new


def _deadzone_curve(val: float, deadzone: float = 0.15) -> float:
    mag = abs(float(val))
    if mag <= deadzone:
        return 0.0
    scaled = min(1.0, (mag - deadzone) / max(1e-6, 1.0 - deadzone))
    return float(np.copysign(scaled * scaled, val))


def _pico_thumbstick_base_velocity(
    readings: Sequence[Mapping[str, Any]],
    prev_vel: np.ndarray,
    *,
    max_vx: float = 0.25,
    max_vy: float = 0.20,
    max_wz: float = 0.45,
) -> np.ndarray:
    """Map PICO left/right controller thumbsticks (``axis``) to planar base ``[vx, vy, wz]``."""
    lx, ly = 0.0, 0.0
    rx, ry = 0.0, 0.0
    for r in readings:
        if not r.get("ready", False):
            continue
        axis = r.get("axis", (0.0, 0.0))
        if axis is None or len(axis) < 2:
            continue
        hand = str(r.get("hand", "")).lower()
        if hand == "left":
            lx, ly = _deadzone_curve(axis[0]), _deadzone_curve(axis[1])
        elif hand == "right":
            rx, ry = _deadzone_curve(axis[0]), _deadzone_curve(axis[1])

    vx = (ly if abs(ly) > 1e-4 else ry) * max_vx
    vy = -lx * max_vy
    wz = -rx * max_wz
    target = np.array([vx, vy, wz], dtype=np.float32)
    smoothed = 0.4 * np.asarray(prev_vel, dtype=np.float32) + 0.6 * target
    smoothed[np.abs(smoothed) < 1e-3] = 0.0
    return smoothed


def _run_pico_teleop(
    env: Any,
    *,
    zmq_addr: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Run a live PICO VR teleoperation loop against ``env``."""
    from rlinf.envs.real.wrappers.teleop.builder import build_teleop

    addr = zmq_addr or os.environ.get("R2_PICO_ZMQ_ADDR", "ipc:///tmp/vr_data.ipc")
    active_arms = tuple(env.config.active_arms)
    devices: list[Any] = (
        [{"pico": {"drives": side}} for side in active_arms]
        if len(active_arms) > 1
        else ["pico"]
    )
    has_gripper = getattr(env, "_ee_type", "") == "gripper"
    orig_freq = float(env.config.step_frequency)
    orig_step_limits = tuple(env.config.joint_step_limits)
    is_live_robot = not env.config.is_dummy and orig_freq < 1000.0
    if is_live_robot:
        env.config.step_frequency = max(orig_freq, 50.0)
        env.config.joint_step_limits = (
            0.045,
            0.045,
            0.045,
            0.045,
            0.05,
            0.05,
            0.05,
        )

    pico_cfg: dict[str, Any] = {
        "zmq_addr": addr,
        "control_trigger": "grip",
        "control_threshold": 0.85,
        "gripper_trigger": "trigger",
        "gripper_close_threshold": 0.5,
        "position_scale": float(os.environ.get("R2_PICO_SCALE", "1.0")),
        "rotation_scale": float(os.environ.get("R2_PICO_ROT_SCALE", "0.85")),
        "rotation_delta_frame": "operator",
        "ik_backtrack_attempts": 3,
        "max_tick_s": 0.25,
        "max_stale_s": 0.3,
        "calibration": {
            "enabled": True,
            "auto_calibrate_on_start": bool(
                int(os.environ.get("R2_PICO_AUTO_CALIBRATE", "0"))
            ),
            "auto_calibrate_on_grip": bool(
                int(os.environ.get("R2_PICO_CALIBRATE_ON_GRIP", "0"))
            ),
            "required": False,
            "button": "menu_button",
        },
        "left": {
            "gripper_close_button": "X",
            "gripper_open_button": "Y",
        },
        "right": {
            "gripper_close_button": "A",
            "gripper_open_button": "B",
        },
    }

    teleop = build_teleop(
        env,
        {
            "no_gripper": not has_gripper,
            "pico": pico_cfg,
        },
        devices,
    )
    # Allow left and right controllers to operate independently even if one
    # controller is resting on the desk or temporarily out of camera view.
    teleop.interlock = None
    host = _get_connection_host(env)
    enable_base = bool(getattr(env.hardware, "enable_base", False))
    base_vel = np.zeros(3, dtype=np.float32)
    try:
        if not env.config.is_dummy:
            if host is not None:
                with contextlib.suppress(Exception):
                    if not host.diagnose().healthy:
                        host.clear_errors()
            env._get_observation()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if all(entry.device._device.ready for entry in teleop.group.entries):
                break
            time.sleep(0.01)
        teleop.reset(env)
        policy_action = env.get_joint_positions()
        # Clear the initial release_grip latch if controllers are already streaming idle packets.
        teleop.read(env, policy_action)
        readings = [entry.device._device.last_reading for entry in teleop.group.entries]
        stream_ready = any(r.get("ready", False) for r in readings)
        print(
            f"  PICO teleoperation active on {addr} "
            f"(arms={active_arms}, gripper={has_gripper}, base={enable_base}, "
            f"rate={env.config.step_frequency:.0f}Hz, stream_ready={stream_ready})."
        )
        if not stream_ready:
            print(
                "  [WARN] No ZeroMQ VR data received yet on "
                f"{addr}. Ensure start_pico_service.sh is running."
            )
        else:
            for r in readings:
                hand = r.get("hand", "?")
                valid = not r.get("invalid_pose", False)
                print(
                    f"  PICO {hand} controller: tracking={'OK' if valid else 'IDLE/UNTRACKED (wake controller)'}, "
                    f"grip={r.get('control_value', 0.0):.2f}, trigger={r.get('trigger', 0.0):.2f}"
                )
        print(
            "  Controls:\n"
            "    - Side grip (squeeze & hold): move left/right arm\n"
            "    - Index trigger (squeeze=close, release=open) or X/Y (left) & A/B (right): control grippers\n"
            "    - Left thumbstick: drive chassis forward/back & strafe left/right\n"
            "    - Right thumbstick: turn chassis left/right (and drive fwd/back when left stick is idle)\n"
            "    - Menu button: recalibrate headset yaw | Ctrl-C: stop"
        )
        step_idx = 0
        while max_steps is None or step_idx < max_steps:
            sample = teleop.read(env, policy_action)
            applied = (
                sample.action.astype(np.float32)
                if sample.action is not None
                else policy_action.copy()
            )
            if enable_base and applied.size >= 3:
                live_readings = [
                    entry.device._device.last_reading for entry in teleop.group.entries
                ]
                base_vel = _pico_thumbstick_base_velocity(live_readings, base_vel)
                applied[-3:] = base_vel
            try:
                _, _, _, _, step_info = env.step(applied)
            except RuntimeError as exc:
                print(
                    f"\n  [WARN] Hardware fault during PICO teleop ({exc}); "
                    "recovering and holding arms. Release and re-squeeze grip to continue."
                )
                if host is not None:
                    with contextlib.suppress(Exception):
                        host.recover(trigger="pico_teleop_fault")
                env._get_observation()
                policy_action = env.get_joint_positions()
                for entry in teleop.group.entries:
                    entry.device.hold_until_released("fault_recovery")
                step_idx += 1
                continue
            if step_info.get("robot_fault"):
                print(
                    f"\n  [WARN] Arm fault recovered ({step_info['robot_fault']}); "
                    "holding arms. Release and re-squeeze grip to continue."
                )
                env._get_observation()
                policy_action = env.get_joint_positions()
                for entry in teleop.group.entries:
                    entry.device.hold_until_released("fault_recovery")
            else:
                policy_action = applied.copy()
            step_idx += 1
        last_obs = env._get_observation()
    except KeyboardInterrupt:
        # ROS 2 can deliver one Ctrl-C twice; a second one must not cut the
        # teardown short.
        with _deferred_interrupts():
            print("\n  PICO teleoperation stopped.")
            last_obs = env._get_observation()
    finally:
        with _deferred_interrupts():
            if enable_base and host is not None:
                try:
                    host.stop_base()
                except Exception:
                    pass
            env.config.step_frequency = orig_freq
            env.config.joint_step_limits = orig_step_limits
            teleop.close()
    return last_obs


def _run_interactive(
    env: Any,
    *,
    step_rad: float,
    reset: bool,
    commands: Optional[Sequence[str]] = None,
    pico_zmq_addr: Optional[str] = None,
) -> None:
    if reset:
        env.reset()
    host = _get_connection_host(env)
    obs = env._get_observation()
    joints = np.array(obs["state"]["arm_joint_position"], dtype=np.float32)
    num_arms = len(env.config.active_arms)
    if (
        "gripper_position" in obs["state"]
        and len(obs["state"]["gripper_position"]) >= num_arms
    ):
        grips = np.array(obs["state"]["gripper_position"][:num_arms], dtype=np.float32)
    else:
        grips = np.ones(num_arms, dtype=np.float32)

    if host is not None:
        rstate = host.get_state()
        waist_q = np.array(rstate.waist_positions, dtype=np.float32)
        neck_q = np.array(rstate.neck_positions, dtype=np.float32)
    else:
        waist_q = np.zeros(5, dtype=np.float32)
        neck_q = np.zeros(2, dtype=np.float32)

    if not commands:
        print(HELP)
    cmd_queue = list(commands) if commands else None

    while True:
        if cmd_queue is not None:
            if not cmd_queue:
                return
            raw_line = cmd_queue.pop(0).strip()
            if not raw_line:
                continue
            print(f"r2> {raw_line}")
            words = raw_line.split()
        else:
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
        if cmd in ("where", "status"):
            obs = env._get_observation()
            _print_status(env, obs, host)
            continue
        if cmd in ("pico", "teleop", "vr"):
            addr = words[1] if len(words) >= 2 else pico_zmq_addr
            obs = _run_pico_teleop(env, zmq_addr=addr)
            joints = np.array(obs["state"]["arm_joint_position"], dtype=np.float32)
            if (
                "gripper_position" in obs["state"]
                and len(obs["state"]["gripper_position"]) >= num_arms
            ):
                grips = np.array(
                    obs["state"]["gripper_position"][:num_arms], dtype=np.float32
                )
            _print_status(env, obs, host)
            continue
        if cmd in ("diag", "diagnose", "health"):
            diag_target = getattr(env, "robot", None) or host
            if diag_target is None or not hasattr(diag_target, "diagnose"):
                print("  diagnose: unavailable (dummy mode)")
            else:
                status = diag_target.diagnose()
                print(f"  diagnose: {status.to_dict()}")
            continue
        if cmd in ("recover", "clear", "clear_errors"):
            rec_target = getattr(env, "robot", None) or host
            if rec_target is None or not hasattr(rec_target, "clear_errors"):
                print("  recover: unavailable (dummy mode)")
            else:
                ok = rec_target.clear_errors()
                status = rec_target.diagnose()
                obs = env._get_observation()
                joints = np.array(obs["state"]["arm_joint_position"], dtype=np.float32)
                if host is not None:
                    rstate = host.get_state()
                    waist_q = np.array(rstate.waist_positions, dtype=np.float32)
                    neck_q = np.array(rstate.neck_positions, dtype=np.float32)
                print(f"  recover: ok={ok} | status={status.to_dict()}")
            continue

        target_joints = joints.copy()
        needs_arm_step = False
        gripper_changed = False

        if cmd == "pose" and len(words) >= 2:
            cmd = words[1].lower()
            words = [cmd] + list(words[2:])

        if cmd == "home":
            if host is not None:
                rstate = host.get_state()
                waist_start = np.array(rstate.waist_positions, dtype=np.float32)
                waist_q = np.zeros_like(waist_start)
                if waist_q.size >= 5:
                    waist_q[0] = max(0.01, min(float(waist_start[0]), 0.015))
                    waist_q[1] = max(0.02, min(float(waist_start[1]), 0.03))
                neck_q = np.zeros_like(
                    np.array(rstate.neck_positions, dtype=np.float32)
                )
                host.move_body_smooth(
                    waist=waist_q,
                    neck=neck_q,
                    max_velocity=0.1,
                )
                rstate = host.get_state()
                for idx, side in enumerate(env.config.active_arms):
                    q_meas = (
                        rstate.left_joint_positions
                        if side == "left"
                        else rstate.right_joint_positions
                    )
                    joints[idx * 7 : (idx + 1) * 7] = np.asarray(
                        q_meas, dtype=np.float32
                    )
            for idx, side in enumerate(env.config.active_arms):
                target_joints[idx * 7 : (idx + 1) * 7] = (
                    R2_LEFT_HOME_QPOS if side == "left" else R2_RIGHT_HOME_QPOS
                )
            needs_arm_step = True

        elif cmd == "ready":
            if host is not None:
                rstate = host.get_state()
                for idx, side in enumerate(env.config.active_arms):
                    q_meas = (
                        rstate.left_joint_positions
                        if side == "left"
                        else rstate.right_joint_positions
                    )
                    joints[idx * 7 : (idx + 1) * 7] = np.asarray(
                        q_meas, dtype=np.float32
                    )
            for idx, side in enumerate(env.config.active_arms):
                target_joints[idx * 7 : (idx + 1) * 7] = (
                    R2_LEFT_READY_QPOS if side == "left" else R2_RIGHT_READY_QPOS
                )
            needs_arm_step = True
        elif cmd == "open":
            grips[:] = 1.0
            needs_arm_step = True
            gripper_changed = True
        elif cmd == "close":
            grips[:] = 0.0
            needs_arm_step = True
            gripper_changed = True
        elif cmd in ("l", "left", "r", "right") and len(words) >= 2:
            side = "left" if cmd in ("l", "left") else "right"
            if side not in env.config.active_arms:
                print(f"Arm {side!r} is not active in this environment")
                continue
            arm_idx = env.config.active_arms.index(side)
            sub_op = words[1].lower()
            if sub_op in ("open", "close"):
                grips[arm_idx] = 1.0 if sub_op == "open" else 0.0
                needs_arm_step = True
                gripper_changed = True
            else:
                q_slice = slice(arm_idx * 7, (arm_idx + 1) * 7)
                try:
                    q_updated = _apply_arm_command(
                        side, words[1:], target_joints[q_slice], waist_q, step_rad
                    )
                except ValueError:
                    print("Invalid numeric argument; type help")
                    continue
                if q_updated is None:
                    print(f"Unknown arm command {' '.join(words)!r}; type help")
                    continue
                target_joints[q_slice] = q_updated
                needs_arm_step = True
        elif cmd in ("waist", "w") and len(words) >= 2:
            if host is None:
                print("Waist control requires a connected R2 host")
                continue
            sub_op = words[1].lower()
            mult = float(words[2]) if len(words) >= 3 else 1.0
            delta = float(mult * step_rad)
            is_real_rate = (
                not env.config.is_dummy and float(env.config.step_frequency) < 1000.0
            )
            if is_real_rate:
                waist_q = np.array(host.get_state().waist_positions, dtype=np.float32)
            w_target = waist_q.copy()
            if sub_op == "home":
                if w_target.size >= 5:
                    w_target[0] = max(0.01, min(float(waist_q[0]), 0.015))
                    w_target[1] = max(0.02, min(float(waist_q[1]), 0.03))
                    w_target[2:] = 0.0
                else:
                    w_target[:] = 0.0
            elif sub_op in ("down", "squat") and w_target.size >= 3:
                sq_delta = min(delta, 0.03)
                w_target[0] = max(0.01, float(w_target[0])) + sq_delta
                w_target[1] = max(0.02, float(w_target[1])) + 2.0 * sq_delta
                w_target[2] += sq_delta
            elif sub_op in ("up", "stand") and w_target.size >= 3:
                sq_delta = min(delta, 0.03)
                w_target[0] = max(0.01, float(w_target[0] - sq_delta))
                w_target[1] = max(0.02, float(w_target[1] - 2.0 * sq_delta))
                w_target[2] = max(0.0, float(w_target[2] - sq_delta))
            elif sub_op in ("fwd", "forward", "bow"):
                idx_p = 2 if w_target.size >= 5 else 1
                w_target[idx_p] -= delta
            elif sub_op in ("back", "backward"):
                idx_p = 2 if w_target.size >= 5 else 1
                w_target[idx_p] += delta
            elif sub_op == "left":
                idx_r = 3 if w_target.size >= 5 else 2
                w_target[idx_r] -= delta
            elif sub_op == "right":
                idx_r = 3 if w_target.size >= 5 else 2
                w_target[idx_r] += delta
            elif sub_op == "turn_left":
                w_target[-1] += delta
            elif sub_op == "turn_right":
                w_target[-1] -= delta
            else:
                try:
                    w_idx = int(sub_op) - 1
                except ValueError:
                    print(f"Unknown waist command {sub_op!r}; type help")
                    continue
                if not (0 <= w_idx < w_target.size):
                    print(f"Waist joint index must be 1..{w_target.size}")
                    continue
                w_target[w_idx] += delta

            if w_target.size == 5:
                w_target = np.clip(w_target, R2_WAIST_LIMITS_LOW, R2_WAIST_LIMITS_HIGH)
            host.move_body_smooth(
                waist=w_target.astype(np.float32),
                max_velocity=0.1,
            )
            waist_q = w_target.astype(np.float32)
            obs = env._get_observation()
            _print_status(env, obs, host)
            continue
        elif cmd in ("head", "neck", "h") and len(words) >= 2:
            if host is None:
                print("Head/neck control requires a connected R2 host")
                continue
            sub_op = words[1].lower()
            mult = float(words[2]) if len(words) >= 3 else 1.0
            delta = float(mult * step_rad)
            is_real_rate = (
                not env.config.is_dummy and float(env.config.step_frequency) < 1000.0
            )
            if is_real_rate:
                neck_q = np.array(host.get_state().neck_positions, dtype=np.float32)
            n_target = neck_q.copy()
            if sub_op == "home":
                n_target[:] = 0.0
            elif sub_op == "left":
                n_target[0] += delta
            elif sub_op == "right":
                n_target[0] -= delta
            elif sub_op == "up":
                n_target[1] += delta
            elif sub_op == "down":
                n_target[1] -= delta
            else:
                print(f"Unknown head command {sub_op!r}; type help")
                continue
            n_target = np.clip(n_target, R2_NECK_LIMITS_LOW, R2_NECK_LIMITS_HIGH)
            host.move_body_smooth(
                neck=n_target.astype(np.float32),
                max_velocity=0.15,
            )
            neck_q = n_target.astype(np.float32)
            obs = env._get_observation()
            _print_status(env, obs, host)
            continue
        elif cmd in ("base", "b") and len(words) >= 2:
            if host is None:
                print("Base control requires a connected R2 host")
                continue
            sub_op = words[1].lower()
            mult = float(words[2]) if len(words) >= 3 else 1.0
            speed_scale = float(words[3]) if len(words) >= 4 else 1.0
            base_dirs: dict[str, np.ndarray] = {
                "fwd": np.array([0.15, 0.0, 0.0], dtype=np.float32),
                "forward": np.array([0.15, 0.0, 0.0], dtype=np.float32),
                "back": np.array([-0.15, 0.0, 0.0], dtype=np.float32),
                "backward": np.array([-0.15, 0.0, 0.0], dtype=np.float32),
                "left": np.array([0.0, 0.15, 0.0], dtype=np.float32),
                "right": np.array([0.0, -0.15, 0.0], dtype=np.float32),
                "turn_left": np.array([0.0, 0.0, 0.25], dtype=np.float32),
                "turn_right": np.array([0.0, 0.0, -0.25], dtype=np.float32),
            }
            if sub_op == "stop":
                host.stop_base()
                obs = env._get_observation()
                _print_status(env, obs, host)
                continue
            if sub_op not in base_dirs:
                print(f"Unknown base command {sub_op!r}; type help")
                continue
            sign = -1.0 if mult < 0.0 else 1.0
            abs_mult = abs(mult)
            if abs_mult >= 1.0:
                vel = base_dirs[sub_op] * float(sign * speed_scale)
                duration = 1.0 * abs_mult
            else:
                vel = base_dirs[sub_op] * float(mult * speed_scale)
                duration = 1.0
            pulse_info = host.move_base_pulse(
                vel,
                duration=duration,
                rate_hz=max(float(env.config.step_frequency), 50.0),
            )
            obs = env._get_observation()
            print(
                f"  base pulse ({duration:.2f}s): vel=",
                _fmt_vec(pulse_info["base_velocity"], 3),
                "| base pose:",
                _fmt_vec(pulse_info["base_pose"], 3),
            )
            continue
        else:
            print(f"Unknown command {cmd!r}; type help")
            continue

        if needs_arm_step:
            obs = _step_arm_smooth(
                env,
                joints,
                target_joints,
                grips,
                step_rad,
                gripper_changed=gripper_changed,
            )
            joints = (
                target_joints.copy()
                if env.config.is_dummy
                else np.array(obs["state"]["arm_joint_position"], dtype=np.float32)
            )
            _print_status(env, obs, host)


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
        dest="enable_base",
        action="store_true",
        default=True,
        help="include the omnidirectional wheeled chassis (enabled by default)",
    )
    parser.add_argument(
        "--no-base",
        dest="enable_base",
        action="store_false",
        help="disable the omnidirectional wheeled chassis",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=["head", "wrist_left", "wrist_right"],
        help="R2 camera names to subscribe to (default: head wrist_left wrist_right)",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="disable camera subscriptions",
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
        default=None,
        help="number of batch steps when running non-interactively (default 5)",
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
        "--reset",
        dest="reset",
        action="store_true",
        default=False,
        help="reset active arms to rest configuration before starting (disabled by default)",
    )
    parser.add_argument(
        "--no-reset",
        dest="reset",
        action="store_false",
        help="start from current arm pose instead of resetting first (default)",
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="only connect and print observation without sending joint commands",
    )
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=True,
        help="enter interactive command prompt (enabled by default)",
    )
    parser.add_argument(
        "--no-interactive",
        "--batch",
        dest="interactive",
        action="store_false",
        help="run batch steps instead of entering the interactive prompt",
    )
    parser.add_argument(
        "--commands",
        type=str,
        default=None,
        help="semicolon-separated interactive commands to execute in sequence",
    )
    parser.add_argument(
        "--enable-camera-player",
        action="store_true",
        help="enable the camera preview window (disabled by default)",
    )
    parser.add_argument(
        "--pico",
        action="store_true",
        help="start directly in PICO VR teleoperation mode",
    )
    parser.add_argument(
        "--pico-zmq-addr",
        type=str,
        default=None,
        help="ZeroMQ endpoint for PICO VR stream (default: R2_PICO_ZMQ_ADDR or ipc:///tmp/vr_data.ipc)",
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
        from rlinf.robotics import R2Config, RobotInfo

        enable_base = bool(args.enable_base)
        if args.no_cameras or (
            len(args.cameras) == 1 and args.cameras[0].lower() == "none"
        ):
            camera_list: list[str] = []
        elif len(args.cameras) == 0:
            camera_list = (
                []
                if args.pico and not args.enable_camera_player
                else ["head", "wrist_left", "wrist_right"]
            )
        else:
            camera_list = list(args.cameras)
        cfg = R2Config(
            body_dof=None if args.body_dof == "auto" else int(args.body_dof),
            end_effector_type=(
                None if args.end_effector == "auto" else args.end_effector
            ),
            enable_base=enable_base,
            cameras=camera_list,
            use_compressed_camera=not args.mock,
        )
        robot_info = RobotInfo(
            type="R2",
            model=cfg.hardware_model("R2"),
            config=cfg,
        )

        env = R2ReachEnv(
            {
                "enable_camera_player": args.enable_camera_player,
                "step_frequency": args.fps,
                "reset_joint_speed": float(np.deg2rad(args.reset_speed)),
                "reset_on_init": False,
            },
            robot_info=robot_info,
        )
        resolved_cfg = env.hardware
        print(
            f"R2 hardware config: body_dof={resolved_cfg.body_dof}, "
            f"end_effector_type={resolved_cfg.end_effector_type!r}, "
            f"enable_base={resolved_cfg.enable_base}, "
            f"cameras={resolved_cfg.cameras}"
        )
        try:
            if args.pico:
                if args.reset:
                    env.reset()
                obs = _run_pico_teleop(
                    env,
                    zmq_addr=args.pico_zmq_addr,
                    max_steps=args.steps,
                )
                _print_status(env, obs, _get_connection_host(env))
            else:
                use_batch = (
                    args.no_move
                    or (args.steps is not None and args.commands is None)
                    or not args.interactive
                )
                if not use_batch or args.commands is not None:
                    parsed_cmds = (
                        [c.strip() for c in args.commands.split(";") if c.strip()]
                        if args.commands is not None
                        else None
                    )
                    _run_interactive(
                        env,
                        step_rad=float(np.deg2rad(args.step)),
                        reset=args.reset,
                        commands=parsed_cmds,
                        pico_zmq_addr=args.pico_zmq_addr,
                    )
                else:
                    _run_batch_steps(
                        env,
                        steps=5 if args.steps is None else args.steps,
                        delta_rad=float(np.deg2rad(args.step)),
                        reset=args.reset,
                        no_move=args.no_move,
                    )
        finally:
            with _deferred_interrupts():
                env.close()


if __name__ == "__main__":
    main()
