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

"""Dual-arm GELLO intervention wrapper for joint-space control.

Step-gated (forwarded via env.step) or direct-stream (daemon pushes
targets at ~1 kHz, bypassing env.step's rate gate).
"""

from __future__ import annotations

import threading
import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.gello.gello_joint_expert import GelloJointExpert


class DualGelloJointIntervention(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        left_port: str,
        right_port: str,
        gripper_enabled: bool = True,
        use_delta: bool = False,
        action_scale: float = 0.1,
        direct_stream: bool = False,
        stream_period: float = 0.001,
    ):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled
        self.use_delta = use_delta
        self.action_scale = action_scale
        self.left_expert = GelloJointExpert(port=left_port)
        self.right_expert = GelloJointExpert(port=right_port)
        self.last_intervene = 0.0

        self._direct_stream = direct_stream
        self._stream_period = stream_period
        self._stream_thread: threading.Thread | None = None
        self._stream_running = False
        self._stream_last_gripper_open: list[bool | None] = [None, None]
        self._stream_gate = threading.Event()
        self._stream_gate.set()  # gate open = stream tick allowed
        self._aligned = False

    def _start_stream_thread(self) -> None:
        if self._resolve_controllers() == (None, None):
            return
        self._stream_running = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="DualGelloJointStream",
            daemon=True,
        )
        self._stream_thread.start()



    def _stream_loop(self) -> None:
        # Gripper events are edge-triggered: open/close RPCs are ~100 ms
        # and streaming them at 1 kHz would starve the serial channel.
        left_ctrl, right_ctrl = self._resolve_controllers()
        if left_ctrl is None or right_ctrl is None:
            return

        period = self._stream_period
        ctrls = (left_ctrl, right_ctrl)

        while self._stream_running:
            self._stream_gate.wait()
            if not self._stream_running:
                break

            loop_start = time.time()

            if not (self.left_expert.ready and self.right_expert.ready):
                time.sleep(period)
                continue

            left_q, left_g = self.left_expert.get_action()
            right_q, right_g = self.right_expert.get_action()

            lf = left_ctrl.move_joints(left_q.astype(np.float32))
            rf = right_ctrl.move_joints(right_q.astype(np.float32))
            lf.wait()
            rf.wait()

            if self.gripper_enabled:
                for arm_idx, (ctrl, grip) in enumerate(zip(ctrls, (left_g, right_g))):
                    is_open_now = grip.item() < 0.5
                    prev = self._stream_last_gripper_open[arm_idx]
                    if prev is None:
                        self._stream_last_gripper_open[arm_idx] = is_open_now
                    elif is_open_now != prev:
                        if is_open_now:
                            ctrl.open_gripper()
                        else:
                            ctrl.close_gripper()
                        self._stream_last_gripper_open[arm_idx] = is_open_now

            elapsed = time.time() - loop_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _get_current_joint_positions(self) -> np.ndarray:
        return self.get_wrapper_attr("get_joint_positions")()

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not (self.left_expert.ready and self.right_expert.ready):
            return action, False

        left_q, left_g = self.left_expert.get_action()
        right_q, right_g = self.right_expert.get_action()
        current = self._get_current_joint_positions()  # (2, 7)

        per_arm = []
        for target_q, current_q in zip((left_q, right_q), (current[0], current[1])):
            if self.use_delta:
                delta_q = (target_q - current_q) / self.action_scale
                arm_a = np.clip(delta_q, -1.0, 1.0)
            else:
                arm_a = target_q.copy()
            per_arm.append(arm_a)

        gripper_active = False
        if self.gripper_enabled:
            grippers = []
            for grip in (left_g, right_g):
                g = -(2 * grip - 1.0)
                g = np.clip(g, -1.0, 1.0)
                grippers.append(g)
                if np.abs(g).item() > 0.5:
                    gripper_active = True
            expert_a = np.concatenate(
                [per_arm[0], grippers[0], per_arm[1], grippers[1]], axis=0
            )
        else:
            expert_a = np.concatenate(per_arm, axis=0)

        movement = np.linalg.norm(
            np.concatenate([left_q, right_q]) - np.concatenate([current[0], current[1]])
        )
        if movement > 0.01 or gripper_active:
            self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True
        return action, False

    def _align_to_gello(self) -> bool:
        # Without this, direct-stream's first loop would push a far target
        # straight into the impedance tracker, causing a reference jump.
        self._aligned = False
        if not (self.left_expert.ready and self.right_expert.ready):
            return False
        left_ctrl, right_ctrl = self._resolve_controllers()
        if left_ctrl is None or right_ctrl is None:
            return False
        left_q, _ = self.left_expert.get_action()
        right_q, _ = self.right_expert.get_action()
        lf = left_ctrl.reset_joint(np.asarray(left_q, dtype=np.float64).tolist())
        rf = right_ctrl.reset_joint(np.asarray(right_q, dtype=np.float64).tolist())
        lf.wait()
        rf.wait()
        setattr(self.unwrapped, "_left_state", left_ctrl.get_state().wait()[0])
        setattr(self.unwrapped, "_right_state", right_ctrl.get_state().wait()[0])
        self._aligned = True
        return True

    def reset(self, **kwargs):
        # Skip the inner env's home slew: aligning to GELLO directly avoids
        # a "home → GELLO" double-slew that breaks tracking continuity.
        options = dict(kwargs.get("options") or {})
        options.setdefault("skip_reset_to_home", True)
        kwargs["options"] = options

        self._stream_gate.clear()
        try:
            result = self.env.reset(**kwargs)
            if self._direct_stream:
                self._align_to_gello()
        finally:
            self._stream_gate.set()
            if self._direct_stream and self._aligned and self._stream_thread is None:
                self._start_stream_thread()
        return result

    def step(self, action):
        new_action, replaced = self.action(action)
        if self._direct_stream and self._aligned and self._stream_thread is None:
            self._start_stream_thread()
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1)
        return obs, rew, done, truncated, info

    def close(self):
        self._stream_running = False
        t = self._stream_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        return super().close()
